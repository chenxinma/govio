"""Ladybug CSV 导入工具 - 全量重建与增量 upsert。

与 falkordb_loader 对应，复用同一批 FalkorDB 批量导入约定的 CSV 文件
（节点首列 `:ID(Label)`，边首两列 `:START_ID(Src)` / `:END_ID(Dst)`）。

策略：
- 全量重建（import_csv_to_ladybug）：先删除所有表（先 REL 后 NODE），
  再建表 + COPY FROM DataFrame（ignore_errors 兜底）。
- 增量更新（upsert_csv_to_ladybug）：CREATE IF NOT EXISTS + 批量
  UNWIND $rows MERGE（dict 行，反引号字段访问，规避保留字）。

Ladybug 严格类型且不可忽略类型转换错误，故属性列默认统一为 STRING
（与 falkordb_loader 的 dtype=str 处理一致）；数值型属性
（size/precision/scale/order_no）显式建成 INT64 并在读 CSV 时转数值。

兼容性：边表表头必须含 :START_ID(...) / :END_ID(...)（位置不强制，自动重排），
全空杂散列丢弃并告警；upsert 遇旧版表结构（缺属性列）时跳过写入缺失列并
给出 --mode rebuild 迁移指引，不透出 Binder 裸异常。
"""

import contextlib
import sys
from os import PathLike
from pathlib import Path

import ladybug as lb
import pandas as pd

# 数值型属性列（TDS 导入的 Col/Standard 属性），入图保持 INT64，
# 否则 ORDER BY / 数值比较按字符串语义（"10" < "2"）
_NUMERIC_COLS = {"size", "precision", "scale", "order_no"}

_DEFAULT_BUFFER_POOL = 256 * 1024 * 1024
_DEFAULT_MAX_DB = 1 * 1024 * 1024 * 1024
_BATCH_SIZE = 500

# 与 falkordb_loader 一致的 CSV -> 节点标签映射
_NODE_CSVS = {
    "PhysicalTable.csv": "PhysicalTable",
    "Col.csv": "Col",
    "Datasource.csv": "Datasource",
    "Standard.csv": "Standard",
    "Metric.csv": "Metric",
    "Dimension.csv": "Dimension",
}

# 与 falkordb_loader 一致的 CSV -> 关系类型映射
_EDGE_CSVS = {
    "HAS_COLUMN.csv": "HAS_COLUMN",
    "OWNS.csv": "OWNS",
    "COMPLIES_WITH.csv": "COMPLIES_WITH",
    "RELATES_TO.csv": "RELATES_TO",
    "USES_TABLE.csv": "USES_TABLE",
    "REFERS_COLUMN.csv": "REFERS_COLUMN",
    "DERIVED_FROM.csv": "DERIVED_FROM",
    "DIMENSION_USED.csv": "DIMENSION_USED",
    "SUPERSEDES.csv": "SUPERSEDES",
}


def _bt(name: str) -> str:
    """反引号包裹标识符。"""
    return f"`{name}`"


def _lit(name: str) -> str:
    """单引号字符串字面量（TABLE_INFO 参数），内部单引号转义。"""
    return "'" + name.replace("'", "''") + "'"


def _is_all_empty(series: pd.Series) -> bool:
    """列值全为空（NaN 或空白字符串）判断。"""
    filled = series.fillna("").astype(str).str.strip()
    return bool(filled.eq("").all())


def _connect(
    db_path: str | PathLike,
    buffer_pool_size: int = _DEFAULT_BUFFER_POOL,
    max_db_size: int = _DEFAULT_MAX_DB,
) -> lb.Connection:
    db = lb.Database(
        db_path,
        buffer_pool_size=buffer_pool_size,
        max_db_size=max_db_size,
    )
    return lb.Connection(db)


# ---------------------------------------------------------------------------
# CSV -> schema 解析
# ---------------------------------------------------------------------------


def _node_spec(filepath: Path) -> tuple[str, list[str], pd.DataFrame]:
    """解析节点 CSV：返回 (label, 属性列, df)。

    首列 `:ID(Label)` 改名为 `id`（作为主键），其余列为属性。
    `:ID(...)` 列不在首位时重排兜底；缺失时给出可读错误。
    """
    df = pd.read_csv(filepath, dtype=str).fillna("")
    cols = list(df.columns)
    id_cols = [c for c in cols if c.startswith(":ID(") and c.endswith(")")]
    if not id_cols:
        raise ValueError(
            f"节点表表头不符合规范: {filepath.name} 应以 :ID(NodeType) 开头，"
            f"实际列: {cols}；请重新运行 meta 各导出步骤生成规范 CSV"
        )
    id_col = id_cols[0]
    label = id_col.removeprefix(":ID(").removesuffix(")")
    prop_cols = [c for c in cols if c != id_col]
    df = df[[id_col, *prop_cols]].rename(columns={id_col: "id"})
    for col in prop_cols:
        if col in _NUMERIC_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")
    return label, prop_cols, df


def _edge_spec(filepath: Path) -> tuple[str, str, list[str], pd.DataFrame]:
    """解析边 CSV：返回 (src_label, dst_label, 属性列, df)。

    端点列 `:START_ID(Src)` / `:END_ID(Dst)` 改名 `from` / `to`，其余为属性。
    端点列不在最前时重排兕底（历史产物可能带 source,target 等杂散列）；
    全空杂散列直接丢弃并告警；找不到端点列时给出可读错误，
    不再透出 Ladybug Binder 裸异常。
    """
    df = pd.read_csv(filepath, dtype=str).fillna("")
    cols = list(df.columns)
    start_cols = [c for c in cols if c.startswith(":START_ID(") and c.endswith(")")]
    end_cols = [c for c in cols if c.startswith(":END_ID(") and c.endswith(")")]
    if not start_cols or not end_cols:
        raise ValueError(
            f"边表表头不符合规范: {filepath.name} 应以 :START_ID(...)、:END_ID(...) 开头，"
            f"实际列: {cols}；请重新运行 meta rel/meta meta 生成规范 CSV，"
            "或手动把端点列移到表头最前"
        )
    start_col, end_col = start_cols[0], end_cols[0]
    prop_cols = [c for c in cols if c not in (start_col, end_col)]
    stray = [c for c in prop_cols if _is_all_empty(df[c])]
    if stray:
        print(
            f"⚠ {filepath.name}: 杂散/空列 {stray} 已丢弃（边表属性列不允许全空杂散列）",
            file=sys.stderr,
        )
        prop_cols = [c for c in prop_cols if c not in stray]
    df = df[[start_col, end_col, *prop_cols]]
    src = start_col.removeprefix(":START_ID(").removesuffix(")")
    dst = end_col.removeprefix(":END_ID(").removesuffix(")")
    df = df.rename(columns={start_col: "from", end_col: "to"})
    return src, dst, prop_cols, df


def _node_ddl(label: str, prop_cols: list[str]) -> str:
    cols = [_bt("id") + " STRING PRIMARY KEY"]
    cols += [
        _bt(c) + (" INT64" if c in _NUMERIC_COLS else " STRING") for c in prop_cols
    ]
    return f"CREATE NODE TABLE IF NOT EXISTS {_bt(label)} ({', '.join(cols)})"


def _edge_ddl(rel_type: str, src: str, dst: str, prop_cols: list[str]) -> str:
    cols = [f"FROM {_bt(src)} TO {_bt(dst)}"]
    cols += [_bt(c) + " STRING" for c in prop_cols]
    return f"CREATE REL TABLE IF NOT EXISTS {_bt(rel_type)} ({', '.join(cols)})"


def _drop_all_tables(conn: lb.Connection) -> None:
    """删除所有表：先 REL 后 NODE（外键约束）。"""
    res = conn.execute("CALL SHOW_TABLES() RETURN *")
    rels: list[str] = []
    nodes: list[str] = []
    for row in res.get_all():
        # [id, name, type, database name, comment]
        name, t = row[1], row[2]
        if t == "REL":
            rels.append(name)
        elif t == "NODE":
            nodes.append(name)
    for name in rels + nodes:
        with contextlib.suppress(Exception):
            conn.execute(f"DROP TABLE {_bt(name)}")


# ---------------------------------------------------------------------------
# 全量重建
# ---------------------------------------------------------------------------


def import_csv_to_ladybug(
    csv_dir: Path | str,
    db_path: str | PathLike,
    *,
    buffer_pool_size: int = _DEFAULT_BUFFER_POOL,
    max_db_size: int = _DEFAULT_MAX_DB,
) -> None:
    """全量重建：清空所有表后建表并 COPY 导入（节点先于边）。"""
    csv_path = Path(csv_dir)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 目录不存在: {csv_path}")

    print(f"\n正在重建 Ladybug 图 ({db_path})...")
    conn = _connect(db_path, buffer_pool_size, max_db_size)
    try:
        _drop_all_tables(conn)

        # 节点
        for filename in _NODE_CSVS:
            filepath = csv_path / filename
            if not filepath.exists():
                continue
            label, prop_cols, df = _node_spec(filepath)
            conn.execute(_node_ddl(label, prop_cols))
            if not df.empty:
                conn.execute(
                    f"COPY {_bt(label)} FROM $df (ignore_errors=true)",
                    {"df": df},
                )
            print(f"  {filename}: {len(df)} 行 -> {label}")

        # 边
        for filename, rel_type in _EDGE_CSVS.items():
            filepath = csv_path / filename
            if not filepath.exists():
                continue
            src, dst, prop_cols, df = _edge_spec(filepath)
            conn.execute(_edge_ddl(rel_type, src, dst, prop_cols))
            if not df.empty:
                conn.execute(
                    f"COPY {_bt(rel_type)} FROM $df (ignore_errors=true)",
                    {"df": df},
                )
            print(f"  {filename}: {len(df)} 行 -> {rel_type}")
    finally:
        conn.close()

    print("Ladybug 图已重建")


# ---------------------------------------------------------------------------
# 增量 upsert
# ---------------------------------------------------------------------------


def _batched_execute(conn: lb.Connection, query: str, rows: list[dict]) -> None:
    """按 _BATCH_SIZE 批量执行 UNWIND $rows 查询。"""
    for start in range(0, len(rows), _BATCH_SIZE):
        batch = rows[start : start + _BATCH_SIZE]
        conn.execute(query, {"rows": batch})


def _existing_prop_cols(conn: lb.Connection, label: str) -> set[str] | None:
    """读取已存在表的属性列名集合；表不存在返回 None。

    节点表 TABLE_INFO 返回 [序号, 列名, 类型, 默认值, 是否主键]（含 id 主键），
    边表只返回属性列（不含 FROM/TO 端点）。
    """
    try:
        rows = conn.execute(f"CALL TABLE_INFO({_lit(label)}) RETURN *").get_all()
    except Exception:
        return None
    return {r[1] for r in rows}


def _filter_writable_props(
    conn: lb.Connection,
    label: str,
    prop_cols: list[str],
) -> list[str]:
    """过滤掉目标表 schema 中不存在的属性列（旧版表结构兼容）。

    Ladybug 严格 schema 且不支持 ALTER TABLE 加列，对缺列直接 SET 会报
    "Cannot find property ... for n" 裸异常；这里改为跳过写入并给出可读
    迁移指引（--mode rebuild）。
    """
    existing = _existing_prop_cols(conn, label)
    if existing is None:
        return prop_cols
    missing = [c for c in prop_cols if c not in existing]
    if missing:
        writable = [c for c in prop_cols if c in existing]
        print(
            f"⚠ 图库表 {label} 为旧版表结构，缺少属性列 {missing}，"
            "本次已跳过这些属性的写入。"
            "完整迁移请使用 govio-cli meta graph --mode rebuild 全量重建"
            "（Ladybug 不支持 ALTER TABLE 就地加列）",
            file=sys.stderr,
        )
        return writable
    return prop_cols


def _node_merge(label: str, prop_cols: list[str]) -> str:
    merge = f"MERGE (n:{_bt(label)} {{{_bt('id')}: row.{_bt('id')}}})"
    if not prop_cols:
        return f"UNWIND $rows AS row {merge}"
    sets = ", ".join(f"n.{_bt(c)} = row.{_bt(c)}" for c in prop_cols)
    return f"UNWIND $rows AS row {merge} SET {sets}"


def _edge_merge(rel_type: str, src: str, dst: str, prop_cols: list[str]) -> str:
    base = (
        f"UNWIND $rows AS row "
        f"MATCH (a:{_bt(src)} {{{_bt('id')}: row.{_bt('src')}}}), "
        f"(b:{_bt(dst)} {{{_bt('id')}: row.{_bt('dst')}}}) "
        f"MERGE (a)-[r:{_bt(rel_type)}]->(b)"
    )
    if not prop_cols:
        return base
    sets = ", ".join(f"r.{_bt(c)} = row.{_bt(c)}" for c in prop_cols)
    return f"{base} SET {sets}"


def upsert_csv_to_ladybug(
    csv_dir: Path | str,
    db_path: str | PathLike,
    *,
    buffer_pool_size: int = _DEFAULT_BUFFER_POOL,
    max_db_size: int = _DEFAULT_MAX_DB,
) -> None:
    """增量更新：CREATE IF NOT EXISTS + 批量 MERGE（节点先于边）。

    节点按主键 id MERGE 并 SET 属性；边按 src/dst 端点 MERGE。
    边的 from/to 列在 dict 中改名为 src/dst（规避保留字）。
    目标表为旧版表结构（缺属性列）时跳过写入缺失列并告警，不报裸异常。
    """
    csv_path = Path(csv_dir)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 目录不存在: {csv_path}")

    print(f"\n正在增量更新 Ladybug 图 ({db_path})...")
    conn = _connect(db_path, buffer_pool_size, max_db_size)
    try:
        # 节点
        for filename in _NODE_CSVS:
            filepath = csv_path / filename
            if not filepath.exists():
                continue
            label, prop_cols, df = _node_spec(filepath)
            conn.execute(_node_ddl(label, prop_cols))
            if df.empty:
                continue
            prop_cols = _filter_writable_props(conn, label, prop_cols)
            query = _node_merge(label, prop_cols)
            _batched_execute(conn, query, df.to_dict("records"))
            print(f"  {filename}: {len(df)} 行 -> {label}")

        # 边：from/to 改名 src/dst 以规避保留字
        for filename, rel_type in _EDGE_CSVS.items():
            filepath = csv_path / filename
            if not filepath.exists():
                continue
            src, dst, prop_cols, df = _edge_spec(filepath)
            conn.execute(_edge_ddl(rel_type, src, dst, prop_cols))
            if df.empty:
                continue
            prop_cols = _filter_writable_props(conn, rel_type, prop_cols)
            df = df.rename(columns={"from": "src", "to": "dst"})
            query = _edge_merge(rel_type, src, dst, prop_cols)
            _batched_execute(conn, query, df.to_dict("records"))
            print(f"  {filename}: {len(df)} 行 -> {rel_type}")
    finally:
        conn.close()

    print("Ladybug 增量更新完成")
