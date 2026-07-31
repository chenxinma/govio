"""Ladybug CSV 导入工具 - 全量重建与增量 upsert。

与 falkordb_loader 对应，复用同一批 FalkorDB 批量导入约定的 CSV 文件
（节点首列 `:ID(Label)`，边首两列 `:START_ID(Src)` / `:END_ID(Dst)`）。

策略：
- 全量重建（import_csv_to_ladybug）：先删除所有表（先 REL 后 NODE），
  再建表 + COPY FROM DataFrame（ignore_errors 兜底）。
- 增量更新（upsert_csv_to_ladybug）：CREATE IF NOT EXISTS + 批量
  UNWIND $rows MERGE（dict 行，反引号字段访问，规避保留字）。

Ladybug 严格类型且不可忽略类型转换错误，故所有属性列统一为 STRING，
与 falkordb_loader 的 dtype=str 处理一致。
"""

from os import PathLike
from pathlib import Path

import ladybug as lb
import pandas as pd

_DEFAULT_BUFFER_POOL = 256 * 1024 * 1024
_DEFAULT_MAX_DB = 1 * 1024 * 1024 * 1024
_BATCH_SIZE = 500

# 与 falkordb_loader 一致的 CSV -> 节点标签映射
_NODE_CSVS = {
    "PhysicalTable.csv": "PhysicalTable",
    "Col.csv": "Col",
    "Application.csv": "Application",
    "Standard.csv": "Standard",
    "Metric.csv": "Metric",
    "Dimension.csv": "Dimension",
}

# 与 falkordb_loader 一致的 CSV -> 关系类型映射
_EDGE_CSVS = {
    "HAS_COLUMN.csv": "HAS_COLUMN",
    "USE.csv": "USE",
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
    """
    df = pd.read_csv(filepath, dtype=str).fillna("")
    cols = list(df.columns)
    label = cols[0].removeprefix(":ID(").removesuffix(")")
    prop_cols = cols[1:]
    df = df.rename(columns={cols[0]: "id"})
    return label, prop_cols, df


def _edge_spec(filepath: Path) -> tuple[str, str, list[str], pd.DataFrame]:
    """解析边 CSV：返回 (src_label, dst_label, 属性列, df)。

    首列 `:START_ID(Src)` 改名 `from`，次列 `:END_ID(Dst)` 改名 `to`，其余为属性。
    """
    df = pd.read_csv(filepath, dtype=str).fillna("")
    cols = list(df.columns)
    src = cols[0].removeprefix(":START_ID(").removesuffix(")")
    dst = cols[1].removeprefix(":END_ID(").removesuffix(")")
    prop_cols = cols[2:]
    df = df.rename(columns={cols[0]: "from", cols[1]: "to"})
    return src, dst, prop_cols, df


def _node_ddl(label: str, prop_cols: list[str]) -> str:
    cols = [_bt("id") + " STRING PRIMARY KEY"]
    cols += [_bt(c) + " STRING" for c in prop_cols]
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
        try:
            conn.execute(f"DROP TABLE {_bt(name)}")
        except Exception:
            pass


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
        for filename, _ in _NODE_CSVS.items():
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
        batch = rows[start:start + _BATCH_SIZE]
        conn.execute(query, {"rows": batch})


def _node_merge(label: str, prop_cols: list[str]) -> str:
    merge = f"MERGE (n:{_bt(label)} {{{_bt('id')}: row.{_bt('id')}}})"
    if not prop_cols:
        return f"UNWIND $rows AS row {merge}"
    sets = ", ".join(f"n.{_bt(c)} = row.{_bt(c)}" for c in prop_cols)
    return f"UNWIND $rows AS row {merge} SET {sets}"


def _edge_merge(
    rel_type: str, src: str, dst: str, prop_cols: list[str]
) -> str:
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
    """
    csv_path = Path(csv_dir)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 目录不存在: {csv_path}")

    print(f"\n正在增量更新 Ladybug 图 ({db_path})...")
    conn = _connect(db_path, buffer_pool_size, max_db_size)
    try:
        # 节点
        for filename, _ in _NODE_CSVS.items():
            filepath = csv_path / filename
            if not filepath.exists():
                continue
            label, prop_cols, df = _node_spec(filepath)
            conn.execute(_node_ddl(label, prop_cols))
            if df.empty:
                continue
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
            df = df.rename(columns={"from": "src", "to": "dst"})
            query = _edge_merge(rel_type, src, dst, prop_cols)
            _batched_execute(conn, query, df.to_dict("records"))
            print(f"  {filename}: {len(df)} 行 -> {rel_type}")
    finally:
        conn.close()

    print("Ladybug 增量更新完成")
