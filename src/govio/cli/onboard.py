import csv
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import questionary

from .config import ConfigManager


# ---------------------------------------------------------------------------
# CSV validation helper (still used internally)
# ---------------------------------------------------------------------------

def validate_csv_directory(csv_dir: Path) -> bool:
    """验证 CSV 目录是否包含必需的文件

    Args:
        csv_dir: CSV 目录路径

    Returns:
        bool: 是否有效
    """
    if not csv_dir.exists() or not csv_dir.is_dir():
        return False

    required_files = ["PhysicalTable.csv"]

    for filename in required_files:
        if not (csv_dir / filename).exists():
            return False

    return True


# ---------------------------------------------------------------------------
# FalkorDB import helpers (imported by meta_export.py — DO NOT REMOVE)
# ---------------------------------------------------------------------------

def delete_falkordb_graph(host: str, port: int, graph_name: str) -> None:
    """删除 FalkorDB 中的图（如果存在）"""
    import falkordb

    try:
        client = falkordb.FalkorDB(host=host, port=port)
        client.execute_command("DEL", graph_name)
        print(f"已删除现有图: {graph_name}")
    except Exception:
        pass


def import_csv_to_falkordb(
    csv_dir: Path, host: str, port: int, graph_name: str
) -> None:
    """使用 falkordb-bulk-insert 将 CSV 导入 FalkorDB

    通过 Python API 调用，避免 Windows 上 CLI 入口点损坏的问题。

    Args:
        csv_dir: CSV 文件目录
        host: FalkorDB 主机地址
        port: FalkorDB 端口
        graph_name: 图数据库名称
    """
    csv_path = Path(csv_dir)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 目录不存在: {csv_path}")

    print("\n正在检查并清理已有数据...")
    delete_falkordb_graph(host, port, graph_name)

    node_files = [
        ("PhysicalTable", csv_path / "PhysicalTable.csv"),
        ("Col", csv_path / "Col.csv"),
        ("Application", csv_path / "Application.csv"),
        ("Standard", csv_path / "Standard.csv"),
        ("Metric", csv_path / "Metric.csv"),
        ("Dimension", csv_path / "Dimension.csv"),
    ]

    relation_files = [
        ("HAS_COLUMN", csv_path / "HAS_COLUMN.csv"),
        ("USE", csv_path / "USE.csv"),
    ]

    extra_rel_file = csv_path / "RELATES_TO.csv"
    if extra_rel_file.exists():
        relation_files.append(("RELATES_TO", extra_rel_file))

    # 指标相关的边文件
    metric_rel_files = [
        ("USES_TABLE", csv_path / "USES_TABLE.csv"),
        ("REFERS_COLUMN", csv_path / "REFERS_COLUMN.csv"),
        ("DERIVED_FROM", csv_path / "DERIVED_FROM.csv"),
        ("DIMENSION_USED", csv_path / "DIMENSION_USED.csv"),
        ("SUPERSEDES", csv_path / "SUPERSEDES.csv"),
    ]
    for rel_type, filepath in metric_rel_files:
        if filepath.exists():
            relation_files.append((rel_type, filepath))

    cmd = [sys.executable, "-m", "falkordb_bulk_loader.bulk_insert", graph_name]

    for label, filepath in node_files:
        if filepath.exists():
            cmd.extend(["--nodes-with-label", label, str(filepath)])

    for rel_type, filepath in relation_files:
        if filepath.exists():
            cmd.extend(["--relations-with-type", rel_type, str(filepath)])

    server_url = f"redis://{host}:{port}"
    cmd.extend(["--server-url", server_url])

    print(f"\n正在执行: {' '.join(cmd)}")

    env = {**os.environ, "PYTHONUTF8": "1"}
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    if result.returncode != 0:
        raise RuntimeError(f"falkordb-bulk-insert 失败: {result.stderr}")

    print(result.stdout)
    print("CSV 数据已导入 FalkorDB")


# ---------------------------------------------------------------------------
# Incremental upsert via MERGE semantics (imported by meta_export.py)
# ---------------------------------------------------------------------------

# Known node CSV files -> node labels (order matters: nodes before edges)
_NODE_CSVS = {
    "PhysicalTable.csv": "PhysicalTable",
    "Col.csv": "Col",
    "Application.csv": "Application",
    "Standard.csv": "Standard",
    "Metric.csv": "Metric",
    "Dimension.csv": "Dimension",
}

# Known edge CSV files -> relationship types
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

# Pattern: :ID(Label) or :START_ID(Label) / :END_ID(Label)
_LABEL_RE = re.compile(r":(?:ID|START_ID|END_ID)\((\w+)\)")


def _parse_label(col_name: str) -> str | None:
    m = _LABEL_RE.match(col_name)
    return m.group(1) if m else None


def _cypher_literal(value: str | None) -> str:
    """Convert a Python value to a Cypher literal."""
    if value is None:
        return "null"
    value = value.replace("\\", "\\\\")
    value = value.replace("'", "\\'")
    value = value.replace("\n", "\\n")
    value = value.replace("\r", "\\r")
    value = value.replace("\t", "\\t")
    value = value.replace("\0", "\\0")
    return "'" + value + "'"


def _build_node_merge(
    id_col: str, prop_cols: list[str], label: str, row: list
) -> str:
    """Construct a MERGE query for a single node row.

    Uses the actual ``:ID(Label)`` column name from the CSV header so that
    the MERGE pattern matches nodes created by ``falkordb-bulk-insert``
    (which stores the ID value under the literal header name, not ``node_id``).
    """
    id_val = _cypher_literal(row[0])
    props = ", ".join(
        f"n.`{prop_cols[i]}` = {_cypher_literal(row[i + 1])}"
        for i in range(len(prop_cols))
    )
    return f"MERGE (n:{label} {{`{id_col}`: {id_val}}}) SET {props}"


def _build_edge_merge(
    src_col: str,
    dst_col: str,
    src_label: str,
    dst_label: str,
    rel_type: str,
    prop_cols: list[str],
    row: list,
) -> str:
    """Construct a MERGE query for a single edge row.

    Uses the actual ``:START_ID(Label)`` / ``:END_ID(Label)`` column names
    to match nodes by the same property that ``falkordb-bulk-insert`` stored.
    """
    src_val = _cypher_literal(row[0])
    dst_val = _cypher_literal(row[1])
    base = (
        f"MATCH (a:{src_label} {{`{src_col}`: {src_val}}}), "
        f"(b:{dst_label} {{`{dst_col}`: {dst_val}}}) "
        f"MERGE (a)-[r:{rel_type}]->(b)"
    )
    if prop_cols:
        props = ", ".join(
            f"r.`{prop_cols[i]}` = {_cypher_literal(row[i + 2])}"
            for i in range(len(prop_cols))
        )
        return f"{base} SET {props}"
    return base


def _read_csv_rows(csv_path: Path) -> tuple[list[str], list[list]]:
    """Read CSV and return (header, rows) with empty cells mapped to None."""
    rows: list[list] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for r in reader:
            rows.append([cell if cell != "" else None for cell in r])
    return header, rows


def _upsert(graph, csv_path: Path, build_fn, type_label: str) -> None:
    """Read a CSV and MERGE each row into FalkorDB."""
    header, rows = _read_csv_rows(csv_path)
    for row in rows:
        query = build_fn(header, row)
        graph.query(query)
    print(f"  {csv_path.name}: {len(rows)} {type_label} upserted")


def upsert_csv_to_falkordb(
    csv_dir: Path, host: str, port: int, graph_name: str
) -> None:
    """Incrementally upsert CSV data into FalkorDB using MERGE semantics.

    Unlike ``import_csv_to_falkordb`` which deletes the graph first, this
    function uses MERGE queries so that existing nodes/edges (e.g. those
    created by ``onboard``) are preserved.
    """
    import falkordb

    csv_path = Path(csv_dir)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 目录不存在: {csv_path}")

    print(f"\n正在增量更新 FalkorDB ({host}:{port}/{graph_name})...")

    client = falkordb.FalkorDB(host=host, port=port)
    graph = client.select_graph(graph_name)

    # 1. Upsert nodes
    for filename, label in _NODE_CSVS.items():
        filepath = csv_path / filename
        if not filepath.exists():
            continue
        header, rows = _read_csv_rows(filepath)
        if not header or not header[0].startswith(":ID("):
            continue
        id_col = header[0]
        prop_cols = header[1:]

        def make_node_builder(lbl, ic, pc):
            return lambda h, row: _build_node_merge(ic, pc, lbl, row)

        _upsert(graph, filepath, make_node_builder(label, id_col, prop_cols), "nodes")

    # 2. Upsert edges
    for filename, rel_type in _EDGE_CSVS.items():
        filepath = csv_path / filename
        if not filepath.exists():
            continue
        header, rows = _read_csv_rows(filepath)
        if len(header) < 2 or not header[0].startswith(":START_ID("):
            continue
        src_col = header[0]
        dst_col = header[1]
        src_label = _parse_label(src_col) or ""
        dst_label = _parse_label(dst_col) or ""
        prop_cols = header[2:]

        def make_edge_builder(rt, sc, dc, sl, dl, pc):
            return lambda h, row: _build_edge_merge(sc, dc, sl, dl, rt, pc, row)

        _upsert(
            graph,
            filepath,
            make_edge_builder(rel_type, src_col, dst_col, src_label, dst_label, prop_cols),
            "edges",
        )

    print("FalkorDB 增量更新完成")


# ---------------------------------------------------------------------------
# Datasource configuration (used by onboard flow)
# ---------------------------------------------------------------------------

def prompt_connect_args(existing: dict[str, Any] | None = None) -> dict[str, Any]:
    """交互式输入连接参数（key=value 格式）

    Args:
        existing: 已有的连接参数

    Returns:
        dict: 连接参数字典
    """
    connect_args: dict[str, Any] = {}

    if existing:
        print(f"  当前连接参数: {existing}")
        keep = questionary.confirm(
            "  是否保留现有参数？",
            default=True,
        ).ask()
        if keep:
            return existing

    print("  输入连接参数 (key=value 格式，留空结束):")
    print("  示例: ssl=true, timeout=30")

    while True:
        line = questionary.text("  >").ask()
        if not line:
            break
        if "=" not in line:
            print("  格式错误，请使用 key=value 格式")
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            print("  格式错误，key 不能为空")
            continue
        value = value.strip()
        if value.lower() in ("true", "false"):
            value = value.lower() == "true"
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
        connect_args[key] = value

    return connect_args


def prompt_datasource_config(
    existing_datasources: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """提示用户配置数据源（可选）

    Args:
        existing_datasources: 已有的数据源配置

    Returns:
        dict: 数据源配置字典，None 表示无数据源
    """
    print("\n=== 数据源配置（可选）===\n")
    print("配置数据源供 observe 命令使用")
    print("可添加 MySQL、DuckDB 等数据源\n")

    datasources: dict[str, Any] = (
        dict(existing_datasources) if existing_datasources else {}
    )

    while True:
        if datasources:
            print("已配置的数据源:")
            for name, ds in datasources.items():
                print(f"  - {name}: {ds['url']}")
            print()

        action = questionary.select(
            "操作选项：",
            choices=[
                questionary.Choice("添加数据源", value="add"),
                questionary.Choice("删除数据源", value="del"),
                questionary.Choice("完成配置", value="done"),
            ],
            default="done",
        ).ask()

        if action == "add":
            name = questionary.text("  数据源名称:").ask()
            if not name:
                print("  名称不能为空")
                continue
            url = questionary.text(
                "  URL (如 mysql+pymysql://user:pass@host/db):"
            ).ask()
            if not url:
                print("  URL 不能为空")
                continue
            if name in datasources:
                overwrite = questionary.confirm(
                    f"  数据源 '{name}' 已存在，是否覆盖？",
                    default=False,
                ).ask()
                if not overwrite:
                    print("  已取消添加")
                    continue
            existing_args = datasources.get(name, {}).get("connect_args") or None
            connect_args = prompt_connect_args(existing_args)
            datasources[name] = {"url": url, "connect_args": connect_args}
            print(f"  已添加数据源: {name}")

        elif action == "del":
            if not datasources:
                print("  没有可删除的数据源")
                continue
            names = list(datasources.keys())
            removed = questionary.select(
                "  选择要删除的数据源：",
                choices=names,
            ).ask()
            if removed:
                del datasources[removed]
                print(f"  已删除: {removed}")

        elif action == "done":
            break

    return datasources if datasources else None


# ---------------------------------------------------------------------------
# Onboard main flow (simplified)
# ---------------------------------------------------------------------------

def onboard():
    """Onboard 向导主函数 — 图数据库后端选择 + 数据源配置"""
    config_manager = ConfigManager()

    if config_manager.exists():
        existing_config = config_manager.load()
        has_backend = "graph" in existing_config and "backend" in existing_config.get("graph", {})

        if has_backend:
            print(f"\n检测到已有配置 (backend: {existing_config['graph']['backend']})")
            skip = questionary.confirm(
                "是否跳过图后端配置，仅配置数据源？",
                default=False,
            ).ask()
            if skip:
                full_config = dict(existing_config)
                datasources = prompt_datasource_config(full_config.get("datasources"))
                if datasources is not None:
                    full_config["datasources"] = datasources
                else:
                    full_config.pop("datasources", None)
                config_manager.save(full_config)
                print(f"\n配置已更新: {config_manager.config_path}")
                return

        print("\n配置文件已存在")
        overwrite = questionary.confirm(
            "是否覆盖现有配置？",
            default=False,
        ).ask()
        if not overwrite:
            print("已取消配置")
            return

    # --- Graph backend selection ---
    print("\n=== Govio Onboard 向导 ===\n")

    backend = questionary.select(
        "请选择图数据库后端：",
        choices=[
            questionary.Choice("networkx - 本地 GML 文件", value="networkx"),
            questionary.Choice("falkordb - FalkorDB 图数据库", value="falkordb"),
        ],
        default="networkx",
    ).ask()

    if backend == "networkx":
        print("\n--- NetworkX 配置 ---\n")
        gml_path_input = questionary.text(
            "请输入 GML 文件路径:",
            validate=lambda v: True if Path(v).exists() else "GML 文件不存在",
        ).ask()
        graph_config = {"backend": "networkx", "networkx": {"gml_path": str(Path(gml_path_input))}}
    else:
        print("\n--- FalkorDB 配置 ---\n")
        host = questionary.text(
            "请输入 FalkorDB 主机地址:",
            default="localhost",
        ).ask() or "localhost"

        port_str = questionary.text(
            "请输入 FalkorDB 端口:",
            default="6379",
            validate=lambda v: True if v.isdigit() else "端口必须是数字",
        ).ask() or "6379"
        port = int(port_str)

        graph_name = questionary.text(
            "请输入图数据库名称:",
            default="ontology",
        ).ask() or "ontology"

        graph_config = {"backend": "falkordb", "falkordb": {"host": host, "port": port, "graph": graph_name}}

    full_config: dict[str, Any] = {"graph": graph_config}

    # --- Datasource config ---
    datasources = prompt_datasource_config()
    if datasources:
        full_config["datasources"] = datasources

    config_manager.save(full_config)
    print(f"\n配置已保存到: {config_manager.config_path}")
    print("\nOnboard 完成！")


if __name__ == "__main__":
    onboard()
