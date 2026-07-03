"""FalkorDB CSV loading utilities — bulk import and incremental upsert."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from falkordb import FalkorDB


def delete_falkordb_graph(host: str, port: int, graph_name: str) -> None:
    """删除 FalkorDB 中的图（如果存在）"""
    try:
        client = FalkorDB(host=host, port=port)
        client.execute_command("DEL", graph_name)
        print(f"已删除现有图: {graph_name}")
    except Exception:
        pass


def import_csv_to_falkordb(
    csv_dir: Path, host: str, port: int, graph_name: str
) -> None:
    """使用 falkordb-bulk-insert 将 CSV 导入 FalkorDB（删除旧图后全量插入）"""
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


# --- Incremental upsert via direct Cypher ---

_BATCH_SIZE = 500

_NODE_CSVS = {
    "PhysicalTable.csv": "PhysicalTable",
    "Col.csv": "Col",
    "Application.csv": "Application",
    "Standard.csv": "Standard",
    "Metric.csv": "Metric",
    "Dimension.csv": "Dimension",
}

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


def _build_node_merge_query(label: str, prop_cols: list[str]) -> str:
    """Build: MERGE (n:Label {`prop0`: row[0], `prop1`: row[1], ...})"""
    props = ", ".join(f"`{col}`: row[{i}]" for i, col in enumerate(prop_cols))
    return f"MERGE (n:{label} {{{props}}})"


def _build_edge_merge_query(
    src_label: str, dst_label: str, rel_type: str, prop_cols: list[str]
) -> str:
    """Build: MATCH (a:Src),(b:Dst) WHERE a.`:ID(Src)`=row[0] AND b.`:ID(Dst)`=row[1] MERGE (a)-[r:TYPE]->(b) [SET ...]"""
    base = (
        f"MATCH (a:{src_label}), (b:{dst_label}) "
        f"WHERE a.`:ID({src_label})` = row[0] AND b.`:ID({dst_label})` = row[1] "
        f"MERGE (a)-[r:{rel_type}]->(b)"
    )
    if prop_cols:
        sets = ", ".join(f"r.`{col}` = row[{i + 2}]" for i, col in enumerate(prop_cols))
        return f"{base} SET {sets}"
    return base


def _execute_batch(graph, query: str, rows: list[list]) -> int:
    """Execute a single batch: CYPHER rows=[...] UNWIND $rows AS row <query>"""
    rows_json = json.dumps(rows, ensure_ascii=False)
    command = f"CYPHER rows={rows_json} UNWIND $rows AS row {query}"
    result = graph.query(command)
    return result.nodes_created + result.relationships_created + result.properties_set


def upsert_csv_to_falkordb(
    csv_dir: Path, host: str, port: int, graph_name: str
) -> None:
    """Incremental upsert via direct Cypher MERGE (batch 500 rows)."""
    csv_path = Path(csv_dir)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 目录不存在: {csv_path}")

    print(f"\n正在增量更新 FalkorDB ({host}:{port}/{graph_name})...")
    client = FalkorDB(host=host, port=port)
    graph = client.select_graph(graph_name)

    # Node CSVs
    for filename, label in _NODE_CSVS.items():
        filepath = csv_path / filename
        if not filepath.exists():
            continue
        df = pd.read_csv(filepath, dtype=str).fillna("")
        cols = list(df.columns)
        prop_cols = cols
        query = _build_node_merge_query(label, prop_cols)
        print(f"  {filename}: {query}")
        total = 0
        for start in range(0, len(df), _BATCH_SIZE):
            batch = df.iloc[start:start + _BATCH_SIZE].values.tolist()
            total += _execute_batch(graph, query, batch)
        print(f"    {len(df)} rows, {total} effects")

    # Edge CSVs
    for filename, rel_type in _EDGE_CSVS.items():
        filepath = csv_path / filename
        if not filepath.exists():
            continue
        df = pd.read_csv(filepath, dtype=str).fillna("")
        cols = list(df.columns)
        if len(cols) < 2 or not cols[0].startswith(":START_ID("):
            continue
        src_label = cols[0].removeprefix(":START_ID(").removesuffix(")")
        dst_label = cols[1].removeprefix(":END_ID(").removesuffix(")")
        prop_cols = cols[2:]
        query = _build_edge_merge_query(src_label, dst_label, rel_type, prop_cols)
        print(f"  {filename}: {query}")
        total = 0
        for start in range(0, len(df), _BATCH_SIZE):
            batch = df.iloc[start:start + _BATCH_SIZE].values.tolist()
            total += _execute_batch(graph, query, batch)
        print(f"    {len(df)} rows, {total} effects")

    print("FalkorDB 增量更新完成")
