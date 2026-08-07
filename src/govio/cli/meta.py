"""meta 命令组 — 知识图库维护

提供元数据同步、导出、数据标准推荐、配置管理等功能。

子命令拆分设计：每个 sync 子命令都是独立可运行的步骤，通过 output 目录的 CSV 文件作为共享状态，
支持增量合并（幂等）。可以按任意顺序运行，多次运行安全。
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import questionary

from .config import ConfigManager, MetaConfigManager
from govio.core.graph_factory import GraphFactory
from govio.core.assets_generator import AssetsGenerator
from govio.graph.falkordb_loader import import_csv_to_falkordb, upsert_csv_to_falkordb
from govio.graph.ladybug_loader import import_csv_to_ladybug, upsert_csv_to_ladybug
from govio.metadata.database import TDSLoader
from govio.metadata.application import AppInfoLoader
from govio.metadata.standard import StandardLoader
from govio.metadata.duckdb_loader import DuckDBLoader
from govio.metadata.relationship import load_relationships
from govio.metadata.metric import MetricLoader
from govio.metadata.node_id import assign_node_ids, write_node_csv

SKILLS_ASSETS_DIR = Path("skills/govio/assets")


# ---------------------------------------------------------------------------
# CSV merge helpers — 增量合并的核心机制
# ---------------------------------------------------------------------------

def merge_node_csv(
    new_df: pd.DataFrame, csv_path: Path, node_type: str, key_col: str,
) -> pd.DataFrame:
    """将新节点数据与已有 CSV 合并，按业务键去重（新的覆盖旧的）。

    读取已有 CSV 时会去掉 :ID(NodeType) 列头前缀以统一列名，
    合并后再通过 write_node_csv 写回标准格式。

    Returns:
        合并后的 DataFrame（已重置索引，含 node_id 列）
    """
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        # 已有 CSV 的第一列是 :ID(NodeType)，重命名为 node_id 以统一
        id_col = f":ID({node_type})"
        if id_col in existing.columns:
            existing = existing.rename(columns={id_col: "node_id"})
        combined = pd.concat([existing, new_df], ignore_index=True)
        dedup_col = key_col if key_col in combined.columns else None
        if dedup_col:
            merged = combined.drop_duplicates(subset=[dedup_col], keep="last")
        else:
            print(f"⚠ merge_node_csv: 未找到去重列 '{key_col}'，跳过去重", file=sys.stderr)
            merged = combined
    else:
        merged = new_df

    merged = merged.reset_index(drop=True)
    write_node_csv(merged, csv_path, node_type)
    return merged


def merge_edge_csv(
    new_df: pd.DataFrame, csv_path: Path, dedup_cols: list[str],
) -> pd.DataFrame:
    """将新边数据与已有 CSV 合并，按复合键去重（新的覆盖旧的）。

    Returns:
        合并后的 DataFrame
    """
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
        cols = [c for c in dedup_cols if c in combined.columns]
        if cols:
            merged = combined.drop_duplicates(subset=cols, keep="last")
        else:
            print(f"⚠ merge_edge_csv: 未找到去重列 {dedup_cols}，跳过去重", file=sys.stderr)
            merged = combined
    else:
        merged = new_df

    merged = merged.reset_index(drop=True)
    merged.to_csv(csv_path, index=False)
    return merged


def _load_csv_with_node_ids(
    csv_path: Path, node_type: str, key_col: str,
) -> pd.DataFrame:
    """加载节点 CSV，若缺少 node_id 列则重新生成并写回。"""
    df = pd.read_csv(csv_path)
    if "node_id" not in df.columns:
        assign_node_ids(df, node_type, key_col)
        df.to_csv(csv_path, index=False)
    return df


# ---------------------------------------------------------------------------
# Graph / assets helpers
# ---------------------------------------------------------------------------

def _update_graph(output: Path, graph_mode: str) -> bool:
    """更新图数据库。返回是否成功。"""
    from govio.metadata.gen_networkx import build_graph

    graph_config = ConfigManager().load()
    graph = graph_config.get("graph") or {}
    backend = graph.get("backend")
    incremental = (graph_mode == "update")

    if backend == "falkordb":
        falkordb_cfg = graph.get("falkordb", {})
        host = falkordb_cfg.get("host", "localhost")
        port = falkordb_cfg.get("port", 6379)
        graph_name = falkordb_cfg.get("graph", "ontology")
        try:
            if incremental:
                upsert_csv_to_falkordb(output, host, port, graph_name)
                print("✓ FalkorDB 数据已更新")
            else:
                import_csv_to_falkordb(output, host, port, graph_name)
                print("✓ FalkorDB 数据已重建")
        except Exception as e:
            print(f"❌ 导入 FalkorDB 失败: {e}")
            return False
    elif backend == "networkx":
        networkx_cfg = graph.get("networkx", {})
        gml_path = networkx_cfg.get("gml_path", str(SKILLS_ASSETS_DIR / "ontology.gml"))
        label = "更新" if incremental else "重建"
        print(f"\n正在从 CSV {label} GML 文件 ({gml_path})...")
        try:
            build_graph(str(output), gml_path, incremental=incremental)
            print(f"✓ GML 文件已{label}")
        except Exception as e:
            print(f"❌ {label} GML 失败: {e}")
            return False
    elif backend == "ladybug":
        ladybug_cfg = graph.get("ladybug", {})
        db_path_val = ladybug_cfg.get("db_path")
        bp = ladybug_cfg.get("buffer_pool_size", 256 * 1024 * 1024)
        maxdb = ladybug_cfg.get("max_db_size", 1 * 1024 * 1024 * 1024)
        if not db_path_val:
            print("❌ Ladybug 配置缺少 db_path，跳过图数据更新")
            return False
        try:
            if incremental:
                upsert_csv_to_ladybug(output, db_path_val, buffer_pool_size=bp, max_db_size=maxdb)
                print("✓ Ladybug 数据已更新")
            else:
                import_csv_to_ladybug(output, db_path_val, buffer_pool_size=bp, max_db_size=maxdb)
                print("✓ Ladybug 数据已重建")
        except Exception as e:
            print(f"❌ 导入 Ladybug 失败: {e}")
            return False
    else:
        print("提示: 未配置 graph backend，跳过图数据更新")
    return True


def _generate_assets() -> None:
    """生成 schema.md、name 索引、metrics_index.md 等 assets。"""
    print("\n正在生成 assets...")
    try:
        graph_config = ConfigManager().load()
        graph = graph_config.get("graph") or {}
        graph_obj = GraphFactory.create(graph)
        generator = AssetsGenerator(graph_obj, SKILLS_ASSETS_DIR)
        generator.generate_all()
        print(f"✓ Assets 已生成到: {SKILLS_ASSETS_DIR}")
    except Exception as e:
        print(f"❌ 生成 assets 失败: {e}")


# ---------------------------------------------------------------------------
# merge / export helpers（保留兼容）
# ---------------------------------------------------------------------------

def merge_metadata(
    df_tds: pd.DataFrame, df_duck: pd.DataFrame, key: str,
) -> pd.DataFrame:
    """TDS full + DuckDB incremental. DuckDB wins on conflict."""
    combined = pd.concat([df_tds, df_duck], ignore_index=True)
    return combined.drop_duplicates(subset=[key], keep="last").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step functions — 可独立调用的管线步骤
# ---------------------------------------------------------------------------

def step_meta_export(
    output: Path,
    source: str,
    db_path: str = "",
    schemas: list[str] | None = None,
    db_name: str | None = None,
    kundb: str = "",
    workspace_uuid: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """步骤：导出元数据（PhysicalTable, Col, HAS_COLUMN）。

    Returns:
        (df_tables, df_columns) 或 None（出错时）
    """
    output.mkdir(parents=True, exist_ok=True)

    if source in ("duckdb", "both") and not db_path:
        print("错误: DuckDB 模式需要指定 --db 路径", file=sys.stderr)
        return None

    if source == "tds" and not kundb:
        print("错误: TDS 模式需要指定 --kundb", file=sys.stderr)
        return None

    # 加载元数据
    if source == "tds":
        print("从 TDS 读取元数据...")
        tds_schemas = schemas if schemas else None
        tds_loader = TDSLoader(kundb, workspace_uuid, tds_schemas)
        df_tables = tds_loader.PhysicalTable
        df_columns = tds_loader.Col
    elif source == "duckdb":
        print("从 DuckDB 读取元数据...")
        duck_loader = DuckDBLoader(db_path, schemas or [])
        df_tables = duck_loader.PhysicalTable
        df_columns = duck_loader.Col
    else:
        print("从 TDS + DuckDB 合并读取元数据...")
        tds_schemas = schemas if schemas else None
        tds_loader = TDSLoader(kundb, workspace_uuid, tds_schemas)
        tds_tables = tds_loader.PhysicalTable
        tds_columns = tds_loader.Col
        duck_loader = DuckDBLoader(db_path, schemas or [])
        duck_tables = duck_loader.PhysicalTable
        duck_columns = duck_loader.Col
        df_tables = merge_metadata(tds_tables, duck_tables, "full_table_name")
        df_columns = merge_metadata(tds_columns, duck_columns, "column")

    # Assign IDs
    df_tables = df_tables.reset_index(drop=True)
    df_columns = df_columns.reset_index(drop=True)
    assign_node_ids(df_tables, "PhysicalTable", "full_table_name")
    assign_node_ids(df_columns, "Col", "column")

    # Merge + write node CSVs
    pt_path = output / "PhysicalTable.csv"
    col_path = output / "Col.csv"
    df_tables = merge_node_csv(df_tables, pt_path, "PhysicalTable", "full_table_name")
    df_columns = merge_node_csv(df_columns, col_path, "Col", "column")

    # HAS_COLUMN edge
    df_has_column = pd.merge(
        df_tables[["full_table_name", "node_id"]].rename(
            columns={"node_id": ":START_ID(PhysicalTable)"}
        ),
        df_columns[["full_table_name", "node_id"]].rename(
            columns={"node_id": ":END_ID(Col)"}
        ),
        on="full_table_name",
        how="inner",
    )[[":START_ID(PhysicalTable)", ":END_ID(Col)"]]

    hc_path = output / "HAS_COLUMN.csv"
    merge_edge_csv(df_has_column, hc_path, [":START_ID(PhysicalTable)", ":END_ID(Col)"])

    print(f"✓ 元数据已导出: {len(df_tables)} 张表, {len(df_columns)} 个字段")
    return df_tables, df_columns


def step_app_export(
    output: Path,
    app_list_file: str,
    app_map_file: str,
    db_name: str | None = None,
) -> None:
    """步骤：导出应用清单（Application）和 USE 边。"""
    output.mkdir(parents=True, exist_ok=True)

    df_app_db_map = pd.read_json(app_map_file, orient="records")
    app_names = [db_name] if db_name else df_app_db_map["name"].to_list()
    app_loader = AppInfoLoader(app_list_file, app_names)
    df_apps = app_loader.Application.reset_index(drop=True)
    assign_node_ids(df_apps, "Application", "app_id")

    # Merge Application node CSV
    app_path = output / "Application.csv"
    df_apps = merge_node_csv(df_apps, app_path, "Application", "app_id")

    # USE edge — 需要 PhysicalTable.csv 已存在
    pt_path = output / "PhysicalTable.csv"
    if not pt_path.exists():
        print("⚠ PhysicalTable.csv 不存在，跳过 USE 边生成。请先运行 meta sync meta")
        return

    df_tables = _load_csv_with_node_ids(pt_path, "PhysicalTable", "full_table_name")

    df_app_table = pd.merge(
        df_app_db_map,
        df_tables[["schema", "node_id"]].rename(
            columns={"node_id": ":END_ID(PhysicalTable)"}
        ),
        on="schema",
        how="inner",
    )
    df_use = pd.merge(
        df_apps[["name", "node_id"]].rename(
            columns={"node_id": ":START_ID(Application)"}
        ),
        df_app_table,
        on="name",
        how="inner",
    )[[":START_ID(Application)", ":END_ID(PhysicalTable)"]]

    use_path = output / "USE.csv"
    merge_edge_csv(df_use, use_path, [":START_ID(Application)", ":END_ID(PhysicalTable)"])

    print(f"✓ 应用数据已导出: {len(df_apps)} 个应用, {len(df_use)} 条 USE 边")


def step_rel_export(
    output: Path,
    relationship_file: str,
) -> None:
    """步骤：导出表关系（RELATES_TO 边）。"""
    output.mkdir(parents=True, exist_ok=True)

    pt_path = output / "PhysicalTable.csv"
    col_path = output / "Col.csv"
    if not pt_path.exists() or not col_path.exists():
        print("❌ 需要先导入元数据（PhysicalTable.csv, Col.csv），请先运行 meta sync meta")
        return

    df_tables = _load_csv_with_node_ids(pt_path, "PhysicalTable", "full_table_name")
    df_columns = _load_csv_with_node_ids(col_path, "Col", "column")

    table_idx_to_id = df_tables["node_id"].tolist()

    try:
        df_relates_to = load_relationships(relationship_file, df_tables, df_columns)
        if not df_relates_to.empty:
            df_relates_to["source"] = [
                table_idx_to_id[i] for i in df_relates_to["source"]
            ]
            df_relates_to["target"] = [
                table_idx_to_id[i] for i in df_relates_to["target"]
            ]
            # 重命名为图导入所需的列名格式，保留元数据列
            df_relates_to = df_relates_to.rename(columns={
                "source": ":START_ID(PhysicalTable)",
                "target": ":END_ID(PhysicalTable)",
            })

        rel_path = output / "RELATES_TO.csv"
        merge_edge_csv(
            df_relates_to, rel_path,
            [":START_ID(PhysicalTable)", ":END_ID(PhysicalTable)", "relationship_type"],
        )
        print(f"✓ RELATES_TO 已导出: {len(df_relates_to)} 个关系 来自[{relationship_file}]")
    except Exception as e:
        print(f"❌ 无法加载关系文件: {e}")


def step_std_export(
    output: Path,
    kundb: str,
    workspace_uuid: str,
) -> None:
    """步骤：导出数据标准（Standard 节点）。"""
    output.mkdir(parents=True, exist_ok=True)

    std_loader = StandardLoader(kundb, workspace_uuid)
    df_stds = std_loader.Standard.reset_index(drop=True)
    assign_node_ids(df_stds, "Standard", "standard_id")

    std_path = output / "Standard.csv"
    merge_node_csv(df_stds, std_path, "Standard", "standard_id")

    print(f"✓ 数据标准已导出: {len(df_stds)} 个标准")


def step_compliance_export(
    output: Path,
    kundb: str,
    workspace_uuid: str,
) -> None:
    """步骤：从 TDS 导出已有标准-字段关联（COMPLIES_WITH 边）。"""
    output.mkdir(parents=True, exist_ok=True)

    col_path = output / "Col.csv"
    std_path = output / "Standard.csv"
    if not col_path.exists():
        print("❌ 需要先导入元数据（Col.csv），请先运行 meta sync meta")
        return
    if not std_path.exists():
        print("❌ 需要先导入数据标准（Standard.csv），请先运行 meta sync std")
        return

    df_columns = _load_csv_with_node_ids(col_path, "Col", "column")
    df_stds = _load_csv_with_node_ids(std_path, "Standard", "standard_id")

    std_loader = StandardLoader(kundb, workspace_uuid)
    df_compliance = std_loader.StdCompliance  # 已贯标列

    if df_compliance.empty:
        print("✓ TDS 中无已有标准关联数据")
        return

    # 将 column 字段映射为 node_id
    col_id_map = df_columns.set_index("column")["node_id"].to_dict()
    std_id_map = df_stds.set_index("standard_id")["node_id"].to_dict()

    df_compliance[":START_ID(Col)"] = df_compliance["column"].map(col_id_map)
    df_compliance[":END_ID(Standard)"] = df_compliance["standard_id"].map(std_id_map)

    # 过滤掉映射失败的行
    df_complies = df_compliance.dropna(subset=[":START_ID(Col)", ":END_ID(Standard)"])[
        [":START_ID(Col)", ":END_ID(Standard)"]
    ]

    if df_complies.empty:
        print("⚠ 无匹配的标准-字段关联（可能需要先导入相关元数据和标准）")
        return

    comp_path = output / "COMPLIES_WITH.csv"
    merge_edge_csv(df_complies, comp_path, [":START_ID(Col)", ":END_ID(Standard)"])

    print(f"✓ COMPLIES_WITH 已导出: {len(df_complies)} 条关联")


def step_metric_export(
    output: Path,
    metric_file: str,
) -> bool:
    """步骤：导出指标维度定义（Metric, Dimension 节点 + 5 种边）。

    Returns:
        True 表示成功，False 表示失败。
    """
    output.mkdir(parents=True, exist_ok=True)

    pt_path = output / "PhysicalTable.csv"
    col_path = output / "Col.csv"
    if not pt_path.exists() or not col_path.exists():
        print("❌ 需要先导入元数据（PhysicalTable.csv, Col.csv），请先运行 meta sync meta", file=sys.stderr)
        return False

    df_tables = _load_csv_with_node_ids(pt_path, "PhysicalTable", "full_table_name")
    df_columns = _load_csv_with_node_ids(col_path, "Col", "column")

    table_idx_to_id = df_tables["node_id"].tolist()
    col_idx_to_id = df_columns["node_id"].tolist()

    try:
        metric_loader = MetricLoader(metric_file, df_tables, df_columns)
        df_metrics = metric_loader.Metric.reset_index(drop=True)
        df_dimensions = metric_loader.Dimension.reset_index(drop=True)

        assign_node_ids(df_metrics, "Metric", "code")
        assign_node_ids(df_dimensions, "Dimension", "code")

        # Merge node CSVs
        df_metrics = merge_node_csv(df_metrics, output / "Metric.csv", "Metric", "code")
        df_dimensions = merge_node_csv(df_dimensions, output / "Dimension.csv", "Dimension", "code")

        metric_idx_to_id = df_metrics["node_id"].tolist()
        dim_idx_to_id = df_dimensions["node_id"].tolist()

        # USES_TABLE 边
        uses_table = metric_loader.uses_table_edges.copy()
        if not uses_table.empty:
            uses_table[":START_ID(Metric)"] = [
                metric_idx_to_id[i] for i in uses_table[":START_ID(Metric)"]
            ]
            uses_table[":END_ID(PhysicalTable)"] = [
                table_idx_to_id[i] for i in uses_table[":END_ID(PhysicalTable)"]
            ]
            merge_edge_csv(
                uses_table, output / "USES_TABLE.csv",
                [":START_ID(Metric)", ":END_ID(PhysicalTable)"],
            )

        # REFERS_COLUMN 边
        refers_col = metric_loader.refers_column_edges.copy()
        if not refers_col.empty:
            refers_col[":START_ID(Metric)"] = [
                metric_idx_to_id[i] for i in refers_col[":START_ID(Metric)"]
            ]
            refers_col[":END_ID(Col)"] = [
                col_idx_to_id[i] for i in refers_col[":END_ID(Col)"]
            ]
            merge_edge_csv(
                refers_col, output / "REFERS_COLUMN.csv",
                [":START_ID(Metric)", ":END_ID(Col)"],
            )

        # DERIVED_FROM 边
        derived_from = metric_loader.derived_from_edges.copy()
        if not derived_from.empty:
            derived_from[":START_ID(Metric)"] = [
                metric_idx_to_id[i] for i in derived_from[":START_ID(Metric)"]
            ]
            derived_from[":END_ID(Metric)"] = [
                metric_idx_to_id[i] for i in derived_from[":END_ID(Metric)"]
            ]
            merge_edge_csv(
                derived_from, output / "DERIVED_FROM.csv",
                [":START_ID(Metric)", ":END_ID(Metric)"],
            )

        # DIMENSION_USED 边
        dim_used = metric_loader.dimension_used_edges.copy()
        if not dim_used.empty:
            dim_used[":START_ID(Metric)"] = [
                metric_idx_to_id[i] for i in dim_used[":START_ID(Metric)"]
            ]
            dim_used[":END_ID(Dimension)"] = [
                dim_idx_to_id[i] for i in dim_used[":END_ID(Dimension)"]
            ]
            merge_edge_csv(
                dim_used, output / "DIMENSION_USED.csv",
                [":START_ID(Metric)", ":END_ID(Dimension)"],
            )

        # SUPERSEDES 边
        supersedes = metric_loader.supersedes_edges.copy()
        if not supersedes.empty:
            supersedes[":START_ID(Metric)"] = [
                metric_idx_to_id[i] for i in supersedes[":START_ID(Metric)"]
            ]
            supersedes[":END_ID(Metric)"] = [
                metric_idx_to_id[i] for i in supersedes[":END_ID(Metric)"]
            ]
            merge_edge_csv(
                supersedes, output / "SUPERSEDES.csv",
                [":START_ID(Metric)", ":END_ID(Metric)"],
            )

        print(
            f"✓ 指标数据已导出: {len(df_metrics)} 个指标, "
            f"{len(df_dimensions)} 个维度"
        )
        return True
    except Exception as e:
        print(f"❌ 无法加载指标定义文件: {e}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Full pipeline — 向后兼容的完整管线
# ---------------------------------------------------------------------------

def meta_export(
    db_path: str,
    schemas: list[str] | None,
    db_name: str | None,
    output: Path,
    graph_mode: str = "dry_run",
    source: str = "auto",
):
    """完整管线：元数据 → 应用 → 标准 → 关系 → 指标 → 图更新 → assets。

    source: "tds" | "duckdb" | "both" | "auto"（auto 按 db_path 有无自动判断）

    注意：完整管线会先清空 output 目录的 CSV，再从头生成。
    如需增量导入请使用各子命令（meta sync meta / app / std 等）。
    """
    output.mkdir(parents=True, exist_ok=True)

    # 完整管线模式：清理旧 CSV，从头生成
    for csv_file in output.glob("*.csv"):
        csv_file.unlink()

    # 自动推断 source
    if source == "auto":
        if db_path and db_name:
            source = "duckdb"
        elif db_path:
            source = "duckdb"
        else:
            source = "tds"

    if source in ("duckdb", "both") and not db_path:
        print("错误: DuckDB 模式需要指定 --db 路径", file=sys.stderr)
        sys.exit(1)

    if source in ("duckdb", "both") and not schemas and not db_name:
        print("错误: DuckDB 模式必须指定 --schemas 或 --db-name", file=sys.stderr)
        sys.exit(1)

    if source == "tds" and db_name:
        print("错误: 单库模式（--db-name）仅适用于 DuckDB 数据源", file=sys.stderr)
        sys.exit(1)

    # --- Load config ---
    meta_cfg = MetaConfigManager()
    if meta_cfg.exists():
        meta_config = meta_cfg.load()
        kundb = meta_config.get("kundb", "")
        workspace_uuid = meta_config.get("workspace_uuid", "82ee37374b314a938bf28170ab4db7cf")
        app_list_file = meta_config.get("app_list", "")
        app_map_file = meta_config.get("app_map", "")
        relationship_file = meta_config.get("relationship")
        metric_file = meta_config.get("metric")
    else:
        main_config = ConfigManager().load()
        metadata = main_config.get("metadata") or {}
        kundb = metadata.get("kundb", "")
        workspace_uuid = metadata.get("workspace_uuid", "82ee37374b314a938bf28170ab4db7cf")
        app_list_file = metadata.get("app_list", "")
        app_map_file = metadata.get("app_map", "")
        relationship_file = metadata.get("relationship")
        metric_file = metadata.get("metric")

    required = [app_list_file, app_map_file]
    if source in ("tds", "both"):
        required.append(kundb)
    if not all(required):
        print("❌ 配置缺少必要字段，请检查 metadata 中的 kundb, app_list, app_map")
        sys.exit(1)

    df_app_db_map = pd.read_json(app_map_file, orient="records")

    if db_name and db_name not in df_app_db_map["name"].values:
        print(
            f"错误: --db-name '{db_name}' 不在 app_map 中，可用: "
            f"{df_app_db_map['name'].tolist()}",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- 决定要抽取的 schema 集合 ---
    if db_name:
        app_schemas = df_app_db_map.loc[
            df_app_db_map["name"] == db_name, "schema"
        ].tolist()
        if schemas:
            effective_schemas = [s for s in app_schemas if s in schemas]
            if not effective_schemas:
                print(
                    f"警告: --db-name '{db_name}' 的 schema {app_schemas} "
                    f"与 --schemas {schemas} 无交集，将导出空结果",
                    file=sys.stderr,
                )
        else:
            effective_schemas = app_schemas
    else:
        effective_schemas = schemas or []

    # --- Step 1: 元数据 ---
    result = step_meta_export(
        output, source=source, db_path=db_path,
        schemas=effective_schemas, db_name=db_name,
        kundb=kundb, workspace_uuid=workspace_uuid,
    )
    if result is None:
        sys.exit(1)
    df_tables, df_columns = result

    # --- Step 2: 应用 ---
    step_app_export(output, app_list_file, app_map_file, db_name)

    # --- Step 3: 数据标准 ---
    if source == "duckdb":
        print("提示: DuckDB 模式跳过 Standard 数据标准的读取")
        # 写入空 Standard.csv 以保持一致的文件结构
        df_empty_std = pd.DataFrame(columns=["standard_id"])
        assign_node_ids(df_empty_std, "Standard", "standard_id")
        write_node_csv(df_empty_std, output / "Standard.csv", "Standard")
    else:
        step_std_export(output, kundb, workspace_uuid)

    # --- Step 4: 已有合规关联 ---
    if source != "duckdb":
        step_compliance_export(output, kundb, workspace_uuid)

    # --- Step 5: 表关系 ---
    if relationship_file:
        step_rel_export(output, relationship_file)

    # --- Step 6: 指标维度 ---
    if metric_file and source != "tds":
        if not step_metric_export(output, metric_file):
            sys.exit(1)
    elif metric_file and source == "tds":
        print("提示: TDS-only 模式跳过指标定义加载")

    # --- Summary ---
    n_tables = len(pd.read_csv(output / "PhysicalTable.csv")) if (output / "PhysicalTable.csv").exists() else 0
    n_cols = len(pd.read_csv(output / "Col.csv")) if (output / "Col.csv").exists() else 0
    n_apps = len(pd.read_csv(output / "Application.csv")) if (output / "Application.csv").exists() else 0
    n_stds = len(pd.read_csv(output / "Standard.csv")) if (output / "Standard.csv").exists() else 0
    n_rel = len(pd.read_csv(output / "RELATES_TO.csv")) if (output / "RELATES_TO.csv").exists() else 0
    n_metric = len(pd.read_csv(output / "Metric.csv")) if (output / "Metric.csv").exists() else 0
    print(f"\n成功导出: {n_tables} 张表, {n_cols} 个字段, "
          f"{n_apps} 个应用, {n_stds} 个标准, {n_rel} 个数据关系, {n_metric} 个指标")

    if graph_mode == "dry_run":
        return

    # --- Update/rebuild graph ---
    _update_graph(output, graph_mode)

    # --- Generate assets ---
    _generate_assets()

    print("\n✅ meta-export 完成！")


# ---------------------------------------------------------------------------
# CLI command handlers — 子命令
# ---------------------------------------------------------------------------

def _load_schemas_from_app_map(app_map_file: str) -> list[str]:
    """从 app_map.json 读取所有 schema"""
    with open(app_map_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return list({row["schema"] for row in data})


def _load_meta_config() -> dict:
    """加载 meta 配置，兼容新旧格式。"""
    meta_cfg = MetaConfigManager()
    if meta_cfg.exists():
        return meta_cfg.load()
    main_config = ConfigManager().load()
    return main_config.get("metadata") or {}


def cmd_sync_meta(args: argparse.Namespace) -> None:
    """meta sync meta — 导入 TDS/DuckDB 元数据（PhysicalTable, Col, HAS_COLUMN）"""
    if hasattr(args, "source") and args.source:
        # CLI 模式
        source = args.source
        db_path = args.db or ""
        schemas = args.schemas.split(",") if args.schemas else None
        output = Path(args.output) if args.output else Path("./output")
        kundb = args.kundb or ""
        workspace_uuid = args.workspace_uuid or ""
    else:
        # 交互模式
        config = _load_meta_config()
        source = questionary.select(
            "数据来源:",
            choices=[
                questionary.Choice("TDS — 仅从元数据库读取", value="tds"),
                questionary.Choice("DuckDB — 仅从 DuckDB 读取", value="duckdb"),
                questionary.Choice("Both — TDS + DuckDB 合并", value="both"),
            ],
        ).ask()

        kundb = args.kundb or config.get("kundb", "")
        workspace_uuid = args.workspace_uuid or config.get("workspace_uuid", "82ee37374b314a938bf28170ab4db7cf")

        if source in ("tds", "both") and not kundb:
            kundb = questionary.text("元数据库 URL:").ask() or ""

        db_path = ""
        schemas = None
        if source in ("duckdb", "both"):
            db_path = questionary.text("DuckDB 数据库文件路径:").ask() or ""
            if not db_path:
                print("错误: DuckDB 模式必须指定数据库路径", file=sys.stderr)
                sys.exit(1)
            schemas_input = questionary.text("要导出的 schema 列表（逗号分隔，留空导出全部）:").ask() or ""
            schemas = [s.strip() for s in schemas_input.split(",") if s.strip()] if schemas_input else None

        output = Path(questionary.text("CSV 输出目录:", default="./output").ask() or "./output")

    result = step_meta_export(
        output, source=source, db_path=db_path,
        schemas=schemas, kundb=kundb, workspace_uuid=workspace_uuid,
    )
    if result is None:
        sys.exit(1)


def cmd_sync_app(args: argparse.Namespace) -> None:
    """meta sync app — 导入应用清单（Application 节点 + USE 边）"""
    if hasattr(args, "app_list") and args.app_list and args.app_map:
        app_list_file = args.app_list
        app_map_file = args.app_map
        output = Path(args.output) if args.output else Path("./output")
        db_name = getattr(args, "db_name", None)
    else:
        config = _load_meta_config()
        app_list_file = args.app_list or config.get("app_list", "")
        app_map_file = args.app_map or config.get("app_map", "")

        if not app_list_file:
            app_list_file = questionary.text("应用清单 Excel 文件路径:").ask() or ""
        if not app_map_file:
            app_map_file = questionary.text("应用映射 JSON 文件路径:").ask() or ""

        output = Path(args.output) if args.output else Path(
            questionary.text("CSV 输出目录:", default=config.get("csv_dir", "./output")).ask() or "./output"
        )
        db_name = None

    if not app_list_file or not app_map_file:
        print("❌ 需要指定 --app-list 和 --app-map", file=sys.stderr)
        sys.exit(1)

    step_app_export(output, app_list_file, app_map_file, db_name)


def cmd_sync_rel(args: argparse.Namespace) -> None:
    """meta sync rel — 导入表关系（RELATES_TO 边）"""
    if hasattr(args, "file") and args.file:
        relationship_file = args.file
        output = Path(args.output) if args.output else Path("./output")
    else:
        config = _load_meta_config()
        relationship_file = args.file or config.get("relationship", "")
        if not relationship_file:
            relationship_file = questionary.text("表关系 JSON 文件路径:").ask() or ""
        output = Path(args.output) if args.output else Path(
            questionary.text("CSV 输出目录:", default=config.get("csv_dir", "./output")).ask() or "./output"
        )

    if not relationship_file:
        print("❌ 需要指定 --file", file=sys.stderr)
        sys.exit(1)

    step_rel_export(output, relationship_file)


def cmd_sync_std(args: argparse.Namespace) -> None:
    """meta sync std — 导入数据标准（Standard 节点）"""
    kundb = getattr(args, "kundb", None)
    workspace_uuid = getattr(args, "workspace_uuid", None)
    output = getattr(args, "output", None)

    config = _load_meta_config()

    if not kundb:
        kundb = config.get("kundb", "")
    if not workspace_uuid:
        workspace_uuid = config.get("workspace_uuid", "82ee37374b314a938bf28170ab4db7cf")

    if not kundb:
        kundb = questionary.text("元数据库 URL:").ask() or ""

    if not kundb:
        print("❌ 需要指定 --kundb 或在配置中设置 kundb", file=sys.stderr)
        sys.exit(1)

    output = Path(output) if output else Path(
        questionary.text("CSV 输出目录:", default=config.get("csv_dir", "./output")).ask() or "./output"
    )

    step_std_export(output, kundb, workspace_uuid)


def cmd_sync_compliance(args: argparse.Namespace) -> None:
    """meta sync compliance — 从 TDS 导出已有标准-字段关联（COMPLIES_WITH 边）"""
    kundb = getattr(args, "kundb", None)
    workspace_uuid = getattr(args, "workspace_uuid", None)
    output = getattr(args, "output", None)

    config = _load_meta_config()

    if not kundb:
        kundb = config.get("kundb", "")
    if not workspace_uuid:
        workspace_uuid = config.get("workspace_uuid", "82ee37374b314a938bf28170ab4db7cf")

    if not kundb:
        kundb = questionary.text("元数据库 URL:").ask() or ""

    if not kundb:
        print("❌ 需要指定 --kundb 或在配置中设置 kundb", file=sys.stderr)
        sys.exit(1)

    output = Path(output) if output else Path(
        questionary.text("CSV 输出目录:", default=config.get("csv_dir", "./output")).ask() or "./output"
    )

    step_compliance_export(output, kundb, workspace_uuid)


def cmd_sync_metric(args: argparse.Namespace) -> None:
    """meta sync metric — 导入指标维度定义（Metric, Dimension + 边）"""
    if hasattr(args, "file") and args.file:
        metric_file = args.file
        output = Path(args.output) if args.output else Path("./output")
    else:
        config = _load_meta_config()
        metric_file = args.file or config.get("metric", "")
        if not metric_file:
            metric_file = questionary.text("指标定义 JSON 文件路径:").ask() or ""
        output = Path(args.output) if args.output else Path(
            questionary.text("CSV 输出目录:", default=config.get("csv_dir", "./output")).ask() or "./output"
        )

    if not metric_file:
        print("❌ 需要指定 --file", file=sys.stderr)
        sys.exit(1)

    step_metric_export(output, metric_file)


def cmd_sync_graph(args: argparse.Namespace) -> None:
    """meta sync graph — 更新图数据库 + 生成 assets"""
    output = Path(args.output) if args.output else Path("./output")
    graph_mode = args.mode if hasattr(args, "mode") and args.mode else "update"

    if not output.exists():
        print(f"❌ 输出目录不存在: {output}", file=sys.stderr)
        sys.exit(1)

    _update_graph(output, graph_mode)
    _generate_assets()

    print("\n✅ graph 更新完成！")


# ---------------------------------------------------------------------------
# 原有 CLI command handlers（保留兼容）
# ---------------------------------------------------------------------------

def cmd_sync(args: argparse.Namespace) -> None:
    """meta sync — 完整/增量同步管线"""

    if args.db is not None or args.schemas or args.db_name or args.output:
        # 命令行模式（至少传了一个参数）
        db_path = args.db or ""
        schemas = args.schemas.split(",") if args.schemas else None
        db_name = args.db_name
        output = Path(args.output) if args.output else Path("./output")
        graph_mode = "dry_run" if args.dry_run else "update"
        source = "auto"
    else:
        # 交互模式：从 meta_config.yaml 读取配置
        meta_cfg = MetaConfigManager()
        try:
            config = meta_cfg.load_or_migrate()
        except FileNotFoundError:
            print("错误: meta 配置文件不存在且无法迁移，请先运行 govio-cli meta config", file=sys.stderr)
            sys.exit(1)

        # 显示当前配置
        print("当前 meta 配置:")
        cfg_fields = [
            ("kundb", "元数据库"),
            ("workspace_uuid", "工作区 UUID"),
            ("app_list", "应用清单"),
            ("app_map", "应用映射"),
            ("relationship", "表关系"),
            ("metric", "指标定义"),
            ("csv_dir", "CSV 输出目录"),
        ]
        for key, label in cfg_fields:
            val = config.get(key)
            display = val if val else "(未设置)"
            print(f"  {label}: {display}")
        print()

        # 数据源选择
        source = questionary.select(
            "数据来源:",
            choices=[
                questionary.Choice("TDS — 仅从元数据库读取", value="tds"),
                questionary.Choice("DuckDB — 仅从 DuckDB 读取", value="duckdb"),
                questionary.Choice("Both — TDS + DuckDB 合并", value="both"),
            ],
        ).ask()

        if source == "tds":
            app_map_file = config.get("app_map", "")
            if not app_map_file:
                print("错误: app_map 未配置，无法获取 schema 列表", file=sys.stderr)
                sys.exit(1)
            schemas = _load_schemas_from_app_map(app_map_file)
            print(f"从 app_map 读取到 {len(schemas)} 个 schema: {schemas}")
            db_path = ""
            db_name = None
        elif source == "duckdb":
            db_path = questionary.text("DuckDB 数据库文件路径:").ask() or ""
            if not db_path:
                print("错误: DuckDB 模式必须指定数据库路径", file=sys.stderr)
                sys.exit(1)
            schemas_input = questionary.text("要导出的 schema 列表（逗号分隔，留空跳过）:").ask() or ""
            db_name = questionary.text("单库模式 app 名称（留空使用全量模式）:").ask() or None
            schemas = [s.strip() for s in schemas_input.split(",") if s.strip()] if schemas_input else None
        else:
            app_map_file = config.get("app_map", "")
            if not app_map_file:
                print("错误: app_map 未配置，无法获取 schema 列表", file=sys.stderr)
                sys.exit(1)
            tds_schemas = _load_schemas_from_app_map(app_map_file)
            print(f"从 app_map 读取到 {len(tds_schemas)} 个 schema: {tds_schemas}")
            db_path = questionary.text("DuckDB 数据库文件路径:").ask() or ""
            if not db_path:
                print("错误: Both 模式必须指定 DuckDB 路径", file=sys.stderr)
                sys.exit(1)
            schemas_input = questionary.text(
                "DuckDB schema 列表（逗号分隔，留空使用 TDS 相同 schema）:",
                default=",".join(tds_schemas),
            ).ask() or ",".join(tds_schemas)
            schemas = [s.strip() for s in schemas_input.split(",") if s.strip()] if schemas_input else tds_schemas
            db_name = None

        output_default = config.get("csv_dir", "") or config.get("output", "")
        output_str = questionary.text("CSV 输出目录:", default=output_default).ask() or output_default
        output = Path(output_str)

        graph_mode = questionary.select(
            "执行模式:",
            choices=[
                questionary.Choice("仅生成 CSV（不更新图库）", value="dry_run"),
                questionary.Choice("生成 CSV 并更新图库（增量 MERGE）", value="update"),
                questionary.Choice("生成 CSV 并重建图库（删除后重新插入）", value="rebuild"),
            ],
        ).ask()

    meta_export(
        db_path=db_path,
        schemas=schemas,
        db_name=db_name,
        output=output,
        graph_mode=graph_mode,
        source=source,
    )


def cmd_recommend(args: argparse.Namespace) -> None:
    """meta recommend — 数据标准推荐"""
    from .std_recommend import std_recommend

    meta_cfg = MetaConfigManager()
    try:
        config = meta_cfg.load_or_migrate()
    except FileNotFoundError:
        print("错误: meta 配置文件不存在且无法迁移，请先运行 govio-cli meta config", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir
    if not output_dir:
        output_dir = config.get("csv_dir", "") or config.get("output", "./output")

    std_recommend(Path(output_dir))


def cmd_config(args: argparse.Namespace) -> None:
    """meta config — 交互式配置 meta_config.yaml"""
    meta_cfg = MetaConfigManager()

    if meta_cfg.exists():
        config = meta_cfg.load()
        print("当前 meta 配置:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        print()

        modify = questionary.confirm("是否修改当前配置？", default=False).ask()
        if not modify:
            return
    else:
        print("meta 配置文件不存在，开始创建...")
        config = {}

    fields = [
        ("kundb", "元数据库 URL", ""),
        ("workspace_uuid", "工作区 UUID", "82ee37374b314a938bf28170ab4db7cf"),
        ("app_list", "应用清单 Excel 文件路径", ""),
        ("app_map", "应用数据库映射 JSON 文件路径", ""),
        ("relationship", "表关系 JSON 文件路径（可选，留空跳过）", ""),
        ("metric", "指标定义 JSON 文件路径（可选，留空跳过）", ""),
        ("csv_dir", "CSV 输出目录", ""),
    ]

    for key, prompt, default_val in fields:
        current = config.get(key, default_val)
        value = questionary.text(f"{prompt}:", default=str(current)).ask() or str(current)
        if key in ("relationship", "metric"):
            config[key] = value if value else None
        else:
            config[key] = value

    meta_cfg.save(config)
    print(f"\n配置已保存到: {meta_cfg.config_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _register_sync_subparsers(sub_sync: argparse._SubParsersAction) -> None:
    """注册 sync 子命令的子解析器"""

    # meta sync meta
    p_meta = sub_sync.add_parser("meta", help="导入 TDS/DuckDB 元数据（PhysicalTable, Col, HAS_COLUMN）")
    p_meta.add_argument("--source", choices=["tds", "duckdb", "both"], help="数据来源")
    p_meta.add_argument("--db", type=str, help="DuckDB 数据库文件路径")
    p_meta.add_argument("--schemas", type=str, help="要导出的 schema 列表，逗号分隔")
    p_meta.add_argument("--kundb", type=str, help="TDS 元数据库 URL")
    p_meta.add_argument("--workspace-uuid", type=str, help="工作区 UUID")
    p_meta.add_argument("--output", type=str, help="CSV 输出目录")
    p_meta.set_defaults(func=cmd_sync_meta)

    # meta sync app
    p_app = sub_sync.add_parser("app", help="导入应用清单（Application 节点 + USE 边）")
    p_app.add_argument("--app-list", type=str, help="应用清单 Excel 文件路径")
    p_app.add_argument("--app-map", type=str, help="应用数据库映射 JSON 文件路径")
    p_app.add_argument("--db-name", type=str, help="单库模式：仅导出指定应用")
    p_app.add_argument("--output", type=str, help="CSV 输出目录")
    p_app.set_defaults(func=cmd_sync_app)

    # meta sync rel
    p_rel = sub_sync.add_parser("rel", help="导入表关系（RELATES_TO 边）")
    p_rel.add_argument("--file", type=str, help="表关系 JSON 文件路径")
    p_rel.add_argument("--output", type=str, help="CSV 输出目录")
    p_rel.set_defaults(func=cmd_sync_rel)

    # meta sync std
    p_std = sub_sync.add_parser("std", help="导入数据标准（Standard 节点）")
    p_std.add_argument("--kundb", type=str, help="TDS 元数据库 URL")
    p_std.add_argument("--workspace-uuid", type=str, help="工作区 UUID")
    p_std.add_argument("--output", type=str, help="CSV 输出目录")
    p_std.set_defaults(func=cmd_sync_std)

    # meta sync compliance
    p_comp = sub_sync.add_parser("compliance", help="从 TDS 导出已有标准-字段关联（COMPLIES_WITH 边）")
    p_comp.add_argument("--kundb", type=str, help="TDS 元数据库 URL")
    p_comp.add_argument("--workspace-uuid", type=str, help="工作区 UUID")
    p_comp.add_argument("--output", type=str, help="CSV 输出目录")
    p_comp.set_defaults(func=cmd_sync_compliance)

    # meta sync metric
    p_metric = sub_sync.add_parser("metric", help="导入指标维度定义（Metric, Dimension + 边）")
    p_metric.add_argument("--file", type=str, help="指标定义 JSON 文件路径")
    p_metric.add_argument("--output", type=str, help="CSV 输出目录")
    p_metric.set_defaults(func=cmd_sync_metric)

    # meta sync graph
    p_graph = sub_sync.add_parser("graph", help="更新图数据库 + 生成 assets")
    p_graph.add_argument("--output", type=str, help="CSV 输出目录")
    p_graph.add_argument("--mode", choices=["update", "rebuild"], default="update", help="更新模式")
    p_graph.set_defaults(func=cmd_sync_graph)


def meta():
    """meta 命令入口"""
    parser = argparse.ArgumentParser(
        prog="govio-cli meta",
        description="知识图库维护 — 元数据同步、推荐、配置",
    )
    sub = parser.add_subparsers(dest="action", required=True)

    # meta sync
    p_sync = sub.add_parser("sync", help="完整/增量同步管线：读取元数据源 → 生成 CSV → 更新图数据 → 生成 assets")
    p_sync.add_argument("--db", type=str, help="DuckDB 数据库文件路径")
    p_sync.add_argument("--schemas", type=str, help="要导出的 schema 列表，逗号分隔")
    p_sync.add_argument("--db-name", type=str, help="单库模式：按 app 名导出单个数据库的相关子图")
    p_sync.add_argument("--output", type=str, help="CSV 输出目录")
    p_sync.add_argument("--dry-run", action="store_true", help="仅生成 CSV 并输出状态，不更新图数据和生成 assets")

    # sync 子命令（meta sync meta / app / rel / std / compliance / metric / graph）
    sync_sub = p_sync.add_subparsers(dest="sync_action")
    _register_sync_subparsers(sync_sub)

    # meta recommend
    p_recommend = sub.add_parser("recommend", help="数据标准推荐")
    p_recommend.add_argument("--output-dir", type=str, help="推荐数据标准的输出目录")

    # meta config
    sub.add_parser("config", help="交互式配置 meta_config.yaml")

    args = parser.parse_args(sys.argv[1:])

    match args.action:
        case "sync":
            if hasattr(args, "sync_action") and args.sync_action:
                # 子命令模式：meta sync meta / app / rel / std / compliance / metric / graph
                args.func(args)
            else:
                # 完整管线模式：meta sync（向后兼容）
                cmd_sync(args)
        case "recommend":
            cmd_recommend(args)
        case "config":
            cmd_config(args)


if __name__ == "__main__":
    meta()
