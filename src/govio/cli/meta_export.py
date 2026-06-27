import sys
from pathlib import Path

import pandas as pd

from govio.cli.config import ConfigManager
from govio.core.graph_factory import GraphFactory
from govio.core.assets_generator import AssetsGenerator
from govio.metadata.database import TDSLoader
from govio.metadata.application import AppInfoLoader
from govio.metadata.standard import StandardLoader
from govio.metadata.duckdb_loader import DuckDBLoader
from govio.metadata.relationship import load_relationships
from govio.metadata.metric import MetricLoader
from govio.metadata.node_id import assign_node_ids, write_node_csv

SKILLS_ASSETS_DIR = Path("skills/govio/assets")


def merge_metadata(
    df_tds: pd.DataFrame, df_duck: pd.DataFrame, key: str
) -> pd.DataFrame:
    """TDS full + DuckDB incremental. DuckDB wins on conflict."""
    combined = pd.concat([df_tds, df_duck], ignore_index=True)
    return combined.drop_duplicates(subset=[key], keep="last").reset_index(drop=True)


def meta_export(
    db_path: str,
    schemas: list[str] | None,
    db_name: str | None,
    output: Path,
    dry_run: bool = True,
):
    output.mkdir(parents=True, exist_ok=True)

    if not schemas and not db_name:
        print("错误: 必须指定 --schemas 或 --db-name 之一", file=sys.stderr)
        sys.exit(1)

    # --- Load config for TDS ---
    config = ConfigManager().load()
    metadata = config.get("metadata") or {}
    kundb = metadata.get("kundb", "")
    workspace_uuid = metadata.get("workspace_uuid", "82ee37374b314a938bf28170ab4db7cf")
    app_list_file = metadata.get("app_list", "")
    app_map_file = metadata.get("app_map", "")
    relationship_file = metadata.get("relationship")
    metric_file = metadata.get("metric")

    if not all([kundb, app_list_file, app_map_file]):
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

    # --- Load TDS metadata ---
    tds_loader = TDSLoader(kundb, workspace_uuid, df_app_db_map["schema"].to_list())
    tds_tables = tds_loader.PhysicalTable
    tds_columns = tds_loader.Col

    # --- Load DuckDB metadata ---
    duck_loader = DuckDBLoader(db_path, schemas)
    duck_tables = duck_loader.PhysicalTable
    duck_columns = duck_loader.Col

    # --- Merge ---
    df_tables = merge_metadata(tds_tables, duck_tables, "full_table_name")
    df_columns = merge_metadata(tds_columns, duck_columns, "column")

    # --- Load apps and standards ---
    app_loader = AppInfoLoader(app_list_file, df_app_db_map["name"].to_list())
    df_apps = app_loader.Application
    std_loader = StandardLoader(kundb, workspace_uuid)
    df_stds = std_loader.Standard

    # --- Assign string IDs ---
    df_tables = df_tables.reset_index(drop=True)
    df_columns = df_columns.reset_index(drop=True)
    df_apps = df_apps.reset_index(drop=True)
    df_stds = df_stds.reset_index(drop=True)
    assign_node_ids(df_tables, "PhysicalTable", "full_table_name")
    assign_node_ids(df_columns, "Col", "column")
    assign_node_ids(df_apps, "Application", "app_id")
    assign_node_ids(df_stds, "Standard", "standard_id")

    files = []

    # --- Node CSVs ---
    write_node_csv(df_tables, output / "PhysicalTable.csv", "PhysicalTable")
    files.append("-n " + str(output / "PhysicalTable.csv"))

    write_node_csv(df_columns, output / "Col.csv", "Col")
    files.append("-n " + str(output / "Col.csv"))

    write_node_csv(df_apps, output / "Application.csv", "Application")
    files.append("-n " + str(output / "Application.csv"))

    write_node_csv(df_stds, output / "Standard.csv", "Standard")
    files.append("-n " + str(output / "Standard.csv"))

    # --- HAS_COLUMN edge ---
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
    df_has_column.to_csv(output / "HAS_COLUMN.csv", index=False)
    files.append("-r " + str(output / "HAS_COLUMN.csv"))

    # --- USE edge ---
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
    df_use.to_csv(output / "USE.csv", index=False)
    files.append("-r " + str(output / "USE.csv"))

    # --- Optional: RELATES_TO ---
    relations_count = 0
    if relationship_file:
        try:
            df_relates_to = load_relationships(relationship_file, df_tables, df_columns)
            relations_count = len(df_relates_to)
            table_idx_to_id = df_tables["node_id"].tolist()
            if not df_relates_to.empty:
                df_relates_to["source"] = [
                    table_idx_to_id[i] for i in df_relates_to["source"]
                ]
                df_relates_to["target"] = [
                    table_idx_to_id[i] for i in df_relates_to["target"]
                ]
            df_relates_to.to_csv(
                output / "RELATES_TO.csv",
                index=False,
                header=[
                    ":START_ID(PhysicalTable)",
                    ":END_ID(PhysicalTable)",
                    "relationship_type",
                    "description",
                    "source_columns",
                    "target_columns",
                ],
            )
            files.append("-r " + str(output / "RELATES_TO.csv"))
            print(f"成功生成 RELATES_TO.csv，包含 {len(df_relates_to)} 个关系 来自[{relationship_file}]")
        except Exception as e:
            print(f"警告: 无法加载关系文件: {e}")

    # --- Optional: metrics ---
    metric_count = 0
    if metric_file:
        try:
            metric_loader = MetricLoader(metric_file, df_tables, df_columns)
            df_metrics = metric_loader.Metric.reset_index(drop=True)
            df_dimensions = metric_loader.Dimension.reset_index(drop=True)

            assign_node_ids(df_metrics, "Metric", "code")
            assign_node_ids(df_dimensions, "Dimension", "code")

            write_node_csv(df_metrics, output / "Metric.csv", "Metric")
            files.append("-n " + str(output / "Metric.csv"))

            write_node_csv(df_dimensions, output / "Dimension.csv", "Dimension")
            files.append("-n " + str(output / "Dimension.csv"))

            # positional index -> node_id 映射
            metric_idx_to_id = df_metrics["node_id"].tolist()
            dim_idx_to_id = df_dimensions["node_id"].tolist()
            table_idx_to_id = df_tables["node_id"].tolist()
            col_idx_to_id = df_columns["node_id"].tolist()

            # USES_TABLE 边
            uses_table = metric_loader.uses_table_edges.copy()
            if not uses_table.empty:
                uses_table[":START_ID(Metric)"] = [
                    metric_idx_to_id[i] for i in uses_table[":START_ID(Metric)"]
                ]
                uses_table[":END_ID(PhysicalTable)"] = [
                    table_idx_to_id[i] for i in uses_table[":END_ID(PhysicalTable)"]
                ]
                uses_table.to_csv(output / "USES_TABLE.csv", index=False)
                files.append("-r " + str(output / "USES_TABLE.csv"))

            # REFERS_COLUMN 边
            refers_col = metric_loader.refers_column_edges.copy()
            if not refers_col.empty:
                refers_col[":START_ID(Metric)"] = [
                    metric_idx_to_id[i] for i in refers_col[":START_ID(Metric)"]
                ]
                refers_col[":END_ID(Col)"] = [
                    col_idx_to_id[i] for i in refers_col[":END_ID(Col)"]
                ]
                refers_col.to_csv(output / "REFERS_COLUMN.csv", index=False)
                files.append("-r " + str(output / "REFERS_COLUMN.csv"))

            # DERIVED_FROM 边
            derived_from = metric_loader.derived_from_edges.copy()
            if not derived_from.empty:
                derived_from[":START_ID(Metric)"] = [
                    metric_idx_to_id[i] for i in derived_from[":START_ID(Metric)"]
                ]
                derived_from[":END_ID(Metric)"] = [
                    metric_idx_to_id[i] for i in derived_from[":END_ID(Metric)"]
                ]
                derived_from.to_csv(output / "DERIVED_FROM.csv", index=False)
                files.append("-r " + str(output / "DERIVED_FROM.csv"))

            # DIMENSION_USED 边
            dim_used = metric_loader.dimension_used_edges.copy()
            if not dim_used.empty:
                dim_used[":START_ID(Metric)"] = [
                    metric_idx_to_id[i] for i in dim_used[":START_ID(Metric)"]
                ]
                dim_used[":END_ID(Dimension)"] = [
                    dim_idx_to_id[i] for i in dim_used[":END_ID(Dimension)"]
                ]
                dim_used.to_csv(output / "DIMENSION_USED.csv", index=False)
                files.append("-r " + str(output / "DIMENSION_USED.csv"))

            # SUPERSEDES 边
            supersedes = metric_loader.supersedes_edges.copy()
            if not supersedes.empty:
                supersedes[":START_ID(Metric)"] = [
                    metric_idx_to_id[i] for i in supersedes[":START_ID(Metric)"]
                ]
                supersedes[":END_ID(Metric)"] = [
                    metric_idx_to_id[i] for i in supersedes[":END_ID(Metric)"]
                ]
                supersedes.to_csv(output / "SUPERSEDES.csv", index=False)
                files.append("-r " + str(output / "SUPERSEDES.csv"))

            print(
                f"成功生成指标数据：{len(df_metrics)} 个指标, "
                f"{len(df_dimensions)} 个维度"
            )
            metric_count = len(df_metrics)
        except Exception as e:
            print(f"警告: 无法加载指标定义文件: {e}")
            exit(1)

    # --- Summary ---
    print(f"成功导出: {len(df_tables)} 张表, {len(df_columns)} 个字段, "
          f"{len(df_apps)} 个应用, {len(df_stds)} 个标准, {relations_count}个数据关系, {metric_count}个指标")
    # print(f"ID 范围: 1 ~ {len(df_tables) + len(df_columns) + len(df_apps) + len(df_stds)}")
    print(f"\nfalkordb-bulk-insert {{GRAPH}} {'  '.join(files)}")

    if dry_run:
        return

    # --- Update graph and generate assets ---
    from govio.cli.onboard import import_csv_to_falkordb
    from govio.metadata.gen_networkx import build_graph

    graph = config.get("graph") or {}
    backend = graph.get("backend")
    if not backend:
        print("警告: 配置中未指定 backend，跳过图数据更新和 assets 生成")
        return

    # Update graph
    if backend == "falkordb":
        falkordb_cfg = graph.get("falkordb", {})
        host = falkordb_cfg.get("host", "localhost")
        port = falkordb_cfg.get("port", 6379)
        graph_name = falkordb_cfg.get("graph", "ontology")
        print(f"\n正在导入 CSV 到 FalkorDB ({host}:{port}/{graph_name})...")
        try:
            import_csv_to_falkordb(output, host, port, graph_name)
            print("✓ FalkorDB 数据已更新")
        except Exception as e:
            print(f"❌ 导入 FalkorDB 失败: {e}")
            return
    elif backend == "networkx":
        networkx_cfg = graph.get("networkx", {})
        gml_path = networkx_cfg.get("gml_path", str(SKILLS_ASSETS_DIR / "ontology.gml"))
        print(f"\n正在从 CSV 生成 GML 文件 ({gml_path})...")
        try:
            build_graph(str(output), gml_path)
            print("✓ GML 文件已更新")
        except Exception as e:
            print(f"❌ 生成 GML 失败: {e}")
            return

    # Generate assets
    print("\n正在生成 assets...")
    try:
        graph_obj = GraphFactory.create(graph)
        generator = AssetsGenerator(graph_obj, SKILLS_ASSETS_DIR)
        generator.generate_all()
        print(f"✓ Assets 已生成到: {SKILLS_ASSETS_DIR}")
    except Exception as e:
        print(f"❌ 生成 assets 失败: {e}")
        return

    print("\n✅ meta-export 完成！")
