"""meta 命令组 — 知识图库维护

提供元数据同步、导出、数据标准推荐、配置管理等功能。
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
from govio.metadata.database import TDSLoader
from govio.metadata.application import AppInfoLoader
from govio.metadata.standard import StandardLoader
from govio.metadata.duckdb_loader import DuckDBLoader
from govio.metadata.relationship import load_relationships
from govio.metadata.metric import MetricLoader
from govio.metadata.node_id import assign_node_ids, write_node_csv

SKILLS_ASSETS_DIR = Path("skills/govio/assets")


# ---------------------------------------------------------------------------
# merge / export helpers
# ---------------------------------------------------------------------------

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
    graph_mode: str = "dry_run",
    source: str = "auto",
):
    """source: "tds" | "duckdb" | "both" | "auto"（auto 按 db_path 有无自动判断）"""
    output.mkdir(parents=True, exist_ok=True)

    # 清理输出目录下的旧 CSV 文件
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

    # --- Load config for TDS ---
    # 优先从 meta_config.yaml 读取，兼容旧的 config.yaml metadata section
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

    # graph 配置始终从 config.yaml 读取
    graph_config = ConfigManager().load()

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

    # --- 决定要抽取的 schema 集合 ---
    if db_name:
        # 单库模式：app_map 里该 app 对应的 schema
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

    # --- Load metadata ---
    if source == "tds":
        print("从 TDS 读取元数据...")
        tds_schemas = effective_schemas if effective_schemas else None
        tds_loader = TDSLoader(kundb, workspace_uuid, tds_schemas)
        df_tables = tds_loader.PhysicalTable
        df_columns = tds_loader.Col
    elif source == "duckdb":
        print("从 DuckDB 读取元数据...")
        duck_loader = DuckDBLoader(db_path, effective_schemas)
        df_tables = duck_loader.PhysicalTable
        df_columns = duck_loader.Col
    else:
        # both: TDS + DuckDB 合并
        print("从 TDS + DuckDB 合并读取元数据...")
        tds_schemas = effective_schemas if effective_schemas else None
        tds_loader = TDSLoader(kundb, workspace_uuid, tds_schemas)
        tds_tables = tds_loader.PhysicalTable
        tds_columns = tds_loader.Col
        duck_loader = DuckDBLoader(db_path, effective_schemas)
        duck_tables = duck_loader.PhysicalTable
        duck_columns = duck_loader.Col
        df_tables = merge_metadata(tds_tables, duck_tables, "full_table_name")
        df_columns = merge_metadata(tds_columns, duck_columns, "column")

    # --- Load apps and standards ---
    app_names = [db_name] if db_name else df_app_db_map["name"].to_list()
    app_loader = AppInfoLoader(app_list_file, app_names)
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
    table_idx_to_id = df_tables["node_id"].tolist()
    col_idx_to_id = df_columns["node_id"].tolist()

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
            relations_count = len(df_relates_to)
            print(f"成功生成 RELATES_TO.csv，包含 {len(df_relates_to)} 个关系 来自[{relationship_file}]")
        except Exception as e:
            print(f"警告: 无法加载关系文件: {e}")

    # --- Optional: metrics (TDS-only 模式跳过) ---
    metric_count = 0
    if metric_file and source != "tds":
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
            sys.exit(1)
    elif metric_file and source == "tds":
        print("提示: TDS-only 模式跳过指标定义加载")

    # --- Summary ---
    print(f"成功导出: {len(df_tables)} 张表, {len(df_columns)} 个字段, "
          f"{len(df_apps)} 个应用, {len(df_stds)} 个标准, {relations_count}个数据关系, {metric_count}个指标")
    print(f"\nfalkordb-bulk-insert {{GRAPH}} {'  '.join(files)}")

    if graph_mode == "dry_run":
        return

    # --- Update/rebuild graph ---
    from govio.metadata.gen_networkx import build_graph

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
            return
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
            return
    else:
        print("提示: 未配置 graph backend，跳过图数据更新")

    # --- Generate assets ---
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


# ---------------------------------------------------------------------------
# CLI command handlers
# ---------------------------------------------------------------------------

def _load_schemas_from_app_map(app_map_file: str) -> list[str]:
    """从 app_map.json 读取所有 schema"""
    with open(app_map_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return list({row["schema"] for row in data})


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
            # TDS 模式：从 app_map 读取 schemas
            app_map_file = config.get("app_map", "")
            if not app_map_file:
                print("错误: app_map 未配置，无法获取 schema 列表", file=sys.stderr)
                sys.exit(1)
            schemas = _load_schemas_from_app_map(app_map_file)
            print(f"从 app_map 读取到 {len(schemas)} 个 schema: {schemas}")
            db_path = ""
            db_name = None
        elif source == "duckdb":
            # DuckDB 模式：需要 db_path + schemas/db_name
            db_path = questionary.text("DuckDB 数据库文件路径:").ask() or ""
            if not db_path:
                print("错误: DuckDB 模式必须指定数据库路径", file=sys.stderr)
                sys.exit(1)
            schemas_input = questionary.text("要导出的 schema 列表（逗号分隔，留空跳过）:").ask() or ""
            db_name = questionary.text("单库模式 app 名称（留空使用全量模式）:").ask() or None
            schemas = [s.strip() for s in schemas_input.split(",") if s.strip()] if schemas_input else None
        else:
            # both 模式：TDS + DuckDB
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

        # CSV 输出目录
        output_default = config.get("csv_dir", "") or config.get("output", "")
        output_str = questionary.text("CSV 输出目录:", default=output_default).ask() or output_default
        output = Path(output_str)

        # 执行模式选择
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

    # meta recommend
    p_recommend = sub.add_parser("recommend", help="数据标准推荐")
    p_recommend.add_argument("--output-dir", type=str, help="推荐数据标准的输出目录")

    # meta config
    sub.add_parser("config", help="交互式配置 meta_config.yaml")

    args = parser.parse_args(sys.argv[1:])

    match args.action:
        case "sync":
            cmd_sync(args)
        case "recommend":
            cmd_recommend(args)
        case "config":
            cmd_config(args)


if __name__ == "__main__":
    meta()
