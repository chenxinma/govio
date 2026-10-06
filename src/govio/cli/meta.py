"""meta 命令组 — 知识图库维护

提供元数据导入、JSON Schema 输出、数据标准推荐等功能。

各导入子命令（meta / std / compliance / rel / metric）完全独立运行，
所有输入通过 CLI 参数显式传入，不依赖配置文件。通过 output 目录的 CSV 文件
作为共享状态，支持增量合并（幂等）。

推荐执行顺序：meta → std → compliance → rel → metric → graph
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .config import ConfigManager
from govio.core.graph_factory import GraphFactory
from govio.core.assets_generator import AssetsGenerator
from govio.graph.falkordb_loader import import_csv_to_falkordb, upsert_csv_to_falkordb
from govio.graph.ladybug_loader import import_csv_to_ladybug, upsert_csv_to_ladybug
from govio.metadata.database import TDSLoader
from govio.metadata.datasource import (
    DatasourceLoader,
    build_owns_edges,
    filter_frames,
    make_datasource_def,
)
from govio.metadata.duckdb_loader import DuckDBLoader
from govio.metadata.metric import MetricLoader
from govio.metadata.node_id import assign_node_ids, write_node_csv
from govio.metadata.relationship import load_relationships
from govio.metadata.standard import StandardLoader

DEFAULT_ASSETS_DIR = Path(".agents/skills/govio/assets")


# ---------------------------------------------------------------------------
# CSV merge helpers — 增量合并的核心机制
# ---------------------------------------------------------------------------


def merge_node_csv(
    new_df: pd.DataFrame,
    csv_path: Path,
    node_type: str,
    key_col: str,
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
            if "node_id" in existing.columns:
                # 两列并存时丢弃 :ID 列（值相同）
                existing = existing.drop(columns=[id_col])
            else:
                existing = existing.rename(columns={id_col: "node_id"})
        combined = pd.concat([existing, new_df], ignore_index=True)
        dedup_col = key_col if key_col in combined.columns else None
        if dedup_col:
            merged = combined.drop_duplicates(subset=[dedup_col], keep="last")
        else:
            print(
                f"⚠ merge_node_csv: 未找到去重列 '{key_col}'，跳过去重", file=sys.stderr
            )
            merged = combined
    else:
        merged = new_df

    merged = merged.reset_index(drop=True)
    write_node_csv(merged, csv_path, node_type)
    return merged


def merge_edge_csv(
    new_df: pd.DataFrame,
    csv_path: Path,
    dedup_cols: list[str],
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
            print(
                f"⚠ merge_edge_csv: 未找到去重列 {dedup_cols}，跳过去重",
                file=sys.stderr,
            )
            merged = combined
    else:
        merged = new_df

    merged = merged.reset_index(drop=True)
    merged.to_csv(csv_path, index=False)
    return merged


def _load_csv_with_node_ids(
    csv_path: Path,
    node_type: str,
    key_col: str,
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


def _update_graph(output: Path, graph_mode: str, assets_dir: Path) -> bool:
    """更新图数据库。返回是否成功。"""
    from govio.metadata.gen_networkx import build_graph

    graph_config = ConfigManager().load()
    graph = graph_config.get("graph") or {}
    backend = graph.get("backend")
    incremental = graph_mode == "update"

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
        gml_path = networkx_cfg.get("gml_path", str(assets_dir / "ontology.gml"))
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
                upsert_csv_to_ladybug(
                    output, db_path_val, buffer_pool_size=bp, max_db_size=maxdb
                )
                print("✓ Ladybug 数据已更新")
            else:
                # rebuild 先删旧文件，避免版本不兼容导致无法打开
                db_file = Path(db_path_val)
                if db_file.exists():
                    db_file.unlink()
                import_csv_to_ladybug(
                    output, db_path_val, buffer_pool_size=bp, max_db_size=maxdb
                )
                print("✓ Ladybug 数据已重建")
        except Exception as e:
            print(f"❌ 导入 Ladybug 失败: {e}")
            return False
    else:
        print("提示: 未配置 graph backend，跳过图数据更新")
    return True


def _clear_graph(assets_dir: Path) -> bool:
    """清空图数据库。返回是否成功。"""
    graph_config = ConfigManager().load()
    graph = graph_config.get("graph") or {}
    backend = graph.get("backend")

    if backend == "falkordb":
        from govio.graph.falkordb_loader import delete_falkordb_graph

        falkordb_cfg = graph.get("falkordb", {})
        host = falkordb_cfg.get("host", "localhost")
        port = falkordb_cfg.get("port", 6379)
        graph_name = falkordb_cfg.get("graph", "ontology")
        try:
            delete_falkordb_graph(host, port, graph_name)
            print(f"✓ FalkorDB 图 '{graph_name}' 已删除")
        except Exception as e:
            print(f"❌ 删除 FalkorDB 图失败: {e}")
            return False
    elif backend == "networkx":
        networkx_cfg = graph.get("networkx", {})
        gml_path = networkx_cfg.get("gml_path", str(assets_dir / "ontology.gml"))
        gml_file = Path(gml_path)
        if gml_file.exists():
            gml_file.unlink()
            print(f"✓ GML 文件已删除: {gml_path}")
        else:
            print("提示: GML 文件不存在，无需删除")
    elif backend == "ladybug":
        ladybug_cfg = graph.get("ladybug", {})
        db_path_val = ladybug_cfg.get("db_path")
        if not db_path_val:
            print("❌ Ladybug 配置缺少 db_path")
            return False
        db_file = Path(db_path_val)
        if db_file.exists():
            db_file.unlink()
            print(f"✓ Ladybug 数据库已删除: {db_path_val}")
        else:
            print("提示: Ladybug 数据库文件不存在，无需删除")
    else:
        print("提示: 未配置 graph backend，跳过")
    return True


def _generate_assets(assets_dir: Path) -> None:
    """生成 schema.md、name 索引、metrics_index.md 等 assets。"""
    print("\n正在生成 assets...")
    try:
        graph_config = ConfigManager().load()
        graph = graph_config.get("graph") or {}
        graph_obj = GraphFactory.create(graph)
        resolved = assets_dir.resolve()
        generator = AssetsGenerator(graph_obj, resolved)
        generator.generate_all()
        print(f"✓ Assets 已生成到: {resolved}")
    except Exception as e:
        print(f"❌ 生成 assets 失败: {e}")


# ---------------------------------------------------------------------------
# Step functions — 可独立调用的管线步骤
# ---------------------------------------------------------------------------


def _describe_duckdb_schemas(db_path: str) -> str:
    """只读列出 DuckDB 文件中可导入的 schema，用于空结果时的错误提示。"""
    try:
        pairs = DuckDBLoader(db_path, []).list_schemas()
    except Exception as e:
        return f"（读取 schema 列表失败: {e}）"
    if not pairs:
        return "（该文件没有可导入的 schema）"
    return "、".join(f"{name}({count} 张表)" for name, count in pairs)


def step_meta_export(
    output: Path,
    source: str,
    db_path: str = "",
    schemas: list[str] | None = None,
    datasource_name: str = "",
    datasource_file: str = "",
    kundb: str = "",
    workspace_uuid: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """步骤：导出元数据（Datasource, PhysicalTable, Col, HAS_COLUMN, OWNS）。

    单次运行覆盖一种信息来源（TDS 或一个实际数据源）。

    Returns:
        (df_tables, df_columns) 或 None（出错时）
    """
    output.mkdir(parents=True, exist_ok=True)

    # 加载元数据
    if source == "tds":
        if not kundb:
            print("错误: TDS 模式需要指定 --kundb", file=sys.stderr)
            return None
        print("从 TDS 读取元数据...")
        try:
            ds_loader = DatasourceLoader(datasource_file)
        except Exception as e:
            print(f"❌ 无法加载数据源声明文件: {e}", file=sys.stderr)
            return None
        ds_map = ds_loader.datasource_map()
        tds_loader = TDSLoader(
            kundb, workspace_uuid, ds_loader.all_schemas(), ds_map
        )
        df_tables = tds_loader.PhysicalTable
        df_columns = tds_loader.Col
        df_tables, df_columns = filter_frames(
            df_tables, df_columns, ds_loader.matches_table
        )
        df_datasources = ds_loader.Datasource
    else:
        if not db_path:
            print("错误: DuckDB 模式需要指定 --db 路径", file=sys.stderr)
            return None
        if not datasource_name:
            print("错误: DuckDB 模式需要指定 --datasource", file=sys.stderr)
            return None
        print("从 DuckDB 读取元数据...")
        declared = None
        if datasource_file:
            try:
                ds_loader = DatasourceLoader(datasource_file)
                declared = ds_loader.get(datasource_name)
            except KeyError:
                print(
                    f"⚠ 声明文件中不存在 datasource '{datasource_name}'，"
                    "按 CLI 参数自动声明"
                )
            except Exception as e:
                print(f"❌ 无法加载数据源声明文件: {e}", file=sys.stderr)
                return None
        ds_def = make_datasource_def(
            datasource_name, "duckdb", schemas or [], declared
        )
        duck_loader = DuckDBLoader(db_path, schemas or [], datasource_name)
        df_tables = duck_loader.PhysicalTable
        df_columns = duck_loader.Col
        df_tables, df_columns = filter_frames(
            df_tables, df_columns, ds_def.matches_table
        )
        df_datasources = pd.DataFrame(
            [ds_def.to_row()],
            columns=["datasource_name", "comment", "source_type", "filter"],
        )
        ds_map = {s: datasource_name for s in ds_def.schemas}

    # 空结果守卫：schema 写错或源库为空时直接失败，不写任何 CSV
    if df_tables.empty:
        schema_desc = ", ".join(schemas) if schemas else "（未指定）"
        msg = f"❌ 未发现任何表: schema [{schema_desc}] 在元数据源中不存在或为空（未写入任何 CSV）"
        if source == "duckdb" and db_path:
            msg += (
                f"\n   {db_path} 可导入的 schema: {_describe_duckdb_schemas(db_path)}"
            )
            msg += "\n   请确认 --schemas 后重试"
        print(msg, file=sys.stderr)
        return None

    # Assign IDs
    df_tables = df_tables.reset_index(drop=True)
    df_columns = df_columns.reset_index(drop=True)
    df_datasources = df_datasources.reset_index(drop=True)
    assign_node_ids(df_tables, "PhysicalTable", "full_table_name")
    assign_node_ids(df_columns, "Col", "column")
    assign_node_ids(df_datasources, "Datasource", "datasource_name")

    # Merge + write node CSVs
    pt_path = output / "PhysicalTable.csv"
    col_path = output / "Col.csv"
    ds_path = output / "Datasource.csv"
    df_tables = merge_node_csv(df_tables, pt_path, "PhysicalTable", "full_table_name")
    df_columns = merge_node_csv(df_columns, col_path, "Col", "column")
    df_datasources = merge_node_csv(
        df_datasources, ds_path, "Datasource", "datasource_name"
    )

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

    # OWNS edge
    df_owns = build_owns_edges(df_datasources, df_tables, ds_map)
    merge_edge_csv(
        df_owns,
        output / "OWNS.csv",
        [":START_ID(Datasource)", ":END_ID(PhysicalTable)"],
    )

    print(
        f"✓ 元数据已导出: {len(df_datasources)} 个数据源, "
        f"{len(df_tables)} 张表, {len(df_columns)} 个字段"
    )
    return df_tables, df_columns


def step_rel_export(
    output: Path,
    relationship_file: str,
) -> None:
    """步骤：导出表关系（RELATES_TO 边）。"""
    output.mkdir(parents=True, exist_ok=True)

    pt_path = output / "PhysicalTable.csv"
    col_path = output / "Col.csv"
    if not pt_path.exists() or not col_path.exists():
        print("❌ 需要先导入元数据（PhysicalTable.csv, Col.csv），请先运行 meta meta")
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
            df_relates_to = df_relates_to.rename(
                columns={
                    "source": ":START_ID(PhysicalTable)",
                    "target": ":END_ID(PhysicalTable)",
                }
            )

        rel_path = output / "RELATES_TO.csv"
        merge_edge_csv(
            df_relates_to,
            rel_path,
            [":START_ID(PhysicalTable)", ":END_ID(PhysicalTable)", "relationship_type"],
        )
        print(
            f"✓ RELATES_TO 已导出: {len(df_relates_to)} 个关系 来自[{relationship_file}]"
        )
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
    datasource_file: str,
) -> None:
    """步骤：从 TDS 导出已有标准-字段关联（COMPLIES_WITH 边）。"""
    output.mkdir(parents=True, exist_ok=True)

    col_path = output / "Col.csv"
    std_path = output / "Standard.csv"
    if not col_path.exists():
        print("❌ 需要先导入元数据（Col.csv），请先运行 meta meta")
        return
    if not std_path.exists():
        print("❌ 需要先导入数据标准（Standard.csv），请先运行 meta std")
        return

    df_columns = _load_csv_with_node_ids(col_path, "Col", "column")
    df_stds = _load_csv_with_node_ids(std_path, "Standard", "standard_id")

    try:
        ds_loader = DatasourceLoader(datasource_file)
    except Exception as e:
        print(f"❌ 无法加载数据源声明文件: {e}", file=sys.stderr)
        return

    std_loader = StandardLoader(kundb, workspace_uuid, ds_loader.datasource_map())
    df_compliance = std_loader.StdCompliance  # 已贯标列

    if df_compliance.empty:
        print("✓ TDS 中无已有标准关联数据")
        return

    # column 字段已由 StandardLoader 生成（全限定标识），直接匹配 Col.csv

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
        print(
            "❌ 需要先导入元数据（PhysicalTable.csv, Col.csv），请先运行 meta meta",
            file=sys.stderr,
        )
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
        df_dimensions = merge_node_csv(
            df_dimensions, output / "Dimension.csv", "Dimension", "code"
        )

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
                uses_table,
                output / "USES_TABLE.csv",
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
                refers_col,
                output / "REFERS_COLUMN.csv",
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
                derived_from,
                output / "DERIVED_FROM.csv",
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
                dim_used,
                output / "DIMENSION_USED.csv",
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
                supersedes,
                output / "SUPERSEDES.csv",
                [":START_ID(Metric)", ":END_ID(Metric)"],
            )

        print(
            f"✓ 指标数据已导出: {len(df_metrics)} 个指标, {len(df_dimensions)} 个维度"
        )
        return True
    except Exception as e:
        print(f"❌ 无法加载指标定义文件: {e}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# CLI command handlers — 独立子命令（CLI-only，无配置文件依赖）
# ---------------------------------------------------------------------------


def cmd_meta(args: argparse.Namespace) -> None:
    """meta meta — 导入 TDS/DuckDB 元数据（Datasource, PhysicalTable, Col, HAS_COLUMN, OWNS）"""
    source = args.source
    db_path = args.db or ""
    schemas = [s.strip() for s in args.schemas.split(",")] if args.schemas else []
    schemas = [s for s in schemas if s]
    output = Path(args.output) if args.output else Path("./output")
    kundb = args.kundb or ""
    workspace_uuid = args.workspace_uuid or ""
    datasource_name = args.datasource or ""
    datasource_file = args.datasources_file or ""

    if source == "tds":
        # TDS 抽取范围由声明文件 filter.schemas 决定，不接受 --schemas
        missing = []
        if not kundb:
            missing.append("--kundb")
        if not workspace_uuid:
            missing.append("--workspace-uuid")
        if not datasource_file:
            missing.append("--datasources-file")
        if missing:
            print(f"❌ TDS 模式需要指定: {', '.join(missing)}", file=sys.stderr)
            sys.exit(1)
    else:
        # DuckDB 过滤由 CLI --schemas 执行：省略会静默导出 0 张表
        if not db_path:
            print("❌ DuckDB 模式需要指定 --db", file=sys.stderr)
            sys.exit(1)
        if not datasource_name:
            print("❌ DuckDB 模式需要指定 --datasource", file=sys.stderr)
            sys.exit(1)
        if not schemas:
            print(
                "❌ 需要指定 --schemas（源库 schema 名，逗号分隔；"
                "DuckDB 文件默认 schema 为 main）",
                file=sys.stderr,
            )
            print(f"\n📖 {db_path} 中可导入的 schema:", file=sys.stderr)
            print(f"   {_describe_duckdb_schemas(db_path)}", file=sys.stderr)
            sys.exit(1)

    result = step_meta_export(
        output,
        source=source,
        db_path=db_path,
        schemas=schemas,
        datasource_name=datasource_name,
        datasource_file=datasource_file,
        kundb=kundb,
        workspace_uuid=workspace_uuid,
    )
    if result is None:
        sys.exit(1)


def cmd_rel(args: argparse.Namespace) -> None:
    """meta rel — 导入表关系（RELATES_TO 边）"""
    if not args.file:
        print("❌ 需要指定 --file", file=sys.stderr)
        sys.exit(1)

    output = Path(args.output) if args.output else Path("./output")
    step_rel_export(output, args.file)


def cmd_std(args: argparse.Namespace) -> None:
    """meta std — 导入数据标准（Standard 节点）"""
    if not args.kundb:
        print("❌ 需要指定 --kundb", file=sys.stderr)
        sys.exit(1)

    output = Path(args.output) if args.output else Path("./output")
    step_std_export(output, args.kundb, args.workspace_uuid)


def cmd_compliance(args: argparse.Namespace) -> None:
    """meta compliance — 从 TDS 导出已有标准-字段关联（COMPLIES_WITH 边）"""
    if not args.kundb:
        print("❌ 需要指定 --kundb", file=sys.stderr)
        sys.exit(1)

    output = Path(args.output) if args.output else Path("./output")
    step_compliance_export(
        output, args.kundb, args.workspace_uuid, args.datasources_file
    )


def cmd_metric(args: argparse.Namespace) -> None:
    """meta metric — 导入指标维度定义（Metric, Dimension + 边）"""
    if not args.file:
        print("❌ 需要指定 --file", file=sys.stderr)
        sys.exit(1)

    output = Path(args.output) if args.output else Path("./output")
    if not step_metric_export(output, args.file):
        sys.exit(1)


def cmd_graph(args: argparse.Namespace) -> None:
    """meta graph — 更新/重建/清空图数据库 + 生成 assets"""
    output = Path(args.output) if args.output else Path("./output")
    graph_mode = args.mode if hasattr(args, "mode") and args.mode else "update"
    assets_dir = Path(args.assets_dir) if args.assets_dir else DEFAULT_ASSETS_DIR

    if graph_mode == "clear":
        _clear_graph(assets_dir)
        print("\n✅ graph 清空完成！")
        return

    if not output.exists():
        print(f"❌ 输出目录不存在: {output}", file=sys.stderr)
        sys.exit(1)

    _update_graph(output, graph_mode, assets_dir)
    _generate_assets(assets_dir)

    print("\n✅ graph 更新完成！")


def _resolve_datasource_schemas(
    datasource_name: str, datasource_file: str, csv_dir: Path
) -> list[str]:
    """解析 --datasource 的 filter.schemas：声明文件优先，其次 csv-dir/Datasource.csv"""
    if datasource_file:
        try:
            return DatasourceLoader(datasource_file).schemas_for(datasource_name)
        except Exception as e:
            print(f"❌ 无法加载数据源声明文件: {e}", file=sys.stderr)
            return []
    csv_path = csv_dir / "Datasource.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        rows = df[df["datasource_name"] == datasource_name]
        if not rows.empty:
            flt = json.loads(rows.iloc[0].get("filter") or "{}")
            return [s for s in flt.get("schemas", []) if s]
    return []


def cmd_recommend(args: argparse.Namespace) -> None:
    """meta recommend — 数据标准推荐"""
    from .std_recommend import std_recommend

    if not args.kundb:
        print("❌ 需要指定 --kundb", file=sys.stderr)
        sys.exit(1)
    if not args.datasource:
        print("❌ 需要指定 --datasource", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else Path("./output")
    csv_dir = Path(args.csv_dir) if args.csv_dir else output_dir

    schemas = _resolve_datasource_schemas(
        args.datasource, args.datasources_file or "", csv_dir
    )
    if not schemas:
        print(
            f"❌ 无法解析数据源 '{args.datasource}' 的 filter.schemas："
            "请提供 --datasources-file，或先运行 meta meta 生成 Datasource.csv",
            file=sys.stderr,
        )
        sys.exit(1)

    std_recommend(
        output_dir=output_dir,
        kundb=args.kundb,
        workspace_uuid=args.workspace_uuid,
        datasource_name=args.datasource,
        schemas=schemas,
        csv_dir=csv_dir,
    )


def cmd_schema(args: argparse.Namespace) -> None:
    """meta schema — 输出标准 JSON Schema（供外部 agent 生成标准数据）"""
    from govio.metadata.datasource import SCHEMA_PATH as DATASOURCE_SCHEMA_PATH
    from govio.metadata.metric import SCHEMA_PATH as METRIC_SCHEMA_PATH
    from govio.metadata.relationship import SCHEMA_PATH as RELATIONSHIP_SCHEMA_PATH

    schema_files = {
        "metric": METRIC_SCHEMA_PATH,
        "relationship": RELATIONSHIP_SCHEMA_PATH,
        "datasource": DATASOURCE_SCHEMA_PATH,
    }
    text = schema_files[args.schema_name].read_text(encoding="utf-8")
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"✓ Schema 已写入: {args.output}")
    else:
        print(text, end="")


def cmd_import_schema(args: argparse.Namespace) -> None:
    """meta import-schema — 从已配置数据源导入元数据到图库（仅 DuckDB）"""
    ds_name = args.datasource

    config = ConfigManager().load()
    datasources = config.get("datasources", {})
    if ds_name not in datasources:
        print(f"❌ 数据源 '{ds_name}' 不存在，请先用 onboard 配置", file=sys.stderr)
        sys.exit(1)

    url = datasources[ds_name].get("url", "")
    if not url.startswith("duckdb://"):
        print(
            f"❌ import-schema 仅支持 DuckDB 数据源，'{ds_name}' 的 URL 不是 duckdb:// 开头",
            file=sys.stderr,
        )
        sys.exit(1)

    db_path = url[len("duckdb://") :]
    schemas = [s.strip() for s in args.schemas.split(",") if s.strip()]
    if not schemas:
        print("❌ 需要指定 --schemas", file=sys.stderr)
        sys.exit(1)

    output = Path(args.output) if args.output else Path("./output")
    assets_dir = Path(args.assets_dir) if args.assets_dir else DEFAULT_ASSETS_DIR

    result = step_meta_export(
        output,
        source="duckdb",
        db_path=db_path,
        schemas=schemas,
        datasource_name=ds_name,
    )
    if result is None:
        sys.exit(1)

    _update_graph(output, args.mode, assets_dir)
    _generate_assets(assets_dir)
    print("\n✅ import-schema 完成！")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def meta():
    """meta 命令入口"""
    parser = argparse.ArgumentParser(
        prog="govio-cli meta",
        description="知识图库维护 — 元数据导入、推荐、图更新",
    )
    sub = parser.add_subparsers(dest="action", required=True)

    # --- 导入子命令 ---

    # TDS 参数复用
    tds_args = argparse.ArgumentParser(add_help=False)
    tds_args.add_argument("--kundb", type=str, required=True, help="TDS 元数据库 URL")
    tds_args.add_argument(
        "--workspace-uuid",
        type=str,
        default="82ee37374b314a938bf28170ab4db7cf",
        help="工作区 UUID",
    )

    # meta meta — 元数据导入
    p_meta = sub.add_parser(
        "meta",
        help="导入 TDS/DuckDB 元数据（Datasource, PhysicalTable, Col, HAS_COLUMN, OWNS）",
    )
    p_meta.add_argument(
        "--source", choices=["tds", "duckdb"], required=True, help="信息来源"
    )
    p_meta.add_argument("--db", type=str, help="DuckDB 数据库文件路径")
    p_meta.add_argument(
        "--schemas",
        type=str,
        help="要导出的 schema 列表，逗号分隔（DuckDB 模式必填；文件默认 schema 为 main）",
    )
    p_meta.add_argument(
        "--datasource",
        type=str,
        help="数据源名（DuckDB 模式必填，生成 Datasource 节点与 OWNS 归属）",
    )
    p_meta.add_argument(
        "--datasources-file",
        type=str,
        help="数据源声明 JSON（TDS 模式必填，抽取范围取 filter.schemas）",
    )
    p_meta.add_argument(
        "--kundb", type=str, help="TDS 元数据库 URL（TDS 模式必须）"
    )
    p_meta.add_argument(
        "--workspace-uuid", type=str, help="工作区 UUID（TDS 模式必须）"
    )
    p_meta.add_argument("--output", type=str, help="CSV 输出目录（默认 ./output）")
    p_meta.set_defaults(func=cmd_meta)

    # meta std — 数据标准导入
    p_std = sub.add_parser(
        "std", help="导入数据标准（Standard 节点）", parents=[tds_args]
    )
    p_std.add_argument("--output", type=str, help="CSV 输出目录（默认 ./output）")
    p_std.set_defaults(func=cmd_std)

    # meta compliance — 已有标准关联
    p_comp = sub.add_parser(
        "compliance",
        help="从 TDS 导出已有标准-字段关联（COMPLIES_WITH 边）",
        parents=[tds_args],
    )
    p_comp.add_argument(
        "--datasources-file",
        type=str,
        required=True,
        help="数据源声明 JSON（schema 归属，用于生成全限定列标识）",
    )
    p_comp.add_argument("--output", type=str, help="CSV 输出目录（默认 ./output）")
    p_comp.set_defaults(func=cmd_compliance)

    # meta rel — 表关系导入
    p_rel = sub.add_parser("rel", help="导入表关系（RELATES_TO 边）")
    p_rel.add_argument("--file", type=str, required=True, help="表关系 JSON 文件路径")
    p_rel.add_argument("--output", type=str, help="CSV 输出目录（默认 ./output）")
    p_rel.set_defaults(func=cmd_rel)

    # meta metric — 指标维度导入
    p_metric = sub.add_parser(
        "metric", help="导入指标维度定义（Metric, Dimension + 边）"
    )
    p_metric.add_argument(
        "--file", type=str, required=True, help="指标定义 JSON 文件路径"
    )
    p_metric.add_argument("--output", type=str, help="CSV 输出目录（默认 ./output）")
    p_metric.set_defaults(func=cmd_metric)

    # --- JSON Schema 输出 ---

    p_schema = sub.add_parser(
        "schema", help="输出标准 JSON Schema（供外部 agent 生成标准数据）"
    )
    p_schema.add_argument(
        "schema_name",
        choices=["metric", "relationship", "datasource"],
        help="schema 名称",
    )
    p_schema.add_argument(
        "-o", "--output", type=str, help="输出文件路径（缺省打印到 stdout）"
    )
    p_schema.set_defaults(func=cmd_schema)

    # --- 图更新 ---

    p_graph = sub.add_parser("graph", help="更新图数据库 + 生成 assets")
    p_graph.add_argument("--output", type=str, help="CSV 输出目录（默认 ./output）")
    p_graph.add_argument(
        "--assets-dir",
        type=str,
        help="assets 输出目录（默认 .agents/skills/govio/assets）",
    )
    p_graph.add_argument(
        "--mode",
        choices=["update", "rebuild", "clear"],
        default="update",
        help="更新模式: update=增量, rebuild=重建, clear=清空",
    )
    p_graph.set_defaults(func=cmd_graph)

    # --- 数据标准推荐 ---

    p_recommend = sub.add_parser("recommend", help="数据标准推荐")
    p_recommend.add_argument(
        "--kundb", type=str, required=True, help="TDS 元数据库 URL"
    )
    p_recommend.add_argument(
        "--workspace-uuid",
        type=str,
        default="82ee37374b314a938bf28170ab4db7cf",
        help="工作区 UUID",
    )
    p_recommend.add_argument(
        "--datasource",
        type=str,
        required=True,
        help="数据源名（分析范围取其 filter.schemas）",
    )
    p_recommend.add_argument(
        "--datasources-file",
        type=str,
        help="数据源声明 JSON（缺省时从 --csv-dir 的 Datasource.csv 解析）",
    )
    p_recommend.add_argument(
        "--csv-dir", type=str, help="已导入的 CSV 目录（默认同 --output-dir）"
    )
    p_recommend.add_argument(
        "--output-dir", type=str, help="推荐结果输出目录（默认 ./output）"
    )
    p_recommend.set_defaults(func=cmd_recommend)

    # meta import-schema — 从已配置数据源导入元数据（仅 DuckDB）
    p_import = sub.add_parser(
        "import-schema",
        help="从已配置的 DuckDB 数据源导入元数据到图库",
    )
    p_import.add_argument(
        "--datasource", type=str, required=True, help="数据源名称（仅支持 DuckDB）"
    )
    p_import.add_argument(
        "--schemas",
        type=str,
        required=True,
        help="要导入的 schema 列表，逗号分隔（DuckDB 默认 schema 为 main）",
    )
    p_import.add_argument("--output", type=str, help="CSV 输出目录（默认 ./output）")
    p_import.add_argument(
        "--assets-dir",
        type=str,
        help="assets 输出目录（默认 .agents/skills/govio/assets）",
    )
    p_import.add_argument(
        "--mode",
        choices=["update", "rebuild", "clear"],
        default="update",
        help="更新模式: update=增量, rebuild=重建, clear=清空",
    )
    p_import.set_defaults(func=cmd_import_schema)

    args = parser.parse_args(sys.argv[1:])
    args.func(args)


if __name__ == "__main__":
    meta()
