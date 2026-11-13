"""导出步骤（step_*）—— 每种元数据域的「读源 → 赋 ID → 合并落盘」管线

每个步骤只做一件事：读取上游数据、生成确定性 node_id、通过 `merge` 幂等写入
CSV，并打印用户可读的进度。CLI 参数校验与退出码在 `commands.py`，
本模块以「返回 None/False 表示失败」的方式把退出决策交给调用方。
"""

import sys
from pathlib import Path

import pandas as pd

from govio.metadata.database import TDSLoader
from govio.metadata.datasource import (
    DatasourceLoader,
    build_owns_edges,
    filter_frames,
    make_datasource_def,
)
from govio.metadata.metric import MetricLoader
from govio.metadata.node_id import assign_node_ids
from govio.metadata.relationship import load_relationships
from govio.metadata.standard import StandardLoader

from .common import _build_metadata_loader, _describe_duckdb_schemas
from .constants import SOURCE_LABELS
from .merge import _load_csv_with_node_ids, merge_edge_csv, merge_node_csv


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

    单次运行覆盖一种信息来源（TDS 或一个实际数据源：DuckDB / MySQL / PostgreSQL / Trino）。
    db_path 语义随 source 变化：duckdb 为文件路径，mysql / postgres / trino 为连接 URL。

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
        tds_loader = TDSLoader(kundb, workspace_uuid, ds_loader.all_schemas(), ds_map)
        df_tables = tds_loader.PhysicalTable
        df_columns = tds_loader.Col
        df_tables, df_columns = filter_frames(
            df_tables, df_columns, ds_loader.matches_table
        )
        df_datasources = ds_loader.Datasource
    else:
        label = SOURCE_LABELS.get(source, source)
        db_desc = "文件路径" if source == "duckdb" else "连接 URL"
        if not db_path:
            print(f"错误: {label} 模式需要指定 --db（{db_desc}）", file=sys.stderr)
            return None
        if not datasource_name:
            print(f"错误: {label} 模式需要指定 --datasource", file=sys.stderr)
            return None
        print(f"从 {label} 读取元数据...")
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
        ds_def = make_datasource_def(datasource_name, source, schemas or [], declared)
        try:
            loader = _build_metadata_loader(
                source, db_path, schemas or [], datasource_name
            )
        except ValueError as e:
            print(f"❌ {e}", file=sys.stderr)
            return None
        df_tables = loader.PhysicalTable
        df_columns = loader.Col
        df_tables, df_columns = filter_frames(
            df_tables, df_columns, ds_def.matches_table
        )
        df_datasources = pd.DataFrame(
            [ds_def.to_row()],
            columns=["datasource_name", "comment", "source_type", "filter"],
        )
        ds_map = dict.fromkeys(ds_def.schemas, datasource_name)

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
    except Exception as e:
        print(f"❌ 无法加载指标定义文件: {e}", file=sys.stderr)
        return False
    return True


# ---------------------------------------------------------------------------
# CLI command handlers — 独立子命令（CLI-only，无配置文件依赖）
