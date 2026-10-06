from pathlib import Path

import pandas as pd

from .database import TDSLoader
from .datasource import DatasourceLoader, build_owns_edges, filter_frames
from .standard import StandardLoader
from .recommender import create_recommender
from .relationship import load_relationships
from .metric import MetricLoader
from .node_id import assign_node_ids, write_node_csv


def make_csv(
    output: Path,
    db: str,
    workspace_uuid: str,
    datasource_file: str,
    relationship_file: str | None = None,
    metric_file: str | None = None,
):
    ds_loader = DatasourceLoader(datasource_file)
    ds_map = ds_loader.datasource_map()
    db_loader = TDSLoader(db, workspace_uuid, ds_loader.all_schemas(), ds_map)
    std_loader = StandardLoader(db, workspace_uuid, ds_map)

    df_tables = db_loader.PhysicalTable
    df_columns = db_loader.Col
    df_tables, df_columns = filter_frames(
        df_tables, df_columns, ds_loader.matches_table
    )
    df_datasources = ds_loader.Datasource
    df_stds = std_loader.Standard

    df_tables = df_tables.reset_index(drop=True)
    df_columns = df_columns.reset_index(drop=True)
    df_datasources = df_datasources.reset_index(drop=True)
    df_stds = df_stds.reset_index(drop=True)
    assign_node_ids(df_tables, "PhysicalTable", "full_table_name")
    assign_node_ids(df_columns, "Col", "column")
    assign_node_ids(df_datasources, "Datasource", "datasource_name")
    assign_node_ids(df_stds, "Standard", "standard_id")

    files = []

    write_node_csv(df_tables, output / "PhysicalTable.csv", "PhysicalTable")
    files.append("-n " + str(output / "PhysicalTable.csv"))

    write_node_csv(df_columns, output / "Col.csv", "Col")
    files.append("-n " + str(output / "Col.csv"))

    write_node_csv(df_datasources, output / "Datasource.csv", "Datasource")
    files.append("-n " + str(output / "Datasource.csv"))

    write_node_csv(df_stds, output / "Standard.csv", "Standard")
    files.append("-n " + str(output / "Standard.csv"))

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

    df_owns = build_owns_edges(df_datasources, df_tables, ds_map)
    df_owns.to_csv(output / "OWNS.csv", index=False)
    files.append("-r " + str(output / "OWNS.csv"))

    if relationship_file:
        try:
            df_relates_to = load_relationships(relationship_file, df_tables, df_columns)
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
            print(f"成功生成 RELATES_TO.csv，包含 {len(df_relates_to)} 个关系")
        except Exception as e:
            print(f"警告: 无法加载关系文件: {e}")

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

            metric_idx_to_id = df_metrics["node_id"].tolist()
            dim_idx_to_id = df_dimensions["node_id"].tolist()
            table_idx_to_id = df_tables["node_id"].tolist()
            col_idx_to_id = df_columns["node_id"].tolist()

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
        except Exception as e:
            print(f"警告: 无法加载指标定义文件: {e}")

    s = f"falkordb-bulk-insert {{GRAPH}} {'  '.join(files)}"
    print("Bulk insert usage:")
    print(s)


def data_standard_recommend(
    output: Path,
    db: str,
    workspace_uuid: str,
    schemas: list[str],
    datasource_name: str,
    csv_dir: Path | None = None,
):
    """数据标准推荐：按 schema 圈定分析范围，产出 COMPLIES_WITH.csv

    Args:
        output: 推荐结果输出目录
        db: TDS 元数据库 URL
        workspace_uuid: 工作区 UUID
        schemas: 分析范围（来自 Datasource.filter.schemas）
        datasource_name: 数据源名（全限定标识前缀）
        csv_dir: 已导入的 CSV 目录（读取 Col.csv / Standard.csv），缺省取 output
    """
    csv_dir = csv_dir or output
    ds_map = {schema: datasource_name for schema in schemas}

    std_loader = StandardLoader(db, workspace_uuid, ds_map)
    # 加载数据
    std_compliance = std_loader.StdCompliance  # 已贯标列

    # 创建推荐器
    WEIGHTS = {
        "table": 0.25,  # 表名权重（仅使用从 full_table_name 提取的 table_name）
        "name": 0.35,  # 列名权重
        "comment": 0.25,  # 列注释权重
        "type": 0.05,  # 数据类型权重
        "numeric": 0.10,  # 数值特征权重
    }
    recommender = create_recommender(
        std_compliance=std_compliance,
        weights=WEIGHTS,
        k_neighbors=5,  # 使用5个最近邻
        top_n=3,  # 返回Top 3推荐
    )

    df = pd.DataFrame()

    for schema in schemas:
        db_loader = TDSLoader(db, workspace_uuid, [schema], ds_map)
        all_columns = db_loader.Col  # 所有列
        print("Schema=", schema, " columns=", all_columns.shape[0])

        # 批量推荐
        recommendations = recommender.batch_recommend(all_columns)
        _recommendations_confirm = recommendations[
            recommendations["recommendation_score"] > 0
        ]

        df = pd.concat([df, _recommendations_confirm])

        # 保存结果
        # _recommendations_confirm.to_csv(output / f'_recommendations_{schema}.csv', index=False)

    if (csv_dir / "Col.csv").exists() and (csv_dir / "Standard.csv").exists():
        df = df[["column", "recommended_standard_id"]]
        df_col = pd.read_csv(csv_dir / "Col.csv")[[":ID(Col)", "column"]]
        df_std = pd.read_csv(csv_dir / "Standard.csv")[[":ID(Standard)", "standard_id"]]

        df_colStdId = pd.merge(df, df_col, on="column", how="inner")
        df_complies_with = pd.merge(
            df_colStdId,
            df_std,
            left_on="recommended_standard_id",
            right_on="standard_id",
            how="inner",
        )
        # COMPLIES_WITH
        df_complies_with[[":ID(Col)", ":ID(Standard)"]].rename(
            columns={":ID(Col)": ":START_ID(Col)", ":ID(Standard)": ":END_ID(Standard)"}
        ).to_csv(output / "COMPLIES_WITH.csv", index=False)
