from pathlib import Path

import pandas as pd

from .application import AppInfoLoader
from .database import TDSLoader
from .standard import StandardLoader
from .recommender import create_recommender
from .relationship import load_relationships
from .metric import MetricLoader


def reorder_index(dfs: list[pd.DataFrame], start: int = 1):
    base_index: int = start

    for df in dfs:
        _end_index = base_index + df.shape[0]
        df["index"] = [i for i in range(base_index, _end_index)]
        df.set_index("index", drop=True, inplace=True)
        base_index = _end_index


def make_csv(
    output: Path,
    db: str,
    workspace_uuid: str,
    app_list_file: str,
    df_app_db_map: pd.DataFrame,
    relationship_file: str | None = None,
    metric_file: str | None = None,
):
    db_loader = TDSLoader(db, workspace_uuid, df_app_db_map["schema"].to_list())
    app_loader = AppInfoLoader(app_list_file, df_app_db_map["name"].to_list())
    std_loader = StandardLoader(db, workspace_uuid)

    df_tables = db_loader.PhysicalTable
    df_columns = db_loader.Col
    df_apps = app_loader.Application
    df_stds = std_loader.Standard

    reorder_index([df_tables, df_columns, df_apps, df_stds])

    files = []

    df_tables.to_csv(output / "PhysicalTable.csv", index_label=":ID(PhysicalTable)")
    files.append("-n " + str(output / "PhysicalTable.csv"))

    df_columns.to_csv(output / "Col.csv", index_label=":ID(Col)")
    files.append("-n " + str(output / "Col.csv"))

    df_apps.to_csv(output / "Application.csv", index_label=":ID(Application)")
    files.append("-n " + str(output / "Application.csv"))

    df_stds.to_csv(output / "Standard.csv", index_label=":ID(Standard)")
    files.append("-n " + str(output / "Standard.csv"))

    df_has_column = pd.merge(
        df_tables[["full_table_name"]]
        .reset_index()
        .rename(columns={"index": ":START_ID(PhysicalTable)"}),
        df_columns[["full_table_name"]]
        .reset_index()
        .rename(columns={"index": ":END_ID(Col)"}),
        on="full_table_name",
        how="inner",
    )[[":START_ID(PhysicalTable)", ":END_ID(Col)"]]
    df_has_column.to_csv(output / "HAS_COLUMN.csv", index=False)
    files.append("-r " + str(output / "HAS_COLUMN.csv"))

    df_app_table = pd.merge(
        df_app_db_map,
        df_tables[["schema"]]
        .reset_index()
        .rename(columns={"index": ":END_ID(PhysicalTable)"}),
        on="schema",
        how="inner",
    )
    df_use = pd.merge(
        df_apps[["name"]]
        .reset_index()
        .rename(columns={"index": ":START_ID(Applicatin)"}),
        df_app_table,
        on="name",
        how="inner",
    )[[":START_ID(Applicatin)", ":END_ID(PhysicalTable)"]]

    df_use.to_csv(output / "USE.csv", index=False)
    files.append("-r " + str(output / "USE.csv"))

    if relationship_file:
        try:
            df_relates_to = load_relationships(relationship_file, df_tables, df_columns)
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
            df_metrics = metric_loader.Metric
            df_dimensions = metric_loader.Dimension

            # 计算 Metric/Dimension 的 ID 起始偏移（接续已有节点）
            metric_offset = (
                len(df_tables) + len(df_columns) + len(df_apps) + len(df_stds) + 1
            )
            dim_offset = metric_offset + len(df_metrics)
            reorder_index([df_metrics, df_dimensions], start=metric_offset)

            df_metrics.to_csv(output / "Metric.csv", index_label=":ID(Metric)")
            files.append("-n " + str(output / "Metric.csv"))

            df_dimensions.to_csv(
                output / "Dimension.csv", index_label=":ID(Dimension)"
            )
            files.append("-n " + str(output / "Dimension.csv"))

            # USES_TABLE 边
            uses_table = metric_loader.uses_table_edges.copy()
            if not uses_table.empty:
                uses_table[":START_ID(Metric)"] += metric_offset
                uses_table.to_csv(output / "USES_TABLE.csv", index=False)
                files.append("-r " + str(output / "USES_TABLE.csv"))

            # REFERS_COLUMN 边
            refers_col = metric_loader.refers_column_edges.copy()
            if not refers_col.empty:
                refers_col[":START_ID(Metric)"] += metric_offset
                refers_col.to_csv(output / "REFERS_COLUMN.csv", index=False)
                files.append("-r " + str(output / "REFERS_COLUMN.csv"))

            # DERIVED_FROM 边
            derived_from = metric_loader.derived_from_edges.copy()
            if not derived_from.empty:
                derived_from[":START_ID(Metric)"] += metric_offset
                derived_from[":END_ID(Metric)"] += metric_offset
                derived_from.to_csv(output / "DERIVED_FROM.csv", index=False)
                files.append("-r " + str(output / "DERIVED_FROM.csv"))

            # DIMENSION_USED 边
            dim_used = metric_loader.dimension_used_edges.copy()
            if not dim_used.empty:
                dim_used[":START_ID(Metric)"] += metric_offset
                dim_used[":END_ID(Dimension)"] += dim_offset
                dim_used.to_csv(output / "DIMENSION_USED.csv", index=False)
                files.append("-r " + str(output / "DIMENSION_USED.csv"))

            # SUPERSEDES 边
            supersedes = metric_loader.supersedes_edges
            if not supersedes.empty:
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
    output: Path, db: str, workspace_uuid: str, df_app_db_map: pd.DataFrame
):
    std_loader = StandardLoader(db, workspace_uuid)
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

    for schema in df_app_db_map["schema"].to_list():
        db_loader = TDSLoader(db, workspace_uuid, [schema])
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

    if (output / "Col.csv").exists() and (output / "Standard.csv").exists():
        df = df[["column", "recommended_standard_id"]]
        df_col = pd.read_csv(output / "Col.csv")[[":ID(Col)", "column"]]
        df_std = pd.read_csv(output / "Standard.csv")[[":ID(Standard)", "standard_id"]]

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
            columns={":ID(Col)": "START_ID(Col)", ":ID(Standard)": "END_ID(Standard)"}
        ).to_csv(output / "COMPLIES_WITH.csv", index=False)
