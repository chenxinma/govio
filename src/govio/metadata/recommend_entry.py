"""数据标准推荐入口（老路径 CLI）

按 Datasource.filter.schemas 圈定分析范围，用 StandardRecommender 对未贯标列
做 k-NN 推荐，产出 COMPLIES_WITH.csv（Col -> Standard 贯标边）。调用方：
`govio-cli meta recommend` 与 `govio.cli.std_recommend.std_recommend`。
导出编排见 `export.py`。
"""

from pathlib import Path

import pandas as pd

from .database import TDSLoader
from .recommender import create_recommender
from .standard import StandardLoader


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
    ds_map = dict.fromkeys(schemas, datasource_name)

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
