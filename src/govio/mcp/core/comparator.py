"""表比对核心逻辑"""

from typing import Any

from datacompy.core import Compare
import pandas as pd


class TableComparator:
    """表比对器"""

    def compare_schema(
        self, source: pd.DataFrame, target: pd.DataFrame
    ) -> dict[str, Any]:
        """比对表结构"""
        source_cols = set(source.columns)
        target_cols = set(target.columns)

        common_cols = source_cols & target_cols
        source_only = source_cols - target_cols
        target_only = target_cols - source_cols

        return {
            "match": len(source_only) == 0 and len(target_only) == 0,
            "source_columns": sorted(list(source_cols)),
            "target_columns": sorted(list(target_cols)),
            "common_columns": sorted(list(common_cols)),
            "source_only": sorted(list(source_only)),
            "target_only": sorted(list(target_only)),
        }

    def compare_data(
        self, source: pd.DataFrame, target: pd.DataFrame, join_columns: list[str]
    ) -> dict[str, Any]:
        """比对数据"""
        compare = Compare(df1=source, df2=target, join_columns=join_columns)

        return {
            "match_rate": compare.percent_match,
            "rows_matched": compare.count_matching_rows(),
            "rows_in_source": compare.df1_rows,
            "rows_in_target": compare.df2_rows,
            "rows_only_in_source": compare.df1_unq_rows,
            "rows_only_in_target": compare.df2_unq_rows,
        }

    def compare(
        self, source: pd.DataFrame, target: pd.DataFrame, join_columns: list[str]
    ) -> dict[str, Any]:
        """完整比对"""
        schema_result = self.compare_schema(source, target)
        data_result = self.compare_data(source, target, join_columns)

        return {
            "schema": schema_result,
            "data": data_result,
        }
