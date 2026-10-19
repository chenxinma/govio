"""表比对核心逻辑"""

from typing import Any

import pandas as pd
from datacompy.core import Compare


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
            "source_columns": sorted(source_cols),
            "target_columns": sorted(target_cols),
            "common_columns": sorted(common_cols),
            "source_only": sorted(source_only),
            "target_only": sorted(target_only),
        }

    def compare_data(
        self, source: pd.DataFrame, target: pd.DataFrame, join_columns: list[str]
    ) -> dict[str, Any]:
        """比对数据"""
        compare = Compare(df1=source, df2=target, join_columns=join_columns)

        return {
            "report": compare.report()
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
