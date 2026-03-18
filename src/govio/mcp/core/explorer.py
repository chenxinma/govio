"""关系探查核心逻辑"""

from difflib import SequenceMatcher
from typing import Any

import pandas as pd


class RelationExplorer:
    """关系探查器"""

    def find_column_similarity(
        self, df1: pd.DataFrame, df2: pd.DataFrame
    ) -> list[dict[str, Any]]:
        """查找列名相似的列"""
        similarities = []

        for col1 in df1.columns:
            for col2 in df2.columns:
                ratio = SequenceMatcher(None, col1.lower(), col2.lower()).ratio()
                if ratio > 0.7:
                    similarities.append(
                        {"column": col1, "match_column": col2, "similarity": ratio}
                    )

        return sorted(similarities, key=lambda x: x["similarity"], reverse=True)

    def infer_foreign_keys(
        self,
        source_df: pd.DataFrame,
        source_name: str,
        target_df: pd.DataFrame,
        target_name: str,
    ) -> list[dict[str, Any]]:
        """推断外键关系"""
        relations = []

        for col in source_df.columns:
            if col.endswith("_id") or col.endswith("Id"):
                target_col = col.replace("_id", "").replace("Id", "") + "_id"
                if target_col not in target_df.columns:
                    target_col = col

                if col in target_df.columns:
                    source_values = set(source_df[col].dropna().unique())
                    target_values = set(target_df[col].dropna().unique())

                    if source_values and target_values:
                        overlap = len(source_values & target_values)
                        ratio = overlap / len(source_values) if source_values else 0

                        if ratio > 0.5:
                            relations.append(
                                {
                                    "source_table": source_name,
                                    "source_column": col,
                                    "target_table": target_name,
                                    "target_column": col,
                                    "confidence": ratio,
                                }
                            )

        return relations

    def explore(self, dataframes: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
        """探查所有 DataFrame 之间的关系"""
        all_relations = []
        names = list(dataframes.keys())

        for i, name1 in enumerate(names):
            for name2 in names[i + 1 :]:
                df1 = dataframes[name1]
                df2 = dataframes[name2]

                relations = self.infer_foreign_keys(df1, name1, df2, name2)
                all_relations.extend(relations)

                relations = self.infer_foreign_keys(df2, name2, df1, name1)
                all_relations.extend(relations)

        return all_relations
