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
        """推断外键关系（大小写不敏感）"""
        relations = []

        target_col_map = {col.lower(): col for col in target_df.columns}

        for col in source_df.columns:
            col_lower = col.lower()
            if col_lower.endswith("_id") or col_lower.endswith("id"):
                if col_lower.endswith("_id"):
                    base = col_lower[:-3]
                else:
                    base = col_lower[:-2]
                inferred_target = base + "_id"

                target_col = None
                if inferred_target in target_col_map:
                    target_col = target_col_map[inferred_target]
                elif col_lower in target_col_map:
                    target_col = target_col_map[col_lower]

                if target_col:
                    source_values = set(source_df[col].dropna().unique())
                    target_values = set(target_df[target_col].dropna().unique())

                    if source_values and target_values:
                        overlap = len(source_values & target_values)
                        ratio = overlap / len(source_values) if source_values else 0

                        if ratio > 0.5:
                            relations.append(
                                {
                                    "source_table": source_name,
                                    "source_column": col,
                                    "target_table": target_name,
                                    "target_column": target_col,
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

                similarities = self.find_column_similarity(df1, df2)
                for sim in similarities:
                    all_relations.append(
                        {
                            "type": "column_similarity",
                            "table1": name1,
                            "column1": sim["column"],
                            "table2": name2,
                            "column2": sim["match_column"],
                            "similarity": sim["similarity"],
                        }
                    )

        return all_relations
