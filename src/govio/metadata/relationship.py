"""
govio.metadata.relationship
读取 schema_of_relationships.json 文件，生成物理表之间的关系边数据
"""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


VALID_RELATIONSHIP_TYPES = {"one_to_one", "one_to_many", "many_to_one", "many_to_many"}


class RelationshipLoader:
    """加载和验证表关系JSON文件"""

    def __init__(
        self, json_path: str, df_tables: pd.DataFrame, df_columns: pd.DataFrame
    ):
        """
        Args:
            json_path: JSON文件路径
            df_tables: PhysicalTable DataFrame
            df_columns: Col DataFrame
        """
        self.json_path = Path(json_path)
        self.df_tables = df_tables
        self.df_columns = df_columns
        self._validate_inputs()

    def _validate_inputs(self):
        """验证输入参数"""
        if not self.json_path.exists():
            raise FileNotFoundError(f"关系文件不存在: {self.json_path}")

        if self.df_tables.empty:
            raise ValueError("PhysicalTable DataFrame 为空")

        if self.df_columns.empty:
            raise ValueError("Col DataFrame 为空")

    def load_json(self) -> dict[str, Any]:
        """加载JSON文件"""
        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "version" not in data:
            raise ValueError("JSON 缺少 version 字段")

        if "relationships" not in data:
            raise ValueError("JSON 缺少 relationships 字段")

        return data

    def validate_relationship(self, rel: dict[str, Any], index: int) -> bool:
        """
        验证单个关系的有效性

        Args:
            rel: 关系字典
            index: 关系索引（用于错误消息）

        Returns:
            bool: 是否有效
        """
        required_fields = ["source", "target", "relationship_type"]
        for field in required_fields:
            if field not in rel:
                logger.warning(f"关系 {index} 缺少必需字段 '{field}'，跳过")
                return False

        if rel["relationship_type"] not in VALID_RELATIONSHIP_TYPES:
            logger.warning(
                f"关系 {index} 的 relationship_type '{rel['relationship_type']}' 无效，"
                f"有效值: {VALID_RELATIONSHIP_TYPES}，跳过"
            )
            return False

        if "PhysicalTable" not in rel["source"] or "Cols" not in rel["source"]:
            logger.warning(
                f"关系 {index} 的 source 缺少 PhysicalTable 或 Cols 字段，跳过"
            )
            return False

        if "PhysicalTable" not in rel["target"] or "Cols" not in rel["target"]:
            logger.warning(
                f"关系 {index} 的 target 缺少 PhysicalTable 或 Cols 字段，跳过"
            )
            return False

        return True

    def _validate_table_exists(self, table_name: str, context: str) -> bool:
        """
        验证表是否存在

        Args:
            table_name: 表名
            context: 上下文描述（用于错误消息）

        Returns:
            bool: 是否存在
        """
        table_names = self.df_tables.get(
            "full_table_name", self.df_tables.get("name", pd.Series())
        )

        if table_name not in table_names.values:
            logger.warning(
                f"{context}: 表 '{table_name}' 不存在于 PhysicalTable 数据中，跳过"
            )
            return False

        return True

    def _validate_column_exists(
        self, table_name: str, column_name: str, context: str
    ) -> tuple[bool, str | None, str | None]:
        """
        验证列是否存在（大小写不敏感匹配）

        Args:
            table_name: 表名
            column_name: 列名
            context: 上下文描述（用于错误消息）

        Returns:
            tuple[bool, str | None, str | None]: (是否存在, 实际列名或None, 实际表名或None)
        """
        full_col_name = f"{table_name}.{column_name}"

        if "column" in self.df_columns.columns:
            col_names = self.df_columns["column"]
        else:
            logger.warning(
                f"{context}: 列 '{full_col_name}' 无法验证，DataFrame 缺少列名列，跳过"
            )
            return False, None, None

        if full_col_name in col_names.values:
            return True, column_name, \
                self.df_columns[self.df_columns["column"]==full_col_name].get("full_table_name", pd.Series()).values[0]

        col_names_lower = {str(c).lower(): c for c in col_names.values}
        full_col_name_lower = full_col_name.lower()

        if full_col_name_lower in col_names_lower:
            actual_col_name = col_names_lower[full_col_name_lower]
            actual_simple_name = actual_col_name.split(".")[-1]
            logger.info(
                f"{context}: 列名大小写不匹配，'{column_name}' -> '{actual_simple_name}'"
            )
            return True, actual_simple_name, \
                self.df_columns[self.df_columns["column"]==actual_col_name].get("full_table_name", pd.Series()).values[0]


        logger.warning(f"{context}: 列 '{full_col_name}' 不存在于 Col 数据中，跳过")
        return False, None, None
    
    def _get_table_id(self, full_table_name:str)-> int:
        return self.df_tables[self.df_tables["full_table_name"] == full_table_name].index[0]


    def validate_table_and_columns(self, rel: dict[str, Any], index: int) -> bool:
        """
        验证关系中的表和列是否存在

        Args:
            rel: 关系字典
            index: 关系索引

        Returns:
            bool: 是否有效
        """
        source_table = rel["source"]["PhysicalTable"]
        target_table = rel["target"]["PhysicalTable"]

        if not self._validate_table_exists(source_table, f"关系 {index} source"):
            return False

        if not self._validate_table_exists(target_table, f"关系 {index} target"):
            return False

        normalized_source_cols = []
        full_table_names = set()
        for col in rel["source"]["Cols"]:
            valid, actual_col, full_table_name = self._validate_column_exists(
                source_table, col, f"关系 {index} source"
            )
            if not valid:
                return False
            normalized_source_cols.append(actual_col)
            full_table_names.add(full_table_name)
        rel["source"]["Cols"] = normalized_source_cols
        if len(full_table_names) > 1:
            logger.warning(f"source 表 {source_table}, 元数据定义出现重复")
            return False
        rel["source"]["ID"] = self._get_table_id(full_table_names.pop())


        normalized_target_cols = []
        full_table_names = set()
        for col in rel["target"]["Cols"]:
            valid, actual_col, full_table_name = self._validate_column_exists(
                target_table, col, f"关系 {index} target"
            )
            if not valid:
                return False
            normalized_target_cols.append(actual_col)
            full_table_names.add(full_table_name)
        rel["target"]["Cols"] = normalized_target_cols
        if len(full_table_names) > 1:
            logger.warning(f"target 表 {target_table}, 元数据定义出现重复")
            return False
        rel["target"]["ID"] = self._get_table_id(full_table_names.pop())

        return True

    def _convert_to_edge_row(self, rel: dict[str, Any]) -> dict[str, Any]:
        """
        将单个关系转换为边数据行

        Args:
            rel: 关系字典

        Returns:
            dict: 边数据行
        """
        return {
            "source": rel["source"]["ID"],
            "target": rel["target"]["ID"],
            "relationship_type": rel["relationship_type"],
            "description": rel.get("description", ""),
            "source_columns": ",".join(rel["source"]["Cols"]),
            "target_columns": ",".join(rel["target"]["Cols"]),
        }

    def load_relationships(self) -> pd.DataFrame:
        """
        加载并验证所有关系，返回边数据 DataFrame

        Returns:
            pd.DataFrame: 包含边数据的 DataFrame
        """
        data = self.load_json()
        relationships = data["relationships"]
        
        edge_rows = []
        for idx, rel in enumerate(relationships):
            if not self.validate_relationship(rel, idx):
                continue

            if not self.validate_table_and_columns(rel, idx):
                continue

            edge_row = self._convert_to_edge_row(rel)
            edge_rows.append(edge_row)
        

        if not edge_rows:
            logger.warning("没有有效的关系数据")
            return pd.DataFrame(
                columns=[
                    "source",
                    "target",
                    "relationship_type",
                    "description",
                    "source_columns",
                    "target_columns",
                ]
            )

        df = pd.DataFrame(edge_rows)
        logger.info(f"成功加载 {len(df)} 个表关系")

        return df


def load_relationships(
    json_path: str, df_tables: pd.DataFrame, df_columns: pd.DataFrame
) -> pd.DataFrame:
    """
    便捷函数：加载关系JSON并返回边数据

    Args:
        json_path: JSON文件路径
        df_tables: PhysicalTable DataFrame
        df_columns: Col DataFrame

    Returns:
        pd.DataFrame: 边数据 DataFrame
    """
    loader = RelationshipLoader(json_path, df_tables, df_columns)
    return loader.load_relationships()
