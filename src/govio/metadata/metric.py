"""
govio.metadata.metric
读取指标定义 JSON 文件，校验后生成 Metric/Dimension 节点和相关边的 DataFrame
"""

import json
import logging
from pathlib import Path

import jsonschema
import pandas as pd

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).parent / "metric_schema.json"


class MetricLoader:
    """加载和校验指标定义 JSON 文件，生成图节点和边的 DataFrame"""

    def __init__(
        self, metric_file: str, df_tables: pd.DataFrame, df_columns: pd.DataFrame
    ):
        """
        Args:
            metric_file: 指标定义 JSON 文件路径
            df_tables: PhysicalTable DataFrame（用于校验 source_tables 引用）
            df_columns: Col DataFrame（用于校验 REFERS_COLUMN 引用）
        """
        self.metric_file = Path(metric_file)
        self.df_tables = df_tables
        self.df_columns = df_columns
        self._data: dict | None = None
        self._metric_df: pd.DataFrame | None = None
        self._dimension_df: pd.DataFrame | None = None
        self._uses_table_edges: pd.DataFrame | None = None
        self._refers_column_edges: pd.DataFrame | None = None
        self._derived_from_edges: pd.DataFrame | None = None
        self._dimension_used_edges: pd.DataFrame | None = None
        self._supersedes_edges: pd.DataFrame | None = None

        self._load_and_validate()

    def _load_and_validate(self):
        """加载 JSON 文件并执行所有校验"""
        if not self.metric_file.exists():
            raise FileNotFoundError(f"指标定义文件不存在: {self.metric_file}")

        with open(self.metric_file, "r", encoding="utf-8") as f:
            self._data = json.load(f)

        self._validate_schema()
        self._validate_semantics()
        self._build_dataframes()

    def _validate_schema(self):
        """使用 JSON Schema 校验文件结构"""
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema = json.load(f)

        jsonschema.validate(instance=self._data, schema=schema)

    def _validate_semantics(self):
        """校验语义约束：source_tables 引用、derived_from 引用、DAG 无环"""
        if not self._data or "metrics" not in self._data:
            raise Exception("No metric loaded.")

        metrics = self._data["metrics"]
        metric_codes = {m["code"] for m in metrics}

        # 校验 source_tables 中的 full_table_name 在 PhysicalTable 中存在
        if not self.df_tables.empty and "full_table_name" in self.df_tables.columns:
            valid_tables = set(self.df_tables["full_table_name"].values)
            for m in metrics:
                if m["type"] == "atomic":
                    for st in m.get("source_tables", []):
                        if st["full_table_name"] not in valid_tables:
                            raise ValueError(
                                f"指标 '{m['code']}' 的 source_table "
                                f"'{st['full_table_name']}' 不存在于 PhysicalTable 数据中"
                            )

        # 校验 derived_from 中的 code 在本文件内定义
        for m in metrics:
            if m["type"] == "derived":
                for dep_code in m.get("derived_from", []):
                    if dep_code not in metric_codes:
                        raise ValueError(
                            f"指标 '{m['code']}' 的 derived_from "
                            f"'{dep_code}' 不在本文件定义的指标中"
                        )

        # 校验 dimensions 中的 code 在 shared_dimensions 中定义
        shared_dims = self._data.get("shared_dimensions", [])
        dim_codes = {d["code"] for d in shared_dims}
        for m in metrics:
            for dim_ref in m.get("dimensions", []):
                if dim_ref["code"] not in dim_codes:
                    raise ValueError(
                        f"指标 '{m['code']}' 引用的维度 '{dim_ref['code']}' "
                        f"不在 shared_dimensions 中定义"
                    )

        # 校验 DERIVED_FROM 图无环
        self._check_no_cycles(metrics)

    def _check_no_cycles(self, metrics: list[dict]):
        """检查 derived_from 关系是否构成有向无环图"""
        adj: dict[str, list[str]] = {}
        for m in metrics:
            adj[m["code"]] = m.get("derived_from", [])

        visited: set[str] = set()
        in_stack: set[str] = set()

        def dfs(node: str) -> bool:
            visited.add(node)
            in_stack.add(node)
            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in in_stack:
                    return True
            in_stack.remove(node)
            return False

        for code in adj:
            if code not in visited:
                if dfs(code):
                    raise ValueError("指标 derived_from 关系存在循环依赖")

    def _build_dataframes(self):
        """从校验后的数据构建所有 DataFrame"""
        if not self._data or "metrics" not in self._data:
            raise Exception("No metric loaded.")

        metrics = self._data["metrics"]
        shared_dims = self._data.get("shared_dimensions", [])

        # Dimension 节点
        dim_rows = []
        for d in shared_dims:
            dim_rows.append(
                {
                    "code": d["code"],
                    "name": d["name"],
                    "granularity": d.get("granularity"),
                    "values_example": d.get("values_example"),
                }
            )
        self._dimension_df = pd.DataFrame(dim_rows)

        # Metric 节点
        metric_rows = []
        for m in metrics:
            metric_rows.append(
                {
                    "code": m["code"],
                    "name": m["name"],
                    "business_definition": m["business_definition"],
                    "type": m["type"],
                    "formula": m.get("formula"),
                    "unit": m["unit"],
                    "data_type": m["data_type"],
                    "owner": m.get("owner"),
                    "update_frequency": m.get("update_frequency"),
                    "statistical_scope": m.get("statistical_scope"),
                    "time_scope": m.get("time_scope"),
                    "source_layer": m["source_layer"],
                    "version": m.get("version", 1),
                    "effective_from": m.get("effective_from"),
                }
            )
        self._metric_df = pd.DataFrame(metric_rows)

        # 使用 metric code -> 行索引的映射（后续在 make_csv 中会通过 assign_node_ids
        # 替换为 string node ID）
        # 先用 code 作为临时键，assign_node_ids 会基于该索引生成实际 string ID
        code_to_idx: dict[str, int] = {}
        for i, m in enumerate(metrics):
            code_to_idx[m["code"]] = i

        # 维度 code -> 行索引映射
        dim_code_to_idx: dict[str, int] = {}
        for i, d in enumerate(shared_dims):
            dim_code_to_idx[d["code"]] = i

        # USES_TABLE 边
        uses_table_rows = []
        if not self.df_tables.empty and "full_table_name" in self.df_tables.columns:
            for m in metrics:
                if m["type"] == "atomic":
                    for st in m.get("source_tables", []):
                        table_matches = self.df_tables[
                            self.df_tables["full_table_name"] == st["full_table_name"]
                        ]
                        if not table_matches.empty:
                            table_id = table_matches.index[0]
                            uses_table_rows.append(
                                {
                                    ":START_ID(Metric)": code_to_idx[m["code"]],
                                    ":END_ID(PhysicalTable)": table_id,
                                }
                            )
        self._uses_table_edges = pd.DataFrame(
            uses_table_rows,
            columns=[":START_ID(Metric)", ":END_ID(PhysicalTable)"],
        )

        # REFERS_COLUMN 边
        refers_column_rows = []
        if not self.df_columns.empty and "column" in self.df_columns.columns:
            for m in metrics:
                if m["type"] == "atomic":
                    for st in m.get("source_tables", []):
                        for col_ref in st.get("columns", []):
                            full_col_name = (
                                f"{st['full_table_name']}.{col_ref['column_name']}"
                            )
                            col_matches = self.df_columns[
                                self.df_columns["column"] == full_col_name
                            ]
                            if not col_matches.empty:
                                col_id = col_matches.index[0]
                                refers_column_rows.append(
                                    {
                                        ":START_ID(Metric)": code_to_idx[m["code"]],
                                        ":END_ID(Col)": col_id,
                                        "role": col_ref["role"],
                                    }
                                )
        self._refers_column_edges = pd.DataFrame(
            refers_column_rows,
            columns=[":START_ID(Metric)", ":END_ID(Col)", "role"],
        )

        # DERIVED_FROM 边
        derived_from_rows = []
        for m in metrics:
            if m["type"] == "derived":
                for dep_code in m.get("derived_from", []):
                    derived_from_rows.append(
                        {
                            ":START_ID(Metric)": code_to_idx[m["code"]],
                            ":END_ID(Metric)": code_to_idx[dep_code],
                        }
                    )
        self._derived_from_edges = pd.DataFrame(
            derived_from_rows,
            columns=[":START_ID(Metric)", ":END_ID(Metric)"],
        )

        # DIMENSION_USED 边
        dimension_used_rows = []
        for m in metrics:
            for dim_ref in m.get("dimensions", []):
                dimension_used_rows.append(
                    {
                        ":START_ID(Metric)": code_to_idx[m["code"]],
                        ":END_ID(Dimension)": dim_code_to_idx[dim_ref["code"]],
                        "usage_type": dim_ref["usage_type"],
                    }
                )
        self._dimension_used_edges = pd.DataFrame(
            dimension_used_rows,
            columns=[":START_ID(Metric)", ":END_ID(Dimension)", "usage_type"],
        )

        # SUPERSEDES 边（初始为空，后续版本管理时使用）
        self._supersedes_edges = pd.DataFrame(
            columns=[":START_ID(Metric)", ":END_ID(Metric)", "change_description"]
        )

        logger.info(
            f"成功加载 {len(self._metric_df)} 个指标, "
            f"{len(self._dimension_df)} 个维度"
        )

    @property
    def Metric(self) -> pd.DataFrame:
        """Metric 节点 DataFrame"""
        return self._metric_df # pyright:ignore

    @property
    def Dimension(self) -> pd.DataFrame:
        """Dimension 节点 DataFrame"""
        return self._dimension_df # pyright:ignore

    @property
    def uses_table_edges(self) -> pd.DataFrame:
        """USES_TABLE 边 DataFrame: Metric -> PhysicalTable"""
        return self._uses_table_edges # pyright:ignore

    @property
    def refers_column_edges(self) -> pd.DataFrame:
        """REFERS_COLUMN 边 DataFrame: Metric -> Col"""
        return self._refers_column_edges # pyright:ignore

    @property
    def derived_from_edges(self) -> pd.DataFrame:
        """DERIVED_FROM 边 DataFrame: Metric -> Metric"""
        return self._derived_from_edges # pyright:ignore

    @property
    def dimension_used_edges(self) -> pd.DataFrame:
        """DIMENSION_USED 边 DataFrame: Metric -> Dimension"""
        return self._dimension_used_edges # pyright:ignore

    @property
    def supersedes_edges(self) -> pd.DataFrame:
        """SUPERSEDES 边 DataFrame: Metric -> Metric"""
        return self._supersedes_edges # pyright:ignore


def load_metrics(
    metric_file: str, df_tables: pd.DataFrame, df_columns: pd.DataFrame
) -> MetricLoader:
    """
    便捷函数：加载指标定义 JSON 并返回 MetricLoader 实例

    Args:
        metric_file: 指标定义 JSON 文件路径
        df_tables: PhysicalTable DataFrame
        df_columns: Col DataFrame

    Returns:
        MetricLoader: 已加载和校验的 MetricLoader 实例
    """
    return MetricLoader(metric_file, df_tables, df_columns)
