"""Datasource 声明文件加载、过滤匹配与全限定标识构造。

Datasource 是治理侧的信息记录节点（datasource_name / name / source_type / filter），
与 observe 的连接配置（config.yaml datasources）仅靠 datasource_name 名字契约关联，
不记录任何可达性状态。

声明文件结构由 datasource_schema.json 定义，可用
``govio-cli meta schema datasource`` 输出给外部工具生成。
"""

import json
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import jsonschema
import pandas as pd

SCHEMA_PATH = Path(__file__).parent / "datasource_schema.json"

DEFAULT_INCLUDE_TABLES = ["*"]


def qualify(datasource_name: str, identifier: str) -> str:
    """给 `schema.table` / `schema.table.column` 标识加数据源前缀

    Args:
        datasource_name: 数据源名（business key）
        identifier: `schema.table` 或 `schema.table.column` 形式标识

    Returns:
        str: `<datasource_name>.<identifier>`
    """
    return f"{datasource_name}.{identifier}"


def resolve_datasource(schema: str, datasource_map: dict[str, str]) -> str:
    """按 schema 反查 datasource_name，未归属时抛 ValueError

    Args:
        schema: schema 名
        datasource_map: schema -> datasource_name 映射

    Returns:
        str: datasource_name
    """
    ds_name = datasource_map.get(schema)
    if ds_name is None:
        raise ValueError(
            f"schema '{schema}' 未归属任何 datasource，无法生成全限定标识"
        )
    return ds_name


@dataclass
class DatasourceDef:
    """单个数据源声明

    Attributes:
        datasource_name: 数据源英文唯一名（business key，与 observe 的
            config.datasources key 一致）
        source_type: 数据源类型（duckdb/mysql/postgres/oracle/hive/trino/...）
        comment: 中文备注（可选）
        filter: 治理范围声明（schemas / include_tables / exclude_tables）
    """

    datasource_name: str
    source_type: str
    comment: str = ""
    filter: dict[str, Any] = field(default_factory=dict)

    @property
    def schemas(self) -> list[str]:
        """纳入治理范围的 schema 列表"""
        return list(self.filter.get("schemas", []))

    @property
    def filter_json(self) -> str:
        """filter 的 JSON 序列化（Datasource 节点属性存储格式）"""
        return json.dumps(self.filter, ensure_ascii=False)

    def matches_table(self, schema: str, table_name: str) -> bool:
        """按 filter 判断表是否纳入治理范围

        include_tables / exclude_tables 为 fnmatch glob，大小写不敏感；
        exclude 优先于 include，include 缺省 ["*"]。

        Args:
            schema: schema 名（精确匹配 filter.schemas）
            table_name: 表名

        Returns:
            bool: 是否纳入
        """
        if schema not in self.schemas:
            return False
        name = table_name.lower()
        for pattern in self.filter.get("exclude_tables", []):
            if fnmatch(name, str(pattern).lower()):
                return False
        includes = self.filter.get("include_tables") or DEFAULT_INCLUDE_TABLES
        return any(fnmatch(name, str(pattern).lower()) for pattern in includes)

    def to_row(self) -> dict[str, str]:
        """Datasource 节点行（CSV 列序）"""
        return {
            "datasource_name": self.datasource_name,
            "comment": self.comment,
            "source_type": self.source_type,
            "filter": self.filter_json,
        }


def make_datasource_def(
    datasource_name: str,
    source_type: str,
    schemas: list[str],
    declared: DatasourceDef | None = None,
) -> DatasourceDef:
    """构造非 TDS 导入本次生效的声明

    实际过滤由 CLI `--schemas` 执行，filter 记录本次生效的范围；
    已有声明时保留其 include_tables / exclude_tables。

    Args:
        datasource_name: 数据源名
        source_type: 数据源类型
        schemas: 本次导入的 schema 列表（CLI `--schemas`）
        declared: 声明文件中的既有定义（可选）

    Returns:
        DatasourceDef: 本次生效的声明
    """
    if declared is None:
        return DatasourceDef(
            datasource_name=datasource_name,
            source_type=source_type,
            filter={"schemas": list(schemas)},
        )
    return DatasourceDef(
        datasource_name=datasource_name,
        source_type=declared.source_type,
        comment=declared.comment,
        filter={
            "schemas": list(schemas),
            "include_tables": declared.filter.get("include_tables")
            or DEFAULT_INCLUDE_TABLES,
            "exclude_tables": declared.filter.get("exclude_tables", []),
        },
    )


def filter_frames(
    df_tables: pd.DataFrame,
    df_columns: pd.DataFrame,
    matches,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按 filter 过滤表与列，列随所属表一并剔除

    Args:
        df_tables: PhysicalTable DataFrame
        df_columns: Col DataFrame
        matches: 谓词 (schema, table_name) -> bool

    Returns:
        (过滤后的表, 过滤后的列)
    """
    if df_tables.empty:
        return df_tables, df_columns
    keep = df_tables.apply(
        lambda row: matches(str(row["schema"]), str(row["table_name"])), axis=1
    )
    df_tables = df_tables[keep].reset_index(drop=True)
    if not df_columns.empty:
        kept_tables = set(df_tables["full_table_name"])
        df_columns = df_columns[
            df_columns["full_table_name"].isin(kept_tables)
        ].reset_index(drop=True)
    return df_tables, df_columns


def build_owns_edges(
    df_datasources: pd.DataFrame,
    df_tables: pd.DataFrame,
    schema_map: dict[str, str],
) -> pd.DataFrame:
    """构建 OWNS 边（Datasource -> PhysicalTable）

    Args:
        df_datasources: Datasource DataFrame，需带 node_id 列
        df_tables: PhysicalTable DataFrame，需带 node_id 与 schema 列
        schema_map: schema -> datasource_name 映射

    Returns:
        DataFrame: 两列 :START_ID(Datasource) / :END_ID(PhysicalTable)
    """
    columns = [":START_ID(Datasource)", ":END_ID(PhysicalTable)"]
    if df_tables.empty:
        return pd.DataFrame(columns=columns)
    ds_ids = df_datasources.set_index("datasource_name")["node_id"].to_dict()
    rows = []
    for schema, node_id in zip(df_tables["schema"], df_tables["node_id"], strict=False):
        ds_name = schema_map.get(str(schema))
        ds_id = ds_ids.get(ds_name or "")
        if ds_id is None:
            continue
        rows.append({":START_ID(Datasource)": ds_id, ":END_ID(PhysicalTable)": node_id})
    return pd.DataFrame(rows, columns=columns)


class DatasourceLoader:
    """加载并校验 datasource 声明文件，生成 Datasource 节点数据与归属映射"""

    def __init__(self, datasource_file: str | Path) -> None:
        self.datasource_file = Path(datasource_file)
        self._defs: dict[str, DatasourceDef] = {}
        self._load()

    def _load(self) -> None:
        if not self.datasource_file.exists():
            raise FileNotFoundError(f"数据源声明文件不存在: {self.datasource_file}")
        with open(self.datasource_file, encoding="utf-8") as f:
            data = json.load(f)
        with open(SCHEMA_PATH, encoding="utf-8") as f:
            schema = json.load(f)
        jsonschema.validate(instance=data, schema=schema)

        for item in data["datasources"]:
            ds_def = DatasourceDef(
                datasource_name=item["datasource_name"],
                source_type=item["source_type"],
                comment=item.get("comment", ""),
                filter=item.get("filter", {}),
            )
            if ds_def.datasource_name in self._defs:
                raise ValueError(f"datasource_name 重复: {ds_def.datasource_name}")
            self._defs[ds_def.datasource_name] = ds_def

        self._check_schema_ownership()

    def _check_schema_ownership(self) -> None:
        """同一 schema 只能归属一个 datasource（TDS 抽取范围靠 schema 反查归属）"""
        seen: dict[str, str] = {}
        for ds_def in self._defs.values():
            for schema in ds_def.schemas:
                owner = seen.get(schema)
                if owner is not None and owner != ds_def.datasource_name:
                    raise ValueError(
                        f"schema '{schema}' 归属多个 datasource: "
                        f"{owner} / {ds_def.datasource_name}"
                    )
                seen[schema] = ds_def.datasource_name

    @property
    def defs(self) -> list[DatasourceDef]:
        """全部数据源声明"""
        return list(self._defs.values())

    @property
    def Datasource(self) -> pd.DataFrame:
        """Datasource 节点 DataFrame"""
        return pd.DataFrame(
            [d.to_row() for d in self._defs.values()],
            columns=["datasource_name", "comment", "source_type", "filter"],
        )

    def get(self, datasource_name: str) -> DatasourceDef:
        """按名取声明，不存在时抛 KeyError"""
        if datasource_name not in self._defs:
            raise KeyError(f"声明文件中不存在 datasource: {datasource_name}")
        return self._defs[datasource_name]

    def schemas_for(self, datasource_name: str) -> list[str]:
        """该数据源 filter.schemas"""
        return self.get(datasource_name).schemas

    def datasource_for_schema(self, schema: str) -> str | None:
        """schema -> datasource_name 反查，未归属返回 None"""
        for ds_def in self._defs.values():
            if schema in ds_def.schemas:
                return ds_def.datasource_name
        return None

    def all_schemas(self) -> list[str]:
        """全部 filter.schemas 并集（TDS 抽取范围）"""
        return [s for d in self._defs.values() for s in d.schemas]

    def datasource_map(self) -> dict[str, str]:
        """schema -> datasource_name 映射（全限定标识构造用）"""
        return {
            s: d.datasource_name for d in self._defs.values() for s in d.schemas
        }

    def matches_table(self, schema: str, table_name: str) -> bool:
        """按 schema 归属数据源的 filter 判断表是否纳入治理范围"""
        ds_name = self.datasource_for_schema(schema)
        if ds_name is None:
            return False
        return self._defs[ds_name].matches_table(schema, table_name)
