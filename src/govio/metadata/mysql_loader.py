"""MySQL 元数据加载器

通过 pymysql 读取 ``information_schema.tables`` / ``information_schema.columns``，
抽取表与列元数据（含表注释、列注释），产出与 TrinoLoader / DuckDBLoader 相同
列结构的 DataFrame，data_entity_type 使用 MYSQL_TABLE / MYSQL_COLUMN。
"""

import re

import pandas as pd
import pymysql

from .database import MetadataLoader
from .datasource import qualify

# 匹配 MySQL 类型：base 或 base(params)，如 decimal(10,2)、varchar(64)、bigint
_TYPE_RE = re.compile(r"^(\w+)\s*(?:\(([^)]*)\))?$")
# 数值类型的 unsigned/zerofill 后缀（如 decimal(10,2) unsigned、int(10) unsigned zerofill），
# 解析参数前剥离，否则整串不匹配会导致 precision/scale 静默丢失
_MODIFIER_RE = re.compile(r"(?:\s+(?:unsigned|zerofill))+$", re.IGNORECASE)

# 输出 DataFrame 列结构，与 TrinoLoader / DuckDBLoader 保持一致
_TABLE_COLUMNS = [
    "full_table_name",
    "schema",
    "table_name",
    "name",
    "data_entity_type",
    "database_name",
]
_COLUMN_COLUMNS = [
    "column",
    "column_name",
    "name",
    "full_table_name",
    "data_entity_type",
    "dtype",
    "size",
    "precision",
    "scale",
    "order_no",
    "data_type",
]


def _parse_type(type_str: str) -> tuple[int, int, int]:
    """从 MySQL COLUMN_TYPE 解析 (size, precision, scale)。

    decimal(p,s) -> (0, p, s)；decimal(p) -> (0, p, 0)
    varchar(n) / char(n) -> (n, 0, 0)；无参 varchar -> (0, 0, 0)
    其他类型（bigint 等）-> (0, 0, 0)

    数值类型的 unsigned / zerofill 后缀在解析前剥离
    （如 decimal(10,2) unsigned -> (0, 10, 2)），dtype 仍保留原始 COLUMN_TYPE。
    """
    m = _TYPE_RE.match(_MODIFIER_RE.sub("", type_str.strip()))
    if not m:
        return 0, 0, 0
    base = m.group(1).lower()
    params = m.group(2)
    if params is None:
        return 0, 0, 0
    nums = [int(p.strip()) for p in params.split(",") if p.strip().isdigit()]
    if base == "decimal":
        if len(nums) >= 2:
            return 0, nums[0], nums[1]
        if len(nums) == 1:
            return 0, nums[0], 0
        return 0, 0, 0
    if base in ("varchar", "char"):
        return (nums[0] if nums else 0), 0, 0
    return 0, 0, 0


class MysqlLoader(MetadataLoader):
    """从 MySQL 实例加载表与列元数据。

    使用 pymysql 读取 ``information_schema.tables`` / ``information_schema.columns``，
    仅抽取 BASE TABLE（不含视图），支持 ``schemas`` 过滤（MySQL 中 schema 即库）
    与 ``datasource_name`` 全限定。产出与 TrinoLoader / DuckDBLoader 相同列结构的
    DataFrame，data_entity_type 使用 MYSQL_TABLE / MYSQL_COLUMN。
    """

    def __init__(
        self,
        host: str,
        schemas: list[str],
        port: int = 3306,
        user: str = "root",
        password: str = "",
        database: str = "",
        datasource_name: str = "",
    ) -> None:
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.schemas = schemas
        self.datasource_name = datasource_name

    def _connect(self):
        return pymysql.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database,
            charset="utf8mb4",
        )

    def _fetch(self, sql: str, params: tuple[str, ...]) -> pd.DataFrame:
        """执行参数化 SQL 并返回 DataFrame，列名取自 cursor.description。"""
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(sql, params)
            names = [d[0] for d in cur.description]
            rows = cur.fetchall()
            return pd.DataFrame(rows, columns=names)
        finally:
            conn.close()

    @staticmethod
    def _schema_in_clause(schemas: list[str]) -> str:
        """生成 TABLE_SCHEMA IN (...) 的 %s 占位符，值经参数化传递。"""
        return ", ".join(["%s"] * len(schemas))

    def load_tables(self) -> pd.DataFrame:
        """读取 information_schema.tables 表清单（含表注释）。

        仅抽取 BASE TABLE；表注释为空时 name 回退为 table_name。

        Returns:
            pd.DataFrame: 包含表信息的 DataFrame
        """
        if not self.schemas:
            return pd.DataFrame(columns=_TABLE_COLUMNS)
        sql = (
            "SELECT TABLE_SCHEMA AS `schema`, TABLE_NAME AS table_name, "
            "TABLE_COMMENT AS table_comment "
            "FROM information_schema.tables "
            "WHERE TABLE_TYPE = 'BASE TABLE' "
            f"AND TABLE_SCHEMA IN ({self._schema_in_clause(self.schemas)}) "
            "ORDER BY TABLE_SCHEMA, TABLE_NAME"
        )
        df = self._fetch(sql, tuple(self.schemas))
        records: list[dict] = []
        for schema, table_name, comment in zip(
            df["schema"].tolist(),
            df["table_name"].tolist(),
            df["table_comment"].tolist(),
            strict=False,
        ):
            # TABLE_COMMENT 可能为 None / NaN / ""，非空字符串才作为展示名
            name = comment if isinstance(comment, str) and comment else str(table_name)
            full_table_name = f"{schema}.{table_name}"
            if self.datasource_name:
                full_table_name = qualify(self.datasource_name, full_table_name)
            records.append(
                {
                    "full_table_name": full_table_name,
                    "schema": str(schema),
                    "table_name": str(table_name),
                    "name": name,
                    "data_entity_type": "MYSQL_TABLE",
                    "database_name": self.database,
                }
            )
        return pd.DataFrame(records, columns=_TABLE_COLUMNS)

    def load_columns(self) -> pd.DataFrame:
        """读取 information_schema.columns 列清单（含列注释与类型解析）。

        只返回 BASE TABLE 的列；列注释为空时 name 回退为 column_name；
        order_no 取 ORDINAL_POSITION（每表从 1 递增）。

        Returns:
            pd.DataFrame: 包含列信息、数据类型的 DataFrame
        """
        if not self.schemas:
            return pd.DataFrame(columns=_COLUMN_COLUMNS)
        sql = (
            "SELECT c.TABLE_SCHEMA AS table_schema, c.TABLE_NAME AS table_name, "
            "c.COLUMN_NAME AS column_name, c.COLUMN_TYPE AS column_type, "
            "c.COLUMN_COMMENT AS column_comment, "
            "c.ORDINAL_POSITION AS ordinal_position "
            "FROM information_schema.columns c "
            "JOIN information_schema.tables t "
            "ON t.TABLE_SCHEMA = c.TABLE_SCHEMA "
            "AND t.TABLE_NAME = c.TABLE_NAME "
            "AND t.TABLE_TYPE = 'BASE TABLE' "
            f"WHERE c.TABLE_SCHEMA IN ({self._schema_in_clause(self.schemas)}) "
            "ORDER BY c.TABLE_SCHEMA, c.TABLE_NAME, c.ORDINAL_POSITION"
        )
        df = self._fetch(sql, tuple(self.schemas))
        records: list[dict] = []
        for schema, table_name, column_name, column_type, comment, ordinal in zip(
            df["table_schema"].tolist(),
            df["table_name"].tolist(),
            df["column_name"].tolist(),
            df["column_type"].tolist(),
            df["column_comment"].tolist(),
            df["ordinal_position"].tolist(),
            strict=False,
        ):
            # COLUMN_COMMENT 可能为 None / NaN / ""，非空字符串才作为展示名
            name = comment if isinstance(comment, str) and comment else str(column_name)
            size, precision, scale = _parse_type(str(column_type))
            column_id = f"{schema}.{table_name}.{column_name}"
            full_table_name = f"{schema}.{table_name}"
            if self.datasource_name:
                column_id = qualify(self.datasource_name, column_id)
                full_table_name = qualify(self.datasource_name, full_table_name)
            dtype = str(column_type)
            records.append(
                {
                    "column": column_id,
                    "column_name": str(column_name),
                    "name": name,
                    "full_table_name": full_table_name,
                    "data_entity_type": "MYSQL_COLUMN",
                    "dtype": dtype,
                    "size": size,
                    "precision": precision,
                    "scale": scale,
                    "order_no": int(ordinal),
                    "data_type": dtype,
                }
            )
        return pd.DataFrame(records, columns=_COLUMN_COLUMNS)
