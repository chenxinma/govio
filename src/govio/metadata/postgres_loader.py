"""PostgreSQL 元数据加载器

表与列来自 ``information_schema.tables`` / ``information_schema.columns``，
注释来自 pg_catalog：表注释 ``obj_description``、列注释 ``col_description``。
产出与 DuckDBLoader / TrinoLoader 相同列结构的 DataFrame，
data_entity_type 使用 ``POSTGRES_TABLE`` / ``POSTGRES_COLUMN``。
"""

import pandas as pd
import psycopg

from .database import MetadataLoader
from .datasource import qualify

# 表清单 + 表注释；obj_description 需要 pg_class 的 OID，按命名空间+表名关联。
# information_schema.tables 含表/视图/外表，与 TrinoLoader 的 SHOW TABLES 范围一致。
_TABLES_SQL = """
SELECT t.table_schema AS "schema",
       t.table_name AS table_name,
       obj_description(c.oid, 'pg_class') AS table_comment
FROM information_schema.tables t
JOIN pg_catalog.pg_namespace n ON n.nspname = t.table_schema
JOIN pg_catalog.pg_class c ON c.relnamespace = n.oid AND c.relname = t.table_name
WHERE t.table_schema = ANY(%s)
ORDER BY t.table_schema, t.table_name
"""

# 列清单 + 列注释；ordinal_position 即 pg_attribute.attnum，
# 故 col_description(pc.oid, c.ordinal_position) 取列注释。
_COLUMNS_SQL = """
SELECT c.table_schema AS table_schema,
       c.table_name AS table_name,
       c.column_name AS column_name,
       c.data_type AS data_type,
       c.character_maximum_length AS character_maximum_length,
       c.numeric_precision AS numeric_precision,
       c.numeric_scale AS numeric_scale,
       c.ordinal_position AS ordinal_position,
       col_description(pc.oid, c.ordinal_position) AS column_comment
FROM information_schema.columns c
JOIN pg_catalog.pg_namespace n ON n.nspname = c.table_schema
JOIN pg_catalog.pg_class pc ON pc.relnamespace = n.oid AND pc.relname = c.table_name
WHERE c.table_schema = ANY(%s)
ORDER BY c.table_schema, c.table_name, c.ordinal_position
"""

# 输出 DataFrame 列结构与 TrinoLoader / DuckDBLoader 保持一致
_TABLE_OUTPUT_COLUMNS = [
    "full_table_name",
    "schema",
    "table_name",
    "name",
    "data_entity_type",
    "database_name",
]

_COLUMN_OUTPUT_COLUMNS = [
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


def _parse_type(
    data_type: str,
    character_maximum_length: float | None,
    numeric_precision: float | None,
    numeric_scale: float | None,
) -> tuple[int, int, int]:
    """从 information_schema 类型属性解析 (size, precision, scale)。

    character varying / character / bit -> (character_maximum_length, 0, 0)
    numeric / decimal -> (0, numeric_precision, numeric_scale)
    其他类型 -> (0, 0, 0)。整型的 numeric_precision 是位宽语义，
    不计入 precision，与 TrinoLoader 的 decimal 语义保持一致。

    判空统一用 pd.notna：_fetch 经 pd.DataFrame 承接整型+NULL 混合列时
    会被推断为 float64，SQL NULL 变成 NaN（既非 None 也非 falsy），
    用 ``is not None`` 或真值判断会在 int(NaN) 上抛 ValueError。

    Args:
        data_type: information_schema.columns.data_type 类型名
        character_maximum_length: 字符/位串最大长度（无则为 None/NaN）
        numeric_precision: 数值精度（无则为 None/NaN）
        numeric_scale: 数值小数位（无则为 None/NaN）

    Returns:
        tuple[int, int, int]: (size, precision, scale)
    """
    base = data_type.strip().lower()
    if base in ("character varying", "character", "bit", "bit varying"):
        length = (
            int(character_maximum_length) if pd.notna(character_maximum_length) else 0
        )
        return length, 0, 0
    if base in ("numeric", "decimal"):
        precision = int(numeric_precision) if pd.notna(numeric_precision) else 0
        scale = int(numeric_scale) if pd.notna(numeric_scale) else 0
        return 0, precision, scale
    return 0, 0, 0


class PostgresLoader(MetadataLoader):
    """从 PostgreSQL 库加载表与列元数据。

    用 information_schema.tables / information_schema.columns 抽取表与列，
    pg_catalog 的 obj_description / col_description 补充表注释与列注释。
    注释为空时 name 回退为表名/列名；支持 datasource_name 全限定
    （qualify）与 schema 过滤。data_entity_type 使用
    POSTGRES_TABLE / POSTGRES_COLUMN。
    """

    def __init__(
        self,
        host: str,
        schemas: list[str],
        dbname: str = "postgres",
        port: int = 5432,
        user: str = "postgres",
        password: str = "",
        datasource_name: str = "",
    ) -> None:
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.dbname = dbname
        self.schemas = schemas
        self.datasource_name = datasource_name

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            dbname=self.dbname,
        )

    def _fetch(self, sql: str, params: tuple | list | None = None) -> pd.DataFrame:
        """执行 SQL 并返回 DataFrame，列名取自 cursor.description。"""
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(sql, params)
            names = [d.name for d in cur.description]
            rows = cur.fetchall()
            return pd.DataFrame(rows, columns=names)
        finally:
            conn.close()

    def load_tables(self) -> pd.DataFrame:
        """读取表清单与表注释。

        表注释为空（NULL 或 ""）时 name 回退为 table_name；
        datasource_name 非空时 full_table_name 加数据源前缀。
        """
        if not self.schemas:
            return pd.DataFrame(columns=_TABLE_OUTPUT_COLUMNS)
        df = self._fetch(_TABLES_SQL, (self.schemas,))
        records: list[dict] = []
        for row in df.itertuples(index=False):
            schema = str(row.schema)
            table_name = str(row.table_name)
            comment = row.table_comment
            name = comment if isinstance(comment, str) and comment else table_name
            full_table_name = f"{schema}.{table_name}"
            if self.datasource_name:
                full_table_name = qualify(self.datasource_name, full_table_name)
            records.append(
                {
                    "full_table_name": full_table_name,
                    "schema": schema,
                    "table_name": table_name,
                    "name": name,
                    "data_entity_type": "POSTGRES_TABLE",
                    "database_name": self.dbname,
                }
            )
        return pd.DataFrame(records, columns=_TABLE_OUTPUT_COLUMNS)

    def load_columns(self) -> pd.DataFrame:
        """读取列清单、列注释与类型参数。

        order_no 按 ordinal_position 排序后每表从 1 递增
        （ordinal_position 即 attnum，删列后可能有空洞，故重排）；
        列注释为空时 name 回退为 column_name。
        """
        if not self.schemas:
            return pd.DataFrame(columns=_COLUMN_OUTPUT_COLUMNS)
        df = self._fetch(_COLUMNS_SQL, (self.schemas,))
        records: list[dict] = []
        order_no = 0
        prev_table: tuple[str, str] | None = None
        for row in df.itertuples(index=False):
            schema = str(row.table_schema)
            table_name = str(row.table_name)
            column_name = str(row.column_name)
            if (schema, table_name) != prev_table:
                prev_table = (schema, table_name)
                order_no = 0
            order_no += 1
            comment = row.column_comment
            name = comment if isinstance(comment, str) and comment else column_name
            dtype = str(row.data_type)
            size, precision, scale = _parse_type(
                dtype,
                row.character_maximum_length,
                row.numeric_precision,
                row.numeric_scale,
            )
            column_id = f"{schema}.{table_name}.{column_name}"
            full_table_name = f"{schema}.{table_name}"
            if self.datasource_name:
                column_id = qualify(self.datasource_name, column_id)
                full_table_name = qualify(self.datasource_name, full_table_name)
            records.append(
                {
                    "column": column_id,
                    "column_name": column_name,
                    "name": name,
                    "full_table_name": full_table_name,
                    "data_entity_type": "POSTGRES_COLUMN",
                    "dtype": dtype,
                    "size": size,
                    "precision": precision,
                    "scale": scale,
                    "order_no": order_no,
                    "data_type": dtype,
                }
            )
        return pd.DataFrame(records, columns=_COLUMN_OUTPUT_COLUMNS)
