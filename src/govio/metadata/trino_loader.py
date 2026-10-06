import re

import pandas as pd
from trino import dbapi

from .database import MetadataLoader
from .datasource import qualify

# 匹配 Trino 类型：base 或 base(params)，如 decimal(10,2)、varchar(19)、bigint
_TYPE_RE = re.compile(r"^(\w+)\s*(?:\(([^)]*)\))?$")


def _parse_type(type_str: str) -> tuple[int, int, int]:
    """从 Trino 类型字符串解析 (size, precision, scale)。

    decimal(p,s) -> (0, p, s)；decimal(p) -> (0, p, 0)
    varchar(n) / char(n) -> (n, 0, 0)；无参 varchar -> (0, 0, 0)
    其他类型 -> (0, 0, 0)
    """
    m = _TYPE_RE.match(type_str.strip())
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


class TrinoLoader(MetadataLoader):
    """从 Trino 集群加载表与列元数据。

    使用 Trino 原生 ``SHOW TABLES`` / ``SHOW COLUMNS`` 语句读取元数据，
    不依赖 ``information_schema`` 的 comment 列（该列在不同连接器/版本下
    可用性不一致），兼容性更好。产出与 DuckDBLoader 相同列结构的
    DataFrame，data_entity_type 使用 TRINO_TABLE / TRINO_COLUMN。
    """

    def __init__(
        self,
        host: str,
        catalog: str,
        schemas: list[str],
        port: int = 8080,
        user: str = "trino",
        http_scheme: str = "http",
        auth: object | None = None,
        datasource_name: str = "",
    ) -> None:
        self.host = host
        self.port = port
        self.user = user
        self.http_scheme = http_scheme
        self.auth = auth
        self.catalog = catalog
        self.schemas = schemas
        self.datasource_name = datasource_name

    def _connect(self):
        return dbapi.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            catalog=self.catalog,
            http_scheme=self.http_scheme,
            auth=self.auth,
        )

    def _fetch(self, sql: str) -> pd.DataFrame:
        """执行 SQL 并返回 DataFrame，列名取自 cursor.description。"""
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(sql)
            names = [d.name for d in cur.description]
            rows = cur.fetchall()
            return pd.DataFrame(rows, columns=names)
        finally:
            conn.close()

    @staticmethod
    def _qualify(*parts: str) -> str:
        """生成 "p1"."p2"..." 限定名，转义内部双引号。"""
        return ".".join('"' + str(p).replace('"', '""') + '"' for p in parts)

    def load_tables(self) -> pd.DataFrame:
        """SHOW TABLES FROM catalog.schema 逐 schema 读取表清单。

        SHOW TABLES 不返回表注释，name 回退为 table_name。
        """
        records: list[dict] = []
        for schema in self.schemas:
            sql = f"SHOW TABLES FROM {self._qualify(self.catalog, schema)}"
            df = self._fetch(sql)
            for table_name in df.iloc[:, 0].tolist():
                full_table_name = f"{schema}.{table_name}"
                if self.datasource_name:
                    full_table_name = qualify(self.datasource_name, full_table_name)
                records.append(
                    {
                        "full_table_name": full_table_name,
                        "schema": schema,
                        "table_name": str(table_name),
                        "name": str(table_name),
                        "data_entity_type": "TRINO_TABLE",
                        "database_name": self.catalog,
                    }
                )
        return pd.DataFrame(
            records,
            columns=[
                "full_table_name",
                "schema",
                "table_name",
                "name",
                "data_entity_type",
                "database_name",
            ],
        )

    def load_columns(self) -> pd.DataFrame:
        """SHOW COLUMNS FROM catalog.schema.table 逐表读取列与注释。"""
        tables = self.load_tables()
        col: list = []
        column_name: list[str] = []
        name: list[str] = []
        full_table_name: list[str] = []
        dtype: list[str] = []
        size: list[int] = []
        precision: list[int] = []
        scale: list[int] = []
        order_no: list[int] = []
        for schema, table_name in zip(
            tables["schema"].tolist(), tables["table_name"].tolist(), strict=False
        ):
            sql = f"SHOW COLUMNS FROM {self._qualify(self.catalog, schema, table_name)}"
            df = self._fetch(sql)
            cnames = df["Column"].astype(str).tolist()
            ctypes = df["Type"].astype(str).tolist()
            comments = df["Comment"].tolist()
            for i, (cname, ctype, comment) in enumerate(
                zip(cnames, ctypes, comments, strict=False), start=1
            ):
                # Comment 可能为 None / NaN / ""，非空字符串才作为展示名
                disp = comment if isinstance(comment, str) and comment else cname
                _size, _prec, _scale = _parse_type(ctype)
                col_id = f"{schema}.{table_name}.{cname}"
                ftn_id = f"{schema}.{table_name}"
                if self.datasource_name:
                    col_id = qualify(self.datasource_name, col_id)
                    ftn_id = qualify(self.datasource_name, ftn_id)
                col.append(col_id)
                column_name.append(cname)
                name.append(disp)
                full_table_name.append(ftn_id)
                dtype.append(ctype)
                size.append(_size)
                precision.append(_prec)
                scale.append(_scale)
                order_no.append(i)
        return pd.DataFrame(
            {
                "column": col,
                "column_name": column_name,
                "name": name,
                "full_table_name": full_table_name,
                "data_entity_type": "TRINO_COLUMN",
                "dtype": dtype,
                "size": size,
                "precision": precision,
                "scale": scale,
                "order_no": order_no,
                "data_type": dtype,
            }
        )
