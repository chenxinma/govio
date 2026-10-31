"""PostgresLoader 单元测试 — monkeypatch 打桩 psycopg 连接，不依赖真实数据库。"""

import pytest

from govio.metadata.database import MetadataLoader
from govio.metadata.postgres_loader import PostgresLoader, _parse_type

_TABLE_COLUMNS = [
    "full_table_name",
    "schema",
    "table_name",
    "name",
    "data_entity_type",
    "database_name",
]

_COL_COLUMNS = [
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


class FakeCursor:
    """内存版游标，按 SQL 分发预置的元数据行并模拟 ANY(%s) 的 schema 过滤。"""

    def __init__(self, conn):
        self._conn = conn
        self._names: list[str] = []
        self._rows: list[tuple] = []

    def execute(self, sql, params=None):
        schemas = list(params[0]) if params else []
        self._names, self._rows = [], []
        if "information_schema.tables" in sql:
            self._names = ["schema", "table_name", "table_comment"]
            rows = [
                (schema, table_name, comment)
                for schema, entries in self._conn.tables.items()
                if schema in schemas
                for table_name, comment in entries
            ]
            self._rows = sorted(rows, key=lambda r: (r[0], r[1]))
        elif "information_schema.columns" in sql:
            self._names = [
                "table_schema",
                "table_name",
                "column_name",
                "data_type",
                "character_maximum_length",
                "numeric_precision",
                "numeric_scale",
                "ordinal_position",
                "column_comment",
            ]
            rows = [
                (schema, *row)
                for schema, entries in self._conn.columns.items()
                if schema in schemas
                for row in entries
            ]
            self._rows = sorted(rows, key=lambda r: (r[0], r[1], r[7]))
        return self

    @property
    def description(self):
        return [type("D", (), {"name": n})() for n in self._names]

    def fetchall(self):
        return self._rows


class FakePostgresConn:
    """内存版 PostgreSQL 连接。

    tables: {schema: [(table_name, table_comment), ...]}
    columns: {schema: [(table_name, column_name, data_type,
                        character_maximum_length, numeric_precision,
                        numeric_scale, ordinal_position, column_comment), ...]}
    """

    def __init__(self, tables, columns):
        self.tables = tables
        self.columns = columns

    def cursor(self):
        return FakeCursor(self)

    def close(self):
        pass


def make_loader(monkeypatch, tables, columns, **kwargs):
    """构造一个打桩了 psycopg.connect 的 PostgresLoader。"""
    captured: dict = {}

    def _connect(**kw):
        captured.update(kw)
        return FakePostgresConn(tables, columns)

    monkeypatch.setattr("govio.metadata.postgres_loader.psycopg.connect", _connect)
    kwargs.setdefault("schemas", ["test_schema"])
    kwargs.setdefault("dbname", "test_db")
    loader = PostgresLoader(host="localhost", **kwargs)
    loader._captured = captured  # type: ignore[attr-defined]
    return loader


@pytest.fixture
def fake_dataset():
    tables = {"test_schema": [("users", "用户表"), ("orders", None)]}
    columns = {
        "test_schema": [
            ("users", "id", "bigint", None, 64, 0, 1, "用户ID"),
            ("users", "name", "character varying", 19, None, None, 2, None),
            ("orders", "order_id", "integer", None, 32, 0, 1, ""),
            ("orders", "amount", "numeric", None, 10, 2, 2, "金额"),
        ],
    }
    return tables, columns


def test_postgres_loader_is_metadata_loader(monkeypatch, fake_dataset):
    """PostgresLoader 是 MetadataLoader 的子类。"""
    tables, columns = fake_dataset
    loader = make_loader(monkeypatch, tables, columns)
    assert isinstance(loader, MetadataLoader)


def test_postgres_loader_load_tables(monkeypatch, fake_dataset):
    """load_tables 返回预期的表结构与数据。"""
    tables, columns = fake_dataset
    loader = make_loader(monkeypatch, tables, columns)
    df = loader.load_tables()
    assert len(df) == 2
    assert list(df.columns) == _TABLE_COLUMNS
    assert all(df["data_entity_type"] == "POSTGRES_TABLE")
    assert all(df["database_name"] == "test_db")
    assert set(df["schema"]) == {"test_schema"}
    assert set(df["table_name"]) == {"users", "orders"}
    assert set(df["full_table_name"]) == {"test_schema.users", "test_schema.orders"}
    users_row = df[df["table_name"] == "users"].iloc[0]
    assert users_row["name"] == "用户表"


def test_postgres_loader_load_columns(monkeypatch, fake_dataset):
    """load_columns 返回预期的列结构与数据。"""
    tables, columns = fake_dataset
    loader = make_loader(monkeypatch, tables, columns)
    df = loader.load_columns()
    assert len(df) == 4
    assert list(df.columns) == _COL_COLUMNS
    assert all(df["data_entity_type"] == "POSTGRES_COLUMN")
    assert set(df["dtype"]) == {"bigint", "character varying", "integer", "numeric"}
    # data_type 与 dtype 同值（与 TrinoLoader / DuckDBLoader 一致）
    assert df["data_type"].tolist() == df["dtype"].tolist()


def test_postgres_loader_properties(monkeypatch, fake_dataset):
    """PhysicalTable 和 Col 属性正确委托。"""
    tables, columns = fake_dataset
    loader = make_loader(monkeypatch, tables, columns)
    assert len(loader.PhysicalTable) == 2
    assert len(loader.Col) == 4


def test_postgres_loader_comments_fallback(monkeypatch, fake_dataset):
    """表/列注释正确加载；None / 空字符串注释回退为表名/列名。"""
    tables, columns = fake_dataset
    loader = make_loader(monkeypatch, tables, columns)
    df_cols = loader.load_columns()
    assert df_cols[df_cols["column_name"] == "id"].iloc[0]["name"] == "用户ID"
    assert df_cols[df_cols["column_name"] == "amount"].iloc[0]["name"] == "金额"
    # 列注释为 None，回退为列名
    assert df_cols[df_cols["column_name"] == "name"].iloc[0]["name"] == "name"
    # 列注释为 ""，回退为列名
    assert df_cols[df_cols["column_name"] == "order_id"].iloc[0]["name"] == "order_id"
    # 表注释为 None，回退为表名
    df_tables = loader.load_tables()
    assert df_tables[df_tables["table_name"] == "orders"].iloc[0]["name"] == "orders"


def test_postgres_loader_empty_string_comment_fallback(monkeypatch):
    """表注释为 "" 时同样回退为表名。"""
    tables = {"s": [("logs", "")]}
    columns = {"s": [("logs", "level", "text", None, None, None, 1, "")]}
    loader = make_loader(monkeypatch, tables, columns, schemas=["s"])
    df_tables = loader.load_tables()
    assert df_tables.iloc[0]["name"] == "logs"
    df_cols = loader.load_columns()
    assert df_cols.iloc[0]["name"] == "level"


def test_postgres_loader_type_parsing(monkeypatch, fake_dataset):
    """load_columns 将类型属性解析为 size/precision/scale。"""
    tables, columns = fake_dataset
    loader = make_loader(monkeypatch, tables, columns)
    df = loader.load_columns().set_index("column_name")
    # character varying(19) -> size=19
    assert df.loc["name", "size"] == 19
    assert df.loc["name", "precision"] == 0
    assert df.loc["name", "scale"] == 0
    # numeric(10,2) -> precision=10, scale=2
    assert df.loc["amount", "size"] == 0
    assert df.loc["amount", "precision"] == 10
    assert df.loc["amount", "scale"] == 2
    # bigint / integer 的 numeric_precision 是位宽语义，不计入 precision
    assert df.loc["id", "size"] == 0
    assert df.loc["id", "precision"] == 0
    assert df.loc["id", "scale"] == 0
    assert df.loc["order_id", "precision"] == 0


def test_postgres_loader_order_no(monkeypatch, fake_dataset):
    """order_no 每表按列顺序从 1 递增。"""
    tables, columns = fake_dataset
    loader = make_loader(monkeypatch, tables, columns)
    df = loader.load_columns()
    users_cols = df[df["full_table_name"] == "test_schema.users"].sort_values(
        "order_no"
    )
    assert users_cols["order_no"].tolist() == [1, 2]
    assert users_cols["column_name"].tolist() == ["id", "name"]
    orders_cols = df[df["full_table_name"] == "test_schema.orders"].sort_values(
        "order_no"
    )
    assert orders_cols["order_no"].tolist() == [1, 2]


def test_postgres_loader_order_no_renumbers_gaps(monkeypatch):
    """ordinal_position 有空洞（删列）时 order_no 仍从 1 连续递增。"""
    tables = {"s": [("t", None)]}
    columns = {
        "s": [
            ("t", "a", "integer", None, 32, 0, 1, None),
            ("t", "c", "integer", None, 32, 0, 3, None),
        ],
    }
    loader = make_loader(monkeypatch, tables, columns, schemas=["s"])
    df = loader.load_columns()
    assert df["order_no"].tolist() == [1, 2]
    assert df["column_name"].tolist() == ["a", "c"]


def test_postgres_loader_empty_schemas(monkeypatch, fake_dataset):
    """空 schema 列表返回带正确列名的空 DataFrame，且不发起连接。"""
    tables, columns = fake_dataset
    loader = make_loader(monkeypatch, tables, columns, schemas=[])
    df_tables = loader.load_tables()
    assert df_tables.empty
    assert list(df_tables.columns) == _TABLE_COLUMNS
    df_cols = loader.load_columns()
    assert df_cols.empty
    assert list(df_cols.columns) == _COL_COLUMNS
    assert loader._captured == {}


def test_postgres_loader_multiple_schemas(monkeypatch):
    """多个 schema 同时导出，未声明的 schema 被过滤。"""
    tables = {
        "s1": [("t1", None)],
        "s2": [("t2", None), ("t3", None)],
        "other": [("t4", None)],
    }
    columns = {
        "s1": [("t1", "a", "character varying", 10, None, None, 1, "字段a")],
        "s2": [
            ("t2", "b", "integer", None, 32, 0, 1, None),
            ("t3", "c", "bigint", None, 64, 0, 1, None),
        ],
        "other": [("t4", "d", "integer", None, 32, 0, 1, None)],
    }
    loader = make_loader(monkeypatch, tables, columns, schemas=["s1", "s2"])
    df_tables = loader.load_tables()
    assert len(df_tables) == 3
    assert set(df_tables["schema"]) == {"s1", "s2"}
    assert set(df_tables["full_table_name"]) == {"s1.t1", "s2.t2", "s2.t3"}
    df_cols = loader.load_columns()
    assert len(df_cols) == 3
    assert set(df_cols["full_table_name"]) == {"s1.t1", "s2.t2", "s2.t3"}


def test_postgres_loader_datasource_qualify(monkeypatch, fake_dataset):
    """datasource_name 非空时 full_table_name / column 加数据源前缀。"""
    tables, columns = fake_dataset
    loader = make_loader(monkeypatch, tables, columns, datasource_name="pg_src")
    df_tables = loader.load_tables()
    assert set(df_tables["full_table_name"]) == {
        "pg_src.test_schema.users",
        "pg_src.test_schema.orders",
    }
    df_cols = loader.load_columns()
    assert set(df_cols["full_table_name"]) == {
        "pg_src.test_schema.users",
        "pg_src.test_schema.orders",
    }
    assert set(df_cols["column"]) == {
        "pg_src.test_schema.users.id",
        "pg_src.test_schema.users.name",
        "pg_src.test_schema.orders.order_id",
        "pg_src.test_schema.orders.amount",
    }


def test_postgres_loader_connect_kwargs(monkeypatch, fake_dataset):
    """构造参数正确传递给 psycopg.connect。"""
    tables, columns = fake_dataset
    loader = make_loader(
        monkeypatch,
        tables,
        columns,
        port=5433,
        user="alice",
        password="s3cret",
        dbname="sales",
    )
    loader.load_tables()
    assert loader._captured == {
        "host": "localhost",
        "port": 5433,
        "user": "alice",
        "password": "s3cret",
        "dbname": "sales",
    }


@pytest.mark.parametrize(
    "data_type, char_len, num_prec, num_scale, expected",
    [
        ("character varying", 19, None, None, (19, 0, 0)),
        ("character", 3, None, None, (3, 0, 0)),
        ("bit", 1, None, None, (1, 0, 0)),
        ("bit varying", 8, None, None, (8, 0, 0)),
        ("character varying", None, None, None, (0, 0, 0)),
        ("text", None, None, None, (0, 0, 0)),
        ("numeric", None, 10, 2, (0, 10, 2)),
        ("numeric", None, 10, 0, (0, 10, 0)),
        ("numeric", None, None, None, (0, 0, 0)),
        ("decimal", None, 12, 4, (0, 12, 4)),
        ("bigint", None, 64, 0, (0, 0, 0)),
        ("integer", None, 32, 0, (0, 0, 0)),
        ("double precision", None, 53, None, (0, 0, 0)),
        ("timestamp without time zone", None, None, None, (0, 0, 0)),
    ],
)
def test_parse_type(data_type, char_len, num_prec, num_scale, expected):
    """size/precision/scale 从 information_schema 类型属性正确解析。"""
    assert _parse_type(data_type, char_len, num_prec, num_scale) == expected
