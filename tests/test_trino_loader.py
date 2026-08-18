import re

import pytest

from govio.metadata.database import MetadataLoader
from govio.metadata.trino_loader import TrinoLoader


class FakeTrinoConn:
    """内存版 Trino 连接，支持 SHOW TABLES / SHOW COLUMNS。

    tables: {schema: [table_name, ...]}
    columns: {"schema.table": [(Column, Type, Extra, Comment), ...]}
    """

    def __init__(self, tables, columns):
        self.tables = tables
        self.columns = columns
        self._desc: list[str] = []
        self._rows: list[list] = []

    def cursor(self, **_kw):
        return self

    def execute(self, sql, params=None):
        parts = re.findall(r'"([^"]*)"', sql)
        self._desc, self._rows = [], []
        s = sql.strip()
        if s.startswith("SHOW TABLES"):
            schema = parts[1]
            self._rows = [[t] for t in self.tables.get(schema, [])]
            self._desc = ["Table"]
        elif s.startswith("SHOW COLUMNS"):
            schema, table = parts[1], parts[2]
            self._rows = [list(r) for r in self.columns.get(f"{schema}.{table}", [])]
            self._desc = ["Column", "Type", "Extra", "Comment"]
        return self

    @property
    def description(self):
        return [type("D", (), {"name": n})() for n in self._desc]

    def fetchall(self):
        return self._rows

    def close(self):
        pass


def make_loader(monkeypatch, tables, columns, **kwargs):
    """构造一个打桩了 trino.dbapi.connect 的 TrinoLoader。"""
    captured: dict = {}

    def _connect(**kw):
        captured.update(kw)
        return FakeTrinoConn(tables, columns)

    monkeypatch.setattr("govio.metadata.trino_loader.dbapi.connect", _connect)
    kwargs.setdefault("schemas", ["test_schema"])
    loader = TrinoLoader(
        host="localhost",
        catalog="test_cat",
        **kwargs,
    )
    loader._captured = captured  # type: ignore[attr-defined]
    return loader


@pytest.fixture
def fake_dataset():
    tables = {"test_schema": ["users", "orders"]}
    columns = {
        "test_schema.users": [
            ("id", "bigint", "", "用户ID"),
            ("name", "varchar", "", "用户名"),
        ],
        "test_schema.orders": [
            ("order_id", "integer", "", ""),
            ("amount", "decimal(10,2)", "", ""),
        ],
    }
    return tables, columns


def test_trino_loader_is_metadata_loader(monkeypatch, fake_dataset):
    """TrinoLoader 是 MetadataLoader 的子类。"""
    tables, columns = fake_dataset
    loader = make_loader(monkeypatch, tables, columns)
    assert isinstance(loader, MetadataLoader)


def test_trino_loader_load_tables(monkeypatch, fake_dataset):
    """load_tables 返回预期的表结构与数据。"""
    tables, columns = fake_dataset
    loader = make_loader(monkeypatch, tables, columns)
    df = loader.load_tables()
    assert len(df) == 2
    assert list(df.columns) == [
        "full_table_name",
        "schema",
        "table_name",
        "name",
        "data_entity_type",
        "database_name",
    ]
    assert all(df["data_entity_type"] == "TRINO_TABLE")
    assert all(df["database_name"] == "test_cat")
    assert set(df["schema"]) == {"test_schema"}
    assert set(df["table_name"]) == {"users", "orders"}
    assert set(df["full_table_name"]) == {"test_schema.users", "test_schema.orders"}


def test_trino_loader_load_columns(monkeypatch, fake_dataset):
    """load_columns 返回预期的列结构与数据。"""
    tables, columns = fake_dataset
    loader = make_loader(monkeypatch, tables, columns)
    df = loader.load_columns()
    assert len(df) == 4
    assert "column" in df.columns
    assert "column_name" in df.columns
    assert "name" in df.columns
    assert "full_table_name" in df.columns
    assert "dtype" in df.columns
    assert "data_type" in df.columns
    assert "order_no" in df.columns
    assert all(df["data_entity_type"] == "TRINO_COLUMN")
    # dtype 直接取自 SHOW COLUMNS 的 Type
    assert set(df["dtype"]) == {"bigint", "varchar", "integer", "decimal(10,2)"}


def test_trino_loader_properties(monkeypatch, fake_dataset):
    """PhysicalTable 和 Col 属性正确委托。"""
    tables, columns = fake_dataset
    loader = make_loader(monkeypatch, tables, columns)
    assert len(loader.PhysicalTable) == 2
    assert len(loader.Col) == 4


def test_trino_loader_column_comments(monkeypatch, fake_dataset):
    """列注释被正确加载；空注释回退为列名。"""
    tables, columns = fake_dataset
    loader = make_loader(monkeypatch, tables, columns)
    df = loader.load_columns()
    id_row = df[df["column_name"] == "id"].iloc[0]
    assert id_row["name"] == "用户ID"
    # order_id 无注释，回退为列名
    order_id_row = df[df["column_name"] == "order_id"].iloc[0]
    assert order_id_row["name"] == "order_id"


def test_trino_loader_table_name_fallback(monkeypatch, fake_dataset):
    """SHOW TABLES 无注释，name 回退为 table_name。"""
    tables, columns = fake_dataset
    loader = make_loader(monkeypatch, tables, columns)
    df = loader.load_tables()
    users_row = df[df["table_name"] == "users"].iloc[0]
    assert users_row["name"] == "users"


def test_trino_loader_order_no(monkeypatch, fake_dataset):
    """order_no 按列出现顺序从 1 递增。"""
    tables, columns = fake_dataset
    loader = make_loader(monkeypatch, tables, columns)
    df = loader.load_columns()
    users_cols = df[df["full_table_name"] == "test_schema.users"].sort_values(
        "order_no"
    )
    assert users_cols["order_no"].tolist() == [1, 2]
    assert users_cols["column_name"].tolist() == ["id", "name"]


def test_trino_loader_empty_schemas(monkeypatch, fake_dataset):
    """空 schema 列表返回带正确列名的空 DataFrame。"""
    tables, columns = fake_dataset
    loader = make_loader(monkeypatch, tables, columns, schemas=[])
    df_tables = loader.load_tables()
    assert df_tables.empty
    assert "full_table_name" in df_tables.columns
    df_cols = loader.load_columns()
    assert df_cols.empty
    assert "column" in df_cols.columns


def test_trino_loader_multiple_schemas(monkeypatch):
    """多个 schema 同时导出。"""
    tables = {"s1": ["t1"], "s2": ["t2", "t3"]}
    columns = {
        "s1.t1": [("a", "varchar", "", "字段a")],
        "s2.t2": [("b", "integer", "", "")],
        "s2.t3": [("c", "bigint", "", "")],
    }
    loader = make_loader(monkeypatch, tables, columns, schemas=["s1", "s2"])
    df_tables = loader.load_tables()
    assert len(df_tables) == 3
    assert set(df_tables["schema"]) == {"s1", "s2"}
    df_cols = loader.load_columns()
    assert len(df_cols) == 3


def test_trino_loader_uses_quoted_identifiers():
    """_qualify 生成双引号限定符并转义内部双引号。"""
    assert TrinoLoader._qualify("cat", "sch") == '"cat"."sch"'
    inner_quote = 'sc"h'
    assert TrinoLoader._qualify("cat", inner_quote) == '"cat"."sc""h"'


def test_trino_loader_connect_kwargs(monkeypatch, fake_dataset):
    """构造参数正确传递给 trino.dbapi.connect。"""
    tables, columns = fake_dataset
    loader = make_loader(
        monkeypatch,
        tables,
        columns,
        port=8443,
        user="alice",
        http_scheme="https",
    )
    loader.load_tables()
    kw = loader._captured
    assert kw["host"] == "localhost"
    assert kw["port"] == 8443
    assert kw["user"] == "alice"
    assert kw["catalog"] == "test_cat"
    assert kw["http_scheme"] == "https"


@pytest.mark.parametrize(
    "type_str, expected",
    [
        ("decimal(10,2)", (0, 10, 2)),
        ("decimal(38)", (0, 38, 0)),
        ("varchar(19)", (19, 0, 0)),
        ("char(3)", (3, 0, 0)),
        ("varchar", (0, 0, 0)),
        ("bigint", (0, 0, 0)),
        ("integer", (0, 0, 0)),
        ("boolean", (0, 0, 0)),
        ("date", (0, 0, 0)),
    ],
)
def test_parse_type(type_str, expected):
    """size/precision/scale 从 Trino 类型字符串正确解析。"""
    from govio.metadata.trino_loader import _parse_type

    assert _parse_type(type_str) == expected


def test_trino_loader_type_parsing(monkeypatch):
    """load_columns 将 decimal/varchar 类型解析为 size/precision/scale。"""
    tables = {"s": ["t"]}
    columns = {
        "s.t": [
            ("price", "decimal(10,2)", "", "价格"),
            ("code", "varchar(19)", "", ""),
            ("flag", "boolean", "", ""),
            ("unbounded", "varchar", "", ""),
        ],
    }
    loader = make_loader(monkeypatch, tables, columns, schemas=["s"])
    df = loader.load_columns().set_index("column_name")
    assert df.loc["price", "precision"] == 10
    assert df.loc["price", "scale"] == 2
    assert df.loc["price", "size"] == 0
    assert df.loc["code", "size"] == 19
    assert df.loc["flag", "size"] == 0
    assert df.loc["flag", "precision"] == 0
    assert df.loc["unbounded", "size"] == 0
