import pytest

from govio.metadata.database import MetadataLoader
from govio.metadata.mysql_loader import MysqlLoader, _parse_type


class FakeMysqlConn:
    """内存版 MySQL 连接，按 information_schema 语义返回表/列数据。

    tables: {schema: [(table_name, table_comment), ...]}，仅含 BASE TABLE
    columns: {(schema, table): [(column_name, column_type, column_comment, ordinal), ...]}
    """

    def __init__(self, tables, columns):
        self.tables = tables
        self.columns = columns
        self.executed: list[tuple[str, tuple]] = []
        self.closed = False

    def cursor(self):
        return FakeMysqlCursor(self)

    def close(self):
        self.closed = True


class FakeMysqlCursor:
    def __init__(self, conn: FakeMysqlConn):
        self.conn = conn
        self._desc: list[str] = []
        self._rows: list[tuple] = []

    def execute(self, sql, params=None):
        params = tuple(params or ())
        self.conn.executed.append((sql, params))
        self._desc, self._rows = [], []
        if "information_schema.columns" in sql:
            self._desc = [
                "table_schema",
                "table_name",
                "column_name",
                "column_type",
                "column_comment",
                "ordinal_position",
            ]
            # 模拟 JOIN information_schema.tables ... TABLE_TYPE = 'BASE TABLE'：
            # 只返回表清单（仅含 BASE TABLE）中出现的表的列
            for schema in params:
                for table_name, _comment in self.conn.tables.get(schema, []):
                    rows = self.conn.columns.get((schema, table_name), [])
                    self._rows.extend(
                        (schema, table_name, cname, ctype, ccomment, ordinal)
                        for cname, ctype, ccomment, ordinal in rows
                    )
        elif "information_schema.tables" in sql:
            self._desc = ["schema", "table_name", "table_comment"]
            for schema in params:
                self._rows.extend(
                    (schema, table_name, comment)
                    for table_name, comment in self.conn.tables.get(schema, [])
                )
        return self

    @property
    def description(self):
        return [(name, None, None, None, None, None, None) for name in self._desc]

    def fetchall(self):
        return self._rows


def make_loader(monkeypatch, tables, columns, **kwargs):
    """构造一个打桩了 pymysql.connect 的 MysqlLoader。"""
    captured: dict = {}
    conns: list[FakeMysqlConn] = []

    def _connect(**kw):
        captured.update(kw)
        conn = FakeMysqlConn(tables, columns)
        conns.append(conn)
        return conn

    monkeypatch.setattr("govio.metadata.mysql_loader.pymysql.connect", _connect)
    kwargs.setdefault("schemas", ["test_db"])
    loader = MysqlLoader(host="localhost", **kwargs)
    loader._captured = captured  # type: ignore[attr-defined]
    loader._conns = conns  # type: ignore[attr-defined]
    return loader


@pytest.fixture
def fake_dataset():
    tables = {"test_db": [("users", "用户表"), ("orders", "")]}
    columns = {
        ("test_db", "users"): [
            ("id", "bigint", "用户ID", 1),
            ("name", "varchar(64)", "", 2),
        ],
        ("test_db", "orders"): [
            ("order_id", "bigint", "", 1),
            ("amount", "decimal(10,2)", "金额", 2),
        ],
    }
    return tables, columns


def test_mysql_loader_is_metadata_loader(monkeypatch, fake_dataset):
    """MysqlLoader 是 MetadataLoader 的子类。"""
    tables, columns = fake_dataset
    loader = make_loader(monkeypatch, tables, columns)
    assert isinstance(loader, MetadataLoader)


def test_mysql_loader_load_tables(monkeypatch, fake_dataset):
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
    assert all(df["data_entity_type"] == "MYSQL_TABLE")
    assert set(df["schema"]) == {"test_db"}
    assert set(df["table_name"]) == {"users", "orders"}
    assert set(df["full_table_name"]) == {"test_db.users", "test_db.orders"}


def test_mysql_loader_load_columns(monkeypatch, fake_dataset):
    """load_columns 返回预期的列结构与数据。"""
    tables, columns = fake_dataset
    loader = make_loader(monkeypatch, tables, columns)
    df = loader.load_columns()
    assert len(df) == 4
    assert list(df.columns) == [
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
    assert all(df["data_entity_type"] == "MYSQL_COLUMN")
    # dtype 直接取自 information_schema 的 COLUMN_TYPE
    assert set(df["dtype"]) == {"bigint", "varchar(64)", "decimal(10,2)"}
    assert set(df["data_type"]) == {"bigint", "varchar(64)", "decimal(10,2)"}
    assert set(df["column"]) == {
        "test_db.users.id",
        "test_db.users.name",
        "test_db.orders.order_id",
        "test_db.orders.amount",
    }


def test_mysql_loader_properties(monkeypatch, fake_dataset):
    """PhysicalTable 和 Col 属性正确委托。"""
    tables, columns = fake_dataset
    loader = make_loader(monkeypatch, tables, columns)
    assert len(loader.PhysicalTable) == 2
    assert len(loader.Col) == 4


def test_mysql_loader_column_comments(monkeypatch, fake_dataset):
    """列注释被正确加载；空注释回退为列名。"""
    tables, columns = fake_dataset
    loader = make_loader(monkeypatch, tables, columns)
    df = loader.load_columns()
    id_row = df[df["column_name"] == "id"].iloc[0]
    assert id_row["name"] == "用户ID"
    # name 无注释，回退为列名
    name_row = df[df["column_name"] == "name"].iloc[0]
    assert name_row["name"] == "name"


def test_mysql_loader_column_comment_none_falls_back(monkeypatch, fake_dataset):
    """列注释为 None 时同样回退为列名。"""
    tables, columns = fake_dataset
    columns = dict(columns)
    columns[("test_db", "users")] = [("id", "bigint", None, 1)]
    loader = make_loader(monkeypatch, tables, columns)
    df = loader.load_columns()
    assert df.iloc[0]["name"] == "id"


def test_mysql_loader_table_name_fallback(monkeypatch, fake_dataset):
    """表注释为空时 name 回退为 table_name。"""
    tables, columns = fake_dataset
    loader = make_loader(monkeypatch, tables, columns)
    df = loader.load_tables()
    orders_row = df[df["table_name"] == "orders"].iloc[0]
    assert orders_row["name"] == "orders"
    users_row = df[df["table_name"] == "users"].iloc[0]
    assert users_row["name"] == "用户表"


def test_mysql_loader_order_no(monkeypatch, fake_dataset):
    """order_no 按 ORDINAL_POSITION 每表从 1 递增。"""
    tables, columns = fake_dataset
    loader = make_loader(monkeypatch, tables, columns)
    df = loader.load_columns()
    users_cols = df[df["full_table_name"] == "test_db.users"].sort_values("order_no")
    assert users_cols["order_no"].tolist() == [1, 2]
    assert users_cols["column_name"].tolist() == ["id", "name"]
    orders_cols = df[df["full_table_name"] == "test_db.orders"].sort_values("order_no")
    assert orders_cols["order_no"].tolist() == [1, 2]
    assert orders_cols["column_name"].tolist() == ["order_id", "amount"]


def test_mysql_loader_type_parsing(monkeypatch):
    """load_columns 将 decimal(10,2)/varchar(64)/bigint 解析为 size/precision/scale。"""
    tables = {"s": [("t", "")]}
    columns = {
        ("s", "t"): [
            ("price", "decimal(10,2)", "", 1),
            ("code", "varchar(64)", "", 2),
            ("flag", "bigint", "", 3),
        ],
    }
    loader = make_loader(monkeypatch, tables, columns, schemas=["s"])
    df = loader.load_columns().set_index("column_name")
    assert df.loc["price", "precision"] == 10
    assert df.loc["price", "scale"] == 2
    assert df.loc["price", "size"] == 0
    assert df.loc["code", "size"] == 64
    assert df.loc["code", "precision"] == 0
    assert df.loc["flag", "size"] == 0
    assert df.loc["flag", "precision"] == 0
    assert df.loc["flag", "scale"] == 0


def test_mysql_loader_empty_schemas(monkeypatch, fake_dataset):
    """空 schema 列表返回带正确列名的空 DataFrame，且不发起查询。"""
    tables, columns = fake_dataset
    loader = make_loader(monkeypatch, tables, columns, schemas=[])
    df_tables = loader.load_tables()
    assert df_tables.empty
    assert list(df_tables.columns) == [
        "full_table_name",
        "schema",
        "table_name",
        "name",
        "data_entity_type",
        "database_name",
    ]
    df_cols = loader.load_columns()
    assert df_cols.empty
    assert "column" in df_cols.columns
    assert loader._conns == []  # type: ignore[attr-defined]


def test_mysql_loader_multiple_schemas(monkeypatch):
    """多个 schema 同时导出。"""
    tables = {"s1": [("t1", "")], "s2": [("t2", ""), ("t3", "")]}
    columns = {
        ("s1", "t1"): [("a", "varchar(19)", "字段a", 1)],
        ("s2", "t2"): [("b", "integer", "", 1)],
        ("s2", "t3"): [("c", "bigint", "", 1)],
    }
    loader = make_loader(monkeypatch, tables, columns, schemas=["s1", "s2"])
    df_tables = loader.load_tables()
    assert len(df_tables) == 3
    assert set(df_tables["schema"]) == {"s1", "s2"}
    df_cols = loader.load_columns()
    assert len(df_cols) == 3
    assert set(df_cols["full_table_name"]) == {"s1.t1", "s2.t2", "s2.t3"}


def test_mysql_loader_datasource_qualify(monkeypatch, fake_dataset):
    """datasource_name 为 full_table_name 与 column 加全限定前缀。"""
    tables, columns = fake_dataset
    loader = make_loader(monkeypatch, tables, columns, datasource_name="ds1")
    df_tables = loader.load_tables()
    assert set(df_tables["full_table_name"]) == {
        "ds1.test_db.users",
        "ds1.test_db.orders",
    }
    df_cols = loader.load_columns()
    assert "ds1.test_db.users.id" in set(df_cols["column"])
    assert set(df_cols["full_table_name"]) == {
        "ds1.test_db.users",
        "ds1.test_db.orders",
    }


def test_mysql_loader_connect_kwargs(monkeypatch, fake_dataset):
    """构造参数正确传递给 pymysql.connect。"""
    tables, columns = fake_dataset
    loader = make_loader(
        monkeypatch,
        tables,
        columns,
        port=3307,
        user="alice",
        password="pw",
        database="test_db",
    )
    loader.load_tables()
    kw = loader._captured
    assert kw["host"] == "localhost"
    assert kw["port"] == 3307
    assert kw["user"] == "alice"
    assert kw["password"] == "pw"
    assert kw["database"] == "test_db"


def test_mysql_loader_database_name(monkeypatch, fake_dataset):
    """database_name 列取构造参数 database。"""
    tables, columns = fake_dataset
    loader = make_loader(monkeypatch, tables, columns, database="test_db")
    df = loader.load_tables()
    assert all(df["database_name"] == "test_db")


def test_mysql_loader_closes_connection(monkeypatch, fake_dataset):
    """查询后连接被关闭。"""
    tables, columns = fake_dataset
    loader = make_loader(monkeypatch, tables, columns)
    loader.load_tables()
    assert loader._conns[0].closed  # type: ignore[attr-defined]


def test_mysql_loader_schema_filter_is_parameterized(monkeypatch, fake_dataset):
    """schema 过滤用 %s 占位符参数化传递，值不拼进 SQL。"""
    tables, columns = fake_dataset
    loader = make_loader(monkeypatch, tables, columns, schemas=["a", "b"])
    loader.load_tables()
    sql, params = loader._conns[0].executed[0]  # type: ignore[attr-defined]
    assert "IN (%s, %s)" in sql
    assert params == ("a", "b")


@pytest.mark.parametrize(
    "type_str, expected",
    [
        ("decimal(10,2)", (0, 10, 2)),
        ("decimal(38)", (0, 38, 0)),
        ("varchar(64)", (64, 0, 0)),
        ("char(3)", (3, 0, 0)),
        ("varchar", (0, 0, 0)),
        ("bigint", (0, 0, 0)),
        ("bigint unsigned", (0, 0, 0)),
        ("int(11)", (0, 0, 0)),
        ("enum('a','b')", (0, 0, 0)),
    ],
)
def test_parse_type(type_str, expected):
    """size/precision/scale 从 MySQL COLUMN_TYPE 正确解析。"""
    assert _parse_type(type_str) == expected
