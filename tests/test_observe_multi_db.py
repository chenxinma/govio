"""observe 查询执行多数据库链路测试。

覆盖：
1. `_resolve_sqlalchemy_target` 的 scheme 归一（mysql / postgres / trino+https）
2. `DatabaseManager` 把归一后的 URL 与合并后的 connect_args 传给 create_engine
3. `execute_sql` 对 postgresql / trino / mysql 走统一 SQLAlchemy 引擎路径
4. `load_dataframe` 查询执行链路打桩验证

打桩 create_engine / pandas.read_sql，不依赖真实数据库（仅驱动可用性用例离线创建引擎）。
"""

from types import SimpleNamespace

import pandas as pd
import pytest
import sqlalchemy

from govio.observe_data.config import DataSourceConfig
from govio.observe_data.core.database import DatabaseManager, _resolve_sqlalchemy_target
from govio.observe_data.tools.load_dataframe import load_dataframe

# ---------------------------------------------------------------------------
# scheme 归一
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected_url", "expected_args"),
    [
        (
            "mysql://u:p@h:3306/db",
            "mysql+pymysql://u:p@h:3306/db",
            {},
        ),
        (
            "mysql+pymysql://u:p@h:3306/db",
            "mysql+pymysql://u:p@h:3306/db",
            {},
        ),
        (
            "postgresql://u:p@h:5432/db",
            "postgresql+psycopg://u:p@h:5432/db",
            {},
        ),
        (
            "postgres://u:p@h:5432/db",
            "postgresql+psycopg://u:p@h:5432/db",
            {},
        ),
        (
            "postgresql+psycopg://u:p@h:5432/db",
            "postgresql+psycopg://u:p@h:5432/db",
            {},
        ),
        (
            "trino://u@h:8080/catalog",
            "trino://u@h:8080/catalog",
            {},
        ),
        (
            "trino+https://u@h:8443/catalog",
            "trino://u@h:8443/catalog",
            {"http_scheme": "https"},
        ),
    ],
)
def test_resolve_sqlalchemy_target(url, expected_url, expected_args):
    """无驱动限定的 scheme 归一到已安装驱动，其余原样返回。"""
    assert _resolve_sqlalchemy_target(url) == (expected_url, expected_args)


def test_normalized_urls_resolve_with_installed_drivers():
    """归一后的 mysql / postgresql / trino URL 均可离线创建引擎（不发起连接）。"""
    for url, dialect in [
        ("mysql://u:p@h:3306/db", "mysql"),
        ("postgresql://u:p@h:5432/db", "postgresql"),
        ("trino://u@h:8080/catalog", "trino"),
    ]:
        sa_url, _ = _resolve_sqlalchemy_target(url)
        engine = sqlalchemy.create_engine(sa_url)
        assert engine.dialect.name == dialect
        engine.dispose()


# ---------------------------------------------------------------------------
# DatabaseManager -> create_engine 参数传递
# ---------------------------------------------------------------------------


class FakeConnection:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


class FakeEngine:
    def connect(self):
        return FakeConnection()


@pytest.fixture
def engine_calls(monkeypatch) -> dict:
    """打桩 sqlalchemy.create_engine，记录 URL 与 connect_args。"""
    calls: dict = {}
    engines: list[FakeEngine] = []

    def _fake_create_engine(url, connect_args=None):
        calls["url"] = url
        calls["connect_args"] = connect_args
        engine = FakeEngine()
        engines.append(engine)
        return engine

    monkeypatch.setattr("sqlalchemy.create_engine", _fake_create_engine)
    calls["engines"] = engines
    return calls


def test_create_engine_receives_normalized_postgres_url(engine_calls):
    """postgres URL 归一后传给 create_engine，config.connect_args 原样透传。"""
    manager = DatabaseManager(
        {
            "pg": DataSourceConfig(
                url="postgresql://u:p@h:5432/db",
                connect_args={"sslmode": "require"},
            )
        }
    )
    manager.get_engine("pg")
    assert engine_calls["url"] == "postgresql+psycopg://u:p@h:5432/db"
    assert engine_calls["connect_args"] == {"sslmode": "require"}


def test_create_engine_trino_https_merges_connect_args(engine_calls):
    """trino+https 归一为 trino:// 并注入 http_scheme，config.connect_args 优先。"""
    manager = DatabaseManager(
        {
            "trino": DataSourceConfig(
                url="trino+https://u@h:8443/catalog",
                connect_args={"verify": False},
            )
        }
    )
    manager.get_engine("trino")
    assert engine_calls["url"] == "trino://u@h:8443/catalog"
    assert engine_calls["connect_args"] == {"http_scheme": "https", "verify": False}


def test_create_engine_mysql_alias_without_driver_suffix(engine_calls):
    """mysql://（无驱动限定）归一为 mysql+pymysql://。"""
    manager = DatabaseManager({"my": DataSourceConfig(url="mysql://u:p@h:3306/db")})
    manager.get_engine("my")
    assert engine_calls["url"] == "mysql+pymysql://u:p@h:3306/db"


# ---------------------------------------------------------------------------
# execute_sql 路由
# ---------------------------------------------------------------------------


def test_execute_sql_runs_through_sqlalchemy_engine(engine_calls, monkeypatch):
    """postgresql / mysql 数据源的 SQL 通过 SQLAlchemy 引擎执行。"""
    captured: dict = {}

    def _fake_read_sql(sql, conn):
        captured["sql"] = str(sql)
        captured["conn"] = conn
        return pd.DataFrame({"x": [1]})

    monkeypatch.setattr("pandas.read_sql", _fake_read_sql)
    manager = DatabaseManager(
        {"pg": DataSourceConfig(url="postgresql://u:p@h:5432/db")}
    )
    df = manager.execute_sql("pg", "SELECT 1 AS x")
    assert list(df["x"]) == [1]
    assert captured["sql"] == "SELECT 1 AS x"
    assert isinstance(captured["conn"], FakeConnection)


def test_execute_sql_unknown_datasource_raises():
    """未注册数据源报 ValueError。"""
    manager = DatabaseManager({})
    with pytest.raises(ValueError, match="数据源不存在"):
        manager.execute_sql("nope", "SELECT 1")


def test_execute_sql_duckdb_native(tmp_path):
    """duckdb 数据源仍走原生连接，不受 scheme 归一影响。"""
    import duckdb

    db_path = tmp_path / "t.duckdb"
    conn = duckdb.connect(str(db_path))
    conn.execute("CREATE TABLE a (x INTEGER)")
    conn.execute("INSERT INTO a VALUES (1), (2)")
    conn.close()

    manager = DatabaseManager({"local": DataSourceConfig(url=f"duckdb://{db_path}")})
    df = manager.execute_sql("local", "SELECT count(*) AS n FROM a")
    assert int(df["n"].iloc[0]) == 2


# ---------------------------------------------------------------------------
# load_dataframe 查询执行链路
# ---------------------------------------------------------------------------


class FakeStore:
    """打桩 ObserveStore：记录 store() 调用。"""

    def __init__(self):
        self.stored = None

    def store(self, name, df, datasource, sql):
        self.stored = {"name": name, "df": df, "datasource": datasource, "sql": sql}
        return SimpleNamespace(
            name=name,
            rows=len(df),
            columns=len(df.columns),
            column_info=[{"name": c, "dtype": str(df[c].dtype)} for c in df.columns],
        )


class FakeDbManager:
    """打桩 DatabaseManager：记录 execute_sql() 调用。"""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.calls: list[tuple[str, str]] = []

    def execute_sql(self, datasource: str, sql: str) -> pd.DataFrame:
        self.calls.append((datasource, sql))
        return self.df


def test_load_dataframe_executes_and_stores():
    """load_dataframe 把查询结果存入 store，返回 rows/columns。"""
    df = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
    db_manager = FakeDbManager(df)
    store = FakeStore()

    result = load_dataframe(store, db_manager, "trino", "result", "SELECT * FROM t")

    assert result["success"] is True
    assert result["name"] == "result"
    assert result["rows"] == 2
    assert result["columns"] == 2
    assert db_manager.calls == [("trino", "SELECT * FROM t")]
    assert store.stored["datasource"] == "trino"
    assert store.stored["sql"] == "SELECT * FROM t"


def test_load_dataframe_reports_sql_error():
    """查询执行失败时返回 success=False 与错误信息。"""

    class FailingDbManager:
        def execute_sql(self, datasource: str, sql: str) -> pd.DataFrame:
            raise RuntimeError("connection refused")

    result = load_dataframe(
        FakeStore(), FailingDbManager(), "postgres", "result", "SELECT 1"
    )
    assert result["success"] is False
    assert "connection refused" in result["error"]
