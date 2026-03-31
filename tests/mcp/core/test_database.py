import tempfile

import pytest

from govio.mcp.config import DataSourceConfig
from govio.mcp.core.database import DatabaseManager


def test_database_manager_sqlalchemy():
    configs = {"sqlite_db": DataSourceConfig(url="sqlite:///:memory:")}
    manager = DatabaseManager(configs)

    engine = manager.get_engine("sqlite_db")
    assert engine is not None


def test_database_manager_duckdb():
    with tempfile.TemporaryDirectory() as tmpdir:
        configs = {"local_data": DataSourceConfig(url=f"duckdb://{tmpdir}")}
        manager = DatabaseManager(configs)

        df = manager.execute_sql("local_data", "SELECT 1 as a")
        assert df is not None
        assert list(df.columns) == ["a"]


def test_database_manager_duckdb_invalid_dir():
    configs = {"invalid": DataSourceConfig(url="duckdb:///nonexistent/path")}

    with pytest.raises(RuntimeError, match="初始化数据源"):
        DatabaseManager(configs)


def test_database_manager_unknown_datasource():
    configs = {"sqlite_db": DataSourceConfig(url="sqlite:///:memory:")}
    manager = DatabaseManager(configs)

    with pytest.raises(ValueError, match="数据源不存在"):
        manager.get_engine("unknown")
