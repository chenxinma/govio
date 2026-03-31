"""测试数据库管理器"""

import pytest

from govio.mcp.config import DataSourceConfig
from govio.mcp.core.database import DatabaseManager


def test_database_manager_with_duckdb_no_file_path():
    datasources = {
        "testdb": DataSourceConfig(
            url="duckdb:///:memory:",
            connect_args={},
            file_path=None,
        )
    }

    manager = DatabaseManager(datasources)
    assert manager.get_duckdb_conn("testdb") is not None


def test_database_manager_execute_sql_duckdb():
    datasources = {
        "testdb": DataSourceConfig(
            url="duckdb:///:memory:",
            connect_args={},
            file_path=None,
        )
    }

    manager = DatabaseManager(datasources)
    df = manager.execute_sql("testdb", "SELECT 1 AS a, 2 AS b")
    assert len(df) == 1
    assert list(df.columns) == ["a", "b"]
