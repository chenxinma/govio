"""测试数据库管理器"""

import tempfile


from govio.mcp.config import DataSourceConfig
from govio.mcp.core.database import DatabaseManager


def test_database_manager_with_duckdb():
    with tempfile.TemporaryDirectory() as tmpdir:
        datasources = {"testdb": DataSourceConfig(url=f"duckdb://{tmpdir}")}

        manager = DatabaseManager(datasources)
        result = manager.execute_sql("testdb", "SELECT 1 AS value")
        assert result is not None
        assert result["value"][0] == 1


def test_database_manager_execute_sql_duckdb():
    with tempfile.TemporaryDirectory() as tmpdir:
        datasources = {"testdb": DataSourceConfig(url=f"duckdb://{tmpdir}")}

        manager = DatabaseManager(datasources)
        df = manager.execute_sql("testdb", "SELECT 1 AS a, 2 AS b")
        assert len(df) == 1
        assert list(df.columns) == ["a", "b"]
