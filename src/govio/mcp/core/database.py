"""数据库连接管理"""

from pathlib import Path

import duckdb
import pandas as pd
from duckdb import DuckDBPyConnection
from sqlalchemy import Engine, create_engine, text

from govio.mcp.config import DataSourceConfig


class DatabaseManager:
    """数据库连接管理器"""

    def __init__(self, datasources: dict[str, DataSourceConfig]) -> None:
        self._datasources = datasources
        self._engines: dict[str, Engine] = {}
        self._duckdb_conns: dict[str, DuckDBPyConnection] = {}
        self._duckdb_dirs: dict[str, str] = {}
        self._init_connections()

    def _init_connections(self) -> None:
        """初始化所有数据源连接"""
        for name, config in self._datasources.items():
            try:
                if config.url.startswith("duckdb://"):
                    dir_path = config.url[9:]
                    if not Path(dir_path).exists():
                        raise ValueError(f"目录不存在: {dir_path}")
                    conn = duckdb.connect(":memory:")
                    self._duckdb_conns[name] = conn
                    self._duckdb_dirs[name] = dir_path
                else:
                    self._engines[name] = create_engine(
                        config.url, connect_args=config.connect_args
                    )
            except Exception as e:
                raise RuntimeError(f"初始化数据源 '{name}' 失败: {e}") from e

    def get_engine(self, datasource: str) -> Engine:
        """获取数据源引擎（仅用于 SQLAlchemy 数据源）"""
        if datasource not in self._engines:
            raise ValueError(f"数据源不存在或不是 SQLAlchemy 类型: {datasource}")
        return self._engines[datasource]

    def execute_sql(self, datasource: str, sql: str) -> pd.DataFrame:
        """执行 SQL 并返回 DataFrame"""
        if datasource in self._duckdb_conns:
            conn = self._duckdb_conns[datasource]
            return conn.execute(sql).df()
        elif datasource in self._engines:
            engine = self._engines[datasource]
            with engine.connect() as conn:
                return pd.read_sql(text(sql), conn)
        else:
            raise ValueError(f"数据源不存在: {datasource}")
