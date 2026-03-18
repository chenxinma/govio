"""数据库连接管理"""

import pandas as pd
from sqlalchemy import Engine, create_engine, text

from govio.mcp.config import DataSourceConfig


class DatabaseManager:
    """数据库连接管理器"""

    def __init__(self, datasources: dict[str, DataSourceConfig]) -> None:
        self._datasources = datasources
        self._engines: dict[str, Engine] = {}
        self._init_engines()

    def _init_engines(self) -> None:
        """初始化所有数据源连接"""
        for name, config in self._datasources.items():
            url = config.to_url()
            self._engines[name] = create_engine(url)

    def get_engine(self, datasource: str) -> Engine:
        """获取数据源引擎"""
        if datasource not in self._engines:
            raise ValueError(f"数据源不存在: {datasource}")
        return self._engines[datasource]

    def execute_sql(self, datasource: str, sql: str) -> pd.DataFrame:
        """执行 SQL 并返回 DataFrame"""
        engine = self.get_engine(datasource)
        with engine.connect() as conn:
            return pd.read_sql(text(sql), conn)
