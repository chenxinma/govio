"""数据库连接管理

按需连接：构造 DatabaseManager 不建立实际连接，首次执行 SQL / 取引擎时
才初始化对应数据源；duckdb / sqlalchemy / pandas 均按需导入。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from govio.observe_data.config import DataSourceConfig

if TYPE_CHECKING:
    import pandas as pd
    from duckdb import DuckDBPyConnection
    from sqlalchemy import Engine

# 需要归一的 SQLAlchemy scheme：无驱动限定的默认方言未随项目安装，
# trino 的 http/https 由 dbapi 的 http_scheme 控制（trino 方言只注册 trino 驱动，
# trino+http / trino+https 直接交给 create_engine 会触发 NoSuchModuleError）
_SQLALCHEMY_URL_ALIASES: dict[str, str] = {
    "mysql": "mysql+pymysql",
    "postgres": "postgresql+psycopg",
    "postgresql": "postgresql+psycopg",
}

# trino+http / trino+https 归一为 trino:// 并以 connect_args.http_scheme 指定协议
_TRINO_URL_HTTP_SCHEMES: dict[str, str] = {
    "trino+http": "http",
    "trino+https": "https",
}


def _resolve_sqlalchemy_target(url: str) -> tuple[str, dict[str, Any]]:
    """归一 SQLAlchemy 连接目标：返回 (URL, 追加的 connect_args)。

    mysql:// -> mysql+pymysql://（默认 mysqldb 驱动未安装）；
    postgresql:// / postgres:// -> postgresql+psycopg://（项目使用 psycopg3）；
    trino+http:// / trino+https:// -> trino:// + connect_args http_scheme
    （trino 方言只注册 trino 驱动）；
    其余 URL（含已限定驱动写法）原样返回。
    """
    scheme, sep, rest = url.partition("://")
    if not sep:
        return url, {}
    normalized = _SQLALCHEMY_URL_ALIASES.get(scheme.lower())
    if normalized:
        return f"{normalized}://{rest}", {}
    http_scheme = _TRINO_URL_HTTP_SCHEMES.get(scheme.lower())
    if http_scheme:
        return f"trino://{rest}", {"http_scheme": http_scheme}
    return url, {}


class DatabaseManager:
    """数据库连接管理器"""

    def __init__(self, datasources: dict[str, DataSourceConfig]) -> None:
        self._datasources = datasources
        self._engines: dict[str, Engine] = {}
        self._duckdb_conns: dict[str, DuckDBPyConnection] = {}
        self._duckdb_dirs: dict[str, str] = {}

    def _ensure(self, name: str) -> None:
        """按需初始化指定数据源连接（首次使用时才真正连接）"""
        if name not in self._datasources:
            return
        if name in self._engines or name in self._duckdb_conns:
            return

        import duckdb
        from sqlalchemy import create_engine

        config = self._datasources[name]
        try:
            # 使用 resolved_url 自动解密密码
            url = config.resolved_url
            if url.startswith("duckdb://"):
                db_path = url[9:]
                if not Path(db_path).exists():
                    raise ValueError(f"目录不存在: {db_path}")
                if Path(db_path).is_dir():
                    dir_path = db_path
                    conn = duckdb.connect(":memory:")
                    conn.execute(f"SET file_search_path = '{dir_path}'")
                    self._duckdb_conns[name] = conn
                    self._duckdb_dirs[name] = dir_path
                else:
                    conn = duckdb.connect(database=db_path, read_only=True)
                    self._duckdb_conns[name] = conn
                    self._duckdb_dirs[name] = str(Path(db_path).parent)
            else:
                sa_url, extra_connect_args = _resolve_sqlalchemy_target(url)
                self._engines[name] = create_engine(
                    sa_url,
                    connect_args={**extra_connect_args, **config.connect_args},
                )
        except Exception as e:
            raise RuntimeError(f"初始化数据源 '{name}' 失败: {e}") from e

    def get_engine(self, datasource: str) -> Engine:
        """获取数据源引擎（仅用于 SQLAlchemy 数据源）"""
        self._ensure(datasource)
        if datasource not in self._engines:
            raise ValueError(f"数据源不存在或不是 SQLAlchemy 类型: {datasource}")
        return self._engines[datasource]

    def execute_sql(self, datasource: str, sql: str) -> pd.DataFrame:
        """执行 SQL 并返回 DataFrame"""
        self._ensure(datasource)
        if datasource in self._duckdb_conns:
            conn = self._duckdb_conns[datasource]
            return conn.execute(sql).df()
        if datasource in self._engines:
            from sqlalchemy import text

            engine = self._engines[datasource]
            with engine.connect() as conn:
                import pandas as pd

                return pd.read_sql(text(sql), conn)
        else:
            raise ValueError(f"数据源不存在: {datasource}")
