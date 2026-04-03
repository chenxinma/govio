"""MCP 服务器入口"""

import argparse
import asyncio
import logging
from pathlib import Path

from mcp.server import FastMCP

from govio.mcp.config import Config, load_config
from govio.mcp.core.database import DatabaseManager
from govio.mcp.core.dataframe_store import DataFrameStore
from govio.mcp.tools.compare_tables import compare_tables
from govio.mcp.tools.explore_relations import explore_relations
from govio.mcp.tools.list_dataframes import list_dataframes
from govio.mcp.tools.load_dataframe import load_dataframe
from govio.mcp.tools.release_dataframe import release_dataframe
from govio.mcp.tools.visualize_relations import visualize_relations

logger = logging.getLogger(__name__)

_store = DataFrameStore()
_db_manager: DatabaseManager | None = None
_config: Config | None = None

mcp = FastMCP("govio-data-exploration")


def init_server(config: Config) -> None:
    """初始化服务器"""
    global _db_manager, _config
    _config = config
    _db_manager = DatabaseManager(config.datasources)


@mcp.tool()
def show_datasource() -> str:
    """显示已配置的数据源及数据库协议

    Returns:
        数据源配置列表
    """
    if _config is None:
        return "错误: 配置未初始化"

    if not _config.datasources:
        return "暂无配置的数据源"

    lines = ["已配置的数据源:"]
    for name, ds_config in _config.datasources.items():
        protocol = (
            ds_config.url.split("://")[0] if "://" in ds_config.url else "unknown"
        )
        lines.append(f"  - {name}: {protocol} ")

    return "\n".join(lines)


@mcp.tool()
def load_df(datasource: str, name: str, sql: str) -> str:
    """执行 SQL 并将结果存入内存 DataFrame

    Args:
        datasource: 数据源名称
        name: DataFrame 名称
        sql: 查询 SQL

    Returns:
        加载结果
    """
    if _db_manager is None:
        return "错误: 数据库管理器未初始化"
    result = load_dataframe(
        store=_store, db_manager=_db_manager, datasource=datasource, name=name, sql=sql
    )
    return str(result)


@mcp.tool()
def list_dfs() -> str:
    """列出已加载的 DataFrame 清单

    Returns:
        DataFrame 清单
    """
    result = list_dataframes(store=_store)
    return str(result)


@mcp.tool()
def release_df(name: str) -> str:
    """释放 DataFrame

    Args:
        name: DataFrame 名称

    Returns:
        释放结果
    """
    result = release_dataframe(store=_store, name=name)
    return str(result)


@mcp.tool()
def compare_dfs(source_df: str, target_df: str, join_columns: list[str]) -> str:
    """比对两个 DataFrame

    Args:
        source_df: 源 DataFrame 名称
        target_df: 目标 DataFrame 名称
        join_columns: 用于比对的列

    Returns:
        比对结果
    """
    result = compare_tables(
        store=_store,
        source_df=source_df,
        target_df=target_df,
        join_columns=join_columns,
    )
    return str(result)


@mcp.tool()
def explore_df_relations(dataframes: list[str] | None = None) -> str:
    """探查 DataFrame 之间的关系

    Args:
        dataframes: DataFrame 名称列表，为空则探查全部

    Returns:
        关系列表
    """
    result = explore_relations(store=_store, dataframes=dataframes)
    return str(result)


@mcp.tool()
def visualize_df_relations(relations: list[dict]) -> str:
    """生成关系图谱

    Args:
        relations: 关系列表

    Returns:
        可视化数据
    """
    result = visualize_relations(relations=relations)
    return str(result)


def create_server(config: Config) -> FastMCP:
    """创建 MCP 服务器"""
    init_server(config)
    return mcp


def main():
    """主入口"""
    parser = argparse.ArgumentParser(description="Govio MCP 数据探查服务")
    parser.add_argument(
        "--datasource-config", type=str, required=True, help="数据源配置文件路径"
    )
    parser.add_argument("--port", type=int, default=8000, help="服务端口")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务地址")

    args = parser.parse_args()

    config = load_config(Path(args.datasource_config))
    init_server(config)

    logging.basicConfig(level=logging.INFO)
    logger.info(f"启动 MCP 服务: http://{args.host}:{args.port}")

    mcp.settings.host = args.host
    mcp.settings.port = args.port
    asyncio.run(mcp.run_streamable_http_async())


if __name__ == "__main__":
    main()
