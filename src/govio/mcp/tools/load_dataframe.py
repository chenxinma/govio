"""load_dataframe 工具"""

from typing import Any

from govio.mcp.core.database import DatabaseManager
from govio.mcp.core.dataframe_store import DataFrameStore


def load_dataframe(
    store: DataFrameStore,
    db_manager: DatabaseManager,
    datasource: str,
    name: str,
    sql: str,
) -> dict[str, Any]:
    """加载 DataFrame 到内存

    Args:
        store: DataFrame 存储
        db_manager: 数据库管理器
        datasource: 数据源名称
        name: DataFrame 名称
        sql: 查询 SQL

    Returns:
        加载结果
    """
    try:
        df = db_manager.execute_sql(datasource, sql)
        info = store.store(name, df)

        return {
            "success": True,
            "name": info.name,
            "rows": info.rows,
            "columns": info.columns,
            "column_info": info.column_info,
        }
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": f"SQL 执行错误: {str(e)}"}
