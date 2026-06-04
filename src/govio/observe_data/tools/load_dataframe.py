"""load_dataframe 工具"""

from typing import Any

import duckdb

from ..core.observe_store import ObserveStore
from ..core.database import DatabaseManager


def load_dataframe(
    store: ObserveStore,
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
        info = store.store(name, df, datasource, sql)

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


def load_from_memory(
    store: ObserveStore,
    name: str,
    sql: str,
) -> dict[str, Any]:
    """从已加载的 DataFrame 中查询

    将 ObserveStore 中所有注册的 DataFrame 加载进 DuckDB 内存，
    按名称注册为表后执行 SQL 查询。

    Args:
        store: DataFrame 存储
        name: 新 DataFrame 名称
        sql: 查询 SQL

    Returns:
        加载结果
    """
    try:
        infos = store.list()
        if not infos:
            return {"success": False, "error": "没有已加载的 DataFrame"}

        df_dict: dict[str, Any] = {}
        for info in infos:
            df = store.get(info.name)
            if df is not None:
                df_dict[info.name] = df

        if not df_dict:
            return {"success": False, "error": "没有可用的 DataFrame 文件"}

        conn = duckdb.connect(":memory:")
        for tbl_name, tbl_df in df_dict.items():
            conn.register(tbl_name, tbl_df)
        result_df = conn.execute(sql).df()

        result_info = store.store(name, result_df, "memory", sql)

        return {
            "success": True,
            "name": result_info.name,
            "rows": result_info.rows,
            "columns": result_info.columns,
            "column_info": result_info.column_info,
            "source_tables": list(df_dict.keys()),
        }
    except Exception as e:
        return {"success": False, "error": f"SQL 执行错误: {str(e)}"}
