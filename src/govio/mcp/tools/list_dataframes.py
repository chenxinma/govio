"""list_dataframes 工具"""

from typing import Any

from govio.mcp.core.dataframe_store import DataFrameStore


def list_dataframes(store: DataFrameStore) -> dict[str, Any]:
    """列出已加载的 DataFrame

    Args:
        store: DataFrame 存储

    Returns:
        DataFrame 清单
    """
    infos = store.list()

    return {
        "dataframes": [
            {
                "name": info.name,
                "rows": info.rows,
                "columns": info.columns,
                "column_info": info.column_info,
            }
            for info in infos
        ]
    }
