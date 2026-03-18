"""compare_tables 工具"""

from typing import Any

from govio.mcp.core.comparator import TableComparator
from govio.mcp.core.dataframe_store import DataFrameStore


def compare_tables(
    store: DataFrameStore, source_df: str, target_df: str, join_columns: list[str]
) -> dict[str, Any]:
    """比对两个 DataFrame

    Args:
        store: DataFrame 存储
        source_df: 源 DataFrame 名称
        target_df: 目标 DataFrame 名称
        join_columns: 用于比对的列

    Returns:
        比对结果
    """
    source = store.get(source_df)
    if source is None:
        return {"success": False, "error": f"DataFrame '{source_df}' 不存在"}

    target = store.get(target_df)
    if target is None:
        return {"success": False, "error": f"DataFrame '{target_df}' 不存在"}

    comparator = TableComparator()
    result = comparator.compare(source, target, join_columns)

    return {"success": True, **result}
