"""release_dataframe 工具"""

from typing import Any

from govio.observe_data.core.observe_store import ObserveStore


def release_dataframe(store: ObserveStore, name: str) -> dict[str, Any]:
    """释放 DataFrame

    Args:
        store: DataFrame 存储
        name: DataFrame 名称

    Returns:
        释放结果
    """
    if store.get(name) is None:
        return {"success": False, "error": f"DataFrame '{name}' 不存在"}

    store.release(name)

    return {"success": True, "message": f"DataFrame '{name}' 已释放"}


def release_all_dataframes(store: ObserveStore) -> dict[str, Any]:
    """释放所有 DataFrame

    Args:
        store: DataFrame 存储

    Returns:
        释放结果，包含已释放的 DataFrame 名称列表
    """
    dataframes = store.list()
    if not dataframes:
        return {"success": True, "released": [], "count": 0}

    released = []
    for df in dataframes:
        store.release(df.name)
        released.append(df.name)

    return {"success": True, "released": released, "count": len(released)}
