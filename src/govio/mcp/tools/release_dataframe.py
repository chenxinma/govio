"""release_dataframe 工具"""

from typing import Any

from govio.cli.observe_store import ObserveStore


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
