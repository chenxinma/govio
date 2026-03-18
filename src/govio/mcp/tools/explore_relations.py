"""explore_relations 工具"""

from typing import Any

from govio.mcp.core.dataframe_store import DataFrameStore
from govio.mcp.core.explorer import RelationExplorer


def explore_relations(
    store: DataFrameStore, dataframes: list[str] | None = None
) -> dict[str, Any]:
    """探查 DataFrame 之间的关系

    Args:
        store: DataFrame 存储
        dataframes: DataFrame 名称列表，为空则探查全部

    Returns:
        关系列表
    """
    if dataframes is None:
        infos = store.list()
        dataframes = [info.name for info in infos]

    df_dict = {}
    for name in dataframes:
        df = store.get(name)
        if df is None:
            return {"success": False, "error": f"DataFrame '{name}' 不存在"}
        df_dict[name] = df

    explorer = RelationExplorer()
    relations = explorer.explore(df_dict)

    return {"success": True, "relations": relations}
