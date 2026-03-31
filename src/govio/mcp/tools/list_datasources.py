"""list_datasources 工具"""

from typing import Any

from govio.mcp.core.database import DatabaseManager


def list_datasources(db_manager: DatabaseManager) -> dict[str, Any]:
    """列出可用的数据源

    Args:
        db_manager: 数据库管理器

    Returns:
        数据源清单
    """
    datasources = []
    for name in db_manager._datasources.keys():
        config = db_manager._datasources[name]
        url = config.get_url()
        driver = url.split(":")[0] if ":" in url else "unknown"
        datasources.append(
            {
                "name": name,
                "driver": driver,
                "url": url,
            }
        )

    return {"success": True, "datasources": datasources}
