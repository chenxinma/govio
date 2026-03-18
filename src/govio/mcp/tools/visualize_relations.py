"""visualize_relations 工具"""

from typing import Any

from govio.mcp.core.visualizer import RelationVisualizer


def visualize_relations(relations: list[dict[str, Any]]) -> dict[str, Any]:
    """生成关系图谱

    Args:
        relations: 关系列表

    Returns:
        可视化数据
    """
    visualizer = RelationVisualizer()
    result = visualizer.visualize(relations)

    return {"success": True, **result}
