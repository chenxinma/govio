"""visualize_relations 工具"""

from typing import Any

from ..core.visualizer import RelationVisualizer


def visualize_relations(
    relations: list[dict[str, Any]] | dict[str, Any],
) -> dict[str, Any]:
    """生成关系图谱

    Args:
        relations: 关系列表，或包含 "relations" 键的字典

    Returns:
        可视化数据
    """
    # 处理包装格式
    if isinstance(relations, dict) and "relations" in relations:
        relations = relations["relations"]

    # 规范化列相似性关系到统一格式
    normalized = []
    for rel in relations:
        if rel.get("type") == "column_similarity":
            normalized.append(
                {
                    "source_table": rel["table1"],
                    "source_column": rel["column1"],
                    "target_table": rel["table2"],
                    "target_column": rel["column2"],
                    "confidence": rel.get("similarity", 0),
                }
            )
        else:
            normalized.append(rel)

    visualizer = RelationVisualizer()
    result = visualizer.visualize(normalized)

    return {"success": True, **result}
