"""关系可视化"""

from typing import Any

import networkx as nx


class RelationVisualizer:
    """关系可视化器"""

    def to_networkx(self, relations: list[dict[str, Any]]) -> nx.DiGraph:
        """转换为 NetworkX 图"""
        graph = nx.DiGraph()

        for rel in relations:
            source = rel["source_table"]
            target = rel["target_table"]

            if not graph.has_node(source):
                graph.add_node(source, type="table")
            if not graph.has_node(target):
                graph.add_node(target, type="table")

            graph.add_edge(
                source,
                target,
                source_column=rel.get("source_column", ""),
                target_column=rel.get("target_column", ""),
                confidence=rel.get("confidence", 0),
            )

        return graph

    def to_json(self, relations: list[dict[str, Any]]) -> dict[str, Any]:
        """转换为 JSON 格式"""
        graph = self.to_networkx(relations)

        nodes = []
        for node, data in graph.nodes(data=True):
            nodes.append({"id": node, "label": node, **data})

        edges = []
        for source, target, data in graph.edges(data=True):
            edges.append({"source": source, "target": target, **data})

        return {"nodes": nodes, "edges": edges}

    def visualize(self, relations: list[dict[str, Any]]) -> dict[str, Any]:
        """生成可视化数据"""
        return self.to_json(relations)
