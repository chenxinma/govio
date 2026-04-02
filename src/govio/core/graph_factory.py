from typing import Any
from govio import NetworkXGraph, FalkorDBGraph


class GraphFactory:
    """图对象工厂，根据配置创建不同的图对象"""

    @staticmethod
    def create(config: dict[str, Any]):
        """根据配置创建图对象

        Args:
            config: 配置字典

        Returns:
            NetworkXGraph 或 FalkorDBGraph 实例

        Raises:
            ValueError: 不支持的 backend 类型
        """
        backend = config.get("backend")

        if backend == "networkx":
            gml_path = config["networkx"]["gml_path"]
            return NetworkXGraph(gml_path)

        elif backend == "falkordb":
            falkordb_config = config["falkordb"]
            return FalkorDBGraph(
                graph=falkordb_config["graph"],
                host=falkordb_config.get("host", "localhost"),
                port=falkordb_config.get("port", 6379),
            )

        else:
            raise ValueError(f"不支持的 backend: {backend}")
