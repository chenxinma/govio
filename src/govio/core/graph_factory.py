from typing import Any

from ..graph import FalkorDBGraph, NetworkXGraph


class GraphFactory:
    """图对象工厂，根据配置创建不同的图对象"""

    @staticmethod
    def create(config: dict[str, Any]) -> NetworkXGraph | FalkorDBGraph:
        """根据配置创建图对象

        Args:
            config: 配置字典

        Returns:
            NetworkXGraph 或 FalkorDBGraph 实例

        Raises:
            ValueError: 配置缺少必需字段或不支持的 backend 类型
            FileNotFoundError: GML 文件不存在 (NetworkX)
        """
        if "backend" not in config:
            raise ValueError("配置缺少 'backend' 字段")

        backend = config["backend"]

        if backend == "networkx":
            if "networkx" not in config:
                raise ValueError("NetworkX backend 需要 'networkx' 配置")
            if "gml_path" not in config["networkx"]:
                raise ValueError("NetworkX 配置缺少 'gml_path' 字段")

            gml_path = config["networkx"]["gml_path"]
            return NetworkXGraph(gml_path)

        elif backend == "falkordb":
            if "falkordb" not in config:
                raise ValueError("FalkorDB backend 需要 'falkordb' 配置")

            falkordb_config = config["falkordb"]
            required_fields = ["host", "port", "graph"]
            for field in required_fields:
                if field not in falkordb_config:
                    raise ValueError(f"FalkorDB 配置缺少 '{field}' 字段")

            return FalkorDBGraph(
                graph=falkordb_config["graph"],
                host=falkordb_config.get("host", "localhost"),
                port=falkordb_config.get("port", 6379),
            )

        else:
            raise ValueError(f"不支持的 backend: {backend}")
