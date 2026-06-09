import shutil
import yaml
from pathlib import Path
from typing import Any

# 旧格式中属于 metadata section 的字段
_METADATA_KEYS = {"kundb", "workspace_uuid", "app_list", "app_map", "relationship", "metric", "csv_dir"}
# 旧格式中属于 graph section 的字段
_GRAPH_KEYS = {"backend", "networkx", "falkordb"}


class ConfigManager:
    """管理 govio 配置文件"""

    def __init__(self, config_path: Path | None = None) -> None:
        if config_path is None:
            self.config_path = Path.home() / ".govio" / "config.yaml"
        else:
            self.config_path = config_path

        self.config_path.parent.mkdir(parents=True, exist_ok=True)

    def exists(self) -> bool:
        """检查配置文件是否存在"""
        return self.config_path.exists()

    def load(self) -> dict[str, Any]:
        """加载配置文件，自动迁移旧格式"""
        if not self.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        if self._is_old_format(config):
            config = self._migrate(config)

        return config

    def save(self, config: dict[str, Any]) -> None:
        """保存配置文件"""
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

    def _is_old_format(self, config: dict[str, Any]) -> bool:
        """检测是否为旧的扁平格式"""
        return "kundb" in config or ("backend" in config and "graph" not in config)

    def _migrate(self, config: dict[str, Any]) -> dict[str, Any]:
        """将旧扁平格式迁移为新的嵌套格式"""
        backup_path = self.config_path.with_suffix(".yaml.bak")
        shutil.copy2(self.config_path, backup_path)

        new_config: dict[str, Any] = {}

        metadata = {}
        for key in _METADATA_KEYS:
            if key in config:
                metadata[key] = config[key]
        if metadata:
            new_config["metadata"] = metadata

        graph = {}
        for key in _GRAPH_KEYS:
            if key in config:
                graph[key] = config[key]
        if graph:
            new_config["graph"] = graph

        if "datasources" in config:
            new_config["datasources"] = config["datasources"]

        self.save(new_config)

        return new_config

    def validate(self, config: dict[str, Any]) -> bool:
        """验证配置的有效性

        支持新格式（嵌套）和旧格式（扁平）的验证。
        """
        if "graph" in config:
            graph = config["graph"]
            if "backend" not in graph:
                raise ValueError("配置缺少 'graph.backend' 字段")
            backend = graph["backend"]
            if backend not in ["networkx", "falkordb"]:
                raise ValueError(f"不支持的 backend: {backend}")
            if backend == "networkx":
                if "networkx" not in graph:
                    raise ValueError("NetworkX backend 需要 'networkx' 配置")
                if "gml_path" not in graph["networkx"]:
                    raise ValueError("NetworkX 配置缺少 'gml_path' 字段")
            elif backend == "falkordb":
                if "falkordb" not in graph:
                    raise ValueError("FalkorDB backend 需要 'falkordb' 配置")
                for field in ["host", "port", "graph"]:
                    if field not in graph["falkordb"]:
                        raise ValueError(f"FalkorDB 配置缺少 '{field}' 字段")
        elif "backend" in config:
            backend = config["backend"]
            if backend not in ["networkx", "falkordb"]:
                raise ValueError(f"不支持的 backend: {backend}")
            if backend == "networkx":
                if "networkx" not in config:
                    raise ValueError("NetworkX backend 需要 'networkx' 配置")
                if "gml_path" not in config["networkx"]:
                    raise ValueError("NetworkX 配置缺少 'gml_path' 字段")
            elif backend == "falkordb":
                if "falkordb" not in config:
                    raise ValueError("FalkorDB backend 需要 'falkordb' 配置")
                for field in ["host", "port", "graph"]:
                    if field not in config["falkordb"]:
                        raise ValueError(f"FalkorDB 配置缺少 '{field}' 字段")
        else:
            raise ValueError("配置缺少 'backend' 字段")

        csv_dir = config.get("metadata", {}).get("csv_dir") or config.get("csv_dir")
        if csv_dir:
            csv_path = Path(csv_dir)
            if not csv_path.exists():
                raise ValueError(f"csv_dir 不存在: {csv_path}")

        if "graph_dir" in config:
            graph_path = Path(config["graph_dir"])
            if not graph_path.exists():
                raise ValueError(f"graph_dir 不存在: {graph_path}")

        datasources = config.get("datasources")
        if datasources:
            if not isinstance(datasources, dict):
                raise ValueError("datasources 必须为字典类型")
            for name, ds_data in datasources.items():
                if not isinstance(ds_data, dict):
                    raise ValueError(f"数据源 '{name}' 配置必须为字典类型")
                if "url" not in ds_data:
                    raise ValueError(f"数据源 '{name}' 缺少 'url' 字段")

        return True
