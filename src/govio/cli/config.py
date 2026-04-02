import yaml
from pathlib import Path
from typing import Any


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
        """加载配置文件"""
        if not self.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def save(self, config: dict[str, Any]) -> None:
        """保存配置文件"""
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

    def validate(self, config: dict[str, Any]) -> bool:
        """验证配置的有效性

        Args:
            config: 配置字典

        Returns:
            bool: 是否有效

        Raises:
            ValueError: 配置无效时抛出
        """
        if "backend" not in config:
            raise ValueError("配置缺少 'backend' 字段")

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
            required_fields = ["host", "port", "graph"]
            for field in required_fields:
                if field not in config["falkordb"]:
                    raise ValueError(f"FalkorDB 配置缺少 '{field}' 字段")

        if "csv_dir" in config:
            csv_path = Path(config["csv_dir"])
            if not csv_path.exists():
                raise ValueError(f"csv_dir 不存在: {csv_path}")

        if "graph_dir" in config:
            graph_path = Path(config["graph_dir"])
            if not graph_path.exists():
                raise ValueError(f"graph_dir 不存在: {graph_path}")

        return True
