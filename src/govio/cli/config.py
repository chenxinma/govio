import shutil
import yaml
from pathlib import Path
from typing import Any

from govio.crypto import encrypt_value, parse_password_from_url


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
        """加载配置文件，自动迁移明文密码为加密存储"""
        if not self.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        if self._has_plaintext_passwords(config):
            config = self._migrate_passwords(config)

        return config

    def save(self, config: dict[str, Any]) -> None:
        """保存配置文件"""
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

    # ------------------------------------------------------------------
    # 密码加密迁移
    # ------------------------------------------------------------------

    @staticmethod
    def _has_plaintext_passwords(config: dict[str, Any]) -> bool:
        """检测数据源中是否存在明文密码"""
        datasources = config.get("datasources", {})
        for ds_data in datasources.values():
            if not isinstance(ds_data, dict):
                continue
            url = ds_data.get("url", "")
            _, password = parse_password_from_url(url)
            if password and not ds_data.get("encrypted_password"):
                return True
        return False

    def _migrate_passwords(self, config: dict[str, Any]) -> dict[str, Any]:
        """将数据源中的明文密码迁移为加密存储"""
        backup_path = self.config_path.with_suffix(".yaml.bak")
        shutil.copy2(self.config_path, backup_path)

        datasources = config.get("datasources", {})
        for name, ds_data in datasources.items():
            if not isinstance(ds_data, dict):
                continue
            url = ds_data.get("url", "")
            masked_url, password = parse_password_from_url(url)
            if password and not ds_data.get("encrypted_password"):
                ds_data["url"] = masked_url
                ds_data["encrypted_password"] = encrypt_value(password)

        self.save(config)
        return config

    @staticmethod
    def _validate_backend(scope: dict[str, Any]) -> None:
        """验证 backend 相关配置（networkx/falkordb/ladybug）"""
        if "backend" not in scope:
            raise ValueError("配置缺少 'backend' 字段")
        backend = scope["backend"]
        if backend not in ["networkx", "falkordb", "ladybug"]:
            raise ValueError(f"不支持的 backend: {backend}")
        if backend == "networkx":
            if "networkx" not in scope:
                raise ValueError("NetworkX backend 需要 'networkx' 配置")
            if "gml_path" not in scope["networkx"]:
                raise ValueError("NetworkX 配置缺少 'gml_path' 字段")
        elif backend == "falkordb":
            if "falkordb" not in scope:
                raise ValueError("FalkorDB backend 需要 'falkordb' 配置")
            for field in ["host", "port", "graph"]:
                if field not in scope["falkordb"]:
                    raise ValueError(f"FalkorDB 配置缺少 '{field}' 字段")
        elif backend == "ladybug":
            if "ladybug" not in scope:
                raise ValueError("Ladybug backend 需要 'ladybug' 配置")
            if "db_path" not in scope["ladybug"]:
                raise ValueError("Ladybug 配置缺少 'db_path' 字段")

    def validate(self, config: dict[str, Any]) -> bool:
        """验证配置的有效性"""
        if "graph" in config:
            self._validate_backend(config["graph"])
        else:
            raise ValueError("配置缺少 'graph' 字段")

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

