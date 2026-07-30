"""数据源配置加载"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from govio.crypto import decrypt_value, reconstruct_url


@dataclass
class DataSourceConfig:
    """数据源配置

    Attributes:
        url: 连接 URL（密码已脱敏为 ***，或无密码）
        connect_args: 额外连接参数
        encrypted_password: Fernet 加密的密码（若有）
    """

    url: str
    connect_args: dict[str, Any] = field(default_factory=dict)
    encrypted_password: str | None = None

    @property
    def resolved_url(self) -> str:
        """返回含真实密码的完整 URL"""
        if self.encrypted_password:
            password = decrypt_value(self.encrypted_password)
            return reconstruct_url(self.url, password)
        return self.url


@dataclass
class Config:
    """配置"""

    datasources: dict[str, DataSourceConfig]


def load_config(path: Path) -> Config:
    """加载配置文件"""
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    datasources = {}
    for name, ds_data in data.get("datasources", {}).items():
        datasources[name] = DataSourceConfig(
            url=ds_data.get("url", ""),
            connect_args=ds_data.get("connect_args", {}),
            encrypted_password=ds_data.get("encrypted_password"),
        )

    return Config(datasources=datasources)
