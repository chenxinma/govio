"""数据源配置加载"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DataSourceConfig:
    """数据源配置"""

    url: str
    connect_args: dict[str, Any] = field(default_factory=dict)


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
        )

    return Config(datasources=datasources)
