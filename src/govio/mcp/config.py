"""数据源配置加载"""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataSourceConfig:
    """数据源配置"""

    driver: str
    host: str = ""
    port: int = 0
    database: str = ""
    username: str = ""
    password: str = ""

    def to_url(self) -> str:
        """转换为 SQLAlchemy 连接 URL"""
        if self.driver.startswith("sqlite"):
            return f"{self.driver}:///{self.database}"

        auth = ""
        if self.username:
            auth = self.username
            if self.password:
                auth += f":{self.password}"
            auth += "@"

        port_str = f":{self.port}" if self.port else ""
        return f"{self.driver}://{auth}{self.host}{port_str}/{self.database}"


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
            driver=ds_data.get("driver", ""),
            host=ds_data.get("host", ""),
            port=ds_data.get("port", 0),
            database=ds_data.get("database", ""),
            username=ds_data.get("username", ""),
            password=ds_data.get("password", ""),
        )

    return Config(datasources=datasources)
