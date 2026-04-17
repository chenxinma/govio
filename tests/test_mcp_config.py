"""测试数据源配置"""

import json
import tempfile
from pathlib import Path

import pytest

from govio.observe_data.config import DataSourceConfig, load_config


def test_load_config_success():
    config_data = {"datasources": {"testdb": {"url": "sqlite:///:memory:"}}}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config_data, f)
        f.flush()
        config = load_config(Path(f.name))

    assert "testdb" in config.datasources
    assert config.datasources["testdb"].url == "sqlite:///:memory:"


def test_load_config_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_config(Path("/nonexistent/config.json"))


def test_datasource_config_with_url():
    config = DataSourceConfig(url="mysql+pymysql://user:pass@localhost:3306/testdb")
    assert config.url == "mysql+pymysql://user:pass@localhost:3306/testdb"
    assert config.connect_args == {}


def test_datasource_config_with_connect_args():
    config = DataSourceConfig(
        url="trino://user:pass@host:8080/db", connect_args={"http_scheme": "https"}
    )
    assert config.url == "trino://user:pass@host:8080/db"
    assert config.connect_args == {"http_scheme": "https"}
