"""测试数据源配置"""

import json
import tempfile
from pathlib import Path

import pytest

from govio.mcp.config import DataSourceConfig, load_config


def test_load_config_success():
    config_data = {
        "datasources": {"testdb": {"driver": "sqlite", "database": ":memory:"}}
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config_data, f)
        f.flush()
        config = load_config(Path(f.name))

    assert "testdb" in config.datasources
    assert config.datasources["testdb"].driver == "sqlite"


def test_load_config_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_config(Path("/nonexistent/config.json"))


def test_datasource_config_to_url():
    config = DataSourceConfig(
        driver="mysql+pymysql",
        host="localhost",
        port=3306,
        database="testdb",
        username="user",
        password="pass",
    )

    url = config.to_url()
    assert url == "mysql+pymysql://user:pass@localhost:3306/testdb"


def test_datasource_config_to_url_sqlite():
    config = DataSourceConfig(driver="sqlite", database="/path/to/db.sqlite")

    url = config.to_url()
    assert url == "sqlite:////path/to/db.sqlite"
