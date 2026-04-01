import pytest
from pathlib import Path
import tempfile
import json

from govio.mcp.config import DataSourceConfig, load_config


def test_datasource_config_url():
    config = DataSourceConfig(url="trino://user:pass@host:8080/db")
    assert config.url == "trino://user:pass@host:8080/db"
    assert config.connect_args == {}


def test_datasource_config_with_connect_args():
    config = DataSourceConfig(
        url="trino://user:pass@host:8080/db", connect_args={"http_scheme": "https"}
    )
    assert config.url == "trino://user:pass@host:8080/db"
    assert config.connect_args == {"http_scheme": "https"}


def test_load_config():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(
            {
                "datasources": {
                    "ds1": {
                        "url": "trino://user:pass@host:8080/db",
                        "connect_args": {"http_scheme": "https"},
                    },
                    "ds2": {"url": "duckdb:///data/warehouse"},
                }
            },
            f,
        )
        f.flush()

        config = load_config(Path(f.name))

        assert "ds1" in config.datasources
        assert "ds2" in config.datasources
        assert config.datasources["ds1"].url == "trino://user:pass@host:8080/db"
        assert config.datasources["ds1"].connect_args == {"http_scheme": "https"}
        assert config.datasources["ds2"].url == "duckdb:///data/warehouse"
        assert config.datasources["ds2"].connect_args == {}


def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config(Path("/nonexistent/config.json"))
