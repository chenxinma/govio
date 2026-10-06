import tempfile
from pathlib import Path

import pytest

from govio.cli.config import ConfigManager


def test_config_manager_save_and_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        manager = ConfigManager(config_path)

        # 新格式配置不会被迁移
        config = {"graph": {"backend": "networkx", "networkx": {"gml_path": "test.gml"}}}

        manager.save(config)
        assert config_path.exists()

        loaded = manager.load()
        assert loaded == config


def test_config_manager_exists():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        manager = ConfigManager(config_path)

        assert not manager.exists()

        config = {"backend": "networkx", "networkx": {"gml_path": "test.gml"}}
        manager.save(config)

        assert manager.exists()


def test_config_manager_validate_networkx():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        manager = ConfigManager(config_path)

        valid_config = {"graph": {"backend": "networkx", "networkx": {"gml_path": "test.gml"}}}

        assert manager.validate(valid_config) is True

        invalid_config = {"graph": {"backend": "networkx", "networkx": {}}}

        with pytest.raises(ValueError):
            manager.validate(invalid_config)


def test_config_manager_validate_falkordb():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        manager = ConfigManager(config_path)

        valid_config = {
            "graph": {
                "backend": "falkordb",
                "falkordb": {"host": "localhost", "port": 6379, "graph": "ontology"},
            },
        }

        assert manager.validate(valid_config) is True

        invalid_config = {"graph": {"backend": "falkordb", "falkordb": {"host": "localhost"}}}

        with pytest.raises(ValueError):
            manager.validate(invalid_config)


def test_config_manager_load_missing_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "nonexistent.yaml"
        manager = ConfigManager(config_path)

        with pytest.raises(FileNotFoundError):
            manager.load()


def test_config_manager_validate_unsupported_backend():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        manager = ConfigManager(config_path)

        invalid_config = {"graph": {"backend": "unsupported"}}

        with pytest.raises(ValueError, match="不支持的 backend"):
            manager.validate(invalid_config)


def test_config_manager_validate_missing_backend():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        manager = ConfigManager(config_path)

        invalid_config = {}

        with pytest.raises(ValueError, match="配置缺少 'graph' 字段"):
            manager.validate(invalid_config)





def test_new_config_not_migrated():
    """新格式配置不应被迁移"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        manager = ConfigManager(config_path)

        new_config = {
            "metadata": {"kundb": "mysql://host/db", "csv_dir": "/tmp/csv"},
            "graph": {"backend": "networkx", "networkx": {"gml_path": "test.gml"}},
        }
        manager.save(new_config)

        loaded = manager.load()
        assert loaded == new_config


def test_validate_new_structure():
    """验证新嵌套结构"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        manager = ConfigManager(config_path)

        valid_config = {
            "graph": {"backend": "networkx", "networkx": {"gml_path": "test.gml"}},
        }
        assert manager.validate(valid_config) is True

        invalid_config = {
            "graph": {"backend": "networkx"},  # 缺少 networkx 配置
        }
        with pytest.raises(ValueError):
            manager.validate(invalid_config)
