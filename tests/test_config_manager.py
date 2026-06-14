import pytest
from pathlib import Path
import tempfile
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

        valid_config = {"backend": "networkx", "networkx": {"gml_path": "test.gml"}}

        assert manager.validate(valid_config) is True

        invalid_config = {"backend": "networkx", "networkx": {}}

        with pytest.raises(ValueError):
            manager.validate(invalid_config)


def test_config_manager_validate_falkordb():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        manager = ConfigManager(config_path)

        valid_config = {
            "backend": "falkordb",
            "falkordb": {"host": "localhost", "port": 6379, "graph": "ontology"},
        }

        assert manager.validate(valid_config) is True

        invalid_config = {"backend": "falkordb", "falkordb": {"host": "localhost"}}

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

        invalid_config = {"backend": "unsupported"}

        with pytest.raises(ValueError, match="不支持的 backend"):
            manager.validate(invalid_config)


def test_config_manager_validate_missing_backend():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        manager = ConfigManager(config_path)

        invalid_config = {}

        with pytest.raises(ValueError, match="配置缺少 'backend' 字段"):
            manager.validate(invalid_config)


def test_migrate_old_flat_config():
    """旧扁平格式应自动迁移为嵌套结构"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        manager = ConfigManager(config_path)

        # 写入旧格式
        old_config = {
            "kundb": "mysql+pymysql://user:pass@host/db",
            "workspace_uuid": "test-uuid",
            "app_list": "app.xlsx",
            "app_map": "app.json",
            "relationship": "rel.json",
            "metric": "metric.json",
            "csv_dir": "/tmp/csv",
            "backend": "networkx",
            "networkx": {"gml_path": "test.gml"},
            "datasources": {"ds1": {"url": "mysql://localhost/db"}},
        }
        import yaml
        with open(config_path, "w") as f:
            yaml.dump(old_config, f)

        # 加载应自动迁移
        loaded = manager.load()

        # 验证新结构
        assert "metadata" in loaded
        assert loaded["metadata"]["kundb"] == "mysql+pymysql://user:pass@host/db"
        assert loaded["metadata"]["workspace_uuid"] == "test-uuid"
        assert loaded["metadata"]["csv_dir"] == "/tmp/csv"

        assert "graph" in loaded
        assert loaded["graph"]["backend"] == "networkx"
        assert loaded["graph"]["networkx"]["gml_path"] == "test.gml"

        assert "datasources" in loaded
        assert loaded["datasources"]["ds1"]["url"] == "mysql://localhost/db"

        # 旧字段不应存在
        assert "kundb" not in loaded
        assert "backend" not in loaded

        # 备份文件应存在
        assert (Path(tmpdir) / "config.yaml.bak").exists()


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
