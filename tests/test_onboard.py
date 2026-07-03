from pathlib import Path
import tempfile
from unittest.mock import patch, MagicMock

import pytest


def create_test_csv_files(csv_dir: Path):
    """创建测试用的 CSV 文件"""
    csv_dir.mkdir(parents=True, exist_ok=True)

    (csv_dir / "PhysicalTable.csv").write_text(
        """:ID(PhysicalTable),name,full_table_name
table1,表1,SCHEMA.TABLE1
""",
        encoding="utf-8",
    )

    (csv_dir / "Col.csv").write_text(
        """:ID(Col),name,column_name,full_table_name
col1,字段1,COL1,SCHEMA.TABLE1
""",
        encoding="utf-8",
    )

    (csv_dir / "Application.csv").write_text(
        """:ID(Application),name,app_name_en
app1,应用1,APP1
""",
        encoding="utf-8",
    )

    (csv_dir / "Standard.csv").write_text(
        """:ID(Standard),name
std1,标准1
""",
        encoding="utf-8",
    )

    (csv_dir / "HAS_COLUMN.csv").write_text(
        """:START_ID(PhysicalTable),:END_ID(Col)
table1,col1
""",
        encoding="utf-8",
    )

    (csv_dir / "USE.csv").write_text(
        """:START_ID(Application),:END_ID(PhysicalTable)
app1,table1
""",
        encoding="utf-8",
    )


def test_validate_csv_directory():
    from govio.cli.onboard import validate_csv_directory

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_dir = Path(tmpdir) / "csv"
        csv_dir.mkdir()

        (csv_dir / "PhysicalTable.csv").write_text(
            ":ID(PhysicalTable),name\n", encoding="utf-8"
        )

        assert validate_csv_directory(csv_dir) is True

        empty_dir = Path(tmpdir) / "empty"
        empty_dir.mkdir()

        assert validate_csv_directory(empty_dir) is False


def test_onboard_networkx_workflow(monkeypatch, tmp_path):
    import importlib
    from govio.cli.config import ConfigManager

    onboard_module = importlib.import_module("govio.cli.onboard")

    config_path = tmp_path / ".govio" / "config.yaml"

    def mock_config_manager():
        return ConfigManager(config_path)

    monkeypatch.setattr(onboard_module, "ConfigManager", mock_config_manager)

    gml_path = tmp_path / "ontology.gml"
    gml_path.touch()

    with patch.object(onboard_module, "questionary") as mock_q:
        # select backend -> networkx
        mock_q.select.return_value.ask.return_value = "networkx"
        mock_q.Choice = MagicMock(side_effect=lambda label, value: (label, value))
        # text for gml path
        mock_q.text.return_value.ask.return_value = str(gml_path)
        # prompt_datasource_config -> None (skip)
        monkeypatch.setattr(onboard_module, "prompt_datasource_config", lambda *a, **kw: None)

        onboard_module.onboard()

    assert config_path.exists()

    saved_config = ConfigManager(config_path).load()
    assert saved_config["graph"]["backend"] == "networkx"
    assert saved_config["graph"]["networkx"]["gml_path"] == str(gml_path)


def test_onboard_falkordb_workflow(monkeypatch, tmp_path):
    import importlib
    from govio.cli.config import ConfigManager

    onboard_module = importlib.import_module("govio.cli.onboard")

    config_path = tmp_path / ".govio" / "config.yaml"

    def mock_config_manager():
        return ConfigManager(config_path)

    monkeypatch.setattr(onboard_module, "ConfigManager", mock_config_manager)

    with patch.object(onboard_module, "questionary") as mock_q:
        # select backend -> falkordb
        mock_q.select.return_value.ask.return_value = "falkordb"
        mock_q.Choice = MagicMock(side_effect=lambda label, value: (label, value))
        # text inputs: host, port, graph_name
        mock_q.text.return_value.ask.side_effect = ["localhost", "6379", "test_graph"]
        # prompt_datasource_config -> None (skip)
        monkeypatch.setattr(onboard_module, "prompt_datasource_config", lambda *a, **kw: None)

        onboard_module.onboard()

    saved_config = ConfigManager(config_path).load()
    assert saved_config["graph"]["backend"] == "falkordb"
    assert saved_config["graph"]["falkordb"]["host"] == "localhost"
    assert saved_config["graph"]["falkordb"]["port"] == 6379
    assert saved_config["graph"]["falkordb"]["graph"] == "test_graph"


def test_onboard_skip_backend_when_existing(monkeypatch, tmp_path):
    """测试已有配置时跳过图后端配置，仅配置数据源"""
    import importlib
    from govio.cli.config import ConfigManager

    onboard_module = importlib.import_module("govio.cli.onboard")

    config_path = tmp_path / ".govio" / "config.yaml"

    # 预先创建已有配置
    existing_config = {
        "graph": {"backend": "networkx", "networkx": {"gml_path": "/tmp/test.gml"}},
    }
    ConfigManager(config_path).save(existing_config)

    def mock_config_manager():
        return ConfigManager(config_path)

    monkeypatch.setattr(onboard_module, "ConfigManager", mock_config_manager)

    with patch.object(onboard_module, "questionary") as mock_q:
        # confirm: skip backend, only datasource
        mock_q.confirm.return_value.ask.return_value = True
        # prompt_datasource_config -> None (skip)
        monkeypatch.setattr(onboard_module, "prompt_datasource_config", lambda *a, **kw: None)

        onboard_module.onboard()

    saved_config = ConfigManager(config_path).load()
    assert saved_config["graph"]["backend"] == "networkx"


class TestPromptConnectArgs:
    """测试 prompt_connect_args 函数"""

    def test_empty_input(self):
        """测试空输入返回空字典"""
        from govio.cli.onboard import prompt_connect_args

        with patch("govio.cli.onboard.questionary") as mock_q:
            mock_q.text.return_value.ask.return_value = ""
            result = prompt_connect_args()
            assert result == {}

    def test_single_kv(self):
        """测试单个 key-value 输入"""
        from govio.cli.onboard import prompt_connect_args

        with patch("govio.cli.onboard.questionary") as mock_q:
            mock_q.text.return_value.ask.side_effect = ["ssl=true", ""]
            result = prompt_connect_args()
            assert result == {"ssl": True}

    def test_multiple_kv(self):
        """测试多个 key-value 输入"""
        from govio.cli.onboard import prompt_connect_args

        with patch("govio.cli.onboard.questionary") as mock_q:
            mock_q.text.return_value.ask.side_effect = ["ssl=true", "timeout=30", "name=test", ""]
            result = prompt_connect_args()
            assert result == {"ssl": True, "timeout": 30, "name": "test"}

    def test_invalid_format_then_valid(self):
        """测试格式错误后继续输入"""
        from govio.cli.onboard import prompt_connect_args

        with patch("govio.cli.onboard.questionary") as mock_q:
            mock_q.text.return_value.ask.side_effect = ["invalid", "key=value", ""]
            result = prompt_connect_args()
            assert result == {"key": "value"}

    def test_keep_existing(self):
        """测试保留已有参数"""
        from govio.cli.onboard import prompt_connect_args

        existing = {"ssl": True, "timeout": 30}
        with patch("govio.cli.onboard.questionary") as mock_q:
            mock_q.confirm.return_value.ask.return_value = True
            result = prompt_connect_args(existing)
            assert result == existing

    def test_replace_existing(self):
        """测试替换已有参数"""
        from govio.cli.onboard import prompt_connect_args

        existing = {"ssl": True}
        with patch("govio.cli.onboard.questionary") as mock_q:
            mock_q.confirm.return_value.ask.return_value = False
            mock_q.text.return_value.ask.side_effect = ["timeout=60", ""]
            result = prompt_connect_args(existing)
            assert result == {"timeout": 60}

    def test_float_value(self):
        """测试浮点数值"""
        from govio.cli.onboard import prompt_connect_args

        with patch("govio.cli.onboard.questionary") as mock_q:
            mock_q.text.return_value.ask.side_effect = ["ratio=0.5", ""]
            result = prompt_connect_args()
            assert result == {"ratio": 0.5}
