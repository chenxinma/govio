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


def test_prompt_backend_choice():
    from govio.cli.onboard import prompt_backend_choice

    with patch("govio.cli.onboard.questionary") as mock_q:
        mock_q.select.return_value.ask.return_value = "networkx"
        mock_q.Choice = MagicMock(side_effect=lambda label, value: (label, value))
        result = prompt_backend_choice()
        assert result == "networkx"

        mock_q.select.return_value.ask.return_value = "falkordb"
        result = prompt_backend_choice()
        assert result == "falkordb"


def test_prompt_networkx_config(monkeypatch, tmp_path):
    import importlib

    onboard_module = importlib.import_module("govio.cli.onboard")

    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    (csv_dir / "PhysicalTable.csv").write_text(
        ":ID(PhysicalTable),name\n", encoding="utf-8"
    )

    onboard_module.SKILLS_ASSETS_DIR = tmp_path / "assets"
    onboard_module.SKILLS_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    with patch.object(onboard_module, "questionary") as mock_q:
        mock_q.confirm.return_value.ask.return_value = True
        mock_q.text.return_value.ask.return_value = str(csv_dir)

        monkeypatch.setattr(
            "govio.cli.onboard.build_graph", lambda *a, **kw: None
        )

        config = onboard_module.prompt_networkx_config()

    assert config["backend"] == "networkx"
    assert "gml_path" in config["networkx"]


def test_prompt_falkordb_config(monkeypatch, tmp_path):
    from govio.cli.onboard import prompt_falkordb_config

    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()

    with patch("govio.cli.onboard.questionary") as mock_q:
        mock_q.text.return_value.ask.side_effect = ["localhost", "6379", "test_graph"]
        monkeypatch.setattr(
            "govio.cli.onboard.import_csv_to_falkordb", lambda *a, **kw: None
        )

        config = prompt_falkordb_config(csv_dir)

    assert config["backend"] == "falkordb"
    assert config["falkordb"]["host"] == "localhost"
    assert config["falkordb"]["port"] == 6379
    assert config["falkordb"]["graph"] == "test_graph"


def test_onboard_networkx_workflow(monkeypatch, tmp_path):
    import importlib
    from govio.cli.config import ConfigManager

    onboard_module = importlib.import_module("govio.cli.onboard")

    csv_dir = tmp_path / "csv"
    create_test_csv_files(csv_dir)

    config_path = tmp_path / ".govio" / "config.yaml"

    onboard_module.SKILLS_ASSETS_DIR = tmp_path / "assets"
    onboard_module.SKILLS_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    def mock_config_manager():
        return ConfigManager(config_path)

    monkeypatch.setattr(onboard_module, "ConfigManager", mock_config_manager)

    monkeypatch.setattr(
        onboard_module,
        "prompt_csv_config",
        lambda _cm: {
            "kundb": "mysql+pymysql://user:pass@host/db",
            "app_list": "app_list.xlsx",
            "app_map": "app_map.json",
            "relationship": None,
            "metric": None,
            "csv_dir": str(csv_dir),
            "workspace_uuid": "82ee37374b314a938bf28170ab4db7cf",
            "output_dir": str(csv_dir),
        },
    )

    monkeypatch.setattr(onboard_module, "generate_csv", lambda _cfg: None)
    monkeypatch.setattr(onboard_module, "prompt_backend_choice", lambda: "networkx")
    monkeypatch.setattr(
        onboard_module,
        "prompt_networkx_config",
        lambda: {
            "backend": "networkx",
            "networkx": {"gml_path": str(tmp_path / "assets" / "ontology.gml")},
            "csv_dir": str(csv_dir),
        },
    )
    monkeypatch.setattr(onboard_module, "prompt_datasource_config", lambda *a, **kw: None)

    mock_graph = MagicMock()
    monkeypatch.setattr(
        "govio.cli.onboard.GraphFactory",
        MagicMock(create=MagicMock(return_value=mock_graph)),
    )
    mock_generator = MagicMock()
    monkeypatch.setattr(
        "govio.cli.onboard.AssetsGenerator",
        MagicMock(return_value=mock_generator),
    )

    onboard_module.onboard()

    assert config_path.exists()

    saved_config = ConfigManager(config_path).load()
    assert saved_config["graph"]["backend"] == "networkx"
    assert saved_config["metadata"]["csv_dir"] is not None


def test_onboard_new_falkordb(monkeypatch, tmp_path):
    """测试 onboard --new-falkordb 跳过 CSV 生成直接导入 FalkorDB"""
    import importlib
    from govio.cli.config import ConfigManager

    onboard_module = importlib.import_module("govio.cli.onboard")

    csv_dir = tmp_path / "csv"
    create_test_csv_files(csv_dir)

    onboard_module.SKILLS_ASSETS_DIR = tmp_path / "assets"
    onboard_module.SKILLS_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    config_path = tmp_path / ".govio" / "config.yaml"

    def mock_config_manager():
        return ConfigManager(config_path)

    monkeypatch.setattr(onboard_module, "ConfigManager", mock_config_manager)

    monkeypatch.setattr(
        "govio.cli.onboard.import_csv_to_falkordb", lambda *a, **kw: None
    )

    with patch.object(onboard_module, "questionary") as mock_q:
        mock_q.text.return_value.ask.side_effect = ["localhost", "6379", "test_graph"]

        mock_graph = MagicMock()
        monkeypatch.setattr(
            "govio.cli.onboard.GraphFactory",
            MagicMock(create=MagicMock(return_value=mock_graph)),
        )
        mock_generator = MagicMock()
        monkeypatch.setattr(
            "govio.cli.onboard.AssetsGenerator",
            MagicMock(return_value=mock_generator),
        )

        onboard_module.onboard(new_falkordb=csv_dir)

    saved_config = ConfigManager(config_path).load()
    assert saved_config["graph"]["backend"] == "falkordb"
    assert saved_config["metadata"]["csv_dir"] == str(csv_dir)
    assert saved_config["graph"]["falkordb"]["host"] == "localhost"
    assert saved_config["graph"]["falkordb"]["graph"] == "test_graph"


def test_onboard_new_networkx(monkeypatch, tmp_path):
    """测试 onboard --new-networkx 跳过 CSV 生成直接生成 GML"""
    import importlib
    from govio.cli.config import ConfigManager

    onboard_module = importlib.import_module("govio.cli.onboard")

    csv_dir = tmp_path / "csv"
    create_test_csv_files(csv_dir)

    onboard_module.SKILLS_ASSETS_DIR = tmp_path / "assets"
    onboard_module.SKILLS_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    config_path = tmp_path / ".govio" / "config.yaml"

    def mock_config_manager():
        return ConfigManager(config_path)

    monkeypatch.setattr(onboard_module, "ConfigManager", mock_config_manager)

    monkeypatch.setattr(
        "govio.cli.onboard.build_graph", lambda *a, **kw: None
    )

    mock_graph = MagicMock()
    monkeypatch.setattr(
        "govio.cli.onboard.GraphFactory",
        MagicMock(create=MagicMock(return_value=mock_graph)),
    )
    mock_generator = MagicMock()
    monkeypatch.setattr(
        "govio.cli.onboard.AssetsGenerator",
        MagicMock(return_value=mock_generator),
    )

    onboard_module.onboard(new_networkx=csv_dir)

    saved_config = ConfigManager(config_path).load()
    assert saved_config["graph"]["backend"] == "networkx"
    assert saved_config["metadata"]["csv_dir"] == str(csv_dir)
    assert "gml_path" in saved_config["graph"]["networkx"]


def test_onboard_new_falkordb_and_new_networkx_exclusive(monkeypatch, tmp_path, capsys):
    """测试 --new-falkordb 和 --new-networkx 不能同时使用"""
    import importlib

    onboard_module = importlib.import_module("govio.cli.onboard")

    with pytest.raises(SystemExit):
        onboard_module.onboard(new_falkordb=tmp_path / "csv", new_networkx=tmp_path / "csv")

    captured = capsys.readouterr()
    assert "不能同时使用" in captured.out


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
