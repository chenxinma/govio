from pathlib import Path
import tempfile
from unittest.mock import patch

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


def test_prompt_backend_choice(monkeypatch):
    from govio.cli.onboard import prompt_backend_choice

    # Test choice 1 (networkx)
    monkeypatch.setattr("builtins.input", lambda _: "1")
    result = prompt_backend_choice()
    assert result == "networkx"

    # Test choice 2 (falkordb)
    monkeypatch.setattr("builtins.input", lambda _: "2")
    result = prompt_backend_choice()
    assert result == "falkordb"

    # Test default (empty input)
    monkeypatch.setattr("builtins.input", lambda _: "")
    result = prompt_backend_choice()
    assert result == "networkx"


def test_prompt_networkx_config(monkeypatch, tmp_path):
    import importlib

    onboard_module = importlib.import_module("govio.cli.onboard")

    # Create test CSV directory
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    (csv_dir / "PhysicalTable.csv").write_text(
        ":ID(PhysicalTable),name\n", encoding="utf-8"
    )

    # Patch SKILLS_ASSETS_DIR to use temp directory
    onboard_module.SKILLS_ASSETS_DIR = tmp_path / "assets"
    onboard_module.SKILLS_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    # Test generating GML from CSV
    inputs = iter(["yes", str(csv_dir)])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    config = onboard_module.prompt_networkx_config()

    assert config["backend"] == "networkx"
    assert "gml_path" in config["networkx"]


def test_prompt_falkordb_config(monkeypatch, tmp_path):
    from govio.cli.onboard import prompt_falkordb_config

    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()

    inputs = iter(["localhost", "6379", "test_graph"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    monkeypatch.setattr(
        "govio.cli.onboard.import_csv_to_falkordb", lambda *a, **kw: None
    )

    config = prompt_falkordb_config(csv_dir)

    assert config["backend"] == "falkordb"
    assert config["falkordb"]["host"] == "localhost"
    assert config["falkordb"]["port"] == 6379
    assert config["falkordb"]["graph"] == "test_graph"


def test_prompt_backend_choice_invalid_input(monkeypatch, capsys):
    from govio.cli.onboard import prompt_backend_choice

    inputs = iter(["3", "invalid", "1"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    result = prompt_backend_choice()
    assert result == "networkx"

    captured = capsys.readouterr()
    assert "无效选项" in captured.out


def test_prompt_falkordb_config_invalid_port(monkeypatch, capsys, tmp_path):
    from govio.cli.onboard import prompt_falkordb_config

    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()

    inputs = iter(["localhost", "invalid", "6379", "test_graph"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    monkeypatch.setattr(
        "govio.cli.onboard.import_csv_to_falkordb", lambda *a, **kw: None
    )

    config = prompt_falkordb_config(csv_dir)

    assert config["falkordb"]["port"] == 6379

    captured = capsys.readouterr()
    assert "端口必须是数字" in captured.out


def test_onboard_networkx_workflow(monkeypatch, tmp_path):
    import importlib
    from govio.cli.config import ConfigManager

    onboard_module = importlib.import_module("govio.cli.onboard")

    # Create test CSV files
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
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

    # Mock config path to use temp directory
    config_path = tmp_path / ".govio" / "config.yaml"

    # Mock inputs for the wizard
    inputs = iter(
        [
            "1",  # backend choice: networkx
            "yes",  # generate GML
            str(csv_dir),  # CSV directory
        ]
    )
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    # Patch SKILLS_ASSETS_DIR
    onboard_module.SKILLS_ASSETS_DIR = tmp_path / "assets"
    onboard_module.SKILLS_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    # Patch ConfigManager to return instance with temp config path
    def mock_config_manager():
        return ConfigManager(config_path)

    monkeypatch.setattr(onboard_module, "ConfigManager", mock_config_manager)

    # Run onboard
    onboard_module.onboard()

    # Verify outputs
    assert config_path.exists()
    assert (tmp_path / "assets" / "schema.md").exists()
    assert (tmp_path / "assets" / "names" / "node_names.md").exists()


def test_onboard_new_falkordb(monkeypatch, tmp_path):
    """测试 onboard --new-falkordb 跳过 CSV 生成直接导入 FalkorDB"""
    import importlib
    from govio.cli.config import ConfigManager

    onboard_module = importlib.import_module("govio.cli.onboard")

    # Create test CSV files
    csv_dir = tmp_path / "csv"
    create_test_csv_files(csv_dir)

    # Patch SKILLS_ASSETS_DIR
    onboard_module.SKILLS_ASSETS_DIR = tmp_path / "assets"
    onboard_module.SKILLS_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    # Mock config path
    config_path = tmp_path / ".govio" / "config.yaml"

    def mock_config_manager():
        return ConfigManager(config_path)

    monkeypatch.setattr(onboard_module, "ConfigManager", mock_config_manager)

    # Mock FalkorDB import and graph factory
    monkeypatch.setattr(
        "govio.cli.onboard.import_csv_to_falkordb", lambda *a, **kw: None
    )

    # Mock inputs: host, port, graph_name
    inputs = iter(["localhost", "6379", "test_graph"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    # Mock GraphFactory and AssetsGenerator
    from unittest.mock import MagicMock

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

    # Verify config saved with falkordb backend
    saved_config = ConfigManager(config_path).load()
    assert saved_config["backend"] == "falkordb"
    assert saved_config["csv_dir"] == str(csv_dir)
    assert saved_config["falkordb"]["host"] == "localhost"
    assert saved_config["falkordb"]["graph"] == "test_graph"

    # Verify backend.txt
    assert (tmp_path / "assets" / "backend.txt").read_text().strip() == "falkordb"


def test_onboard_new_networkx(monkeypatch, tmp_path):
    """测试 onboard --new-networkx 跳过 CSV 生成直接生成 GML"""
    import importlib
    from govio.cli.config import ConfigManager

    onboard_module = importlib.import_module("govio.cli.onboard")

    # Create test CSV files
    csv_dir = tmp_path / "csv"
    create_test_csv_files(csv_dir)

    # Patch SKILLS_ASSETS_DIR
    onboard_module.SKILLS_ASSETS_DIR = tmp_path / "assets"
    onboard_module.SKILLS_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    # Mock config path
    config_path = tmp_path / ".govio" / "config.yaml"

    def mock_config_manager():
        return ConfigManager(config_path)

    monkeypatch.setattr(onboard_module, "ConfigManager", mock_config_manager)

    # Mock build_graph and graph factory
    from unittest.mock import MagicMock

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

    # Verify config saved with networkx backend
    saved_config = ConfigManager(config_path).load()
    assert saved_config["backend"] == "networkx"
    assert saved_config["csv_dir"] == str(csv_dir)
    assert "gml_path" in saved_config["networkx"]

    # Verify backend.txt
    assert (tmp_path / "assets" / "backend.txt").read_text().strip() == "networkx"


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

        with patch("builtins.input", return_value=""):
            result = prompt_connect_args()
            assert result == {}

    def test_single_kv(self):
        """测试单个 key-value 输入"""
        from govio.cli.onboard import prompt_connect_args

        responses = ["ssl=true", ""]
        with patch("builtins.input", side_effect=responses):
            result = prompt_connect_args()
            assert result == {"ssl": True}

    def test_multiple_kv(self):
        """测试多个 key-value 输入"""
        from govio.cli.onboard import prompt_connect_args

        responses = ["ssl=true", "timeout=30", "name=test", ""]
        with patch("builtins.input", side_effect=responses):
            result = prompt_connect_args()
            assert result == {"ssl": True, "timeout": 30, "name": "test"}

    def test_invalid_format_then_valid(self):
        """测试格式错误后继续输入"""
        from govio.cli.onboard import prompt_connect_args

        responses = ["invalid", "key=value", ""]
        with patch("builtins.input", side_effect=responses):
            result = prompt_connect_args()
            assert result == {"key": "value"}

    def test_keep_existing(self):
        """测试保留已有参数"""
        from govio.cli.onboard import prompt_connect_args

        existing = {"ssl": True, "timeout": 30}
        with patch("builtins.input", return_value=""):
            result = prompt_connect_args(existing)
            assert result == existing

    def test_replace_existing(self):
        """测试替换已有参数"""
        from govio.cli.onboard import prompt_connect_args

        existing = {"ssl": True}
        responses = ["no", "timeout=60", ""]
        with patch("builtins.input", side_effect=responses):
            result = prompt_connect_args(existing)
            assert result == {"timeout": 60}

    def test_float_value(self):
        """测试浮点数值"""
        from govio.cli.onboard import prompt_connect_args

        responses = ["ratio=0.5", ""]
        with patch("builtins.input", side_effect=responses):
            result = prompt_connect_args()
            assert result == {"ratio": 0.5}
