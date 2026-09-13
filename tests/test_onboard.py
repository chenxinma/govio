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


class TestCliDatasource:
    """测试非交互式数据源管理（onboard --add-datasource 等）"""

    @pytest.fixture
    def cm(self, monkeypatch, tmp_path):
        """隔离配置文件与加密密钥"""
        from govio.cli.config import ConfigManager
        import govio.cli.onboard as onboard_module

        config_path = tmp_path / ".govio" / "config.yaml"
        monkeypatch.setattr(
            onboard_module, "ConfigManager", lambda: ConfigManager(config_path)
        )
        monkeypatch.setattr(
            onboard_module, "encrypt_value", lambda p: f"enc:{p}"
        )
        return ConfigManager(config_path)

    def test_add_datasource_basic(self, cm):
        """添加含密码 URL 的数据源，密码脱敏加密存储"""
        from govio.cli.onboard import add_datasource

        entry = add_datasource(
            "prod",
            "mysql+pymysql://user:pw@host:3306/db",
            connect_args={"charset": "utf8mb4"},
        )
        assert entry["url"] == "mysql+pymysql://user:***@host:3306/db"
        assert entry["encrypted_password"] == "enc:pw"
        assert entry["connect_args"] == {"charset": "utf8mb4"}

        saved = cm.load()
        assert saved["datasources"]["prod"]["url"] == "mysql+pymysql://user:***@host:3306/db"

    def test_add_datasource_keeps_existing_graph(self, cm):
        """已有图后端配置时添加数据源不会破坏原配置"""
        from govio.cli.onboard import add_datasource

        cm.save({"graph": {"backend": "networkx", "networkx": {"gml_path": "/tmp/x.gml"}}})
        add_datasource("ds1", "duckdb:///tmp/data.duckdb")

        saved = cm.load()
        assert saved["graph"]["backend"] == "networkx"
        assert saved["datasources"]["ds1"]["url"] == "duckdb:///tmp/data.duckdb"
        assert saved["datasources"]["ds1"]["connect_args"] == {}

    def test_add_datasource_password_separate(self, cm):
        """--password 单独提供时嵌入 URL 后加密"""
        from govio.cli.onboard import add_datasource

        entry = add_datasource(
            "pg", "postgresql://user@host:5432/db", password="s3cret"
        )
        assert entry["url"] == "postgresql://user:***@host:5432/db"
        assert entry["encrypted_password"] == "enc:s3cret"

    def test_add_datasource_password_conflict(self, cm):
        """URL 已含密码时再传 password 报错"""
        from govio.cli.onboard import add_datasource

        with pytest.raises(ValueError, match="URL 中已包含密码"):
            add_datasource("pg", "postgresql://user:pw@host/db", password="other")

    def test_add_datasource_invalid_url(self, cm):
        """缺少协议前缀的 URL 报错"""
        from govio.cli.onboard import add_datasource

        with pytest.raises(ValueError, match="URL 格式无效"):
            add_datasource("bad", "host:3306/db")

    def test_add_datasource_duplicate_requires_overwrite(self, cm):
        """同名数据源默认拒绝，--overwrite 才覆盖"""
        from govio.cli.onboard import add_datasource

        add_datasource("prod", "mysql+pymysql://u:p1@h1/db")
        with pytest.raises(ValueError, match="已存在"):
            add_datasource("prod", "mysql+pymysql://u:p2@h2/db")

        add_datasource(
            "prod", "mysql+pymysql://u:p2@h2/db", overwrite=True
        )
        assert cm.load()["datasources"]["prod"]["url"] == "mysql+pymysql://u:***@h2/db"

    def test_remove_datasource(self, cm):
        """删除数据源，最后一个删除后移除 datasources 键"""
        from govio.cli.onboard import add_datasource, remove_datasource

        add_datasource("a", "duckdb:///a.duckdb")
        add_datasource("b", "duckdb:///b.duckdb")
        remove_datasource("a")
        assert "a" not in cm.load()["datasources"]

        remove_datasource("b")
        assert "datasources" not in cm.load()

        with pytest.raises(ValueError, match="不存在"):
            remove_datasource("a")

    def test_parse_cli_connect_args(self):
        """命令行连接参数解析与类型转换"""
        from govio.cli.onboard import parse_cli_connect_args

        result = parse_cli_connect_args(["ssl=true", "timeout=30", "ratio=0.5", "name=x"])
        assert result == {"ssl": True, "timeout": 30, "ratio": 0.5, "name": "x"}

        assert parse_cli_connect_args(None) == {}

        with pytest.raises(ValueError, match="格式错误"):
            parse_cli_connect_args(["invalid"])
        with pytest.raises(ValueError, match="格式错误"):
            parse_cli_connect_args(["=value"])

    def test_cli_add_datasource_end_to_end(self, cm, capsys):
        """CLI 入口：无配置文件时自动创建并提示图后端未配置"""
        from govio.cli.onboard import onboard_datasource_cli

        onboard_datasource_cli(
            add_name="prod",
            url="mysql+pymysql://user:pw@host:3306/db",
            connect_args=["charset=utf8mb4"],
        )
        saved = cm.load()
        assert saved["datasources"]["prod"]["encrypted_password"] == "enc:pw"
        out = capsys.readouterr().out
        assert "已添加数据源: prod" in out
        assert "尚未配置图后端" in out

    def test_cli_add_without_url_exits(self, cm, capsys):
        """CLI 入口：--add-datasource 缺少 --url 时退出码 1"""
        from govio.cli.onboard import onboard_datasource_cli

        with pytest.raises(SystemExit) as excinfo:
            onboard_datasource_cli(add_name="prod")
        assert excinfo.value.code == 1
        assert "--url" in capsys.readouterr().err

    def test_cli_add_and_remove_conflict_exits(self, cm):
        """CLI 入口：添加与删除互斥"""
        from govio.cli.onboard import onboard_datasource_cli

        with pytest.raises(SystemExit) as excinfo:
            onboard_datasource_cli(add_name="a", remove_name="b")
        assert excinfo.value.code == 1

    def test_cli_overwrite_duplicate_exits_without_flag(self, cm):
        """CLI 入口：同名未加 --overwrite 时退出码 1"""
        from govio.cli.onboard import onboard_datasource_cli

        onboard_datasource_cli(add_name="prod", url="duckdb:///a.duckdb")
        with pytest.raises(SystemExit) as excinfo:
            onboard_datasource_cli(add_name="prod", url="duckdb:///b.duckdb")
        assert excinfo.value.code == 1

        onboard_datasource_cli(
            add_name="prod", url="duckdb:///b.duckdb", overwrite=True
        )
        assert cm.load()["datasources"]["prod"]["url"] == "duckdb:///b.duckdb"
