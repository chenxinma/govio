"""onboard 非交互图后端配置测试（FRD: docs/tasks/X33-onboard-noninteractive/FRD.md §7）"""

import sys
from pathlib import Path

import pytest


class _ExplodingQuestionary:
    """任何 questionary 调用都视为失败（非交互路径不得弹交互）"""

    def __getattr__(self, name):
        raise AssertionError(f"非交互路径不应调用 questionary.{name}")


@pytest.fixture
def cm(monkeypatch, tmp_path):
    """隔离配置文件与加密，questionary 一律炸掉"""
    import govio.cli.onboard as onboard_module
    from govio.cli.config import ConfigManager

    config_path = tmp_path / ".govio" / "config.yaml"
    monkeypatch.setattr(
        onboard_module, "ConfigManager", lambda: ConfigManager(config_path)
    )
    monkeypatch.setattr(onboard_module, "encrypt_value", lambda p: f"enc:{p}")
    monkeypatch.setattr(onboard_module, "questionary", _ExplodingQuestionary())
    return ConfigManager(config_path)


@pytest.fixture
def cli(monkeypatch, tmp_path, cm):
    """通过 govio.cli.main.main() 走完整 CLI 路由（sys.argv + main 侧 ConfigManager）"""
    import importlib

    from govio.cli.config import ConfigManager

    # govio.cli.main 属性被 __init__ re-export 的 main 函数遮蔽，需从 sys.modules 取模块
    main_module = importlib.import_module("govio.cli.main")
    config_path = tmp_path / ".govio" / "config.yaml"
    monkeypatch.setattr(
        main_module, "ConfigManager", lambda: ConfigManager(config_path)
    )

    def run(*argv):
        monkeypatch.setattr(sys, "argv", ["govio-cli", *argv])
        main_module.main()

    return run


def _run_graph_cli(**kwargs):
    from govio.cli.onboard import onboard_graph_cli

    onboard_graph_cli(**kwargs)


class TestGraphCliDefaults:
    """FRD §7-1/2：缺省值与自定义路径"""

    def test_ladybug_default_db_path(self, cm):
        _run_graph_cli(backend="ladybug")

        saved = cm.load()
        assert saved["graph"] == {
            "backend": "ladybug",
            "ladybug": {"db_path": str(Path.home() / ".govio" / "ontology.lbdb")},
        }

    def test_ladybug_custom_db_path(self, cm, tmp_path):
        db = tmp_path / "custom.lbdb"
        _run_graph_cli(backend="ladybug", db_path=str(db))

        assert cm.load()["graph"]["ladybug"]["db_path"] == str(db)

    def test_falkordb_defaults(self, cm):
        _run_graph_cli(backend="falkordb")

        assert cm.load()["graph"] == {
            "backend": "falkordb",
            "falkordb": {"host": "localhost", "port": 6379, "graph": "ontology"},
        }

    def test_falkordb_custom(self, cm):
        _run_graph_cli(backend="falkordb", host="h", port="1234", graph_name="g")

        assert cm.load()["graph"] == {
            "backend": "falkordb",
            "falkordb": {"host": "h", "port": 1234, "graph": "g"},
        }

    def test_networkx_gml_path(self, cm, tmp_path):
        gml = tmp_path / "ontology.gml"
        gml.touch()
        _run_graph_cli(backend="networkx", gml_path=str(gml))

        assert cm.load()["graph"] == {
            "backend": "networkx",
            "networkx": {"gml_path": str(gml)},
        }


class TestGraphCliIdempotent:
    """FRD §7-3：graph 段 upsert 覆盖、datasources 保留不变"""

    def test_switch_backend_preserves_datasources(self, cm):
        from govio.cli.onboard import add_datasource

        _run_graph_cli(backend="ladybug")
        add_datasource("mydb", "duckdb:///x.db", config_manager=cm)

        _run_graph_cli(backend="falkordb", host="h", port="1234", graph_name="g")

        saved = cm.load()
        assert saved["graph"]["backend"] == "falkordb"
        assert saved["graph"]["falkordb"] == {"host": "h", "port": 1234, "graph": "g"}
        assert saved["datasources"]["mydb"]["url"] == "duckdb:///x.db"

    def test_repeat_same_backend_idempotent(self, cm):
        _run_graph_cli(backend="ladybug")
        first = cm.load()
        _run_graph_cli(backend="ladybug")
        assert cm.load() == first


class TestGraphCliWithDatasource:
    """FRD §7-4：图后端 + 数据源增删组合，单次落盘"""

    def test_combined_add(self, cm, tmp_path):
        db = tmp_path / "x.lbdb"
        _run_graph_cli(
            backend="ladybug",
            db_path=str(db),
            add_name="mydb",
            url="duckdb:///x.db",
        )

        saved = cm.load()
        assert saved["graph"]["ladybug"]["db_path"] == str(db)
        assert saved["datasources"]["mydb"]["url"] == "duckdb:///x.db"
        assert saved["datasources"]["mydb"]["connect_args"] == {}

    def test_combined_add_password_encrypted_and_masked(self, cm):
        _run_graph_cli(
            backend="ladybug",
            add_name="pg",
            url="postgresql://user:pw@host:5432/db",
        )

        entry = cm.load()["datasources"]["pg"]
        assert entry["url"] == "postgresql://user:***@host:5432/db"
        assert entry["encrypted_password"] == "enc:pw"

    def test_combined_add_duplicate_then_overwrite(self, cm):
        _run_graph_cli(backend="ladybug", add_name="mydb", url="duckdb:///a.db")

        with pytest.raises(SystemExit) as excinfo:
            _run_graph_cli(backend="ladybug", add_name="mydb", url="duckdb:///b.db")
        assert excinfo.value.code == 1

        _run_graph_cli(
            backend="ladybug", add_name="mydb", url="duckdb:///b.db", overwrite=True
        )
        assert cm.load()["datasources"]["mydb"]["url"] == "duckdb:///b.db"

    def test_combined_remove(self, cm):
        from govio.cli.onboard import add_datasource

        add_datasource("a", "duckdb:///a.duckdb", config_manager=cm)
        _run_graph_cli(backend="ladybug", remove_name="a")

        saved = cm.load()
        assert saved["graph"]["backend"] == "ladybug"
        assert "datasources" not in saved


class TestGraphCliErrors:
    """FRD §7-5：错误路径均 exit 1 且 stderr 含 错误:"""

    def _assert_fail(self, capsys, match):
        err = capsys.readouterr().err
        assert "错误:" in err
        assert match in err

    def test_networkx_without_gml_path(self, cm, capsys):
        with pytest.raises(SystemExit) as excinfo:
            _run_graph_cli(backend="networkx")
        assert excinfo.value.code == 1
        self._assert_fail(capsys, "--gml-path")

    def test_networkx_gml_not_exists(self, cm, capsys, tmp_path):
        with pytest.raises(SystemExit) as excinfo:
            _run_graph_cli(backend="networkx", gml_path=str(tmp_path / "missing.gml"))
        assert excinfo.value.code == 1
        self._assert_fail(capsys, "GML 文件不存在")

    def test_falkordb_port_not_digit(self, cm, capsys):
        with pytest.raises(SystemExit) as excinfo:
            _run_graph_cli(backend="falkordb", port="abc")
        assert excinfo.value.code == 1
        self._assert_fail(capsys, "端口必须是数字")

    def test_mismatched_backend_flag(self, cm, capsys):
        with pytest.raises(SystemExit) as excinfo:
            _run_graph_cli(backend="ladybug", host="x")
        assert excinfo.value.code == 1
        self._assert_fail(capsys, "--host 需配合 --backend falkordb")

    def test_add_without_url(self, cm, capsys):
        with pytest.raises(SystemExit) as excinfo:
            _run_graph_cli(backend="ladybug", add_name="mydb")
        assert excinfo.value.code == 1
        self._assert_fail(capsys, "--url")

    def test_add_and_remove_conflict(self, cm, capsys):
        with pytest.raises(SystemExit) as excinfo:
            _run_graph_cli(
                backend="ladybug",
                add_name="a",
                url="duckdb:///a.db",
                remove_name="b",
            )
        assert excinfo.value.code == 1
        self._assert_fail(capsys, "不能同时使用")

    def test_remove_with_url_rejected(self, cm, capsys):
        with pytest.raises(SystemExit) as excinfo:
            _run_graph_cli(backend="ladybug", remove_name="a", url="duckdb:///a.db")
        assert excinfo.value.code == 1
        self._assert_fail(capsys, "--remove-datasource 不接受")

    def test_no_error_leaves_config_untouched(self, cm, capsys):
        """报错退出不落盘（校验先行）"""
        with pytest.raises(SystemExit):
            _run_graph_cli(backend="networkx")
        assert not cm.exists()


class TestCliRouting:
    """FRD §3.2/§7-6/7：路由与向后兼容"""

    def test_backend_flag_routes_non_interactive(self, cli, cm, tmp_path):
        cli("onboard", "--backend", "ladybug", "--db-path", str(tmp_path / "a.lbdb"))

        saved = cm.load()
        assert saved["graph"]["backend"] == "ladybug"
        assert saved["graph"]["ladybug"]["db_path"] == str(tmp_path / "a.lbdb")

    def test_graph_flag_without_backend_fails(self, cli, cm, capsys):
        with pytest.raises(SystemExit) as excinfo:
            cli("onboard", "--db-path", "x.lbdb")
        assert excinfo.value.code == 1
        assert "--db-path 需配合 --backend ladybug" in capsys.readouterr().err

    def test_combined_via_cli(self, cli, cm):
        cli(
            "onboard",
            "--backend",
            "ladybug",
            "--add-datasource",
            "mydb",
            "--url",
            "duckdb:///x.db",
        )

        saved = cm.load()
        assert saved["graph"]["backend"] == "ladybug"
        assert saved["datasources"]["mydb"]["url"] == "duckdb:///x.db"

    def test_datasource_flags_keep_existing_route(self, cli, cm, monkeypatch):
        """不带 --backend 的数据源旗标仍走 onboard_datasource_cli（零回归）"""
        import govio.cli.onboard as onboard_module

        called = {}
        monkeypatch.setattr(
            onboard_module,
            "onboard_datasource_cli",
            lambda **kw: called.setdefault("kw", kw),
        )
        cli("onboard", "--add-datasource", "a", "--url", "duckdb:///a.db")
        assert called["kw"]["add_name"] == "a"

    def test_bare_onboard_still_interactive(self, cli, monkeypatch):
        """不带 --backend 且无数据源旗标时仍走交互向导"""
        import govio.cli.onboard as onboard_module

        called = []
        monkeypatch.setattr(onboard_module, "onboard", lambda: called.append(1))
        cli("onboard")
        assert called == [1]

    def test_url_without_add_datasource_keeps_error(self, cli, capsys):
        with pytest.raises(SystemExit) as excinfo:
            cli("onboard", "--url", "duckdb:///a.db")
        assert excinfo.value.code == 1
        assert "需配合 --add-datasource" in capsys.readouterr().err

    def test_help_lists_new_flags(self, cli, capsys):
        with pytest.raises(SystemExit) as excinfo:
            cli("onboard", "--help")
        assert excinfo.value.code == 0
        out = capsys.readouterr().out
        for flag in (
            "--backend",
            "--db-path",
            "--host",
            "--port",
            "--graph",
            "--gml-path",
        ):
            assert flag in out
