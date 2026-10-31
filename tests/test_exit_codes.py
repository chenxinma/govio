"""query / observe load 失败退出码契约测试。

覆盖三条约束：
1. query 成功（含空结果）退出 0；执行失败（语法错/后端异常）、连接失败、
   参数缺失或非法退出 1
2. observe load 成功退出 0；SQL 错误、连接失败、内存加载失败退出 1
3. 失败信息走 stderr（❌ 前缀），stdout 不混入错误

全部打桩，不连真实库。
"""

import json
import sys

import pandas as pd
import pytest

from govio.cli import observe as observe_mod
from govio.cli import query as query_mod


class StubConfigManager:
    """打桩 ConfigManager：返回指定配置"""

    def __init__(self, config: dict):
        self._config = config

    def exists(self) -> bool:
        return True

    def load(self) -> dict:
        return self._config


class StubNetworkXGraph:
    """打桩 NetworkXGraph：返回空图，不读真实 GML"""

    def __init__(self, gml_path) -> None:
        self.G = {}


def _setup_networkx_query(monkeypatch, tmp_path) -> None:
    """打桩 networkx 后端的 query 调用链（配置 + 图加载）"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "test.gml").write_text("graph []", encoding="utf-8")
    monkeypatch.setattr(
        query_mod,
        "ConfigManager",
        lambda: StubConfigManager(
            {
                "graph": {
                    "backend": "networkx",
                    "networkx": {"gml_path": "test.gml"},
                }
            }
        ),
    )
    monkeypatch.setattr(query_mod, "NetworkXGraph", StubNetworkXGraph)


# ---------------------------------------------------------------------------
# query：成功 / 空结果 / 执行失败
# ---------------------------------------------------------------------------


def test_query_success_exits_zero(tmp_path, monkeypatch, capsys):
    """查询成功返回 None（进程退出码 0），结果摘要落 stdout。"""
    _setup_networkx_query(monkeypatch, tmp_path)

    result = query_mod.query("result = [['a', 1]]")

    assert result is None
    captured = capsys.readouterr()
    assert "Result output:" in captured.out
    assert "❌" not in captured.err


def test_query_empty_result_exits_zero(tmp_path, monkeypatch, capsys):
    """空结果不是失败：只输出 Data not found.，退出码仍为 0。"""
    _setup_networkx_query(monkeypatch, tmp_path)

    result = query_mod.query("result = []")

    assert result is None
    captured = capsys.readouterr()
    assert captured.out.strip() == "Data not found."
    assert "❌" not in captured.err


@pytest.mark.parametrize(
    "bad_code",
    ["result = [", "raise RuntimeError('boom')"],
    ids=["syntax-error", "runtime-error"],
)
def test_query_exec_failure_exits_one(tmp_path, monkeypatch, capsys, bad_code):
    """查询执行失败（语法错/后端异常）退出码 1，错误走 stderr。"""
    _setup_networkx_query(monkeypatch, tmp_path)

    with pytest.raises(SystemExit) as exc:
        query_mod.query(bad_code)

    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "❌" in captured.err
    assert "查询执行失败" in captured.err
    assert "❌" not in captured.out


def test_query_connection_failure_exits_one(tmp_path, monkeypatch, capsys):
    """图库不可达退出码 1，连接错误走 stderr。"""
    monkeypatch.chdir(tmp_path)

    class DownFalkorDBGraph:
        def __init__(self, **kwargs) -> None:
            raise ConnectionError("connection refused")

    monkeypatch.setattr(
        query_mod,
        "ConfigManager",
        lambda: StubConfigManager({"graph": {"backend": "falkordb", "falkordb": {}}}),
    )
    monkeypatch.setattr(query_mod, "FalkorDBGraph", DownFalkorDBGraph)

    with pytest.raises(SystemExit) as exc:
        query_mod.query("MATCH (n) RETURN n")

    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "❌" in captured.err
    assert "连接图库失败" in captured.err
    assert "❌" not in captured.out


@pytest.mark.parametrize("query_text", [None, "", "   "], ids=["none", "empty", "blank"])
def test_query_missing_code_exits_one(tmp_path, monkeypatch, capsys, query_text):
    """查询语句缺失退出码 1。"""
    _setup_networkx_query(monkeypatch, tmp_path)

    with pytest.raises(SystemExit) as exc:
        query_mod.query(query_text)

    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "❌" in captured.err


def test_query_non_match_cypher_exits_one(tmp_path, monkeypatch, capsys):
    """非 MATCH 查询（参数非法）退出码 1。"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        query_mod,
        "ConfigManager",
        lambda: StubConfigManager({"graph": {"backend": "falkordb", "falkordb": {}}}),
    )

    with pytest.raises(SystemExit) as exc:
        query_mod.query("CREATE (n:X)")

    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "❌" in captured.err
    assert "Please write a MATCH query." in captured.err


# ---------------------------------------------------------------------------
# observe load：成功 / SQL 错误 / 连接失败 / 内存加载失败
# ---------------------------------------------------------------------------


class FakeDbManager:
    """打桩 DatabaseManager：按需返回 DataFrame 或抛错"""

    def __init__(self, df: pd.DataFrame | None = None, error: Exception | None = None):
        self.df = df
        self.error = error

    def execute_sql(self, datasource: str, sql: str) -> pd.DataFrame:
        if self.error is not None:
            raise self.error
        assert self.df is not None
        return self.df


def test_observe_load_success_exits_zero(tmp_path, monkeypatch, capsys):
    """加载成功不抛 SystemExit（退出码 0），stdout 输出 success JSON。"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        observe_mod,
        "get_db_manager",
        lambda config: FakeDbManager(df=pd.DataFrame({"x": [1, 2]})),
    )

    result = observe_mod.cmd_load({}, "t1", "SELECT x FROM t", datasource="d1")

    assert result is None
    payload = json.loads(capsys.readouterr().out)
    assert payload["success"] is True
    assert payload["rows"] == 2


@pytest.mark.parametrize(
    "db_error",
    [RuntimeError("syntax error near SELECT"), ConnectionError("connection refused")],
    ids=["sql-error", "connection-failure"],
)
def test_observe_load_failure_exits_one(tmp_path, monkeypatch, capsys, db_error):
    """SQL 错误/连接失败退出码 1；stdout 保留 success/error JSON，错误信息走 stderr。"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        observe_mod, "get_db_manager", lambda config: FakeDbManager(error=db_error)
    )

    with pytest.raises(SystemExit) as exc:
        observe_mod.cmd_load({}, "t1", "SELECT bad", datasource="d1")

    assert exc.value.code == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["success"] is False
    assert "❌" in captured.err
    assert str(db_error) in captured.err
    assert "❌" not in captured.out


def test_observe_load_memory_failure_exits_one(tmp_path, monkeypatch, capsys):
    """内存加载失败（无已加载 DataFrame）退出码 1，错误走 stderr。"""
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as exc:
        observe_mod.cmd_load({}, "t1", "SELECT 1", memory=True)

    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "❌" in captured.err
    assert "没有已加载的 DataFrame" in captured.err


# ---------------------------------------------------------------------------
# observe 入口兜底：未预期异常转退出码 1，不裸奔 traceback
# ---------------------------------------------------------------------------


def test_observe_entry_unexpected_error_exits_one(tmp_path, monkeypatch, capsys):
    """observe 入口未捕获异常兜底为退出码 1，stderr 简洁错误。"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        observe_mod,
        "ConfigManager",
        lambda: StubConfigManager({"datasources": {"d1": {"url": "duckdb://x"}}}),
    )

    def boom(*args, **kwargs) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(observe_mod, "cmd_load", boom)
    monkeypatch.setattr(
        sys,
        "argv",
        ["observe", "load", "--name", "t1", "--datasource", "d1", "--sql", "SELECT 1"],
    )

    with pytest.raises(SystemExit) as exc:
        observe_mod.observe()

    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "❌" in captured.err
    assert "boom" in captured.err
    assert "Traceback" not in captured.err
