"""query / observe 全部子命令失败退出码契约测试。

覆盖三条约束：
1. query 成功（含空结果）退出 0；执行失败（语法错/后端异常）、连接失败、
   参数缺失或非法退出 1
2. observe load 成功退出 0；SQL 错误、连接失败、内存加载失败退出 1
3. observe 其余子命令（compare/explore/chart/info/show/release/visualize-relations）
   成功（含空结果）退出 0；失败（DataFrame 不存在、参数非法、文件不存在、
   执行错误、success:false）退出 1
4. 失败信息走 stderr（❌ 前缀），stdout 不混入 ❌ 错误（保留结构化 JSON）

全部打桩，不连真实库。
"""

import argparse
import json
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from govio.cli import observe as observe_mod
from govio.cli import query as query_mod
from govio.observe_data.core.observe_store import ObserveStore


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


# ---------------------------------------------------------------------------
# observe compare：成功 / 失败 / 无差异（空结果）
# ---------------------------------------------------------------------------


def _seed_dataframe(name: str, df: pd.DataFrame) -> None:
    """向 ObserveStore 写入一个 DataFrame（测试打桩用）"""
    ObserveStore().store(name, df, datasource="d1", sql="SELECT 1")


def test_observe_compare_success_exits_zero(tmp_path, monkeypatch, capsys):
    """比对成功不抛 SystemExit（退出码 0），stdout 输出 success JSON。"""
    monkeypatch.chdir(tmp_path)
    _seed_dataframe("src_df", pd.DataFrame({"id": [1, 2], "v": [10, 20]}))
    _seed_dataframe("tgt_df", pd.DataFrame({"id": [1, 2], "v": [10, 99]}))

    result = observe_mod.cmd_compare({}, "src_df", "tgt_df", ["id"])

    assert result is None
    payload = json.loads(capsys.readouterr().out)
    assert payload["success"] is True
    assert "schema" in payload and "data" in payload


@pytest.mark.parametrize(
    "source, target",
    [("src_df", "missing"), ("missing", "tgt_df")],
    ids=["target-missing", "source-missing"],
)
def test_observe_compare_missing_dataframe_exits_one(
    tmp_path, monkeypatch, capsys, source, target
):
    """source/target DataFrame 不存在退出码 1；stdout 保留 success:false JSON。"""
    monkeypatch.chdir(tmp_path)
    _seed_dataframe("src_df", pd.DataFrame({"id": [1], "v": [1]}))
    _seed_dataframe("tgt_df", pd.DataFrame({"id": [1], "v": [1]}))

    with pytest.raises(SystemExit) as exc:
        observe_mod.cmd_compare({}, source, target, ["id"])

    assert exc.value.code == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["success"] is False
    assert "❌" in captured.err
    assert "❌" not in captured.out


def test_observe_compare_exec_error_exits_one(tmp_path, monkeypatch, capsys):
    """比对执行错误（join 列不存在）退出码 1，错误走 stderr。"""
    monkeypatch.chdir(tmp_path)
    _seed_dataframe("src_df", pd.DataFrame({"id": [1], "v": [1]}))
    _seed_dataframe("tgt_df", pd.DataFrame({"id": [1], "v": [1]}))

    with pytest.raises(SystemExit) as exc:
        observe_mod.cmd_compare({}, "src_df", "tgt_df", ["no_such_col"])

    assert exc.value.code == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["success"] is False
    assert "❌" in captured.err
    assert "比对执行错误" in captured.err


def test_observe_compare_no_diff_exits_zero(tmp_path, monkeypatch, capsys):
    """无差异（空结果）不是失败：success:true，退出码 0。"""
    monkeypatch.chdir(tmp_path)
    df = pd.DataFrame({"id": [1, 2], "v": [10, 20]})
    _seed_dataframe("src_df", df.copy())
    _seed_dataframe("tgt_df", df.copy())

    result = observe_mod.cmd_compare({}, "src_df", "tgt_df", ["id"])

    assert result is None
    payload = json.loads(capsys.readouterr().out)
    assert payload["success"] is True


# ---------------------------------------------------------------------------
# observe explore：成功 / 失败 / 空结果
# ---------------------------------------------------------------------------


def test_observe_explore_success_exits_zero(tmp_path, monkeypatch, capsys):
    """关系探查成功退出码 0，stdout 输出 success JSON。"""
    monkeypatch.chdir(tmp_path)
    _seed_dataframe("t1", pd.DataFrame({"id": [1, 2], "ref": [10, 20]}))
    _seed_dataframe("t2", pd.DataFrame({"code": [10, 20], "label": ["a", "b"]}))

    result = observe_mod.cmd_explore({}, ["t1", "t2"])

    assert result is None
    payload = json.loads(capsys.readouterr().out)
    assert payload["success"] is True
    assert "foreign_keys" in payload and "column_similarities" in payload


def test_observe_explore_missing_dataframe_exits_one(tmp_path, monkeypatch, capsys):
    """指定 DataFrame 不存在退出码 1；stdout 保留 success:false JSON。"""
    monkeypatch.chdir(tmp_path)
    _seed_dataframe("t1", pd.DataFrame({"id": [1]}))

    with pytest.raises(SystemExit) as exc:
        observe_mod.cmd_explore({}, ["t1", "missing"])

    assert exc.value.code == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["success"] is False
    assert "❌" in captured.err
    assert "❌" not in captured.out


def test_observe_explore_empty_result_exits_zero(tmp_path, monkeypatch, capsys):
    """无任何 DataFrame（空结果）不是失败：success:true 空关系，退出码 0。"""
    monkeypatch.chdir(tmp_path)

    result = observe_mod.cmd_explore({}, None)

    assert result is None
    payload = json.loads(capsys.readouterr().out)
    assert payload["success"] is True
    assert payload["foreign_keys"] == []
    assert payload["column_similarities"] == []


# ---------------------------------------------------------------------------
# observe chart：成功 / 失败 / 空结果
# ---------------------------------------------------------------------------


def test_observe_chart_success_exits_zero(tmp_path, monkeypatch, capsys):
    """图表生成成功退出码 0，输出 PNG 路径。"""
    monkeypatch.chdir(tmp_path)
    _seed_dataframe("t1", pd.DataFrame({"x": [1, 2], "y": [3, 4]}))

    result = observe_mod.cmd_chart({}, "t1", "bar", "x", "y", "out.png")

    assert result is None
    payload = json.loads(capsys.readouterr().out)
    assert payload["success"] is True
    assert (tmp_path / "out.png").exists()


def test_observe_chart_missing_dataframe_exits_one(tmp_path, monkeypatch, capsys):
    """DataFrame 不存在退出码 1；stdout 保留 success:false JSON。"""
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as exc:
        observe_mod.cmd_chart({}, "missing", "bar", "x", "y", "out.png")

    assert exc.value.code == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["success"] is False
    assert "❌" in captured.err


def test_observe_chart_missing_column_exits_one(tmp_path, monkeypatch, capsys):
    """render_chart 返回 success:false（列不存在）时 CLI 转退出码 1。"""
    monkeypatch.chdir(tmp_path)
    _seed_dataframe("t1", pd.DataFrame({"x": [1, 2], "y": [3, 4]}))

    with pytest.raises(SystemExit) as exc:
        observe_mod.cmd_chart({}, "t1", "bar", "x", "no_such_col", "out.png")

    assert exc.value.code == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["success"] is False
    assert "❌" in captured.err
    assert "❌" not in captured.out


def test_observe_chart_empty_result_exits_zero(tmp_path, monkeypatch, capsys):
    """0 行 DataFrame 渲染（空结果）不是失败：退出码 0。"""
    monkeypatch.chdir(tmp_path)
    _seed_dataframe(
        "t1",
        pd.DataFrame({"x": pd.Series([], dtype="int64"), "y": pd.Series([], dtype="int64")}),
    )

    result = observe_mod.cmd_chart({}, "t1", "bar", "x", "y", "out.png")

    assert result is None
    payload = json.loads(capsys.readouterr().out)
    assert payload["success"] is True


# ---------------------------------------------------------------------------
# observe show（info --name）：成功 / 失败 / 空结果
# ---------------------------------------------------------------------------


def test_observe_show_success_exits_zero(tmp_path, monkeypatch, capsys):
    """show 成功退出码 0，stdout 输出 schema + sample JSON。"""
    monkeypatch.chdir(tmp_path)
    _seed_dataframe("t1", pd.DataFrame({"x": [1, 2], "y": ["a", "b"]}))

    result = observe_mod.cmd_show({}, "t1", 10)

    assert result is None
    payload = json.loads(capsys.readouterr().out)
    assert payload["success"] is True
    assert payload["rows"] == 2
    assert len(payload["schema"]) == 2


def test_observe_show_missing_dataframe_exits_one(tmp_path, monkeypatch, capsys):
    """DataFrame 不存在退出码 1；stdout 保留 success:false JSON。"""
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as exc:
        observe_mod.cmd_show({}, "missing", 10)

    assert exc.value.code == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["success"] is False
    assert "❌" in captured.err
    assert "❌" not in captured.out


def test_observe_show_negative_rows_exits_one(tmp_path, monkeypatch, capsys):
    """样本行数为负（参数非法）退出码 1。"""
    monkeypatch.chdir(tmp_path)
    _seed_dataframe("t1", pd.DataFrame({"x": [1, 2]}))

    with pytest.raises(SystemExit) as exc:
        observe_mod.cmd_show({}, "t1", -1)

    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "❌" in captured.err
    assert "样本行数" in captured.err


def test_observe_show_empty_result_exits_zero(tmp_path, monkeypatch, capsys):
    """0 行 DataFrame（空结果）不是失败：success:true sample:[]，退出码 0。"""
    monkeypatch.chdir(tmp_path)
    _seed_dataframe(
        "t1", pd.DataFrame({"x": pd.Series([], dtype="int64")})
    )

    result = observe_mod.cmd_show({}, "t1", 10)

    assert result is None
    payload = json.loads(capsys.readouterr().out)
    assert payload["success"] is True
    assert payload["rows"] == 0
    assert payload["sample"] == []


# ---------------------------------------------------------------------------
# observe release：成功 / 失败 / 空结果
# ---------------------------------------------------------------------------


def test_observe_release_success_exits_zero(tmp_path, monkeypatch, capsys):
    """释放成功退出码 0，stdout 输出 success JSON。"""
    monkeypatch.chdir(tmp_path)
    _seed_dataframe("t1", pd.DataFrame({"x": [1]}))

    result = observe_mod.cmd_release({}, "t1", False)

    assert result is None
    payload = json.loads(capsys.readouterr().out)
    assert payload["success"] is True


def test_observe_release_missing_dataframe_exits_one(tmp_path, monkeypatch, capsys):
    """释放不存在的 DataFrame 退出码 1；stdout 保留 success:false JSON。"""
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as exc:
        observe_mod.cmd_release({}, "missing", False)

    assert exc.value.code == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["success"] is False
    assert "❌" in captured.err
    assert "❌" not in captured.out


def test_observe_release_all_empty_exits_zero(tmp_path, monkeypatch, capsys):
    """--all 且无 DataFrame（空结果）不是失败：released:[]，退出码 0。"""
    monkeypatch.chdir(tmp_path)

    result = observe_mod.cmd_release({}, None, True)

    assert result is None
    payload = json.loads(capsys.readouterr().out)
    assert payload["success"] is True
    assert payload["released"] == []
    assert payload["count"] == 0


def test_observe_release_conflicting_flags_exits_one(tmp_path, monkeypatch, capsys):
    """--name 与 --all 同时指定（参数非法）退出码 1。"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        observe_mod,
        "ConfigManager",
        lambda: StubConfigManager({"datasources": {"d1": {"url": "duckdb://x"}}}),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["observe", "release", "--name", "t1", "--all"],
    )

    with pytest.raises(SystemExit) as exc:
        observe_mod.observe()

    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "❌" in captured.err
    assert "同时指定" in captured.err


# ---------------------------------------------------------------------------
# observe visualize-relations：成功 / 失败 / 空结果
# ---------------------------------------------------------------------------


def _write_relations(tmp_path, content: str):
    """写入关系 JSON 文件并返回路径"""
    path = tmp_path / "relations.json"
    path.write_text(content, encoding="utf-8")
    return str(path)


def test_observe_visualize_success_exits_zero(tmp_path, monkeypatch, capsys):
    """关系可视化成功退出码 0，stdout 输出 nodes/edges JSON。"""
    monkeypatch.chdir(tmp_path)
    rel_file = _write_relations(
        tmp_path,
        json.dumps(
            [
                {
                    "source_table": "a",
                    "source_column": "id",
                    "target_table": "b",
                    "target_column": "aid",
                    "confidence": 0.9,
                }
            ]
        ),
    )

    result = observe_mod.cmd_visualize({}, rel_file)

    assert result is None
    payload = json.loads(capsys.readouterr().out)
    assert payload["success"] is True
    assert len(payload["nodes"]) == 2
    assert len(payload["edges"]) == 1


@pytest.mark.parametrize(
    "content, use_missing_file",
    [("{bad json", False), (None, True)],
    ids=["bad-json", "missing-file"],
)
def test_observe_visualize_failure_exits_one(
    tmp_path, monkeypatch, capsys, content, use_missing_file
):
    """关系文件不存在 / JSON 解析失败退出码 1；stdout 保留 success:false JSON。"""
    monkeypatch.chdir(tmp_path)
    if use_missing_file:
        rel_file = str(tmp_path / "no_such.json")
    else:
        rel_file = _write_relations(tmp_path, content)

    with pytest.raises(SystemExit) as exc:
        observe_mod.cmd_visualize({}, rel_file)

    assert exc.value.code == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["success"] is False
    assert "❌" in captured.err
    assert "❌" not in captured.out


def test_observe_visualize_empty_result_exits_zero(tmp_path, monkeypatch, capsys):
    """空关系列表（空结果）不是失败：nodes:[] edges:[]，退出码 0。"""
    monkeypatch.chdir(tmp_path)
    rel_file = _write_relations(tmp_path, "[]")

    result = observe_mod.cmd_visualize({}, rel_file)

    assert result is None
    payload = json.loads(capsys.readouterr().out)
    assert payload["success"] is True
    assert payload["nodes"] == []
    assert payload["edges"] == []


# ---------------------------------------------------------------------------
# observe info：成功 / 失败 / 空结果
# ---------------------------------------------------------------------------


def _info_args(**overrides) -> argparse.Namespace:
    """构造 cmd_info 所需的 argparse.Namespace"""
    defaults = {"datasource": False, "df": False, "name": None, "rows": 10}
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _stub_db_manager(datasources: dict[str, str]):
    """打桩 get_db_manager：仅提供 list_datasources 所需的 _datasources"""
    return SimpleNamespace(
        _datasources={name: SimpleNamespace(url=url) for name, url in datasources.items()}
    )


def test_observe_info_overview_success_exits_zero(tmp_path, monkeypatch, capsys):
    """info 概览成功退出码 0，输出 datasources + dataframes。"""
    monkeypatch.chdir(tmp_path)
    _seed_dataframe("t1", pd.DataFrame({"x": [1]}))
    monkeypatch.setattr(
        observe_mod,
        "get_db_manager",
        lambda config: _stub_db_manager({"d1": "duckdb://x"}),
    )

    result = observe_mod.cmd_info({}, _info_args())

    assert result is None
    payload = json.loads(capsys.readouterr().out)
    assert payload["datasources"] == ["d1"]
    assert len(payload["dataframes"]["dataframes"]) == 1


def test_observe_info_missing_dataframe_exits_one(tmp_path, monkeypatch, capsys):
    """info --name 指定 DataFrame 不存在退出码 1。"""
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as exc:
        observe_mod.cmd_info({}, _info_args(name="missing"))

    assert exc.value.code == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["success"] is False
    assert "❌" in captured.err


def test_observe_info_empty_result_exits_zero(tmp_path, monkeypatch, capsys):
    """空概览（无数据源、无 DataFrame）不是失败：退出码 0。"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        observe_mod,
        "get_db_manager",
        lambda config: _stub_db_manager({}),
    )

    result = observe_mod.cmd_info({}, _info_args())

    assert result is None
    payload = json.loads(capsys.readouterr().out)
    assert payload["datasources"] == []
    assert payload["dataframes"]["dataframes"] == []
