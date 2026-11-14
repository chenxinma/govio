"""`govio-cli sql caliber` 命令测试（图读取打桩，不连真实图库）。

覆盖：--sql/--sql-file 输入、--metric 可重复、结构化 JSON 输出（stdout 单行、
ensure_ascii）、退出码契约（比对完成含 conflict/unknown 退出 0；缺 SQL/缺
--metric/文件不存在/配置错误退出 1）、未知指标降级、多指标混排。
"""

import json
import sys

import pytest

from govio.cli import sql as sql_mod
from govio.metadata.caliber import CaliberMetric, DimensionDecl, MetricColumn


class StubConfigManager:
    def __init__(self, config: dict | None):
        self._config = config

    def exists(self) -> bool:
        return self._config is not None

    def load(self) -> dict:
        return self._config or {}


class StubGraph:
    pass


KNOWN_METRIC = CaliberMetric(
    id="ME00000001",
    code="m_sales",
    name="销售额",
    formula="SUM(amount)",
    columns=[MetricColumn("amount", "measure")],
)


def _setup(monkeypatch, metric=KNOWN_METRIC, config=None):
    """打桩：配置 + 图工厂 + 图读取"""
    monkeypatch.setattr(
        sql_mod,
        "ConfigManager",
        lambda: StubConfigManager(
            config if config is not None else {"graph": {"backend": "networkx"}}
        ),
    )
    monkeypatch.setattr(
        "govio.core.graph_factory.GraphFactory.create",
        staticmethod(lambda cfg: StubGraph()),
    )
    monkeypatch.setattr(
        "govio.metadata.caliber.read_metric_caliber",
        lambda graph, ref: metric if ref == "m_sales" else None,
    )


def _run(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["govio-cli", *argv])
    sql_mod.sql()


# ---------------------------------------------------------------------------
# 成功路径：JSON 输出 + 退出码 0
# ---------------------------------------------------------------------------


def test_caliber_conflict_result_is_json_on_stdout(monkeypatch, capsys):
    _setup(monkeypatch)
    _run(
        monkeypatch,
        ["caliber", "--sql", "SELECT AVG(price) FROM orders", "--metric", "m_sales"],
    )
    out = capsys.readouterr().out
    payload = json.loads(out.strip().splitlines()[-1])
    assert list(payload.keys()) == ["results"]
    assert payload["results"][0]["metric"] == {
        "id": "ME00000001",
        "code": "m_sales",
        "name": "销售额",
    }
    assert payload["results"][0]["status"] == "conflict"
    aspects = {d["aspect"] for d in payload["results"][0]["differences"]}
    assert aspects == {"aggregation", "column"}


def test_caliber_consistent_metric_exits_zero(monkeypatch, capsys):
    _setup(monkeypatch)
    _run(
        monkeypatch,
        ["caliber", "--sql", "SELECT SUM(amount) FROM orders", "--metric", "m_sales"],
    )
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["results"][0]["status"] == "consistent"


def test_caliber_sql_file_input(monkeypatch, capsys, tmp_path):
    _setup(monkeypatch)
    sql_file = tmp_path / "q.sql"
    sql_file.write_text("SELECT SUM(amount) FROM orders", encoding="utf-8")
    _run(monkeypatch, ["caliber", "--sql-file", str(sql_file), "--metric", "m_sales"])
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["results"][0]["status"] == "consistent"


def test_caliber_sql_text_with_spaces_and_quotes_intact(monkeypatch, capsys):
    _setup(monkeypatch)
    sql_text = "SELECT SUM(amount) FROM orders WHERE name = 'a b  c'"
    _run(monkeypatch, ["caliber", "--sql", sql_text, "--metric", "m_sales"])
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["results"][0]["status"] == "consistent"


# ---------------------------------------------------------------------------
# 未知指标降级（exit 0，单指标 unknown）
# ---------------------------------------------------------------------------


def test_caliber_unknown_metric_degrades_to_unknown(monkeypatch, capsys):
    _setup(monkeypatch)
    _run(monkeypatch, ["caliber", "--sql", "SELECT 1 FROM t", "--metric", "m_nope"])
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    result = payload["results"][0]
    assert result["metric"] == {"code": "m_nope"}
    assert result["status"] == "unknown"
    assert result["differences"][0]["aspect"] == "metric"


def test_caliber_multi_metric_mixed(monkeypatch, capsys):
    _setup(monkeypatch)
    _run(
        monkeypatch,
        [
            "caliber",
            "--sql",
            "SELECT SUM(amount) FROM orders",
            "--metric",
            "m_sales",
            "--metric",
            "m_nope",
        ],
    )
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    results = payload["results"]
    assert len(results) == 2
    assert results[0]["metric"]["code"] == "m_sales"
    assert results[0]["status"] == "consistent"
    assert results[1]["status"] == "unknown"


# ---------------------------------------------------------------------------
# 维度限定实质比对（X19-W5a）：CLI 输出契约不变
# ---------------------------------------------------------------------------


DIMENSION_METRIC = CaliberMetric(
    id="ME00000009",
    code="m_sales",
    name="销售额",
    formula="sum(amount)",
    statistical_scope="仅统计中国的销售额",
    columns=[
        MetricColumn("amount", "measure"),
        MetricColumn("country_id", "dimension_ref"),
    ],
    dimensions=[
        DimensionDecl(
            code="country",
            name="国家",
            usage_type="filter",
            values_example="中国、美国、德国",
        )
    ],
)


def test_caliber_dimension_conflict_keeps_json_contract(monkeypatch, capsys):
    """声明限定「仅中国」而 SQL 无任何等价限定 -> conflict，JSON 形状不变"""
    _setup(monkeypatch, metric=DIMENSION_METRIC)
    _run(
        monkeypatch,
        [
            "caliber",
            "--sql",
            "SELECT SUM(amount) FROM fact_sales",
            "--metric",
            "m_sales",
        ],
    )
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    result = payload["results"][0]
    assert result["status"] == "conflict"
    assert {d["aspect"] for d in result["differences"]} == {"filter"}
    for diff in result["differences"]:
        assert set(diff.keys()) <= {"aspect", "expected", "found", "detail"}


def test_caliber_dimension_restriction_satisfied_is_consistent(monkeypatch, capsys):
    _setup(monkeypatch, metric=DIMENSION_METRIC)
    _run(
        monkeypatch,
        [
            "caliber",
            "--sql",
            "SELECT SUM(f.amount) FROM fact_sales f WHERE f.country_id = '中国'",
            "--metric",
            "m_sales",
        ],
    )
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["results"][0]["status"] == "consistent"
    assert payload["results"][0]["differences"] == []


# ---------------------------------------------------------------------------
# 坏输入：退出码 1，错误只落 stderr
# ---------------------------------------------------------------------------


def test_caliber_missing_sql_exits_one(monkeypatch, capsys):
    _setup(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, ["caliber", "--metric", "m_sales"])
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "缺少 SQL" in captured.err
    assert captured.out == ""


def test_caliber_empty_sql_exits_one(monkeypatch, capsys):
    _setup(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, ["caliber", "--sql", "   ", "--metric", "m_sales"])
    assert exc.value.code == 1
    assert "缺少 SQL" in capsys.readouterr().err


def test_caliber_missing_metric_exits_one(monkeypatch, capsys):
    _setup(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, ["caliber", "--sql", "SELECT 1"])
    assert exc.value.code == 1
    assert "--metric" in capsys.readouterr().err


def test_caliber_sql_file_not_found_exits_one(monkeypatch, capsys):
    _setup(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        _run(
            monkeypatch,
            ["caliber", "--sql-file", "no-such-file.sql", "--metric", "m_sales"],
        )
    assert exc.value.code == 1
    assert "文件不存在" in capsys.readouterr().err


def test_caliber_missing_config_exits_one(monkeypatch, capsys):
    _setup(monkeypatch, config={})
    monkeypatch.setattr(sql_mod, "ConfigManager", lambda: StubConfigManager(None))
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, ["caliber", "--sql", "SELECT 1", "--metric", "m_sales"])
    assert exc.value.code == 1
    assert "配置" in capsys.readouterr().err


def test_caliber_missing_backend_config_exits_one(monkeypatch, capsys):
    _setup(monkeypatch, config={"graph": {}})
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, ["caliber", "--sql", "SELECT 1", "--metric", "m_sales"])
    assert exc.value.code == 1
    assert "backend" in capsys.readouterr().err


def test_caliber_no_subcommand_prints_help(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["govio-cli"])
    with pytest.raises(SystemExit) as exc:
        sql_mod.sql()
    assert exc.value.code == 1
    assert "caliber" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 输出形态纪律
# ---------------------------------------------------------------------------


def test_caliber_stdout_is_pure_json(monkeypatch, capsys):
    """stdout 只有一行 JSON（无日志/无 ❌ 前缀），ensure_ascii 转义对控制台编码免疫"""
    _setup(monkeypatch)
    _run(
        monkeypatch,
        ["caliber", "--sql", "SELECT SUM(amount) FROM orders", "--metric", "m_sales"],
    )
    out = capsys.readouterr().out
    lines = out.strip().splitlines()
    assert len(lines) == 1
    assert out.isascii()
    json.loads(lines[0])
