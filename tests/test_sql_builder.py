"""sql_builder 核心逻辑单元测试"""

import pytest
from govio.core.sql_builder import build_metric_sql


def test_single_atomic_metric_with_dimensions():
    """单原子指标 + 维度分组：应生成 GROUP BY"""
    sql = build_metric_sql(
        metrics=[{
            "code": "bill_income_amt",
            "name": "当月账单收入",
            "type": "原子",
            "source_table": "dws.income_bill_monthly",
        }],
        dimensions=["sales_dept"],
        filters={"report_ym": "2026-05"},
    )
    assert "WITH" in sql
    assert "atomic_income_bill_monthly AS" in sql
    assert "SUM(bill_income_amt) AS bill_income_amt" in sql
    assert "GROUP BY sales_dept" in sql
    assert "report_ym = '2026-05'" in sql
    assert "LIMIT 100" in sql


def test_single_atomic_metric_without_dimensions():
    """单原子指标无维度：无 GROUP BY，直接 SELECT"""
    sql = build_metric_sql(
        metrics=[{
            "code": "bill_income_amt",
            "name": "当月账单收入",
            "type": "原子",
            "source_table": "dws.income_bill_monthly",
        }],
        filters={"report_ym": "2026-05"},
    )
    assert "bill_income_amt AS metric_value" in sql
    assert "GROUP BY" not in sql


def test_multiple_atomic_metrics_same_table():
    """多原子指标同表：合并为单个 atomic_* CTE"""
    sql = build_metric_sql(
        metrics=[
            {"code": "bill_income_amt", "name": "账单收入", "type": "原子", "source_table": "dws.income_bill_monthly"},
            {"code": "signed_amt", "name": "签约额", "type": "原子", "source_table": "dws.income_bill_monthly"},
        ],
        filters={"report_ym": "2026-05"},
    )
    assert sql.count("atomic_income_bill_monthly AS") == 1


def test_multiple_atomic_metrics_different_tables():
    """多原子指标异表：多个 atomic_* CTE，UNION ALL"""
    sql = build_metric_sql(
        metrics=[
            {"code": "bill_income_amt", "name": "账单收入", "type": "原子", "source_table": "dws.income_bill_monthly"},
            {"code": "signed_amt", "name": "签约额", "type": "原子", "source_table": "dws.signed_monthly"},
        ],
        filters={"report_ym": "2026-05"},
    )
    assert "atomic_income_bill_monthly AS" in sql
    assert "atomic_signed_monthly AS" in sql
    assert "UNION ALL" in sql


def test_derived_metric():
    """派生指标：生成 derived_* CTE，公式解析为 t.引用"""
    sql = build_metric_sql(
        metrics=[
            {"code": "signed_amt", "name": "签约额", "type": "原子", "source_table": "dws.signed_monthly"},
            {"code": "bill_income_amt", "name": "账单收入", "type": "原子", "source_table": "dws.income_bill_monthly"},
            {"code": "book_to_bill", "name": "签约覆盖率", "type": "派生", "formula": "signed_amt / bill_income_amt"},
        ],
        filters={"report_ym": "2026-05"},
    )
    assert "derived_book_to_bill AS" in sql
    assert "t.signed_amt / t.bill_income_amt" in sql


def test_report_ym_required():
    """report_ym 缺失：抛 ValueError"""
    with pytest.raises(ValueError, match="report_ym"):
        build_metric_sql(
            metrics=[{
                "code": "bill_income_amt",
                "name": "当月账单收入",
                "type": "原子",
                "source_table": "dws.income_bill_monthly",
            }],
            filters={},
        )


def test_cte_refs_injection():
    """cte_refs：CTE 前置注入"""
    sql = build_metric_sql(
        metrics=[{
            "code": "bill_income_amt",
            "name": "当月账单收入",
            "type": "原子",
            "source_table": "dws.income_bill_monthly",
        }],
        filters={"report_ym": "2026-05"},
        cte_refs={"existing_df": "SELECT * FROM tmp"},
    )
    assert "existing_df AS (SELECT * FROM tmp)" in sql


def test_order_by_and_limit():
    """order_by + limit：尾部子句"""
    sql = build_metric_sql(
        metrics=[{
            "code": "bill_income_amt",
            "name": "当月账单收入",
            "type": "原子",
            "source_table": "dws.income_bill_monthly",
        }],
        filters={"report_ym": "2026-05"},
        order_by="metric_value DESC",
        limit=10,
    )
    assert "ORDER BY metric_value DESC" in sql
    assert "LIMIT 10" in sql


def test_empty_metrics_raises():
    """空 metrics：抛 ValueError"""
    with pytest.raises(ValueError, match="至少需要一个指标"):
        build_metric_sql(metrics=[])


# ---------- CLI 集成测试 ----------

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def _run_cli(*args, stdin_text=None):
    """运行 govio-cli sql build，返回 (returncode, stdout, stderr)"""
    cmd = [sys.executable, "-m", "govio.cli"] + list(args)
    proc = subprocess.run(
        cmd,
        input=stdin_text,
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _write_query_json(path: Path, **overrides):
    """写入测试用 JSON 规格"""
    req = {
        "metrics": [{
            "code": "bill_income_amt",
            "name": "当月账单收入",
            "type": "原子",
            "source_table": "dws.income_bill_monthly",
        }],
        "filters": {"report_ym": "2026-05"},
    }
    req.update(overrides)
    path.write_text(json.dumps(req, ensure_ascii=False), encoding="utf-8")
    return req


def test_cli_build_file_to_stdout():
    """-f 文件输入 → stdout 输出 SQL"""
    with tempfile.TemporaryDirectory() as d:
        qpath = Path(d) / "q.json"
        _write_query_json(qpath)
        rc, out, err = _run_cli("sql", "build", "-f", str(qpath))
        assert rc == 0, f"stderr: {err}"
        assert "WITH" in out
        assert "atomic_income_bill_monthly" in out
        assert out.endswith("\n")


def test_cli_build_file_to_output_file():
    """-f 文件 + -o 输出文件"""
    with tempfile.TemporaryDirectory() as d:
        qpath = Path(d) / "q.json"
        opath = Path(d) / "out.sql"
        _write_query_json(qpath)
        rc, out, err = _run_cli("sql", "build", "-f", str(qpath), "-o", str(opath))
        assert rc == 0, f"stderr: {err}"
        assert out == ""
        content = opath.read_text(encoding="utf-8")
        assert "WITH" in content
        assert content.endswith("\n")


def test_cli_build_stdin_input():
    """stdin 管道输入"""
    req = {
        "metrics": [{
            "code": "bill_income_amt",
            "name": "当月账单收入",
            "type": "原子",
            "source_table": "dws.income_bill_monthly",
        }],
        "filters": {"report_ym": "2026-05"},
    }
    rc, out, err = _run_cli("sql", "build", stdin_text=json.dumps(req))
    assert rc == 0, f"stderr: {err}"
    assert "WITH" in out


def test_cli_build_file_not_found():
    """文件不存在 → exit 1"""
    rc, out, err = _run_cli("sql", "build", "-f", "/nonexistent/q.json")
    assert rc == 1
    assert "文件不存在" in err


def test_cli_build_json_parse_error():
    """JSON 解析失败 → exit 1"""
    with tempfile.TemporaryDirectory() as d:
        qpath = Path(d) / "q.json"
        qpath.write_text("{invalid json", encoding="utf-8")
        rc, out, err = _run_cli("sql", "build", "-f", str(qpath))
        assert rc == 1
        assert "JSON 解析失败" in err


def test_cli_build_value_error_propagation():
    """build_metric_sql ValueError → exit 1"""
    with tempfile.TemporaryDirectory() as d:
        qpath = Path(d) / "q.json"
        req = {
            "metrics": [{
                "code": "bill_income_amt",
                "name": "当月账单收入",
                "type": "原子",
                "source_table": "dws.income_bill_monthly",
            }],
            "filters": {},
        }
        qpath.write_text(json.dumps(req), encoding="utf-8")
        rc, out, err = _run_cli("sql", "build", "-f", str(qpath))
        assert rc == 1
        assert "SQL 组装失败" in err
        assert "report_ym" in err
