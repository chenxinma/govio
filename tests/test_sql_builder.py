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
