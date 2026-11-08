"""口径冲突检测比对逻辑测试（govio.metadata.caliber 纯模块）。

覆盖：consistent / conflict（聚合签名、时间窗、计量列）/ unknown（弱证据降级）/
多指标 / 坏输入（空 SQL）/ 图读取（NetworkX 与 Cypher 后端打桩）。
判定纪律：宁可 unknown 不误报 conflict。
"""

from types import SimpleNamespace

import networkx as nx

from govio.metadata.caliber import (
    CaliberMetric,
    MetricColumn,
    check_caliber,
    compare_metric,
    extract_sql_facts,
    read_metric_caliber,
)


def _metric(**kwargs) -> CaliberMetric:
    defaults = {"id": "ME00000001", "code": "m_sales", "name": "销售额"}
    defaults.update(kwargs)
    return CaliberMetric(**defaults)


# ---------------------------------------------------------------------------
# consistent
# ---------------------------------------------------------------------------


def test_consistent_aggregation_and_columns():
    metric = _metric(
        formula="SUM(amount)",
        columns=[MetricColumn("amount", "measure")],
    )
    result = check_caliber([metric], "SELECT SUM(amount) FROM orders")
    assert result["results"][0]["status"] == "consistent"
    assert result["results"][0]["differences"] == []
    assert result["results"][0]["metric"] == {"id": "ME00000001", "code": "m_sales", "name": "销售额"}


def test_consistent_time_window_match():
    metric = _metric(
        formula="SUM(amount)",
        time_scope="近30天",
        columns=[MetricColumn("amount", "measure")],
    )
    sql = "SELECT SUM(amount) FROM orders WHERE dt >= CURRENT_DATE - INTERVAL '30' DAY"
    assert check_caliber([metric], sql)["results"][0]["status"] == "consistent"


def test_consistent_exclusion_with_negation():
    metric = _metric(
        business_definition="销售金额合计，剔除 refund 订单",
        columns=[MetricColumn("amount", "measure")],
    )
    sql = "SELECT SUM(amount) FROM orders WHERE order_type != 'refund'"
    assert check_caliber([metric], sql)["results"][0]["status"] == "consistent"


def test_consistent_parameterized_period_no_window_expectation():
    # 「统计周期内」是参数化表述，不产出时间窗期望；SQL 自选窗口不算冲突
    metric = _metric(
        business_definition="统计周期内所有订单的销售金额（amount）合计",
    )
    sql = "SELECT SUM(amount) FROM orders WHERE dt >= CURRENT_DATE - INTERVAL '7' DAY"
    assert check_caliber([metric], sql)["results"][0]["status"] == "consistent"


# ---------------------------------------------------------------------------
# conflict（确定性证据）
# ---------------------------------------------------------------------------


def test_conflict_aggregation_function_mismatch():
    metric = _metric(formula="SUM(amount)", columns=[MetricColumn("amount", "measure")])
    result = check_caliber([metric], "SELECT AVG(amount) FROM orders")["results"][0]
    assert result["status"] == "conflict"
    aspects = {d["aspect"] for d in result["differences"]}
    assert "aggregation" in aspects
    diff = next(d for d in result["differences"] if d["aspect"] == "aggregation")
    assert diff["expected"] == "SUM(amount)"
    assert diff["found"] == "AVG(amount)"


def test_conflict_distinct_mismatch():
    metric = _metric(formula="COUNT(DISTINCT user_id)")
    result = check_caliber([metric], "SELECT COUNT(user_id) FROM events")["results"][0]
    assert result["status"] == "conflict"


def test_conflict_time_window_mismatch():
    metric = _metric(
        formula="SUM(amount)",
        time_scope="近30天",
        columns=[MetricColumn("amount", "measure")],
    )
    sql = "SELECT SUM(amount) FROM orders WHERE dt >= CURRENT_DATE - INTERVAL '7' DAY"
    result = check_caliber([metric], sql)["results"][0]
    assert result["status"] == "conflict"
    diff = next(d for d in result["differences"] if d["aspect"] == "time_scope")
    assert "30" in diff["expected"] and "7" in diff["found"]


def test_conflict_measure_column_missing():
    metric = _metric(formula="SUM(amount)", columns=[MetricColumn("amount", "measure")])
    result = check_caliber([metric], "SELECT SUM(price) FROM orders")["results"][0]
    assert result["status"] == "conflict"
    diff = next(d for d in result["differences"] if d["aspect"] == "column")
    assert "amount" in diff["expected"]


def test_conflict_measure_column_missing_from_refers_column():
    metric = _metric(columns=[MetricColumn("order_id", "measure")])
    result = check_caliber([metric], "SELECT COUNT(*) FROM orders")["results"][0]
    # COUNT(*) 计数对象无法确认 → 存疑 unknown，不误报 conflict
    assert result["status"] == "unknown"
    result2 = check_caliber([metric], "SELECT SUM(price) FROM orders")["results"][0]
    assert result2["status"] == "conflict"


# ---------------------------------------------------------------------------
# unknown（弱证据/无法判定）
# ---------------------------------------------------------------------------


def test_unknown_prose_aggregation_hint_mismatch():
    # 中文语义提示（合计→SUM）不一致只给 unknown，不判 conflict
    metric = _metric(business_definition="销售金额合计")
    result = check_caliber([metric], "SELECT AVG(amount) FROM orders")["results"][0]
    assert result["status"] == "unknown"
    assert all(d["aspect"] == "aggregation" for d in result["differences"])


def test_unknown_window_absent_in_sql():
    metric = _metric(formula="SUM(amount)", time_scope="近30天")
    result = check_caliber([metric], "SELECT SUM(amount) FROM orders")["results"][0]
    assert result["status"] == "unknown"


def test_unknown_exclusion_term_absent():
    metric = _metric(business_definition="剔除退款订单的销售金额 SUM(amount)")
    result = check_caliber([metric], "SELECT SUM(amount) FROM orders WHERE status = 'paid'")["results"][0]
    assert result["status"] == "unknown"
    diff = next(d for d in result["differences"] if d["aspect"] == "filter")
    assert "退款订单" in diff["expected"]


def test_unknown_exclusion_term_present_without_negation():
    metric = _metric(business_definition="剔除 refund 订单的销售金额 SUM(amount)")
    result = check_caliber(
        [metric], "SELECT SUM(amount) FROM orders WHERE order_type = 'refund'"
    )["results"][0]
    assert result["status"] == "unknown"
    diff = next(d for d in result["differences"] if d["aspect"] == "filter")
    assert "否定算子" in diff["found"]


def test_unknown_count_star_caveat():
    metric = _metric(
        business_definition="统计周期内订单笔数（COUNT(order_id)）",
        columns=[MetricColumn("order_id", "measure")],
    )
    result = check_caliber([metric], "SELECT COUNT(*) FROM orders")["results"][0]
    assert result["status"] == "unknown"


def test_unknown_empty_sql():
    result = check_caliber([_metric()], "   ")["results"][0]
    assert result["status"] == "unknown"
    assert result["differences"][0]["aspect"] == "sql"


def test_unknown_non_select_sql():
    result = check_caliber([_metric()], "-- 注释而已\n")["results"][0]
    assert result["status"] == "unknown"


# ---------------------------------------------------------------------------
# 多指标
# ---------------------------------------------------------------------------


def test_multi_metric_mixed_statuses():
    ok = _metric(code="m_ok", formula="SUM(amount)", columns=[MetricColumn("amount", "measure")])
    bad = _metric(code="m_bad", formula="SUM(price)")
    sql = "SELECT SUM(amount) FROM orders"
    results = check_caliber([ok, bad], sql)["results"]
    assert [r["metric"]["code"] for r in results] == ["m_ok", "m_bad"]
    assert results[0]["status"] == "consistent"
    assert results[1]["status"] == "conflict"  # formula 计量列 price 未在 SQL 中引用


# ---------------------------------------------------------------------------
# SQL 事实提取
# ---------------------------------------------------------------------------


def test_extract_sql_facts_ignores_literals_and_comments():
    facts = extract_sql_facts(
        "SELECT SUM(amount) -- SUM(fake)\nFROM orders WHERE status = 'not_a_col' /* x */"
    )
    assert any(a.func == "SUM" and a.arg == "amount" for a in facts.aggs)
    assert "not_a_col" not in facts.referenced_columns
    assert "fake" not in facts.referenced_columns
    assert "amount" in facts.referenced_columns


def test_extract_sql_facts_where_and_window():
    facts = extract_sql_facts(
        "SELECT a FROM t WHERE dt >= CURRENT_DATE - INTERVAL '7' DAY AND status = 'paid' GROUP BY a"
    )
    assert facts.window_days == {7}
    assert "status" in facts.where_text
    assert len(facts.where_segments) == 2


# ---------------------------------------------------------------------------
# 图读取（打桩，不连真实图库）
# ---------------------------------------------------------------------------


def test_read_metric_caliber_networkx():
    g = nx.MultiDiGraph()
    g.add_node("ME00000001", node_type="Metric", id="ME00000001", code="m_sales",
               name="销售额", formula="SUM(amount)", business_definition="合计",
               statistical_scope=None, time_scope="近30天")
    g.add_node("CO00000001", node_type="Col", column="ds.main.orders.amount", column_name="amount")
    g.add_edge("ME00000001", "CO00000001", edge_type="REFERS_COLUMN", role="measure")
    graph = SimpleNamespace(G=g)

    by_code = read_metric_caliber(graph, "m_sales")
    assert by_code is not None
    assert by_code.id == "ME00000001"
    assert by_code.columns == [MetricColumn("amount", "measure")]

    by_id = read_metric_caliber(graph, "ME00000001")
    assert by_id is not None and by_id.code == "m_sales"

    assert read_metric_caliber(graph, "m_missing") is None


def test_read_metric_caliber_cypher():
    metric_rows = [["ME00000002", "m_orders", "订单数", "COUNT(order_id)",
                    "订单笔数", None, None]]
    col_rows = [["ME00000002", "m_orders", "order_id", "ds.main.orders.order_id", "measure"]]

    class StubGraph:
        def query(self, query: str, params: dict | None = None):
            # 回归护栏（BUGFIX-caliber-cypher）：属性名必须取 schema.md（id/code）；
            # 把 CSV 装载列名（如 `:ID(Metric)`）写进 Cypher 会触发真实图库 Binder exception。
            assert ":ID(" not in query, f"Cypher 引用了非 schema.md 属性: {query}"
            if "REFERS_COLUMN" in query:
                # `column` 是保留字，必须反引号（真实图库 Parser exception 回归护栏）
                assert "c.`column`" in query, f"Col.column 未反引号: {query}"
                return col_rows
            return metric_rows

    metric = read_metric_caliber(StubGraph(), "m_orders")
    assert metric is not None
    assert metric.formula == "COUNT(order_id)"
    assert metric.columns == [MetricColumn("order_id", "measure")]
    assert read_metric_caliber(StubGraph(), "nope") is None


# ---------------------------------------------------------------------------
# compare_metric 输出形态
# ---------------------------------------------------------------------------


def test_output_shape_omits_missing_metric_keys():
    metric = CaliberMetric(code="m_only", formula="SUM(x)")
    result = compare_metric(metric, extract_sql_facts("SELECT SUM(x) FROM t"))
    assert result["metric"] == {"code": "m_only"}
    assert set(result.keys()) == {"metric", "status", "differences"}
    for d in result["differences"]:
        assert set(d.keys()) <= {"aspect", "expected", "found", "detail"}
