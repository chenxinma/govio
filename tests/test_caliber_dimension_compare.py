"""口径维度限定实质比对测试（X19-W5a：PD-2 定调「计算表达式 + 维度限定条件」）。

覆盖：SQL 维度限定三元组解析（列/算子/值集合）、指标侧维度声明对齐
（statistical_scope 文本 + DIMENSION_USED/values_example + REFERS_COLUMN role）、
consistent/conflict/unknown 三态判定矩阵、判定纪律「宁可 unknown 不误报」、
PD-2 补充的 8 档回归样本（s1-s4 / t1-t4）、图读取维度声明与降级、输出契约不变。
"""

from types import SimpleNamespace

import networkx as nx

from govio.metadata.caliber import (
    CaliberMetric,
    DimensionDecl,
    MetricColumn,
    _dim_requirements,
    check_caliber,
    extract_sql_facts,
    read_metric_caliber,
)

COUNTRY_DIM = DimensionDecl(
    code="country",
    name="国家",
    usage_type="filter",
    values_example="中国、美国、德国",
)
CHANNEL_DIM = DimensionDecl(
    code="channel",
    name="渠道",
    usage_type="filter",
    values_example="online,offline,mail order",
)


def _metric(**kwargs) -> CaliberMetric:
    defaults = {
        "id": "ME00000001",
        "code": "m_revenue",
        "name": "销售额",
        "formula": "sum(amount)",
        "columns": [
            MetricColumn("amount", "measure"),
            MetricColumn("country_id", "dimension_ref"),
            MetricColumn("channel_id", "dimension_ref"),
        ],
    }
    defaults.update(kwargs)
    return CaliberMetric(**defaults)


def _verdict(metric: CaliberMetric, sql: str) -> str:
    return check_caliber([metric], sql)["results"][0]["status"]


def _filter_details(result: dict) -> list[str]:
    return [d["detail"] for d in result["differences"] if d["aspect"] == "filter"]


LIMITED = _metric(statistical_scope="仅统计中国的销售额", dimensions=[COUNTRY_DIM])


# ---------------------------------------------------------------------------
# SQL 维度限定三元组解析
# ---------------------------------------------------------------------------


def _triples(sql: str):
    return {t.column: t for t in extract_sql_facts(sql).dim_predicates}


def test_triple_equality_in_like_and_range():
    sql = (
        "SELECT SUM(amount) FROM t WHERE channel = 'online' "
        "AND country IN ('CN', 'DE') AND note LIKE 'abc%' AND qty > 10"
    )
    triples = _triples(sql)
    assert triples["channel"].op == "="
    assert triples["channel"].values == frozenset({"online"})
    assert triples["country"].op == "IN"
    assert triples["country"].values == frozenset({"cn", "de"})
    assert triples["note"].op == "LIKE"
    # 范围算子只说明「有限定」，取值不可枚举 → 不产出值集合（不猜测）
    assert triples["qty"].op == ">"
    assert triples["qty"].values == frozenset()


def test_triple_column_is_canonicalized_like_expression():
    # 与 `_canonical_expr` 同口径：去表/库限定、小写
    triples = _triples(
        "SELECT SUM(f.amount) FROM fact_sales f WHERE f.channel = 'ONLINE'"
    )
    assert triples["channel"].column == "channel"
    assert triples["channel"].values == frozenset({"online"})


def test_triple_negation_is_not_positive_restriction():
    triples = _triples(
        "SELECT SUM(amount) FROM t WHERE channel != 'offline' "
        "AND country NOT IN ('US') AND region IS NOT NULL"
    )
    assert triples["channel"].op == "!=" and triples["channel"].positive is False
    assert triples["country"].op == "NOT IN" and triples["country"].positive is False
    assert triples["region"].op == "IS NOT NULL" and triples["region"].positive is False


def test_triple_is_null_is_positive_but_not_enumerable():
    triples = _triples("SELECT SUM(amount) FROM t WHERE channel IS NULL")
    assert triples["channel"].op == "IS NULL"
    assert triples["channel"].positive is True
    assert triples["channel"].values == frozenset()


def test_triple_subquery_and_parameter_values_are_not_enumerable():
    triples = _triples(
        "SELECT SUM(amount) FROM t WHERE country_id IN (SELECT id FROM dim_country) "
        "AND channel = :ch"
    )
    assert triples["country_id"].values == frozenset()
    # 参数占位不是字面量：整个谓词不作为维度限定
    assert "channel" not in triples or triples["channel"].values == frozenset()


def test_triple_join_equality_in_where_yields_no_value():
    # `f.order_id = o.order_id` 是连接条件，不是取值限定
    triples = _triples(
        "SELECT SUM(f.amount) FROM f JOIN o ON 1 = 1 WHERE f.order_id = o.order_id"
    )
    assert triples["order_id"].values == frozenset()


def test_triple_between_and_is_not_split_as_two_predicates():
    triples = _triples(
        "SELECT SUM(amount) FROM t WHERE order_date BETWEEN '2023-01-01' "
        "AND '2023-12-31' AND channel = 'online'"
    )
    assert triples["order_date"].op == "BETWEEN"
    assert triples["order_date"].values == frozenset()
    assert triples["channel"].values == frozenset({"online"})


def test_triple_expands_parenthesized_or_group():
    sql = (
        "SELECT SUM(amount) FROM t WHERE (channel = 'online' OR channel = 'mail order') "
        "AND amount > 0"
    )
    found = {
        (t.column, t.op, tuple(sorted(t.values)))
        for t in extract_sql_facts(sql).dim_predicates
    }
    assert ("channel", "=", ("mail order",)) in found
    assert ("channel", "=", ("online",)) in found


def test_triple_drops_not_wrapped_group():
    # `NOT (...)` 包裹的取值无法还原为正/负限定：整体丢弃，不猜测
    facts = extract_sql_facts(
        "SELECT SUM(amount) FROM t WHERE NOT (channel = 'online')"
    )
    assert [p for p in facts.dim_predicates if p.positive] == []


def test_triple_unrecognized_shape_yields_nothing():
    assert (
        extract_sql_facts(
            "SELECT SUM(amount) FROM t WHERE upper(channel) = 'ONLINE'"
        ).dim_predicates
        == []
    )


def test_triple_from_subquery_where_counts_as_restriction():
    sql = (
        "SELECT SUM(x.amount) FROM (SELECT amount FROM fact_sales f "
        "WHERE f.country_name = '中国') x"
    )
    values = {v for p in extract_sql_facts(sql).dim_predicates for v in p.values}
    assert "中国" in values


def test_group_by_columns_and_positional_group_by():
    named = extract_sql_facts(
        "SELECT c.country_name, SUM(f.amount) FROM fact_sales f "
        "JOIN dim_country c ON f.country_id = c.country_id "
        "GROUP BY c.country_name HAVING SUM(f.amount) > 1"
    )
    assert "country_name" in named.group_by_columns
    assert (
        extract_sql_facts("SELECT SUM(amount) FROM t GROUP BY 1").group_by_columns
        == set()
    )


# ---------------------------------------------------------------------------
# 指标侧维度声明对齐
# ---------------------------------------------------------------------------


def test_requirement_from_filter_usage_and_enum_value_in_scope():
    reqs = _dim_requirements(LIMITED)
    assert len(reqs) == 1
    assert reqs[0].expected == frozenset({"中国"})
    assert reqs[0].excluded == frozenset()
    assert "country_id" in reqs[0].columns  # 维度 code 与 REFERS_COLUMN 列名字面匹配


def test_requirement_excluded_value_from_exclusion_phrase():
    metric = _metric(
        statistical_scope="销售额合计，剔除 offline 渠道", dimensions=[CHANNEL_DIM]
    )
    reqs = _dim_requirements(metric)
    assert reqs[0].excluded == frozenset({"offline"})
    assert reqs[0].expected == frozenset()


def test_group_usage_dimension_without_restriction_cue_is_not_a_requirement():
    # 「按国家分组」属查询自由度：不产声明，也不产存疑噪声
    metric = _metric(
        statistical_scope="按国家分组统计销售额",
        dimensions=[
            DimensionDecl(
                code="country",
                name="国家",
                usage_type="group",
                values_example="中国、美国、德国",
            )
        ],
    )
    assert _dim_requirements(metric) == []


def test_group_usage_dimension_with_cue_becomes_a_requirement():
    metric = _metric(
        statistical_scope="只选取中国的订单，按国家分组",
        dimensions=[
            DimensionDecl(
                code="country",
                name="国家",
                usage_type="group",
                values_example="中国、美国",
            )
        ],
    )
    reqs = _dim_requirements(metric)
    assert reqs and reqs[0].expected == frozenset({"中国"})


def test_no_enum_values_means_no_requirement():
    metric = _metric(
        statistical_scope="仅统计中国区",
        dimensions=[
            DimensionDecl(
                code="country", name="国家", usage_type="filter", values_example=None
            )
        ],
    )
    assert _dim_requirements(metric) == []


def test_short_numeric_enum_value_is_not_used_as_evidence():
    # 单字符码值（"1"/"0"）极易在中文描述里误命中，不作为限定证据
    metric = _metric(
        statistical_scope="仅统计 1 类订单",
        dimensions=[
            DimensionDecl(
                code="kind", name="类别", usage_type="filter", values_example="1,2,3"
            )
        ],
    )
    assert _dim_requirements(metric) == []


# ---------------------------------------------------------------------------
# 三态判定矩阵
# ---------------------------------------------------------------------------


def _result(metric, sql):
    return check_caliber([metric], sql)["results"][0]


def test_consistent_when_declared_value_is_restricted_in_sql():
    assert (
        _verdict(
            LIMITED,
            "SELECT SUM(f.amount) FROM fact_sales f WHERE f.country_id = '中国'",
        )
        == "consistent"
    )


def test_conflict_when_declared_restriction_has_no_landing():
    result = _result(LIMITED, "SELECT SUM(f.amount) FROM fact_sales f")
    assert result["status"] == "conflict"
    assert "实质矛盾" in _filter_details(result)[0]


def test_conflict_when_dimension_restricted_to_other_declared_value():
    result = _result(LIMITED, "SELECT SUM(f.amount) FROM f WHERE f.country_id = '德国'")
    assert result["status"] == "conflict"
    assert "未选取的声明取值" in _filter_details(result)[0]


def test_conflict_when_sql_picks_only_excluded_values():
    metric = _metric(
        statistical_scope="销售额合计，剔除 offline 渠道", dimensions=[CHANNEL_DIM]
    )
    result = _result(
        metric, "SELECT SUM(f.amount) FROM f WHERE f.channel_id = 'offline'"
    )
    assert result["status"] == "conflict"
    assert any("剔除" in d for d in _filter_details(result))


def test_consistent_when_exclusion_implemented_by_negation():
    metric = _metric(
        statistical_scope="销售额合计，剔除 offline 渠道", dimensions=[CHANNEL_DIM]
    )
    assert (
        _verdict(metric, "SELECT SUM(f.amount) FROM f WHERE f.channel_id != 'offline'")
        == "consistent"
    )


def test_unknown_when_sql_value_set_exceeds_declaration():
    result = _result(
        LIMITED, "SELECT SUM(f.amount) FROM f WHERE f.country_id IN ('中国','美国')"
    )
    assert result["status"] == "unknown"
    assert "超出指标声明" in _filter_details(result)[0]


def test_unknown_when_same_column_value_is_not_enumerable():
    # 同列限定但取值不可枚举（范围 / IS NULL）：无法比对，只给存疑
    assert (
        _verdict(LIMITED, "SELECT SUM(f.amount) FROM f WHERE f.country_id > 100")
        == "unknown"
    )
    assert (
        _verdict(LIMITED, "SELECT SUM(f.amount) FROM f WHERE f.country_id IS NULL")
        == "unknown"
    )


def test_unknown_when_declaration_may_be_implemented_by_join_condition():
    # WHERE 里的列间等值（连接条件）落在声明维度列上：取值不可枚举 → 存疑不误报
    sql = "SELECT SUM(f.amount) FROM fact_sales f WHERE f.country_id = c.country_id"
    assert _verdict(LIMITED, sql) == "unknown"


def test_conflict_when_same_column_has_only_non_null_check():
    # `IS NOT NULL` 不是取值限定：声明的「仅中国」确实无落点 → 实质矛盾
    result = _result(
        LIMITED, "SELECT SUM(f.amount) FROM f WHERE f.country_id IS NOT NULL"
    )
    assert result["status"] == "conflict"


def test_unknown_when_declaration_appears_only_in_group_by():
    result = _result(
        LIMITED, "SELECT c.country_name, SUM(f.amount) FROM f GROUP BY c.country_id"
    )
    assert result["status"] == "unknown"
    assert "分组而非限定" in _filter_details(result)[0]


def test_unknown_for_extra_restriction_not_declared_by_metric():
    result = _result(
        LIMITED,
        "SELECT SUM(f.amount) FROM f WHERE f.country_id = '中国' AND f.warehouse_id = 'W9'",
    )
    assert result["status"] == "unknown"
    assert "附加了指标未声明的维度限定" in _filter_details(result)[0]


def test_dimension_aspect_silent_without_any_declaration():
    metric = _metric(statistical_scope="全部订单的销售额合计")
    result = _result(
        metric, "SELECT SUM(f.amount) FROM f WHERE f.channel_id = 'online'"
    )
    assert result["status"] == "consistent"
    assert _filter_details(result) == []


def test_time_grain_conflict_when_declared_monthly_but_sql_yearly():
    metric = _metric(time_scope="按月统计")
    assert (
        _verdict(
            metric, "SELECT DATE_TRUNC('month', d) AS x, SUM(amount) FROM t GROUP BY 1"
        )
        == "consistent"
    )
    result = _result(metric, "SELECT YEAR(d) AS x, SUM(amount) FROM t GROUP BY 1")
    assert result["status"] == "conflict"
    assert any(d["aspect"] == "time_scope" for d in result["differences"])


def test_time_grain_silent_when_sql_has_no_explicit_grain():
    metric = _metric(time_scope="按月统计")
    assert _verdict(metric, "SELECT SUM(amount) FROM t") == "consistent"


# ---------------------------------------------------------------------------
# 判定纪律：宁可 unknown 不误报
# ---------------------------------------------------------------------------


def test_no_false_conflict_when_restriction_sits_on_unmappable_column():
    # SQL 限定了 region_code='EMEA'：无法归属到声明维度 → 存疑，不判冲突
    result = _result(
        LIMITED, "SELECT SUM(f.amount) FROM f WHERE r.region_code = 'EMEA'"
    )
    assert result["status"] == "unknown"


def test_no_false_conflict_when_restriction_is_implemented_in_subquery():
    sql = (
        "SELECT SUM(x.amount) FROM (SELECT amount FROM fact_sales f "
        "WHERE f.country_name = '中国') x"
    )
    assert _verdict(LIMITED, sql) == "consistent"


def test_conflict_needs_substantive_evidence_aspects_are_stable():
    allowed = {"aggregation", "filter", "time_scope", "column", "sql", "metric"}
    result = _result(
        LIMITED, "SELECT AVG(total_price) FROM f WHERE f.channel_id = 'online'"
    )
    assert {d["aspect"] for d in result["differences"]} <= allowed


# ---------------------------------------------------------------------------
# PD-2 补充的 8 档回归样本（s1-s4 / t1-t4）
# ---------------------------------------------------------------------------

CHOCOLATE_METRIC = CaliberMetric(
    id="ME00000123",
    code="m_sales_amount",
    name="销售额",
    formula="sum(amount)",
    business_definition="统计周期内所有订单的销售金额（amount）合计",
    statistical_scope="全部国家、渠道与商品的订单",
    time_scope="统计周期内",
    columns=[
        MetricColumn("amount", "measure"),
        MetricColumn("country_id", "dimension_ref"),
        MetricColumn("channel_id", "dimension_ref"),
        MetricColumn("product_id", "dimension_ref"),
        MetricColumn("order_date", "dimension_ref"),
    ],
    dimensions=[
        DimensionDecl(
            code="country",
            name="国家",
            usage_type="group",
            values_example="中国、美国、德国",
        ),
        DimensionDecl(
            code="channel",
            name="渠道",
            usage_type="group",
            values_example="online,offline,mail order",
        ),
    ],
)

# (档位, SQL 要点, 改动前判定（PD-2 补充实测）, 改动后预期)
PD2_SAMPLES = [
    (
        "s1",
        "乘法别名冒充口径",
        "conflict",
        "conflict",
        "SELECT country_id, SUM(price_per_box * boxes_shipped) AS sales_amount FROM fact_sales GROUP BY country_id",
    ),
    (
        "s2",
        "正确口径（ROUND 包装 + 国家分组）",
        "unknown",
        "consistent",
        "SELECT c.country_name, ROUND(SUM(f.amount), 2) AS revenue_amount FROM fact_sales f "
        "JOIN dim_country c ON f.country_id = c.country_id GROUP BY 1",
    ),
    (
        "s3",
        "不含 amount 的其它金额列",
        "conflict",
        "conflict",
        "SELECT SUM(total_price) AS sales_amount FROM fact_orders",
    ),
    (
        "s4",
        "AVG 取代 SUM",
        "conflict",
        "conflict",
        "SELECT AVG(amount) FROM fact_sales",
    ),
    (
        "t1",
        "AVG 改名 revenue_calc",
        "conflict",
        "conflict",
        "SELECT AVG(amount) AS revenue_calc FROM fact_sales",
    ),
    (
        "t2",
        "正确口径（同 s2 第二轮）",
        "unknown",
        "consistent",
        "SELECT ROUND(SUM(f.amount), 2) AS revenue_amount FROM fact_sales f",
    ),
    (
        "t3",
        "不含 amount（同 s3 第二轮）",
        "conflict",
        "conflict",
        "SELECT SUM(net_amount) AS revenue FROM fact_sales",
    ),
    (
        "t4",
        "算术修饰 amount",
        "conflict",
        "conflict",
        "SELECT ROUND(SUM(f.amount * (1 - f.discount_pct / 100)), 2) AS revenue_calc FROM fact_sales f",
    ),
]


def test_pd2_eight_samples_match_expectation():
    got = [
        (_tag, _verdict(CHOCOLATE_METRIC, _sql))
        for _tag, _note, _before, _after, _sql in PD2_SAMPLES
    ]
    assert got == [
        ("s1", "conflict"),
        ("s2", "consistent"),
        ("s3", "conflict"),
        ("s4", "conflict"),
        ("t1", "conflict"),
        ("t2", "consistent"),
        ("t3", "conflict"),
        ("t4", "conflict"),
    ]


def test_pd2_samples_after_are_declared_expectations():
    # 预期表本身入档：改动后判定 == PD-2 补充的预期（s2/t2 可达 consistent，t4 仍 conflict）
    for tag, note, before, after, sql in PD2_SAMPLES:
        assert note
        assert _verdict(CHOCOLATE_METRIC, sql) == after, tag
        assert before in {"conflict", "unknown", "consistent"}


def test_pd2_dimension_aspect_does_not_dilute_correct_caliber():
    # s2/t2 的正确口径不被维度限定维度稀释成 unknown（本指标未声明具体限定取值）
    for tag, _note, _before, _after, sql in PD2_SAMPLES:
        if tag in ("s2", "t2"):
            result = _result(CHOCOLATE_METRIC, sql)
            assert result["differences"] == []


# ---------------------------------------------------------------------------
# 图读取维度声明（NetworkX / Cypher 打桩）
# ---------------------------------------------------------------------------


def _networkx_graph() -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    g.add_node(
        "ME00000001",
        node_type="Metric",
        id="ME00000001",
        code="m_revenue",
        name="销售额",
        formula="sum(amount)",
        statistical_scope="仅统计中国的销售额",
    )
    g.add_node(
        "CO00000001",
        node_type="Col",
        column="ds.main.fact.amount",
        column_name="amount",
    )
    g.add_node(
        "CO00000002",
        node_type="Col",
        column="ds.main.fact.country_id",
        column_name="country_id",
    )
    g.add_node(
        "DI00000001",
        node_type="Dimension",
        code="country",
        name="国家",
        values_example="中国、美国、德国",
    )
    g.add_edge("ME00000001", "CO00000001", edge_type="REFERS_COLUMN", role="measure")
    g.add_edge(
        "ME00000001", "CO00000002", edge_type="REFERS_COLUMN", role="dimension_ref"
    )
    g.add_edge(
        "ME00000001", "DI00000001", edge_type="DIMENSION_USED", usage_type="filter"
    )
    return g


def test_read_metric_caliber_loads_dimension_declarations():
    metric = read_metric_caliber(SimpleNamespace(G=_networkx_graph()), "m_revenue")
    assert metric is not None
    assert metric.dimensions == [COUNTRY_DIM]
    # 图 -> 比对全链路：声明限定中国，SQL 无限定 → conflict
    assert (
        check_caliber([metric], "SELECT SUM(amount) FROM fact_sales")["results"][0][
            "status"
        ]
        == "conflict"
    )


def test_read_metric_caliber_cypher_dimensions():
    metric_rows = [
        ["ME00000002", "m_orders", "订单数", "COUNT(order_id)", "订单笔数", None, None]
    ]
    col_rows = [
        ["ME00000002", "m_orders", "order_id", "ds.main.orders.order_id", "measure"]
    ]
    dim_rows = [
        ["ME00000002", "m_orders", "channel", "渠道", "online,offline", "filter"],
        ["ME00000999", "m_other", "x", "无关", "a", "group"],
    ]

    class StubGraph:
        def query(self, query: str, params: dict | None = None):
            # 回归护栏：Cypher 只引用 schema.md 属性，保留字须反引号
            assert ":ID(" not in query, f"Cypher 引用了非 schema.md 属性: {query}"
            if "DIMENSION_USED" in query:
                return dim_rows
            if "REFERS_COLUMN" in query:
                assert "c.`column`" in query, f"Col.column 未反引号: {query}"
                return col_rows
            return metric_rows

    metric = read_metric_caliber(StubGraph(), "m_orders")
    assert metric is not None
    assert metric.dimensions == [
        DimensionDecl(
            code="channel",
            name="渠道",
            usage_type="filter",
            values_example="online,offline",
        )
    ]


def test_read_dimensions_failure_degrades_to_no_declaration():
    class BrokenGraph:
        def query(self, query: str, params: dict | None = None):
            if "DIMENSION_USED" in query:
                raise RuntimeError("Cannot find label Dimension")
            if "REFERS_COLUMN" in query:
                return [
                    [
                        "ME00000002",
                        "m_orders",
                        "order_id",
                        "ds.main.orders.order_id",
                        "measure",
                    ]
                ]
            return [
                [
                    "ME00000002",
                    "m_orders",
                    "订单数",
                    "COUNT(order_id)",
                    "订单笔数",
                    None,
                    None,
                ]
            ]

    metric = read_metric_caliber(BrokenGraph(), "m_orders")
    assert metric is not None
    assert metric.dimensions == []
    assert metric.formula == "COUNT(order_id)"


# ---------------------------------------------------------------------------
# 输出契约（map 侧消费）：三态取值与 differences 形状不变
# ---------------------------------------------------------------------------


def test_output_contract_unchanged_for_dimension_conflict():
    result = _result(LIMITED, "SELECT SUM(f.amount) FROM fact_sales f")
    assert set(result.keys()) == {"metric", "status", "differences"}
    assert result["status"] in {"consistent", "conflict", "unknown"}
    for diff in result["differences"]:
        assert set(diff.keys()) <= {"aspect", "expected", "found", "detail"}
        assert diff["aspect"] == "filter"
