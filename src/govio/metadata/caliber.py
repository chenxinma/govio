"""
govio.metadata.caliber
口径冲突检测（T2.3）：SQL 与图谱指标定义的最小维度比对。

纯 stdlib 模块（只依赖 re/dataclasses），不触图后端、不连库：
- `extract_sql_facts`：从 SQL 文本提取可比对事实（聚合调用/引用列/WHERE 条件/时间窗）；
- `check_caliber` / `compare_metric`：逐指标比对，输出结构化 JSON 形态
  `{results: [{metric, status, differences}]}`；
- `read_metric_caliber`：从图对象（NetworkXGraph / FalkorDBGraph / LadybugGraph）
  读取指标定义与 REFERS_COLUMN 边，图对象由调用方注入（GraphFactory），本模块不建连。

判定纪律：**宁可 unknown 不误报 conflict**。conflict 仅在证据确定时给出
（聚合签名不匹配 / 明确时间窗不一致 / 计量列完全缺失），语义比对不确定一律 unknown。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# SQL 事实提取
# ---------------------------------------------------------------------------

_COMMENT_RE = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)
_STRING_RE = re.compile(r"'(?:[^']|'')*'|\"[^\"]*\"")
_AGG_RE = re.compile(
    r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(\s*(DISTINCT\s+)?([^()]*)\)", re.IGNORECASE
)
_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
# SELECT 列表（到首个顶层 FROM 为止；无 FROM 则到结尾）——星号判定只看列表内
_SELECT_LIST_RE = re.compile(r"\bSELECT\b(.*?)(?=\bFROM\b|\Z)", re.IGNORECASE | re.DOTALL)
# `t.*` / `tbl.*` 全表展开
_TABLE_STAR_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\s*\.\s*\*")
# count(*) 内的星号不计入 has_star（计数对象不可确认由 has_count_star 表达）
_COUNT_STAR_RE = re.compile(r"\bCOUNT\s*\(\s*\*\s*\)", re.IGNORECASE)
_WHERE_END_RE = re.compile(
    r"\b(GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT|OFFSET|UNION|EXCEPT|INTERSECT)\b", re.IGNORECASE
)
# SQL 侧时间窗：`INTERVAL '7' DAY`、`INTERVAL 30 DAY`、`DATE_SUB(..., 7 DAY)` 等
_SQL_WINDOW_RE = re.compile(
    r"\b(\d+)\s*'?\s*(DAY|DAYS|WEEK|WEEKS|MONTH|MONTHS|YEAR|YEARS|HOUR|HOURS)\b", re.IGNORECASE
)
# 指标定义侧时间窗：「近30天」「最近7日」「过去3个月」「近1年」等
_DEF_WINDOW_RE = re.compile(r"(?:近|最近|过去|向前)\s*(\d+)\s*(天|日|周|个月|月|年)")
# 定义侧剔除类限定：「剔除退款订单」「不含内部测试用户」等
_EXCLUSION_RE = re.compile(
    r"(?:不含|不包括|剔除|排除|去除|过滤掉|不计)\s*掉?\s*([\u4e00-\u9fa5A-Za-z0-9_]{2,12})"
)
# 定义侧显式聚合表达式：「COUNT(order_id)」「SUM(amount)」
_EXPLICIT_AGG_RE = re.compile(
    r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(\s*(DISTINCT\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\)",
    re.IGNORECASE,
)
# 定义侧中文聚合语义（弱证据）：「合计」=SUM、「均值/平均」=AVG、「笔数/个数」=COUNT
_PROSE_AGG_HINTS: tuple[tuple[str, str], ...] = (
    ("合计", "SUM"),
    ("总额", "SUM"),
    ("总量", "SUM"),
    ("平均", "AVG"),
    ("均值", "AVG"),
    ("笔数", "COUNT"),
    ("个数", "COUNT"),
    ("次数", "COUNT"),
    ("数量", "COUNT"),
    ("最大", "MAX"),
    ("最小", "MIN"),
)

_SQL_KEYWORDS = {
    "SELECT", "FROM", "WHERE", "AND", "OR", "NOT", "IN", "ON", "AS", "JOIN",
    "LEFT", "RIGHT", "INNER", "OUTER", "FULL", "CROSS", "GROUP", "BY", "ORDER",
    "HAVING", "LIMIT", "OFFSET", "UNION", "ALL", "DISTINCT", "INSERT", "INTO",
    "VALUES", "UPDATE", "SET", "DELETE", "CREATE", "TABLE", "ALTER", "DROP",
    "INDEX", "VIEW", "IF", "EXISTS", "BETWEEN", "LIKE", "IS", "NULL", "CASE",
    "WHEN", "THEN", "ELSE", "END", "ASC", "DESC", "WITH", "RECURSIVE", "OVER",
    "PARTITION", "CAST", "COALESCE", "INTERVAL", "CURRENT_DATE", "CURRENT_TIMESTAMP",
    "TRUE", "FALSE", "COUNT", "SUM", "AVG", "MIN", "MAX", "DATE", "TIMESTAMP",
}

_UNIT_DAYS = {
    "HOUR": 1 / 24, "HOURS": 1 / 24,
    "DAY": 1, "DAYS": 1, "天": 1, "日": 1,
    "WEEK": 7, "WEEKS": 7, "周": 7,
    "MONTH": 30, "MONTHS": 30, "月": 30, "个月": 30,
    "YEAR": 365, "YEARS": 365, "年": 365,
}


@dataclass
class AggCall:
    """一次聚合调用：函数名 + DISTINCT 标记 + 参数原文"""

    func: str
    distinct: bool
    arg: str


@dataclass
class SqlFacts:
    """SQL 可比对事实。parse_ok=False 表示未解析出任何结构（整体降级 unknown）"""

    parse_ok: bool = False
    aggs: list[AggCall] = field(default_factory=list)
    referenced_columns: set[str] = field(default_factory=set)
    has_star: bool = False
    has_count_star: bool = False
    where_text: str = ""
    where_segments: list[str] = field(default_factory=list)
    window_days: set[int] = field(default_factory=set)


def _strip_literals(sql: str) -> str:
    """字符串字面量替换为等长空白：保留偏移（raw/stripped 同位对齐）与 SQL 结构，
    引号内容不参与标识符/聚合解析，但 WHERE 原文仍可按同位切回供限定词搜索。"""
    return _STRING_RE.sub(lambda m: " " * len(m.group(0)), sql)


def _window_days(value: str, unit: str) -> int:
    return round(int(value) * _UNIT_DAYS.get(unit.upper(), _UNIT_DAYS.get(unit, 1)))


def _split_select_items(select_list: str) -> list[str]:
    """按顶层逗号切分 SELECT 列表（括号内逗号不切）"""
    items: list[str] = []
    depth = 0
    buf: list[str] = []
    for ch in select_list:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(depth - 1, 0)
        if ch == "," and depth == 0:
            items.append("".join(buf))
            buf = []
            continue
        buf.append(ch)
    items.append("".join(buf))
    return items


def _has_select_star(stripped: str) -> bool:
    """SELECT 列表是否含裸星号展开（`SELECT *` / `SELECT t.*`）。

    只在 SELECT 列表内识别，且排除：
    - 乘法/算术运算（`price_per_box * boxes_shipped`、`SUM(price*boxes)`）——
      旧实现 `"*" in sql` 把乘号当星号展开，令计量列缺失只能判 unknown，
      口径冲突永远报不出来；
    - `COUNT(*)`——计数对象不可确认属存疑（has_count_star 表达）。
    """
    m = _SELECT_LIST_RE.search(stripped)
    if not m:
        return False
    select_list = _COUNT_STAR_RE.sub("COUNT(_)", m.group(1))
    if _TABLE_STAR_RE.search(select_list):
        return True
    return any(
        re.fullmatch(r"\s*(?:DISTINCT\s+)?\*\s*", item, re.IGNORECASE)
        for item in _split_select_items(select_list)
    )


def extract_sql_facts(sql: str) -> SqlFacts:
    """从 SQL 文本提取比对事实（纯文本解析，不做完整 SQL 语法分析）"""
    facts = SqlFacts()
    raw = _COMMENT_RE.sub(" ", sql or "")
    stripped = _strip_literals(raw)

    for m in _AGG_RE.finditer(stripped):
        facts.aggs.append(
            AggCall(func=m.group(1).upper(), distinct=bool(m.group(2)), arg=m.group(3).strip())
        )

    for token in _IDENT_RE.findall(stripped):
        if token.upper() not in _SQL_KEYWORDS:
            facts.referenced_columns.add(token.lower())

    facts.has_star = _has_select_star(stripped)
    facts.has_count_star = _COUNT_STAR_RE.search(stripped) is not None

    where_m = re.search(r"\bWHERE\b", stripped, re.IGNORECASE)
    if where_m:
        rest = stripped[where_m.end():]
        end_m = _WHERE_END_RE.search(rest)
        facts.where_text = raw[where_m.end(): where_m.end() + (end_m.start() if end_m else len(rest))].strip()
        facts.where_segments = [
            seg.strip()
            for seg in re.split(r"\b(?:AND|OR)\b", facts.where_text, flags=re.IGNORECASE)
            if seg.strip()
        ]

    for m in _SQL_WINDOW_RE.finditer(raw):
        facts.window_days.add(_window_days(m.group(1), m.group(2)))

    facts.parse_ok = bool(facts.aggs or facts.referenced_columns or facts.where_text)
    return facts


# ---------------------------------------------------------------------------
# 指标侧事实
# ---------------------------------------------------------------------------


@dataclass
class MetricColumn:
    """REFERS_COLUMN 边：列名 + role（measure/filter/dimension_ref）"""

    column_name: str
    role: str


@dataclass
class CaliberMetric:
    """参与比对的指标定义（图谱 Metric 节点 + REFERS_COLUMN 边）"""

    id: str | None = None
    code: str | None = None
    name: str | None = None
    formula: str | None = None
    business_definition: str | None = None
    statistical_scope: str | None = None
    time_scope: str | None = None
    columns: list[MetricColumn] = field(default_factory=list)


def _metric_summary(metric: CaliberMetric) -> dict[str, Any]:
    """输出用 metric 标识：缺失键省略（与 map 侧省略纪律一致）"""
    out: dict[str, Any] = {}
    if metric.id:
        out["id"] = metric.id
    if metric.code:
        out["code"] = metric.code
    if metric.name:
        out["name"] = metric.name
    return out


def _definition_text(metric: CaliberMetric) -> str:
    return " ".join(
        x for x in (metric.business_definition, metric.statistical_scope, metric.time_scope) if x
    )


def _expected_aggs(metric: CaliberMetric) -> tuple[list[AggCall], list[str]]:
    """指标侧聚合期望：formula 显式聚合优先；定义文本显式 FUNC(col) 次之；
    中文语义提示（合计/均值…）为弱证据单独返回。"""
    explicit: list[AggCall] = []
    for text in (metric.formula, metric.business_definition, metric.statistical_scope):
        if not text:
            continue
        for m in _EXPLICIT_AGG_RE.finditer(text):
            call = AggCall(func=m.group(1).upper(), distinct=bool(m.group(2)), arg=m.group(3))
            if call not in explicit:
                explicit.append(call)
    prose = sorted(
        {func for hint, func in _PROSE_AGG_HINTS if hint in _definition_text(metric)}
    )
    return explicit, prose


def _expected_windows(metric: CaliberMetric) -> set[int]:
    """指标定义中的明确时间窗（天）。「统计周期内」等参数化表述不产出期望（不比对）。"""
    days: set[int] = set()
    for m in _DEF_WINDOW_RE.finditer(_definition_text(metric)):
        days.add(_window_days(m.group(1), m.group(2)))
    return days


def _exclusion_terms(metric: CaliberMetric) -> list[str]:
    text = _definition_text(metric)
    terms: list[str] = []
    for m in _EXCLUSION_RE.finditer(text):
        term = m.group(1).strip()
        if term and term not in terms:
            terms.append(term)
    return terms


def _diff(aspect: str, expected: str, found: str, detail: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"aspect": aspect, "expected": expected, "found": found}
    if detail:
        out["detail"] = detail
    return out


_NEGATION_MARKERS = ("NOT", "!=", "<>", "IS NULL", "= 0", "= FALSE", "= '否'", 'NOT LIKE', 'NOT IN')


def _agg_signature(call: AggCall) -> tuple[str, bool]:
    return (call.func, call.distinct)


def _compare_aggregation(
    metric: CaliberMetric, facts: SqlFacts
) -> list[tuple[dict[str, Any], bool]]:
    """聚合口径比对。返回 (差异, 是否确定) 列表：显式 FUNC(...) 证据为确定，中文语义提示为存疑。"""
    explicit, prose = _expected_aggs(metric)
    found_sigs = {_agg_signature(c) for c in facts.aggs}
    found_desc = ", ".join(
        sorted({f"{c.func}({'DISTINCT ' if c.distinct else ''}{c.arg})" for c in facts.aggs})
    ) or "无聚合函数"
    expected_desc = ", ".join(
        sorted({f"{c.func}({'DISTINCT ' if c.distinct else ''}{c.arg})" for c in explicit})
    ) or ", ".join(prose)

    if explicit:
        if any(_agg_signature(c) not in found_sigs for c in explicit):
            return [(_diff("aggregation", expected_desc, found_desc, "SQL 聚合结构与 formula 不一致"), True)]
        return []
    if prose and any(
        (f, False) not in found_sigs and (f, True) not in found_sigs for f in prose
    ):
        # 中文语义提示只是弱证据：不一致只给 unknown
        return [(_diff("aggregation", expected_desc, found_desc, "定义中的聚合语义未能在 SQL 中确认"), False)]
    return []


def _compare_filter(
    metric: CaliberMetric, facts: SqlFacts
) -> list[tuple[dict[str, Any], bool]]:
    """过滤/统计范围比对（剔除类限定）。全部存疑：只给 unknown，不判 conflict。"""
    diffs: list[tuple[dict[str, Any], bool]] = []
    for term in _exclusion_terms(metric):
        hit_segments = [s for s in facts.where_segments if term.lower() in s.lower()]
        if not hit_segments:
            diffs.append((_diff("filter", f"剔除 {term}", "SQL WHERE 未见该限定",
                                "无法确认 SQL 是否实现了该剔除条件"), False))
        elif not any(
            any(marker in seg.upper() or marker in seg for marker in _NEGATION_MARKERS)
            for seg in hit_segments
        ):
            diffs.append((_diff("filter", f"剔除 {term}", "SQL WHERE 提及该词但未见否定算子",
                                "无法确认是否为剔除条件"), False))
    return diffs


def _compare_time_scope(
    metric: CaliberMetric, facts: SqlFacts
) -> list[tuple[dict[str, Any], bool]]:
    """时间窗比对：仅比对定义中明确的窗口（「统计周期内」等参数化表述不比对）。

    双侧都有明确窗口而不相交 → 确定 conflict；SQL 侧无窗口 → 存疑 unknown。
    """
    expected = _expected_windows(metric)
    if not expected:
        return []
    expected_desc = f"时间窗 {sorted(expected)} 天"
    if not facts.window_days:
        return [(_diff("time_scope", expected_desc, "SQL 未见明确时间窗限定",
                       "SQL 可能由参数化条件实现时间范围"), False)]
    if expected & facts.window_days:
        return []
    return [(_diff("time_scope", expected_desc, f"时间窗 {sorted(facts.window_days)} 天",
                   "SQL 时间窗与指标定义不一致"), True)]


def _compare_columns(
    metric: CaliberMetric, facts: SqlFacts
) -> list[tuple[dict[str, Any], bool]]:
    """引用列比对：REFERS_COLUMN（role）列 + 显式聚合参数列 vs SQL 实际引用列。

    measure 列缺失且 SQL 无 * 展开 → 确定 conflict；其余缺失/无法解析 → 存疑 unknown。
    """
    expected_cols = list(metric.columns)
    # formula/定义中显式聚合的参数列按 measure 计入期望
    explicit, _prose = _expected_aggs(metric)
    for c in explicit:
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", c.arg) and not any(
            ec.column_name.lower() == c.arg.lower() for ec in expected_cols
        ):
            expected_cols.append(MetricColumn(column_name=c.arg, role="measure"))
    if not expected_cols:
        return []
    expected_desc = ", ".join(f"{c.column_name}({c.role})" for c in expected_cols)
    found_desc = ", ".join(sorted(facts.referenced_columns)) or "未解析出引用列"

    if not facts.referenced_columns:
        return [(_diff("column", expected_desc, found_desc, "SQL 未解析出引用列，无法比对"), False)]

    diffs: list[tuple[dict[str, Any], bool]] = []
    # 星号展开（SELECT *）与 COUNT(*) 都会让「计量列是否真的没用到」不可确认：
    # 前者可能已展开该列，后者计数对象未知 —— 二者都只给存疑，不误报 conflict。
    ambiguous_expansion = facts.has_star or facts.has_count_star
    for col in expected_cols:
        if col.column_name.lower() in facts.referenced_columns:
            continue
        if col.role == "measure" and not ambiguous_expansion:
            diffs.append((_diff("column", expected_desc, found_desc,
                                f"计量列 {col.column_name} 未在 SQL 中引用"), True))
        elif col.role == "measure":
            reason = "SQL 含 * 展开无法确认" if facts.has_star else "SQL 含 COUNT(*) 计数对象无法确认"
            diffs.append((_diff("column", expected_desc, found_desc,
                                f"计量列 {col.column_name} 未显式引用，{reason}"), False))
        else:
            diffs.append((_diff("column", expected_desc, found_desc,
                                f"{col.role} 列 {col.column_name} 未在 SQL 中引用，可能经由其它表达式实现"), False))
    return diffs


def compare_metric(metric: CaliberMetric, facts: SqlFacts) -> dict[str, Any]:
    """单指标比对：返回 {metric, status, differences}

    status：任一确定差异 → conflict；仅存疑项 → unknown；否则 consistent。
    differences 只含 aspect/expected/found/detail（不带判定，判定收在 status）。
    """
    if not facts.parse_ok:
        return {
            "metric": _metric_summary(metric),
            "status": "unknown",
            "differences": [_diff("sql", "可解析的 SQL", "无法解析 SQL 结构", "SQL 文本未解析出可比对结构")],
        }

    verdicts = (
        _compare_aggregation(metric, facts)
        + _compare_filter(metric, facts)
        + _compare_time_scope(metric, facts)
        + _compare_columns(metric, facts)
    )
    conflict = any(certain for _d, certain in verdicts)
    status = "conflict" if conflict else ("unknown" if verdicts else "consistent")
    return {
        "metric": _metric_summary(metric),
        "status": status,
        "differences": [d for d, _certain in verdicts],
    }


def check_caliber(metrics: list[CaliberMetric], sql: str) -> dict[str, Any]:
    """多指标口径检测入口：SQL 解析一次，逐指标比对"""
    facts = extract_sql_facts(sql)
    return {"results": [compare_metric(m, facts) for m in metrics]}


# ---------------------------------------------------------------------------
# 图读取（图对象由调用方注入，本模块不建连）
# ---------------------------------------------------------------------------


def _metric_from_attrs(
    node_id: str | None, attrs: dict[str, Any], columns: list[MetricColumn]
) -> CaliberMetric:
    return CaliberMetric(
        id=attrs.get("id") or node_id,
        code=attrs.get("code"),
        name=attrs.get("name"),
        formula=attrs.get("formula"),
        business_definition=attrs.get("business_definition"),
        statistical_scope=attrs.get("statistical_scope"),
        time_scope=attrs.get("time_scope"),
        columns=columns,
    )


def _match_ref(ref: str, *candidates: Any) -> bool:
    return any(c is not None and str(c) == ref for c in candidates)


def _read_metric_networkx(graph: Any, ref: str) -> CaliberMetric | None:
    g = graph.G
    for node_key, attrs in g.nodes(data=True):
        if attrs.get("node_type") != "Metric":
            continue
        if not _match_ref(ref, attrs.get("code"), attrs.get("id"), node_key):
            continue
        columns: list[MetricColumn] = []
        for _u, v, edata in g.out_edges(node_key, data=True):
            if edata.get("edge_type") != "REFERS_COLUMN":
                continue
            v_attrs = g.nodes[v]
            name = v_attrs.get("column_name") or str(v_attrs.get("column", "")).rsplit(".", 1)[-1]
            columns.append(MetricColumn(column_name=name or str(v), role=str(edata.get("role") or "")))
        return _metric_from_attrs(str(node_key), attrs, columns)
    return None


def _read_metric_cypher(graph: Any, ref: str) -> CaliberMetric | None:
    # 属性名一律取 schema.md（Metric 主键 id、code）；CSV 装载列名（如 `:ID(Metric)`）
    # 不是图库属性，写进 Cypher 会触发 Binder exception（BUGFIX-caliber-cypher）。
    rows = graph.query(
        "MATCH (m:Metric) RETURN m.id, m.code, m.name, m.formula, "
        "m.business_definition, m.statistical_scope, m.time_scope"
    )
    target: tuple[Any, ...] | None = None
    for row in rows:
        if _match_ref(ref, row[0], row[1]):
            target = row
            break
    if target is None:
        return None

    columns: list[MetricColumn] = []
    for row in graph.query(
        "MATCH (m:Metric)-[r:REFERS_COLUMN]->(c:Col) "
        # `column` 是 Cypher 保留字，必须反引号（Parser exception，见 BUGFIX-caliber-cypher）
        "RETURN m.id, m.code, c.column_name, c.`column`, r.role"
    ):
        if not _match_ref(ref, row[0], row[1]):
            continue
        name = row[2] or str(row[3] or "").rsplit(".", 1)[-1]
        columns.append(MetricColumn(column_name=str(name), role=str(row[4] or "")))

    attrs = {
        "id": target[0],
        "code": target[1],
        "name": target[2],
        "formula": target[3],
        "business_definition": target[4],
        "statistical_scope": target[5],
        "time_scope": target[6],
    }
    return _metric_from_attrs(str(attrs["id"]), attrs, columns)


def read_metric_caliber(graph: Any, ref: str) -> CaliberMetric | None:
    """按 code 或节点 ID 读取指标定义 + REFERS_COLUMN；未找到返回 None

    支持 NetworkXGraph（.G）与 Cypher 后端（FalkorDBGraph / LadybugGraph，.query）。
    """
    if hasattr(graph, "G"):
        return _read_metric_networkx(graph, ref)
    if hasattr(graph, "query"):
        return _read_metric_cypher(graph, ref)
    return None
