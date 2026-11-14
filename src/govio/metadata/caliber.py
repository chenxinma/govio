"""
govio.metadata.caliber
口径冲突检测（T2.3）：SQL 与图谱指标定义的最小维度比对。

SQL 事实提取基于 sqlparse 分词（不依赖方言语法分析），不触图后端、不连库：
- `extract_sql_facts`：从 SQL 文本提取可比对事实（聚合调用/引用列/WHERE 维度限定三元组/时间窗与时间粒度）；
- `check_caliber` / `compare_metric`：逐指标比对，输出结构化 JSON 形态
  `{results: [{metric, status, differences}]}`；
- `read_metric_caliber`：从图对象（NetworkXGraph / FalkorDBGraph / LadybugGraph）
  读取指标定义与 REFERS_COLUMN 边，图对象由调用方注入（GraphFactory），本模块不建连。

判定纪律：**宁可 unknown 不误报 conflict**。conflict 仅在证据确定时给出
（聚合签名不匹配 / 明确时间窗或粒度不一致 / 计量列完全缺失 /
维度限定与指标声明实质矛盾），语义比对不确定一律 unknown。
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import sqlparse
import sqlparse.sql as S
from sqlparse import tokens as T

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQL 事实提取
# ---------------------------------------------------------------------------

_COMMENT_RE = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)
# SQL 侧时间窗：`INTERVAL '7' DAY`、`INTERVAL 30 DAY`、`DATE_SUB(..., 7 DAY)` 等
_SQL_WINDOW_RE = re.compile(
    r"\b(\d+)\s*'?\s*(DAY|DAYS|WEEK|WEEKS|MONTH|MONTHS|YEAR|YEARS|HOUR|HOURS)\b",
    re.IGNORECASE,
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

# sqlparse Function 节点中命中该集合的聚合函数才记录为一次聚合调用
_AGG_FUNCS = frozenset({"COUNT", "SUM", "AVG", "MIN", "MAX"})

_UNIT_DAYS = {
    "HOUR": 1 / 24,
    "HOURS": 1 / 24,
    "DAY": 1,
    "DAYS": 1,
    "天": 1,
    "日": 1,
    "WEEK": 7,
    "WEEKS": 7,
    "周": 7,
    "MONTH": 30,
    "MONTHS": 30,
    "月": 30,
    "个月": 30,
    "YEAR": 365,
    "YEARS": 365,
    "年": 365,
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
    dim_predicates: list[DimPredicate] = field(default_factory=list)
    group_by_columns: set[str] = field(default_factory=set)
    grains: set[str] = field(default_factory=set)


def _window_days(value: str, unit: str) -> int:
    return round(int(value) * _UNIT_DAYS.get(unit.upper(), _UNIT_DAYS.get(unit, 1)))


def _agg_from_function(func: S.Function) -> AggCall | None:
    """从 sqlparse Function 节点提取聚合调用。

    直接取外层 Parenthesis 的原文作为参数，天然支持嵌套表达式
    （`SUM(COALESCE(x, 0))`），规避旧正则 `[^()]*` 在嵌套括号处漏掉聚合的问题。
    """
    name = (func.get_name() or "").upper()
    if name not in _AGG_FUNCS:
        return None
    for child in func.tokens:
        if isinstance(child, S.Parenthesis):
            inner = str(child)[1:-1].strip()
            distinct = bool(re.match(r"^\s*DISTINCT\s+", inner, re.IGNORECASE))
            arg = re.sub(
                r"^\s*DISTINCT\s+", "", inner, count=1, flags=re.IGNORECASE
            ).strip()
            return AggCall(func=name, distinct=distinct, arg=arg)
    return AggCall(func=name, distinct=False, arg="")


def _collect(
    node: Any,
    aggs: list[AggCall],
    columns: set[str],
    excluded: set[str],
) -> None:
    """递归遍历 token 树：收集聚合调用与标识符。

    引号字面量/注释是独立 token 类型，不会混入标识符；函数名与内置名
    （INTERVAL/INT 等 Token.Name.Builtin）进 excluded，最终从引用列中剔除。
    """
    for tok in getattr(node, "tokens", []) or []:
        if isinstance(tok, S.Function):
            agg = _agg_from_function(tok)
            if agg is not None:
                aggs.append(agg)
            if tok.get_name():
                excluded.add(tok.get_name().lower())
            _collect(tok, aggs, columns, excluded)
        elif tok.ttype is T.Name:
            columns.add(tok.value.lower())
        elif tok.ttype is T.Name.Builtin:
            excluded.add(tok.value.lower())
        else:
            _collect(tok, aggs, columns, excluded)


def _detect_select_star(stmt: Any) -> bool:
    """SELECT 列表是否含裸星号展开（`SELECT *` / `SELECT t.*`）。

    只扫描 SELECT 关键字到首个顶层 FROM 之间的选择项；乘号 `a * b` 是
    Token.Operator、星号展开是 Token.Wildcard，分词层天然区分，`COUNT(*)`
    作为 Function 整体跳过（计数对象由 has_count_star 单独表达）。
    """
    seen_select = False
    for tok in getattr(stmt, "tokens", []) or []:
        if tok.is_whitespace:
            continue
        if not seen_select:
            if tok.ttype is T.Keyword.DML and tok.value.upper() == "SELECT":
                seen_select = True
            continue
        if tok.is_keyword and tok.value.upper() == "FROM":
            break
        if isinstance(tok, S.Function):
            continue
        if any(sub.ttype is T.Wildcard for sub in tok.flatten()):
            return True
    return False


def _find_where(stmt: Any) -> S.Where | None:
    for tok in getattr(stmt, "tokens", []) or []:
        if isinstance(tok, S.Where):
            return tok
    return None


def _where_body_tokens(where: S.Where) -> list[Any]:
    """WHERE 关键字之后的顶层 token（子查询作为整体保留，不误切其内 GROUP BY 等）"""
    start = 0
    for i, tok in enumerate(where.tokens):
        if tok.is_keyword and tok.value.upper() == "WHERE":
            start = i + 1
            break
    return where.tokens[start:]


def _where_text(where: S.Where) -> str:
    raw = "".join(tok.value for tok in _where_body_tokens(where))
    return _COMMENT_RE.sub(" ", raw).strip()


def _split_where_segments(where: S.Where) -> list[str]:
    """按顶层 AND/OR 切分 WHERE 条件（子查询内的 AND/OR 不切）"""
    segments: list[str] = []
    buf: list[str] = []
    for tok in _where_body_tokens(where):
        if tok.is_keyword and tok.value.upper() in ("AND", "OR"):
            seg = "".join(buf).strip()
            if seg:
                segments.append(_COMMENT_RE.sub(" ", seg).strip())
            buf = []
        else:
            buf.append(tok.value)
    seg = "".join(buf).strip()
    if seg:
        segments.append(_COMMENT_RE.sub(" ", seg).strip())
    return segments


# ---------------------------------------------------------------------------
# 维度限定三元组：SQL 侧解析（列 / 算子 / 值集合）
# ---------------------------------------------------------------------------

# 谓词形状 `列 算子 值`；算子按长优先排列，避免 NOT IN 被 IN 抢先命中
_PREDICATE_RE = re.compile(
    r"^\s*(?P<lhs>[A-Za-z_][A-Za-z0-9_.]*)\s*"
    r"(?P<op>NOT\s+IN|NOT\s+LIKE|IS\s+NOT\s+NULL|IS\s+NULL"
    r"|!=|<>|>=|<=|=|<|>|\bIN\b|\bLIKE\b|\bBETWEEN\b)"
    r"\s*(?P<rhs>.*)$",
    re.IGNORECASE | re.DOTALL,
)
_CONNECTOR_RE = re.compile(r"\b(BETWEEN|AND|OR)\b", re.IGNORECASE)
# 值侧只有「纯字面量（引号串/数字）列表」才产出取值集合；其余形态一律不猜测
_LITERAL_RE = re.compile(r"'((?:[^']|'')*)'|\"([^\"]*)\"|(-?\d+(?:\.\d+)?)")
_LITERAL_ONLY_RE = re.compile(
    r"^(?:\s*(?:'(?:[^']|'')*'|\"[^\"]*\"|-?\d+(?:\.\d+)?)\s*)"
    r"(?:,\s*(?:'(?:[^']|'')*'|\"[^\"]*\"|-?\d+(?:\.\d+)?)\s*)*$"
)
_OP_CANON = {
    "=": "=",
    "!=": "!=",
    "<>": "!=",
    ">": ">",
    "<": "<",
    ">=": ">=",
    "<=": "<=",
    "IN": "IN",
    "NOT IN": "NOT IN",
    "LIKE": "LIKE",
    "NOT LIKE": "NOT LIKE",
    "IS NULL": "IS NULL",
    "IS NOT NULL": "IS NOT NULL",
    "BETWEEN": "BETWEEN",
}
# 正向限定（限定到值集合）；!=/NOT IN/NOT LIKE/IS NOT NULL 是实现剔除，不是选取
_POSITIVE_OPS = frozenset(
    {"=", ">", "<", ">=", "<=", "IN", "LIKE", "IS NULL", "BETWEEN"}
)
# 可枚举取值的算子：范围/BETWEEN 的右值不是离散取值集合
_VALUE_OPS = frozenset({"=", "!=", "IN", "NOT IN", "LIKE", "NOT LIKE"})
# 「SQL 附加未声明限定」只按等值/集合/前缀匹配判定，范围类归 time_scope 处理
_EXTRA_RESTRICTION_OPS = frozenset({"=", "IN", "LIKE"})
_QUOTE_CHARS = ("'", '"', "`")


@dataclass
class DimPredicate:
    """SQL 侧一个维度限定三元组

    column: `_canonical_expr` 口径的列名；op: 归一算子；
    values: 字面量取值集合（不可枚举时为空集）；positive: 是否「限定到取值」。
    """

    column: str
    op: str
    values: frozenset[str]
    positive: bool
    raw: str


def _structural_mask(text: str) -> list[tuple[int, bool]]:
    """逐字符的 (括号深度, 是否在引号内) 状态，用于只在顶层切分连接词"""
    mask: list[tuple[int, bool]] = []
    depth = 0
    quote = ""
    for ch in text:
        if quote:
            mask.append((depth, True))
            if ch == quote:
                quote = ""
            continue
        if ch in _QUOTE_CHARS:
            mask.append((depth, True))
            quote = ch
            continue
        mask.append((depth, False))
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
    return mask


def _split_predicates(text: str) -> list[str]:
    """按顶层 AND/OR 切分谓词原文；`BETWEEN x AND y` 的 AND 不切"""
    mask = _structural_mask(text)
    parts: list[str] = []
    last = 0
    pending_between_and = False
    for m in _CONNECTOR_RE.finditer(text):
        depth, in_quote = mask[m.start()]
        if in_quote or depth != 0:
            continue
        word = m.group(1).upper()
        if word == "BETWEEN":
            pending_between_and = True
            continue
        if word == "AND" and pending_between_and:
            pending_between_and = False
            continue
        seg = text[last : m.start()].strip()
        if seg:
            parts.append(seg)
        last = m.end()
    tail = text[last:].strip()
    if tail:
        parts.append(tail)
    return parts


def _paren_contents(text: str) -> list[str]:
    """取顶层括号内的原文（括号里的复合条件要展开成独立谓词）"""
    out: list[str] = []
    depth = 0
    start = 0
    quote = ""
    for i, ch in enumerate(text):
        if quote:
            if ch == quote:
                quote = ""
            continue
        if ch in _QUOTE_CHARS:
            quote = ch
        elif ch == "(":
            depth += 1
            if depth == 1:
                start = i + 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                out.append(text[start:i])
    return out


def _literal_values(rhs: str) -> frozenset[str]:
    """值侧 -> 取值集合：仅接受引号串/数字字面量（允许 IN 的括号列表与尾随 ::type）"""
    body = rhs.strip()
    if body.startswith("(") and body.endswith(")"):
        body = body[1:-1]
    body = _CAST_SUFFIX_RE.sub("", body).strip()
    if not body or not _LITERAL_ONLY_RE.match(body):
        return frozenset()
    values: set[str] = set()
    for m in _LITERAL_RE.finditer(body):
        token = next((g for g in m.groups() if g is not None), "")
        token = token.replace("''", "'").strip().lower()
        if token:
            values.add(token)
    return frozenset(values)


def _parse_predicate(seg: str) -> DimPredicate | None:
    """谓词原文 -> 三元组；形状不认识返回 None（宁可不比对，也不猜测）"""
    seg = _COMMENT_RE.sub(" ", seg).strip()
    m = _PREDICATE_RE.match(seg)
    if not m:
        return None
    op_raw = re.sub(r"\s+", " ", m.group("op")).strip().upper()
    op = _OP_CANON.get(op_raw)
    if op is None:
        return None
    column = _canonical_expr(m.group("lhs"))
    if not column:
        return None
    values = _literal_values(m.group("rhs")) if op in _VALUE_OPS else frozenset()
    return DimPredicate(
        column=column,
        op=op,
        values=values,
        positive=op in _POSITIVE_OPS,
        raw=re.sub(r"\s+", " ", seg),
    )


def _iter_predicate_segments(text: str) -> list[str]:
    """切分谓词并展开括号内复合条件；`NOT (...)` 整体丢弃（否定包裹无法还原取值）"""
    out: list[str] = []
    for seg in _split_predicates(text):
        if _PREDICATE_RE.match(seg):
            out.append(seg)
            continue
        if re.match(r"^\s*NOT\b", seg, re.IGNORECASE):
            continue
        inner = _paren_contents(seg)
        if inner:
            out.extend(_iter_predicate_segments(" ".join(inner)))
    return out


def _iter_where_nodes(node: Any, out: list[Any]) -> None:
    """递归收集 WHERE 子句节点（子查询/CTE 内的限定同样算限定）"""
    for tok in getattr(node, "tokens", []) or []:
        if isinstance(tok, S.Where):
            out.append(tok)
        _iter_where_nodes(tok, out)


def _extract_dim_predicates(stmt: Any) -> list[DimPredicate]:
    """SQL -> 维度限定三元组列表，按 (列, 算子, 取值, 正向) 去重"""
    wheres: list[Any] = []
    _iter_where_nodes(stmt, wheres)
    found: dict[tuple[str, str, frozenset[str], bool], DimPredicate] = {}
    for where in wheres:
        for seg in _iter_predicate_segments(_where_text(where)):
            pred = _parse_predicate(seg)
            if pred is not None:
                found.setdefault(
                    (pred.column, pred.op, pred.values, pred.positive), pred
                )
    return list(found.values())


def _group_by_columns(stmt: Any) -> set[str]:
    """GROUP BY 里的列名（位置号 `GROUP BY 1` 取不到名字，自然落空）"""
    out: set[str] = set()
    collecting = False
    for tok in getattr(stmt, "tokens", []) or []:
        if tok.is_keyword:
            value = (tok.value or "").upper()
            if value == "GROUP BY":
                collecting = True
                continue
            if value in ("HAVING", "ORDER BY", "LIMIT", "OFFSET", "UNION", "WINDOW"):
                collecting = False
                continue
        if not collecting:
            continue
        names: set[str] = set()
        excluded: set[str] = set()
        _collect(tok, [], names, excluded)
        out |= {_canonical_expr(n) for n in names - excluded if n.strip()}
    return out


def extract_sql_facts(sql: str) -> SqlFacts:
    """从 SQL 文本提取比对事实（sqlparse 分词，不做方言语义分析）"""
    facts = SqlFacts()
    statements = sqlparse.parse(sql or "")
    if not statements:
        return facts
    stmt = statements[0]

    aggs: list[AggCall] = []
    columns: set[str] = set()
    excluded: set[str] = set()
    _collect(stmt, aggs, columns, excluded)
    facts.aggs = aggs
    facts.referenced_columns = columns - excluded
    facts.has_star = _detect_select_star(stmt)
    facts.has_count_star = any(a.func == "COUNT" and a.arg == "*" for a in aggs)

    where = _find_where(stmt)
    if where is not None:
        facts.where_text = _where_text(where)
        facts.where_segments = _split_where_segments(where)

    facts.dim_predicates = _extract_dim_predicates(stmt)
    facts.group_by_columns = _group_by_columns(stmt)

    raw = _COMMENT_RE.sub(" ", sql or "")
    for m in _SQL_WINDOW_RE.finditer(raw):
        facts.window_days.add(_window_days(m.group(1), m.group(2)))
    facts.grains = _sql_grains(raw)

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
    """参与比对的指标定义（图谱 Metric 节点 + REFERS_COLUMN / DIMENSION_USED 边）"""

    id: str | None = None
    code: str | None = None
    name: str | None = None
    formula: str | None = None
    business_definition: str | None = None
    statistical_scope: str | None = None
    time_scope: str | None = None
    columns: list[MetricColumn] = field(default_factory=list)
    dimensions: list[DimensionDecl] = field(default_factory=list)


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
        x
        for x in (
            metric.business_definition,
            metric.statistical_scope,
            metric.time_scope,
        )
        if x
    )


def _expected_aggs(metric: CaliberMetric) -> tuple[list[AggCall], list[str]]:
    """指标侧聚合期望：formula 显式聚合优先；定义文本显式 FUNC(col) 次之；
    中文语义提示（合计/均值…）为弱证据单独返回。"""
    explicit: list[AggCall] = []
    for text in (metric.formula, metric.business_definition, metric.statistical_scope):
        if not text:
            continue
        for m in _EXPLICIT_AGG_RE.finditer(text):
            call = AggCall(
                func=m.group(1).upper(), distinct=bool(m.group(2)), arg=m.group(3)
            )
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


def _diff(
    aspect: str, expected: str, found: str, detail: str | None = None
) -> dict[str, Any]:
    out: dict[str, Any] = {"aspect": aspect, "expected": expected, "found": found}
    if detail:
        out["detail"] = detail
    return out


_NEGATION_MARKERS = (
    "NOT",
    "!=",
    "<>",
    "IS NULL",
    "= 0",
    "= FALSE",
    "= '否'",
    "NOT LIKE",
    "NOT IN",
)


def _agg_signature(call: AggCall) -> tuple[str, bool]:
    return (call.func, call.distinct)


# 展示层包装（不影响口径语义）：ROUND(x, n) / CAST(x AS t) / x::t
_DISPLAY_WRAP_RE = re.compile(
    r"^\s*(ROUND|CAST)\s*\((.*)\)\s*(?:::\s*[A-Za-z][A-Za-z0-9_ (),]*)?\s*$",
    re.IGNORECASE | re.DOTALL,
)
_QUALIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\s*\.\s*")
# 尾随类型强转（PostgreSQL `expr::numeric`），不影响口径语义
_CAST_SUFFIX_RE = re.compile(r"::\s*[A-Za-z_][A-Za-z0-9_ (),\[\]]*$")


def _strip_display_wrapper(expr: str) -> str:
    """剥掉展示层包装：ROUND(x, 2) / CAST(x AS DECIMAL(18,2)) / x::int → x"""
    e = expr.strip()
    while True:
        m = _DISPLAY_WRAP_RE.match(e)
        if not m:
            return e
        inner = m.group(2)
        func = (m.group(1) or "").upper()
        if func == "ROUND":
            # 去 ROUND 的第二参数（精度），只留待舍入表达式
            depth, cut = 0, None
            for i, ch in enumerate(inner):
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                elif ch == "," and depth == 0:
                    cut = i
                    break
            if cut is not None:
                inner = inner[:cut]
        else:  # CAST(x AS type) → x
            parts = re.split(r"\s+AS\s+", inner, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) == 2:
                inner = parts[0]
        e = inner.strip()


def _canonical_expr(expr: str) -> str:
    """表达式规范化（用于「不同写法实质相同」的判定）：
    剥展示包装 → 去表/库限定 → 压空白 → 小写。

    `ROUND(SUM(f.amount), 2)` 与 `sum(amount)` 归一后相同；
    `SUM(f.amount * (1 - f.discount_pct/100))` 归一后不同（口径被修饰）。
    只做字面层归一（不做代数化简，避免误报）。
    """
    e = _strip_display_wrapper(expr)
    e = _QUALIFIER_RE.sub("", e)
    e = re.sub(r"\s+", "", e)
    e = e.lower()
    # `sum(amount)::numeric` 与 `sum(amount)` 同为展示层强转
    while _CAST_SUFFIX_RE.search(e):
        e = _CAST_SUFFIX_RE.sub("", e)
    return e


def _compare_aggregation(
    metric: CaliberMetric, facts: SqlFacts
) -> list[tuple[dict[str, Any], bool]]:
    """聚合口径比对。返回 (差异, 是否确定) 列表：显式 FUNC(...) 证据为确定，中文语义提示为存疑。

    两类确定差异：
    1. 聚合结构不一致（函数名/DISTINCT 不同）；
    2. **聚合参数表达式不一致**（同函数名但参数被修饰，如指标要 `sum(amount)`，
       SQL 给 `SUM(amount*(1-discount_pct/100))`）——旧实现只比 (func, distinct)，
       这类「列在但表达式被扭曲」漏判。参数按 `_canonical_expr` 归一后比对，
       因此 `SUM(f.amount)`、`ROUND(SUM(amount),2)`、`sum(amount)` 视为同一口径。
    """
    explicit, prose = _expected_aggs(metric)
    found_desc = (
        ", ".join(
            sorted(
                {
                    f"{c.func}({'DISTINCT ' if c.distinct else ''}{c.arg})"
                    for c in facts.aggs
                }
            )
        )
        or "无聚合函数"
    )
    expected_desc = ", ".join(
        sorted(
            {f"{c.func}({'DISTINCT ' if c.distinct else ''}{c.arg})" for c in explicit}
        )
    ) or ", ".join(prose)

    if explicit:
        diffs: list[tuple[dict[str, Any], bool]] = []
        for expected in explicit:
            same_sig = [
                a for a in facts.aggs if _agg_signature(a) == _agg_signature(expected)
            ]
            if not same_sig:
                diffs.append(
                    (
                        _diff(
                            "aggregation",
                            expected_desc,
                            found_desc,
                            "SQL 聚合结构与 formula 不一致",
                        ),
                        True,
                    )
                )
                continue
            if expected.arg and not any(
                _canonical_expr(a.arg) == _canonical_expr(expected.arg)
                for a in same_sig
            ):
                # COUNT(*) 的计数对象不可确认（可能是同一列）→ 存疑，不误报
                if any(a.arg.strip() == "*" for a in same_sig):
                    diffs.append(
                        (
                            _diff(
                                "aggregation",
                                expected_desc,
                                found_desc,
                                f"SQL 使用 COUNT(*)，计数对象无法确认（期望 {expected.func}({expected.arg})）",
                            ),
                            False,
                        )
                    )
                else:
                    diffs.append(
                        (
                            _diff(
                                "aggregation",
                                expected_desc,
                                found_desc,
                                f"聚合参数与 formula 不一致（期望 {expected.func}({expected.arg})）",
                            ),
                            True,
                        )
                    )
        return diffs
    found_sigs = {_agg_signature(c) for c in facts.aggs}
    if prose and any(
        (f, False) not in found_sigs and (f, True) not in found_sigs for f in prose
    ):
        # 中文语义提示只是弱证据：不一致只给 unknown
        return [
            (
                _diff(
                    "aggregation",
                    expected_desc,
                    found_desc,
                    "定义中的聚合语义未能在 SQL 中确认",
                ),
                False,
            )
        ]
    return []


def _compare_filter(
    metric: CaliberMetric, facts: SqlFacts
) -> list[tuple[dict[str, Any], bool]]:
    """过滤/统计范围比对（定义里的剔除类词 vs WHERE 段子串包含）。

    弱证据路径：全部存疑，只给 unknown，不判 conflict。剔除条件是否被**实质**违反，
    由 `_compare_dimensions` 按「维度枚举码值 vs SQL 三元组取值」判定（X19-W5a）。
    """
    diffs: list[tuple[dict[str, Any], bool]] = []
    for term in _exclusion_terms(metric):
        hit_segments = [s for s in facts.where_segments if term.lower() in s.lower()]
        if not hit_segments:
            diffs.append(
                (
                    _diff(
                        "filter",
                        f"剔除 {term}",
                        "SQL WHERE 未见该限定",
                        "无法确认 SQL 是否实现了该剔除条件",
                    ),
                    False,
                )
            )
        elif not any(
            any(marker in seg.upper() or marker in seg for marker in _NEGATION_MARKERS)
            for seg in hit_segments
        ):
            diffs.append(
                (
                    _diff(
                        "filter",
                        f"剔除 {term}",
                        "SQL WHERE 提及该词但未见否定算子",
                        "无法确认是否为剔除条件",
                    ),
                    False,
                )
            )
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
        return [
            (
                _diff(
                    "time_scope",
                    expected_desc,
                    "SQL 未见明确时间窗限定",
                    "SQL 可能由参数化条件实现时间范围",
                ),
                False,
            )
        ]
    if expected & facts.window_days:
        return []
    return [
        (
            _diff(
                "time_scope",
                expected_desc,
                f"时间窗 {sorted(facts.window_days)} 天",
                "SQL 时间窗与指标定义不一致",
            ),
            True,
        )
    ]


def _compare_columns(
    metric: CaliberMetric, facts: SqlFacts
) -> list[tuple[dict[str, Any], bool]]:
    """引用列比对（measure-first）：只对 measure 列给出差异。

    measure 列缺失且 SQL 无 * 展开/COUNT(*) → 确定 conflict；否则存疑 unknown。

    `dimension_ref`（维度引用列，如 product_id/channel_id）缺失**不再产差异**：
    星型模型里一个指标几乎不可能同时引用所有维度引用列，旧实现因此把每一项
    都打成存疑，令 `consistent` 对正常 SQL 不可达（正确口径也只能 unknown），
    口径信号被稀释。维度引用是否使用属查询自由度，不是口径偏差。
    """
    expected_cols = [c for c in metric.columns if c.role == "measure"]
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
        return [
            (
                _diff(
                    "column", expected_desc, found_desc, "SQL 未解析出引用列，无法比对"
                ),
                False,
            )
        ]

    diffs: list[tuple[dict[str, Any], bool]] = []
    # 星号展开（SELECT *）与 COUNT(*) 都会让「计量列是否真的没用到」不可确认：
    # 前者可能已展开该列，后者计数对象未知 —— 二者都只给存疑，不误报 conflict。
    ambiguous_expansion = facts.has_star or facts.has_count_star
    for col in expected_cols:
        if col.column_name.lower() in facts.referenced_columns:
            continue
        if not ambiguous_expansion:
            diffs.append(
                (
                    _diff(
                        "column",
                        expected_desc,
                        found_desc,
                        f"计量列 {col.column_name} 未在 SQL 中引用",
                    ),
                    True,
                )
            )
        else:
            reason = (
                "SQL 含 * 展开无法确认"
                if facts.has_star
                else "SQL 含 COUNT(*) 计数对象无法确认"
            )
            diffs.append(
                (
                    _diff(
                        "column",
                        expected_desc,
                        found_desc,
                        f"计量列 {col.column_name} 未显式引用，{reason}",
                    ),
                    False,
                )
            )
    return diffs


# ---------------------------------------------------------------------------
# 维度限定条件：指标侧声明对齐 + 三态实质比对
# ---------------------------------------------------------------------------

# REFERS_COLUMN 里承担维度角色的 role（enum 见 metric_schema.json）
_DIM_COLUMN_ROLES = frozenset({"dimension_ref", "filter"})
# 声明式限定要求维度承担「过滤/切片」角色；group 维度需正向限定线索词才算限定
_LIMITING_USAGE_TYPES = frozenset({"filter", "slice"})
_RESTRICTION_CUE_RE = re.compile(r"仅|只|限定|限制|范围|针对|口径为|选取")
_NAME_SPLIT_RE = re.compile(r"[^0-9a-z\u4e00-\u9fa5]+")
# 枚举码值分隔：只按分隔符切，不按空格切（避免拆散带空格的取值）
_ENUM_SPLIT_RE = re.compile(r"[,，、;；/|\n]+")
_ENUM_TRIM_RE = re.compile(r"^[\s\[\('\"`]+|[\s\]\)'\"`]+$")
# 时间粒度声明（「近30天」这类窗口由 _DEF_WINDOW_RE 负责，不在此列）
_DEF_GRAIN_RE = re.compile(
    r"按\s*([日天周月季年])|每\s*([日天周月季年])|([日月季年])\s*度|([日天周月季年])\s*粒度"
)
_SQL_GRAIN_FUNC_RE = re.compile(r"\b(YEAR|QUARTER|MONTH|WEEK|DAY)\s*\(", re.IGNORECASE)
_SQL_TRUNC_RE = re.compile(
    r"DATE_TRUNC\s*\(\s*['\"]?(YEAR|QUARTER|MONTH|WEEK|DAY)['\"]?\s*,", re.IGNORECASE
)
_GRAIN_CHARS = {
    "日": "day",
    "天": "day",
    "周": "week",
    "月": "month",
    "季": "quarter",
    "年": "year",
}


@dataclass
class DimensionDecl:
    """DIMENSION_USED 指向的 Dimension 声明（code/name/granularity/values_example + 边属性 usage_type）"""

    code: str | None = None
    name: str | None = None
    usage_type: str | None = None
    values_example: str | None = None


@dataclass
class DimRequirement:
    """指标侧一条维度限定声明（实质判据）"""

    label: str
    usage_type: str
    enum_values: frozenset[str]
    expected: frozenset[str]
    excluded: frozenset[str]
    columns: frozenset[str]


def _name_tokens(value: Any) -> set[str]:
    return {t for t in _NAME_SPLIT_RE.split(str(value or "").lower()) if t}


def _split_enum_values(raw: Any) -> list[str]:
    """Dimension.values_example 字符串 -> 码值列表（只按分隔符切，不做翻译/别名归一）"""
    if raw is None:
        return []
    out: list[str] = []
    for part in _ENUM_SPLIT_RE.split(str(raw)):
        token = _ENUM_TRIM_RE.sub("", part).strip("'\"").lower()
        # 单字符取值（"1"/"0"）在中文描述里极易误命中，不作为限定证据
        if len(token) >= 2 and token not in out:
            out.append(token)
    return out


def _expected_grains(metric: CaliberMetric) -> set[str]:
    """指标声明的时间粒度：按月/月度/每季/日粒度…"""
    grains: set[str] = set()
    for m in _DEF_GRAIN_RE.finditer(_definition_text(metric)):
        char = next((g for g in m.groups() if g), "")
        if char in _GRAIN_CHARS:
            grains.add(_GRAIN_CHARS[char])
    return grains


def _sql_grains(text: str) -> set[str]:
    """SQL 侧时间粒度：YEAR(x)/DATE_TRUNC('month', x) 等显式形态"""
    grains = {m.group(1).lower() for m in _SQL_GRAIN_FUNC_RE.finditer(text)}
    grains |= {m.group(1).lower() for m in _SQL_TRUNC_RE.finditer(text)}
    return grains


def _values_desc(values: Iterable[str]) -> str:
    items = sorted({str(v) for v in values if str(v)})
    return "、".join(f"'{v}'" for v in items[:5]) or "未声明取值"


def _dim_requirements(metric: CaliberMetric) -> list[DimRequirement]:
    """把维度声明读成限定声明：Dimension.values_example 枚举码值 ∩ 指标定义文本。

    - 码值出现在定义文本且不在剔除短语内 → 声明限定的取值（expected）；
    - 码值只出现在剔除短语（剔除/不含/排除…）内 → 声明剔除的取值（excluded）；
    - usage_type=filter/slice 是显式限定维度；group 维度需定义文本有正向限定线索词；
    - 两者都为空的维度不产声明（无声明即不比对，不制造存疑噪声）。
    """
    text = _definition_text(metric).lower()
    if not text or not metric.dimensions:
        return []
    spans = [(m.start(1), m.end(1)) for m in _EXCLUSION_RE.finditer(text)]
    reqs: list[DimRequirement] = []
    for dim in metric.dimensions:
        enum = _split_enum_values(dim.values_example)
        if not enum:
            continue
        usage = (dim.usage_type or "").lower()
        qualifies = usage in _LIMITING_USAGE_TYPES or bool(
            _RESTRICTION_CUE_RE.search(text)
        )
        code_tokens = _name_tokens(dim.code)
        columns = frozenset(
            c.column_name.lower()
            for c in metric.columns
            if c.role in _DIM_COLUMN_ROLES and code_tokens & _name_tokens(c.column_name)
        )
        expected: list[str] = []
        excluded: list[str] = []
        for value in enum:
            positions = [m.start() for m in re.finditer(re.escape(value), text)]
            if not positions:
                continue
            outside = [
                p
                for p in positions
                if not any(s <= p and p + len(value) <= e for s, e in spans)
            ]
            inside = [p for p in positions if p not in outside]
            if outside and qualifies:
                expected.append(value)
            if inside and not outside:
                excluded.append(value)
        if not expected and not excluded:
            continue
        reqs.append(
            DimRequirement(
                label=str(dim.name or dim.code or "维度"),
                usage_type=usage,
                enum_values=frozenset(enum),
                expected=frozenset(expected),
                excluded=frozenset(excluded),
                columns=columns,
            )
        )
    return reqs


def _compare_dimensions(
    metric: CaliberMetric, facts: SqlFacts
) -> list[tuple[dict[str, Any], bool]]:
    """维度限定条件的实质比对：SQL 三元组 vs 指标声明的维度取值。

    判定矩阵（详见 docs/tasks/X19-caliber-dimension-substance/design.md）：

    | 声明取值 expected | SQL 侧证据                       | 判定                |
    |-------------------|----------------------------------|---------------------|
    | 有                | 等值/IN 命中 expected（含跨列） | 无差异（consistent）|
    | 有                | 命中但取值超出 expected          | unknown（取值超出） |
    | 有                | 同维度的其它声明枚举值           | conflict（实质矛盾）|
    | 有                | 限定同一列但取值不可枚举         | unknown             |
    | 有                | 只在 GROUP BY 出现该列           | unknown（分组非限定）|
    | 有                | 另有无法归属声明维度的等值限定   | unknown（不猜映射） |
    | 有                | 整条 SQL 无任何等价落点          | conflict（限定缺失）|
    | 剔除 excluded     | SQL 只取被剔除的取值             | conflict（实质矛盾）|
    | 无声明            | 任意                             | 不产差异（不增噪声）|

    conflict 只在「实质矛盾」给出；SQL 多了指标未声明的限定 → unknown + reason。
    """
    reqs = _dim_requirements(metric)
    if not reqs:
        return []
    predicates = facts.dim_predicates
    covered_columns: set[str] = set()
    all_enum_values: set[str] = set()
    for req in reqs:
        covered_columns |= set(req.columns)
        all_enum_values |= set(req.enum_values)
    positive_enum = [
        p
        for p in predicates
        if p.positive and p.op in _EXTRA_RESTRICTION_OPS and p.values
    ]

    diffs: list[tuple[dict[str, Any], bool]] = []
    unattributed_emitted = False
    for req in reqs:
        violated = [
            p
            for p in predicates
            if p.positive and p.values and p.values <= req.excluded
        ]
        if violated:
            diffs.append(
                (
                    _diff(
                        "filter",
                        f"剔除 {_values_desc(req.excluded)}",
                        violated[0].raw,
                        "SQL 把结果限定到指标声明剔除的取值（实质矛盾）",
                    ),
                    True,
                )
            )
        expected = req.expected
        others = req.enum_values - expected
        hit = [p for p in predicates if p.positive and p.values & expected]
        if hit:
            over = [p for p in hit if not p.values <= expected]
            if over:
                diffs.append(
                    (
                        _diff(
                            "filter",
                            _values_desc(expected),
                            over[0].raw,
                            f"SQL 对维度 {req.label} 的取值超出指标声明的限定取值",
                        ),
                        False,
                    )
                )
            continue
        if not expected:
            continue

        sibling = [
            p
            for p in positive_enum
            if p.values & others and (not req.columns or p.column in req.columns)
        ]
        stray_sibling = [p for p in positive_enum if p.values & others]
        same_col = [p for p in predicates if p.positive and p.column in req.columns]
        if sibling:
            diffs.append(
                (
                    _diff(
                        "filter",
                        _values_desc(expected),
                        sibling[0].raw,
                        f"SQL 把维度 {req.label} 限定到指标未选取的声明取值（实质矛盾）",
                    ),
                    True,
                )
            )
        elif same_col:
            diffs.append(
                (
                    _diff(
                        "filter",
                        _values_desc(expected),
                        same_col[0].raw,
                        f"SQL 限定了维度 {req.label} 所在列，但取值无法与声明枚举比对",
                    ),
                    False,
                )
            )
        elif req.columns & facts.group_by_columns:
            grouped = "、".join(sorted(req.columns & facts.group_by_columns))
            diffs.append(
                (
                    _diff(
                        "filter",
                        _values_desc(expected),
                        f"GROUP BY {grouped}",
                        f"SQL 按维度 {req.label} 分组而非限定，取值范围无法确认",
                    ),
                    False,
                )
            )
        elif stray_sibling or positive_enum:
            # SQL 里有归属不到声明维度的等值限定：可能由它实现该限定，不猜映射
            other = stray_sibling[0] if stray_sibling else positive_enum[0]
            diffs.append(
                (
                    _diff(
                        "filter",
                        _values_desc(expected),
                        other.raw,
                        f"指标声明 {req.label} 限定到 {_values_desc(expected)}，"
                        "但 SQL 的等值限定无法归属到该维度，未确证",
                    ),
                    False,
                )
            )
            unattributed_emitted = True
        else:
            diffs.append(
                (
                    _diff(
                        "filter",
                        _values_desc(expected),
                        "SQL WHERE 未见该维度限定",
                        f"指标声明 {req.label} 限定到 {_values_desc(expected)}，"
                        "SQL 未见任何等价取值限定（实质矛盾）",
                    ),
                    True,
                )
            )

    if not unattributed_emitted:
        extra = [
            p
            for p in positive_enum
            if p.column not in covered_columns and not p.values & all_enum_values
        ]
        if extra:
            desc = "；".join(p.raw for p in sorted(extra, key=lambda x: x.raw)[:3])
            diffs.append(
                (
                    _diff(
                        "filter",
                        "指标声明的维度限定",
                        desc,
                        "SQL 附加了指标未声明的维度限定，可能收窄口径（存疑，不判冲突）",
                    ),
                    False,
                )
            )
    return diffs


def _compare_time_grain(
    metric: CaliberMetric, facts: SqlFacts
) -> list[tuple[dict[str, Any], bool]]:
    """时间粒度比对（time_scope 侧的实质比对补充）。

    指标声明粒度（按月/季度…）且 SQL 有显式粒度（DATE_TRUNC/YEAR(...)）而不相交 →
    确定 conflict；SQL 无显式粒度 → 不产差异（按时间分组属查询自由度，不是口径偏差）。
    """
    expected = _expected_grains(metric)
    if not expected or not facts.grains:
        return []
    if expected & facts.grains:
        return []
    return [
        (
            _diff(
                "time_scope",
                "时间粒度 " + "、".join(sorted(expected)),
                "时间粒度 " + "、".join(sorted(facts.grains)),
                "SQL 时间粒度与指标声明不一致",
            ),
            True,
        )
    ]


def compare_metric(metric: CaliberMetric, facts: SqlFacts) -> dict[str, Any]:
    """单指标比对：返回 {metric, status, differences}

    status：任一确定差异 → conflict；仅存疑项 → unknown；否则 consistent。
    differences 只含 aspect/expected/found/detail（不带判定，判定收在 status）。
    """
    if not facts.parse_ok:
        return {
            "metric": _metric_summary(metric),
            "status": "unknown",
            "differences": [
                _diff(
                    "sql",
                    "可解析的 SQL",
                    "无法解析 SQL 结构",
                    "SQL 文本未解析出可比对结构",
                )
            ],
        }

    verdicts = (
        _compare_aggregation(metric, facts)
        + _compare_filter(metric, facts)
        + _compare_time_scope(metric, facts)
        + _compare_columns(metric, facts)
        + _compare_dimensions(metric, facts)
        + _compare_time_grain(metric, facts)
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
    node_id: str | None,
    attrs: dict[str, Any],
    columns: list[MetricColumn],
    dimensions: list[DimensionDecl] | None = None,
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
        dimensions=list(dimensions or []),
    )


def _match_ref(ref: str, *candidates: Any) -> bool:
    return any(c is not None and str(c) == ref for c in candidates)


def _read_dimensions_networkx(graph: Any, node_key: Any) -> list[DimensionDecl]:
    """读取 DIMENSION_USED -> Dimension 声明；图库缺该标签/边时降级为无维度声明。

    降级而非报错：口径比对不能因为维度声明读不到就整体失败
    （历史缺陷 BUGFIX-caliber-has-star：DIMENSION_USED 查询报错导致检测静默跳过）。
    """
    dims: list[DimensionDecl] = []
    try:
        for _u, v, edata in graph.out_edges(node_key, data=True):
            if edata.get("edge_type") != "DIMENSION_USED":
                continue
            attrs = graph.nodes[v]
            if attrs.get("node_type") != "Dimension":
                continue
            dims.append(
                DimensionDecl(
                    code=attrs.get("code"),
                    name=attrs.get("name"),
                    usage_type=edata.get("usage_type"),
                    values_example=attrs.get("values_example"),
                )
            )
    except Exception as exc:
        logger.warning("读取 DIMENSION_USED 失败，按无维度声明处理：%s", exc)
    return dims


def _read_dimensions_cypher(graph: Any, ref: str) -> list[DimensionDecl]:
    """Cypher 后端读取维度声明（属性名一律取 schema.md，同 BUGFIX-caliber-cypher 纪律）"""
    try:
        rows = graph.query(
            "MATCH (m:Metric)-[r:DIMENSION_USED]->(d:Dimension) "
            "RETURN m.id, m.code, d.code, d.name, d.values_example, r.usage_type"
        )
    except Exception as exc:
        logger.warning("读取 DIMENSION_USED 失败，按无维度声明处理：%s", exc)
        return []
    dims: list[DimensionDecl] = []
    for row in rows:
        if not _match_ref(ref, row[0], row[1]):
            continue
        dims.append(
            DimensionDecl(
                code=str(row[2]) if row[2] is not None else None,
                name=str(row[3]) if row[3] is not None else None,
                values_example=str(row[4]) if row[4] is not None else None,
                usage_type=str(row[5]) if row[5] is not None else None,
            )
        )
    return dims


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
            name = (
                v_attrs.get("column_name")
                or str(v_attrs.get("column", "")).rsplit(".", 1)[-1]
            )
            columns.append(
                MetricColumn(
                    column_name=name or str(v), role=str(edata.get("role") or "")
                )
            )
        return _metric_from_attrs(
            str(node_key), attrs, columns, _read_dimensions_networkx(g, node_key)
        )
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
    return _metric_from_attrs(
        str(attrs["id"]), attrs, columns, _read_dimensions_cypher(graph, ref)
    )


def read_metric_caliber(graph: Any, ref: str) -> CaliberMetric | None:
    """按 code 或节点 ID 读取指标定义 + REFERS_COLUMN；未找到返回 None

    支持 NetworkXGraph（.G）与 Cypher 后端（FalkorDBGraph / LadybugGraph，.query）。
    """
    if hasattr(graph, "G"):
        return _read_metric_networkx(graph, ref)
    if hasattr(graph, "query"):
        return _read_metric_cypher(graph, ref)
    return None
