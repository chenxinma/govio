"""指标问数 SQL 组装器

根据指标元数据生成分析 SQL，支持：
- 原子指标直接查询
- 派生指标通过 CTE 组合
- 维度过滤和分组
- 时间范围过滤
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class MetricInfo:
    """指标元数据"""
    code: str
    name: str
    type: Literal["原子", "派生"]
    source_table: str | None = None
    formula: str | None = None
    dimensions: list[str] = field(default_factory=list)
    time_column: str = "report_ym"


@dataclass
class QueryRequest:
    """查询请求"""
    metrics: list[MetricInfo]
    dimensions: list[str] = field(default_factory=list)
    filters: dict[str, str] = field(default_factory=dict)
    order_by: str | None = None
    limit: int = 100


def build_metric_sql(
    metrics: list[dict],
    dimensions: list[str] | None = None,
    filters: dict[str, str] | None = None,
    order_by: str | None = None,
    limit: int = 100,
    cte_refs: dict[str, str] | None = None,
) -> str:
    """组装指标查询 SQL

    Args:
        metrics: 指标列表，每个指标包含 code, name, type, source_table, formula, time_column, actual_column
        dimensions: 分组维度字段列表，如 ["sales_unit", "sales_dept"]
        filters: 过滤条件，如 {"report_ym": "202605", "sales_unit": "华东区"}
        order_by: 排序字段，如 "metric_value DESC"
        limit: 返回行数限制
        cte_refs: 已加载的 DataFrame CTE 引用，如 {"df_customers": "SELECT * FROM ..."}

    Returns:
        组装好的 SQL 语句
    """
    if not metrics:
        raise ValueError("至少需要一个指标")

    dimensions = dimensions or []
    filters = filters or {}
    cte_refs = cte_refs or {}

    _check_report_ym_required(metrics, dimensions, filters)

    atomic_metrics = [m for m in metrics if m.get("type") == "原子"]
    derived_metrics = [m for m in metrics if m.get("type") == "派生"]

    if not atomic_metrics and not derived_metrics:
        raise ValueError("指标类型必须为'原子'或'派生'")

    cte_parts = []

    for cte_name, cte_sql in cte_refs.items():
        cte_parts.append(f"{cte_name} AS ({cte_sql})")

    tables: dict[str, list[dict]] = {}
    if atomic_metrics:
        for m in atomic_metrics:
            table = m.get("source_table", "")
            if not table:
                raise ValueError(f"原子指标 {m['code']} 缺少 source_table")
            tables.setdefault(table, []).append(m)

        for table, table_metrics in tables.items():
            cte_name = f"atomic_{table.split('.')[-1]}"
            select_parts = []

            for dim in dimensions:
                select_parts.append(f"    {dim}")

            for m in table_metrics:
                metric_col = m.get("actual_column", m["code"])
                if dimensions:
                    select_parts.append(f"    SUM({metric_col}) AS {m['code']}")
                else:
                    select_parts.append(f"    {metric_col} AS {m['code']}")

            where_parts = _build_where_conditions(filters, table_metrics)

            sql = f"{cte_name} AS (\n"
            sql += f"  SELECT\n"
            sql += ",\n".join(select_parts)
            sql += f"\n  FROM {table}"

            if where_parts:
                sql += f"\n  WHERE {' AND '.join(where_parts)}"

            if dimensions:
                sql += f"\n  GROUP BY {', '.join(dimensions)}"

            sql += "\n)"
            cte_parts.append(sql)

    if derived_metrics:
        for m in derived_metrics:
            formula = m.get("formula", "")
            if not formula:
                raise ValueError(f"派生指标 {m['code']} 缺少 formula")

            cte_name = f"derived_{m['code']}"
            select_parts = []

            for dim in dimensions:
                select_parts.append(f"    {dim}")

            formula_expr = _resolve_formula(formula, atomic_metrics)
            select_parts.append(f"    {formula_expr} AS {m['code']}")

            source_cte = _find_source_cte(formula, atomic_metrics, tables if atomic_metrics else {})

            sql = f"{cte_name} AS (\n"
            sql += f"  SELECT\n"
            sql += ",\n".join(select_parts)
            sql += f"\n  FROM {source_cte}"
            sql += "\n)"
            cte_parts.append(sql)

    final_select = []
    for dim in dimensions:
        final_select.append(f"  {dim}")

    all_metrics = atomic_metrics + derived_metrics

    if len(all_metrics) == 1:
        m = all_metrics[0]
        cte_name = _get_cte_name(m, atomic_metrics, tables if atomic_metrics else {})
        final_select.append(f"  {m['code']} AS metric_value")
        final_select.append(f"  '{m['name']}' AS metric_name")
        from_cte = cte_name
    else:
        union_parts = []
        for m in all_metrics:
            cte_name = _get_cte_name(m, atomic_metrics, tables if atomic_metrics else {})
            dim_select = ", ".join(dimensions) if dimensions else "NULL AS dim_key"
            union_parts.append(
                f"SELECT {dim_select}, {m['code']} AS metric_value, '{m['name']}' AS metric_name FROM {cte_name}"
            )

        final_select = [
            "  *"
        ]
        from_cte = "(" + " UNION ALL ".join(union_parts) + ") t"

    sql = "WITH\n"
    sql += ",\n".join(cte_parts)
    sql += "\n\nSELECT\n"
    sql += ",\n".join(final_select)
    sql += f"\nFROM {from_cte}"

    if order_by:
        sql += f"\nORDER BY {order_by}"

    sql += f"\nLIMIT {limit}"

    return sql


def _check_report_ym_required(
    metrics: list[dict],
    dimensions: list[str],
    filters: dict[str, str],
) -> None:
    """校验 report_ym 必须条件"""
    for m in metrics:
        time_col = m.get("time_column", "report_ym")
        need_check = time_col == "report_ym" or time_col in dimensions
        if need_check:
            value = filters.get(time_col, "").strip()
            if not value:
                raise ValueError(
                    f"指标 {m['code']} 的来源表包含 {time_col} 字段，"
                    f"该字段为必须过滤条件（拉链表不同时期数据不可合并）。"
                    f"请在 filters 中指定 {time_col}，如 {{\"{time_col}\": \"最新年月\"}}"
                )


def _build_where_conditions(
    filters: dict[str, str],
    metrics: list[dict],
) -> list[str]:
    """构建 WHERE 条件"""
    conditions = []
    for key, value in filters.items():
        if key in ("report_ym", "ym", "forecast_ym"):
            conditions.append(f"{key} = '{value}'")
        else:
            conditions.append(f"{key} = '{value}'")
    return conditions


def _resolve_formula(formula: str, atomic_metrics: list[dict]) -> str:
    """解析公式中的指标引用"""
    result = formula
    for m in atomic_metrics:
        code = m["code"]
        result = result.replace(code, f"t.{code}")
    return result


def _find_source_cte(
    formula: str,
    atomic_metrics: list[dict],
    tables: dict[str, list[dict]],
) -> str:
    """根据公式找到数据来源 CTE"""
    referenced = []
    for m in atomic_metrics:
        if m["code"] in formula:
            referenced.append(m)

    if not referenced:
        raise ValueError(f"公式 {formula} 中未找到引用的原子指标")

    first_metric = referenced[0]
    table = first_metric.get("source_table", "")
    return f"atomic_{table.split('.')[-1]}"


def _get_cte_name(
    metric: dict,
    atomic_metrics: list[dict],
    tables: dict[str, list[dict]],
) -> str:
    """获取指标对应的 CTE 名称"""
    if metric["type"] == "原子":
        table = metric.get("source_table", "")
        return f"atomic_{table.split('.')[-1]}"
    else:
        return f"derived_{metric['code']}"
