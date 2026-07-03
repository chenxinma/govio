# sql_builder CLI 集成 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `skills/govio-query/scripts/sql_builder.py` 的 SQL 组装逻辑迁入 `govio` 包，提供 `govio-cli sql build` 子命令，删除原脚本并更新 SKILL.md。

**Architecture:** 核心逻辑 `build_metric_sql()` 放 `src/govio/core/sql_builder.py`（可复用库函数），CLI 壳放 `src/govio/cli/sql.py`（argparse + 文件读写 + 错误处理），`main.py` 用 REMAINDER 模式注册 `sql` 命令组。JSON 规格格式不变，零迁移成本。

**Tech Stack:** Python 3.13+，argparse，pytest，无新依赖。

**Spec:** `docs/superpowers/specs/2026-07-03-sql-builder-cli-integration-design.md`

---

## File Structure

| 文件 | 操作 | 职责 |
|------|------|------|
| `src/govio/core/sql_builder.py` | 创建 | `build_metric_sql()` + dataclass + 辅助函数 |
| `src/govio/cli/sql.py` | 创建 | CLI 入口 `sql()` + `_build()` 子函数 |
| `src/govio/cli/main.py` | 修改 | 注册 `sql` 子命令组并 dispatch |
| `src/govio/__init__.py` | 修改 | 导出 `build_metric_sql` |
| `tests/test_sql_builder.py` | 创建 | 单元测试 + CLI 集成测试 |
| `skills/govio-query/scripts/sql_builder.py` | 删除 | 原脚本 |
| `skills/govio-query/SKILL.md` | 修改 | 脚本调用 → `govio-cli sql build` |
| `/data/home/.../govio-query/SKILL.md` | 修改 | 运行时同步 |

---

## Task 1: 迁移 build_metric_sql 到 core 模块

**Files:**
- Create: `src/govio/core/sql_builder.py`
- Test: `tests/test_sql_builder.py`

- [ ] **Step 1: 写失败的单元测试**

创建 `tests/test_sql_builder.py`：

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_sql_builder.py -v`
Expected: 全部 FAIL，错误信息含 `ModuleNotFoundError: No module named 'govio.core.sql_builder'`

- [ ] **Step 3: 创建 src/govio/core/sql_builder.py**

从 `skills/govio-query/scripts/sql_builder.py` 迁入核心逻辑，移除 `main()` 和 argparse 部分：

```python
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

        final_select = ["  *"]
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `uv run pytest tests/test_sql_builder.py -v`
Expected: 9 passed

- [ ] **Step 5: 提交**

```bash
git add src/govio/core/sql_builder.py tests/test_sql_builder.py
git commit -m "feat: migrate build_metric_sql to govio.core.sql_builder"
```

---

## Task 2: 创建 CLI 壳 src/govio/cli/sql.py

**Files:**
- Create: `src/govio/cli/sql.py`
- Test: `tests/test_sql_builder.py`（追加 CLI 集成测试）

- [ ] **Step 1: 追加失败的 CLI 集成测试**

在 `tests/test_sql_builder.py` 末尾追加：

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_sql_builder.py -v -k "cli"`
Expected: CLI 测试 FAIL，错误含 `sql` 子命令不存在（main.py 未注册）

- [ ] **Step 3: 创建 src/govio/cli/sql.py**

```python
"""govio-cli sql 命令组：指标 SQL 组装"""

import argparse
import json
import sys
from pathlib import Path

from govio.core.sql_builder import build_metric_sql


def sql():
    """sql 命令组入口"""
    parser = argparse.ArgumentParser(prog="govio-cli sql")
    sub = parser.add_subparsers(dest="sql_action")

    p_build = sub.add_parser("build", help="从 JSON 规格组装 SQL")
    p_build.add_argument("-f", "--file", help="JSON 规格文件路径")
    p_build.add_argument("-o", "--output", help="输出 SQL 文件路径，省略则打印到 stdout")

    args = parser.parse_args()

    if args.sql_action == "build":
        _build(args)
    else:
        parser.print_help()
        sys.exit(1)


def _build(args):
    """sql build 子命令实现"""
    if args.file:
        try:
            text = Path(args.file).read_text(encoding="utf-8")
        except FileNotFoundError:
            print(f"错误：文件不存在：{args.file}", file=sys.stderr)
            sys.exit(1)
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        print("错误：请通过 -f 指定 JSON 文件或通过 stdin 传入", file=sys.stderr)
        sys.exit(1)

    try:
        req = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"错误：JSON 解析失败：{e}", file=sys.stderr)
        sys.exit(1)

    try:
        sql_text = build_metric_sql(
            metrics=req["metrics"],
            dimensions=req.get("dimensions"),
            filters=req.get("filters"),
            order_by=req.get("order_by"),
            limit=req.get("limit", 100),
            cte_refs=req.get("cte_refs"),
        )
    except (ValueError, KeyError) as e:
        print(f"错误：SQL 组装失败：{e}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        Path(args.output).write_text(sql_text + "\n", encoding="utf-8")
    else:
        print(sql_text)
```

- [ ] **Step 4: 运行测试确认仍失败（main.py 未注册 sql 命令）**

Run: `uv run pytest tests/test_sql_builder.py -v -k "cli"`
Expected: CLI 测试仍 FAIL，`govio-cli sql build` 报未知命令（因 main.py 未 dispatch）

- [ ] **Step 5: 提交**

```bash
git add src/govio/cli/sql.py tests/test_sql_builder.py
git commit -m "feat: add govio-cli sql build subcommand wrapper"
```

---

## Task 3: 在 main.py 注册 sql 命令组

**Files:**
- Modify: `src/govio/cli/main.py`
- Modify: `src/govio/cli/__init__.py`

- [ ] **Step 1: 修改 src/govio/cli/main.py 注册 sql 子命令**

在 `src/govio/cli/main.py` 中：

a) 在 import 区（第 7-10 行附近）追加：

```python
from .sql import sql
```

修改后的 import 块：

```python
from .onboard import onboard
from .observe import observe
from .query import query
from .meta import meta
from .sql import sql
```

b) 在 `p_observe` 子命令注册后（第 61 行后）追加 sql 子命令注册：

```python
    # sql 子命令组
    p_sql = sub.add_parser("sql", help="指标 SQL 组装", add_help=False)
    p_sql.add_argument(
        "sql_args", nargs=argparse.REMAINDER, help="sql 子命令参数"
    )
```

c) 在 dispatch 区（`elif args.action == "observe":` 块后，第 86 行后）追加：

```python
    elif args.action == "sql":
        sys.argv = ["govio-cli"] + args.sql_args + remaining
        sql()
```

- [ ] **Step 2: 运行 CLI 测试确认通过**

Run: `uv run pytest tests/test_sql_builder.py -v -k "cli"`
Expected: 6 CLI 测试 PASS

- [ ] **Step 3: 手动验证命令注册**

Run: `uv run python -m govio.cli sql --help`
Expected: 输出含 `build  从 JSON 规格组装 SQL`

Run: `uv run python -m govio.cli sql build --help`
Expected: 输出含 `-f FILE` 和 `-o OUTPUT`

- [ ] **Step 4: 运行全量测试确认无回归**

Run: `uv run pytest tests/ -v`
Expected: 全部 PASS（含原有测试 + 新增 15 个 sql_builder 测试）

- [ ] **Step 5: 提交**

```bash
git add src/govio/cli/main.py
git commit -m "feat: register sql subcommand group in govio-cli main"
```

---

## Task 4: 导出 build_metric_sql

**Files:**
- Modify: `src/govio/__init__.py`

- [ ] **Step 1: 修改 src/govio/__init__.py**

```python
from .graph.falkordb_graph import FalkorDBGraph
from .graph.networkx_graph import NetworkXGraph
from .cli import main
from .core.sql_builder import build_metric_sql

__all__ = [
    "FalkorDBGraph",
    "NetworkXGraph",
    "main",
    "build_metric_sql",
]
```

- [ ] **Step 2: 验证导出**

Run: `uv run python -c "from govio import build_metric_sql; print(build_metric_sql.__module__)"`
Expected: 输出 `govio.core.sql_builder`

- [ ] **Step 3: 提交**

```bash
git add src/govio/__init__.py
git commit -m "feat: export build_metric_sql from govio package"
```

---

## Task 5: 删除旧脚本，更新 SKILL.md

**Files:**
- Delete: `skills/govio-query/scripts/sql_builder.py`
- Modify: `skills/govio-query/SKILL.md`
- Modify: `/data/home/macx/work/tmp/govio_runtime/.claude/skills/govio-query/SKILL.md`

- [ ] **Step 1: 删除旧脚本**

```bash
git rm skills/govio-query/scripts/sql_builder.py
rmdir skills/govio-query/scripts 2>/dev/null || true
```

- [ ] **Step 2: 更新 skills/govio-query/SKILL.md 的脚本调用**

a) 找到"使用 `scripts/sql_builder.py` 脚本组装 SQL"段落（约第 145 行），将：

```markdown
使用 `scripts/sql_builder.py` 脚本组装 SQL。接受 JSON 文件作为输入。

#### 调用方式

```bash
# 打印到 stdout
uv run python scripts/sql_builder.py query.json

# 输出到文件
uv run python scripts/sql_builder.py query.json -o output.sql
```
```

替换为：

```markdown
使用 `govio-cli sql build` 命令组装 SQL。接受 JSON 文件作为输入。

#### 调用方式

```bash
# 打印到 stdout
govio-cli sql build -f query.json

# 输出到文件
govio-cli sql build -f query.json -o output.sql

# 从 stdin 读取
cat query.json | govio-cli sql build
```
```

b) 找到"#### 场景：环比分析"中的脚本调用（约第 259 行），将：

```bash
uv run python scripts/sql_builder.py current.json -o current.sql
```

替换为：

```bash
govio-cli sql build -f current.json -o current.sql
```

c) 同段落下一行，将：

```bash
uv run python scripts/sql_builder.py compare.json -o compare.sql
```

替换为：

```bash
govio-cli sql build -f compare.json -o compare.sql
```

- [ ] **Step 3: 更新 skills/govio-query/SKILL.md 的资源文件章节**

找到"## 资源文件"章节末尾的 `本 Skill 目录下:` 块，将：

```
本 Skill 目录下:
├── reference-falkordb.md   # FalkorDB Cypher 深度参考
├── reference-networkx.md   # NetworkX Python 深度参考
└── scripts/
     └── sql_builder.py     # SQL 组装脚本（CLI）
```

替换为：

```
本 Skill 目录下:
├── reference-falkordb.md   # FalkorDB Cypher 深度参考
└── reference-networkx.md   # NetworkX Python 深度参考
```

- [ ] **Step 4: 同步更新运行时 SKILL.md**

把源 SKILL.md 复制到运行时：

```bash
cp skills/govio-query/SKILL.md /data/home/macx/work/tmp/govio_runtime/.claude/skills/govio-query/SKILL.md
```

并删除运行时的旧脚本目录：

```bash
rm -rf /data/home/macx/work/tmp/govio_runtime/.claude/skills/govio-query/scripts
```

- [ ] **Step 5: 全仓 grep 确认无残留引用**

Run: `grep -rn "sql_builder.py" --include="*.md" --include="*.py" --include="*.toml" . 2>/dev/null | grep -v "\.venv\|\.git"`
Expected: 无输出（或仅 `docs/superpowers/specs/` 和 `docs/superpowers/plans/` 中的历史文档引用，可忽略）

Run: `grep -rn "scripts/sql_builder" --include="*.md" . 2>/dev/null | grep -v "\.venv\|\.git"`
Expected: 无输出

- [ ] **Step 6: 提交**

```bash
git add skills/govio-query/SKILL.md
git commit -m "refactor: replace sql_builder.py script with govio-cli sql build

Remove skills/govio-query/scripts/sql_builder.py, update SKILL.md to
invoke `govio-cli sql build` CLI. Sync runtime SKILL.md."
```

---

## Task 6: 本地 CLI 烟雾测试

**Files:** 无文件改动，仅验证

- [ ] **Step 1: 构建并安装**

```bash
uv build
WHL=$(ls dist/govio-*.whl | head -1)
uv tool install --from "$WHL" govio --force
```
Expected: 输出 `✓ govio 已安装` 或类似成功信息

- [ ] **Step 2: 验证 sql 子命令可用**

Run: `govio-cli sql --help`
Expected: 输出含 `build  从 JSON 规格组装 SQL`

- [ ] **Step 3: 端到端验证**

```bash
cat > /tmp/test_query.json <<'EOF'
{
  "metrics": [
    {"code": "bill_income_amt", "name": "当月账单收入", "type": "原子", "source_table": "dws.income_bill_monthly"}
  ],
  "dimensions": ["sales_dept"],
  "filters": {"report_ym": "2026-05"},
  "order_by": "metric_value DESC",
  "limit": 10
}
EOF
govio-cli sql build -f /tmp/test_query.json
```
Expected: 输出完整 SQL，含 `WITH`、`atomic_income_bill_monthly`、`GROUP BY sales_dept`、`ORDER BY metric_value DESC`、`LIMIT 10`

- [ ] **Step 4: 验证 -o 输出文件**

```bash
govio-cli sql build -f /tmp/test_query.json -o /tmp/test_out.sql
cat /tmp/test_out.sql
```
Expected: 文件内容与 Step 3 stdout 一致

- [ ] **Step 5: 验证 stdin 管道**

```bash
cat /tmp/test_query.json | govio-cli sql build
```
Expected: 输出与 Step 3 一致

- [ ] **Step 6: 验证错误场景**

```bash
govio-cli sql build -f /nonexistent.json
echo "exit: $?"
```
Expected: stderr 含 `文件不存在`，exit code 1

```bash
echo "{invalid" | govio-cli sql build
echo "exit: $?"
```
Expected: stderr 含 `JSON 解析失败`，exit code 1

- [ ] **Step 7: 清理临时文件**

```bash
rm -f /tmp/test_query.json /tmp/test_out.sql
```

---

## Self-Review 已完成

**Spec 覆盖**：spec 的 10 个实施步骤全部映射到 Task 1-6。
- 步骤 1 → Task 1
- 步骤 2 → Task 2
- 步骤 3 → Task 3
- 步骤 4 → Task 4
- 步骤 5-7 → Task 5
- 步骤 8 → Task 1/2 测试
- 步骤 9 → Task 3 Step 4 + Task 6
- 步骤 10 → Task 6

**Placeholder 扫描**：无 TBD/TODO，所有代码块完整。

**类型一致性**：`build_metric_sql` 签名在 Task 1 定义，Task 2 CLI 调用参数一致（metrics/dimensions/filters/order_by/limit/cte_refs）。
