# sql_builder 集成到 govio-cli 设计

**日期**: 2026-07-03
**状态**: 已批准
**关联**: 2026-07-02-cli-skills-restructure-design.md（CLI 命令组重构）

## 背景

`skills/govio-query/scripts/sql_builder.py` 是指标问数流程中的 SQL 组装脚本，根据指标元数据 JSON 规格生成分析 SQL。当前作为独立脚本运行：

```bash
uv run python scripts/sql_builder.py query.json
```

存在两个问题：

1. **路径不可靠**：SKILL.md 中的 bash 命令 `uv run python scripts/sql_builder.py` 从项目根执行，但脚本位于 `skills/govio-query/scripts/` 下，路径无法解析。运行时部署到 `.claude/skills/govio-query/scripts/` 后路径更不可控。
2. **职责分散**：核心 SQL 组装逻辑（`build_metric_sql()`）是有价值的业务能力，但被困在 skill 脚本中，无法被包内其他模块复用，也不随 `govio-cli` 安装分发。

## 目标

- 将 `build_metric_sql()` 核心逻辑迁入 `govio` 包，作为可复用库函数
- 提供 `govio-cli sql build` 子命令，替代独立脚本
- 删除原脚本，更新 SKILL.md 引用
- 保持 JSON 规格格式不变，零迁移成本

## 非目标

- 不改变 `build_metric_sql()` 的内部实现和 SQL 生成逻辑
- 不增加 CLI 参数模式（如 `--metric code1 --filter k=v`），JSON 文件已能表达全部场景
- 不实现 `sql validate`、`sql explain` 等子命令（仅预留命令组结构）

## 设计

### 架构

```
src/govio/
├── core/
│   └── sql_builder.py      # build_metric_sql() + dataclass（从 skills/ 迁入）
├── cli/
│   ├── sql.py              # 新增：CLI 壳，argparse + 调用 core
│   └── main.py             # 注册 sql 子命令组
└── __init__.py             # 导出 build_metric_sql

skills/govio-query/
├── SKILL.md                # 更新：scripts 调用 → govio-cli sql build
└── (scripts/ 目录删除)
```

**职责分离**：
- `core/sql_builder.py` 放可复用的 `build_metric_sql()` 函数和 `MetricInfo`/`QueryRequest` dataclass
- `cli/sql.py` 只做参数解析、文件读写、调用核心函数、错误处理
- 逻辑可被包内其他模块（如未来的批量指标查询）复用，也可被测试直接调用

### CLI 接口

```bash
# 从 JSON 文件读取，SQL 输出到 stdout
govio-cli sql build -f query.json

# 输出到文件
govio-cli sql build -f query.json -o out.sql

# 从 stdin 读取（管道场景）
cat query.json | govio-cli sql build
```

**参数**：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `-f, --file <path>` | keyword | 二选一 | JSON 规格文件路径 |
| `-o, --output <path>` | keyword | 否 | 输出 SQL 文件路径，省略则 stdout |

**stdin 行为**：未指定 `-f` 且 stdin 非 tty 时（`not sys.stdin.isatty()`），自动从 stdin 读取 JSON。两者都未提供时报错退出。

**JSON 格式**（与现有 `build_metric_sql()` 入参完全一致）：

```json
{
  "metrics": [
    {
      "code": "bill_income_amt",
      "name": "当月账单收入",
      "type": "原子",
      "source_table": "dws.income_bill_monthly",
      "time_column": "report_ym"
    }
  ],
  "dimensions": ["sales_unit", "sales_dept"],
  "filters": {"report_ym": "2026-05"},
  "order_by": "metric_value DESC",
  "limit": 100,
  "cte_refs": {}
}
```

字段说明与 `build_metric_sql()` 签名一致，`metrics` 数组每项支持 `code`/`name`/`type`/`source_table`/`formula`/`actual_column`/`time_column`。

**输出**：纯 SQL 文本，末尾换行。

### main.py 集成

参考现有 `meta`/`observe` 命令组的 REMAINDER 模式，`sql` 也用子命令结构（为未来 `sql validate` 等扩展留空间）：

```python
# main.py - 注册
p_sql = sub.add_parser("sql", help="指标 SQL 组装", add_help=False)
p_sql.add_argument("sql_args", nargs=argparse.REMAINDER, help="sql 子命令参数")

# dispatch
elif args.action == "sql":
    sys.argv = ["govio-cli"] + args.sql_args + remaining
    sql()
```

```python
# cli/sql.py
def sql():
    parser = argparse.ArgumentParser(prog="govio-cli sql")
    sub = parser.add_subparsers(dest="sql_action")

    p_build = sub.add_parser("build", help="从 JSON 规格组装 SQL")
    p_build.add_argument("-f", "--file", help="JSON 规格文件路径")
    p_build.add_argument("-o", "--output", help="输出 SQL 文件路径")

    args = parser.parse_args()
    if args.sql_action == "build":
        _build(args)
    else:
        parser.print_help()
        sys.exit(1)


def _build(args):
    # 读取 JSON：-f 优先，否则 stdin（非 tty 时）
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
        sql = build_metric_sql(
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
        Path(args.output).write_text(sql + "\n", encoding="utf-8")
    else:
        print(sql)
```

### 错误处理

| 场景 | 行为 | exit code |
|------|------|-----------|
| JSON 文件不存在 | stderr: `错误：文件不存在：<path>` | 1 |
| JSON 解析失败 | stderr: `错误：JSON 解析失败：<详情>` | 1 |
| `build_metric_sql()` 抛 `ValueError`（如缺 report_ym） | stderr: `错误：SQL 组装失败：<msg>` | 1 |
| `-f` 和 stdin 都未提供 | stderr: `错误：请通过 -f 指定...` | 1 |
| 成功 | stdout 输出 SQL（或写文件） | 0 |

错误信息保持中文，与现有 CLI 风格一致。不引入新依赖。

### SKILL.md 更新

`skills/govio-query/SKILL.md` 中所有脚本调用替换为 CLI：

| 原写法 | 新写法 |
|--------|--------|
| `uv run python scripts/sql_builder.py query.json` | `govio-cli sql build -f query.json` |
| `uv run python scripts/sql_builder.py query.json -o output.sql` | `govio-cli sql build -f query.json -o output.sql` |
| `uv run python scripts/sql_builder.py current.json -o current.sql` | `govio-cli sql build -f current.json -o current.sql` |
| `uv run python scripts/sql_builder.py compare.json -o compare.sql` | `govio-cli sql build -f compare.json -o compare.sql` |

"资源文件"章节移除 `scripts/sql_builder.py` 条目；"调用方式"章节改用 CLI 命令。

同步更新运行时 `/data/home/macx/work/tmp/govio_runtime/.claude/skills/govio-query/SKILL.md`。

### 包导出

`src/govio/__init__.py` 导出 `build_metric_sql`，供包外复用：

```python
from .core.sql_builder import build_metric_sql
__all__ = ["run", "gml_generate", "FalkorDBGraph", "NetworkXGraph", "build_metric_sql"]
```

### 测试

新增 `tests/test_sql_builder.py`：

**单元测试**（直接调用 `build_metric_sql()`，不经 CLI）：
- 单原子指标：验证 SELECT/FROM/WHERE/GROUP BY 结构
- 多原子指标同表：验证合并为单个 CTE
- 多原子指标异表：验证多个 atomic_* CTE
- 派生指标：验证 `derived_*` CTE 和公式解析
- `report_ym` 校验：缺 filter 时抛 `ValueError`
- `cte_refs` 引用：验证 CTE 前置注入
- `dimensions` 为空：验证无 GROUP BY
- `order_by` / `limit`：验证尾部子句

**CLI 集成测试**（`subprocess` 调用 `govio-cli sql build`）：
- `-f` 文件输入 + stdout 输出
- `-f` 文件输入 + `-o` 文件输出
- stdin 管道输入
- JSON 解析失败 → exit 1
- 缺 report_ym → exit 1

测试 JSON fixture 复用 SKILL.md 中的示例结构。

## 实施步骤

1. 创建 `src/govio/core/sql_builder.py`，从 `skills/govio-query/scripts/sql_builder.py` 迁入 `build_metric_sql()`、`MetricInfo`、`QueryRequest` 及全部辅助函数（`_check_report_ym_required` 等），移除 `main()` 和 argparse 部分
2. 创建 `src/govio/cli/sql.py`，实现 `sql()` 入口和 `_build()` 子函数
3. 修改 `src/govio/cli/main.py`，注册 `sql` 子命令组并 dispatch
4. 修改 `src/govio/__init__.py`，导出 `build_metric_sql`
5. 删除 `skills/govio-query/scripts/sql_builder.py`（及空目录 `scripts/`）
6. 更新 `skills/govio-query/SKILL.md`：替换脚本调用、移除资源文件中的 scripts 条目
7. 同步更新运行时 `/data/home/macx/work/tmp/govio_runtime/.claude/skills/govio-query/SKILL.md`
8. 新增 `tests/test_sql_builder.py`
9. 运行 `uv run pytest tests/test_sql_builder.py` 验证
10. 运行 `uv build && uv tool install --force dist/govio-*.whl` 本地验证 CLI

## 风险与回滚

- **风险**：SKILL.md 更新遗漏导致 skill 仍引用旧脚本路径
  - **缓解**：步骤 5 删除脚本后 grep 全仓 `sql_builder.py` 确认无残留引用
- **风险**：运行时 SKILL.md 未同步
  - **缓解**：步骤 7 显式同步，提交前 diff 确认
- **回滚**：`git revert` 单个 commit 即可恢复脚本和 SKILL.md
