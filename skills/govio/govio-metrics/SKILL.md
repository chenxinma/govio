---
name: govio-metrics
description: 基于指标的问数能力。当用户询问指标数据（如"本月账单收入"、"按部门统计签约额"）、指标血缘、维度分析时触发。通过查询指标元数据、组装 SQL、执行查询完成数据分析。不适用于纯元数据查询（应用、表结构）。
---

# Govio 指标问数

基于知识图谱中的指标元数据，生成分析 SQL 并执行查询。

## 步骤（强制顺序）

**Step 0** ⚠️ 读取 `assets/schema.md`（**仅此一次**）

**Step 1** 读取 `assets/metrics_index.md` 获取指标概览

**Step 2** 使用 `govio-cli query` 查询指标元数据（来源表、维度、公式）

**Step 3** 使用 `scripts/sql_builder.py` 组装 SQL

**Step 4** 使用 `govio-cli observe load` 执行查询

**Step 5** 格式化输出结果

## 工作流程

```
用户问题 → 解析指标 → 查询元数据 → 组装 SQL → 执行查询 → 返回结果
                              ↓
                     来源表 / 维度 / 公式
```

## 指标类型

| 类型 | 说明 | 查询方式 |
|------|------|----------|
| 原子指标 | 直接从来源表获取 | `SELECT metric_col FROM source_table` |
| 派生指标 | 由原子指标计算得出 | 通过 CTE 组合多个原子指标后计算 |

## Step 2: 查询指标元数据

### 查询指标基本信息

```cypher
MATCH (m:Metric {code: "bill_income_amt"})
RETURN m.code, m.name, m.type, m.formula, m.source_layer, m.unit
```

### 查询指标来源表

```cypher
MATCH (m:Metric {code: "bill_income_amt"})-[:USES_TABLE]->(t:PhysicalTable)
RETURN t.full_table_name, t.name
```

### 查询指标维度

```cypher
MATCH (m:Metric {code: "bill_income_amt"})-[d:DIMENSION_USED]->(dim:Dimension)
RETURN dim.code, dim.name, d.usage_type
```

### 查询指标引用列

```cypher
MATCH (m:Metric {code: "bill_income_amt"})-[:REFERS_COLUMN]->(c:Col)
RETURN c.column_name, c.data_type
```

### 查询派生指标血缘

```cypher
MATCH (m:Metric {code: "book_to_bill"})-[:DERIVED_FROM]->(up:Metric)
RETURN up.code, up.name, up.type
```

## Step 3: 组装 SQL

使用 `scripts/sql_builder.py` 脚本组装 SQL。接受 JSON 文件作为输入。

### 脚本路径

```
scripts/sql_builder.py  # 相对于本 SKILL.md 所在目录
```

### 调用方式

```bash
# 打印到 stdout
uv run python scripts/sql_builder.py query.json

# 输出到文件
uv run python scripts/sql_builder.py query.json -o output.sql
```

### JSON 请求格式

示例文件：`scripts/query_example.json`

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
  "filters": {
    "report_ym": "2026-05"
  },
  "order_by": null,
  "limit": 100,
  "cte_refs": {}
}
```

### 字段说明

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `metrics` | list[dict] | 是 | 指标列表，包含 code, name, type, source_table, formula |
| `dimensions` | list[str] | 否 | 分组维度字段，如 `["sales_unit", "sales_dept"]` |
| `filters` | dict[str, str] | 否 | 过滤条件，如 `{"report_ym": "2026-05"}` |
| `order_by` | str | 否 | 排序字段，如 `"metric_value DESC"` |
| `limit` | int | 否 | 返回行数限制，默认 100 |
| `cte_refs` | dict[str, str] | 否 | 已加载 DataFrame 的 CTE 引用 |

### 指标对象字段

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `code` | str | 是 | 指标编码 |
| `name` | str | 是 | 指标名称 |
| `type` | str | 是 | `"原子"` 或 `"派生"` |
| `source_table` | str | 原子必填 | 来源表，如 `"dws.income_bill_monthly"` |
| `formula` | str | 派生必填 | 计算公式，如 `"signed_amt / bill_income_amt"` |
| `time_column` | str | 否 | 时间字段名，默认 `"report_ym"` |

## Step 4: 执行查询

使用 `govio-cli observe load` 执行 SQL 并加载为 DataFrame：

```bash
govio-cli observe load --name <df_name> --datasource <ds_name> --sql "<sql>"
```

### 命名规范

DataFrame 名称格式：`metric_{指标编码}_{时间}`

示例：
- `metric_bill_income_202605`
- `metric_book_to_bill_202605`

### 数据源

从 `~/.govio/config.yaml` 中获取可用数据源，或询问用户指定。

## CTE 组合查询

当需要组合多次查询结果时，使用 CTE (Common Table Expression) 引用已加载的 DataFrame：

### 场景：环比分析

```bash
# 1. 查询当月数据并加载
uv run python scripts/sql_builder.py current.json -o current.sql
govio-cli observe load --name metric_current --datasource dw --sql "$(cat current.sql)"

# 2. 查询上月数据，通过 cte_refs 引用当月结果
# compare.json 中 cte_refs 包含: {"metric_current": "<当月SQL>"}
uv run python scripts/sql_builder.py compare.json -o compare.sql
```

### 场景：多指标组合

先加载基础指标，再通过 CTE 组合计算派生指标：

```bash
# 1. 加载签约数据
govio-cli observe load --name signed_data --datasource dw --sql "SELECT sales_unit, SUM(signed_amt) as signed_amt FROM dws.signed_monthly WHERE report_ym = '2026-05' GROUP BY sales_unit"

# 2. 加载账单数据
govio-cli observe load --name bill_data --datasource dw --sql "SELECT sales_unit, SUM(bill_income_amt) as bill_income_amt FROM dws.income_bill_monthly WHERE report_ym = '2026-05' GROUP BY sales_unit"

# 3. 组合计算签约覆盖率（通过 cte_refs 引用已加载的 DataFrame）
```

## 常见指标查询模板

### 按时间趋势

```json
{
  "metrics": [
    {"code": "bill_income_amt", "name": "当月账单收入", "type": "原子", "source_table": "dws.income_bill_monthly"}
  ],
  "dimensions": ["report_ym"],
  "order_by": "report_ym"
}
```

### 按部门排名

```json
{
  "metrics": [
    {"code": "bill_income_amt", "name": "当月账单收入", "type": "原子", "source_table": "dws.income_bill_monthly"}
  ],
  "dimensions": ["sales_unit", "sales_dept"],
  "filters": {"report_ym": "2026-05"},
  "order_by": "bill_income_amt DESC",
  "limit": 10
}
```

### 派生指标（签约覆盖率）

```json
{
  "metrics": [
    {"code": "signed_amt", "name": "签约额", "type": "原子", "source_table": "dws.signed_monthly"},
    {"code": "bill_income_amt", "name": "账单收入", "type": "原子", "source_table": "dws.income_bill_monthly"},
    {"code": "book_to_bill", "name": "签约覆盖率", "type": "派生", "formula": "signed_amt / bill_income_amt"}
  ],
  "dimensions": ["sales_unit"],
  "filters": {"report_ym": "2026-05"}
}
```

## 输出格式

### 单指标结果

```
📊 查询结果: 当月账单收入 (2026-05)

| 销售单元 | 销售部门 | 指标值 |
|---------|---------|-------|
| 华东区   | 上海分部 | 1,234 |
| 华北区   | 北京分部 | 2,345 |

共 2 条记录
```

### 多指标结果

```
📊 查询结果: 指标对比 (2026-05)

| 销售单元 | 指标名称 | 指标值 |
|---------|---------|-------|
| 华东区   | 当月签约额 | 1,000 |
| 华东区   | 当月账单收入 | 1,234 |
| 华北区   | 当月签约额 | 2,000 |
| 华北区   | 当月账单收入 | 2,345 |

共 4 条记录
```

## 指标概览

`assets/metrics_index.md` 中包含所有指标的索引：

### 原子指标

| 编码 | 名称 | 来源层 | 单位 |
|------|------|--------|------|
| bill_income_amt | 当月账单收入 | DWS | 万元 |
| signed_amt | 当月销售签约额 | DWS | 万元 |
| forecast_income_amt | 预计当月账单收入 | DWS | 万元 |
| ... | ... | ... | ... |

### 派生指标

| 编码 | 名称 | 公式 | 单位 |
|------|------|------|------|
| book_to_bill | 签约覆盖率 | signed_amt / bill_income_amt | 倍 |
| burndown_amt | 存量消耗额 | forecast_income_amt - risk_amt | 万元 |
| ... | ... | ... | ... |

## 排除场景

以下场景**不要**触发本技能：
- 查询应用列表、表结构、字段信息 → 使用 `govio-metadata`
- 数据比对、迁移验证 → 使用 `govio-observe`
- 代码开发、调试

## 资源文件

```
../assets/               # 资源文件在上层govio/assets/
├── schema.md            # 图模式（Step 0 必读）
├── metrics_index.md     # 指标索引
└── names/               # 名称映射
scripts/
├── sql_builder.py       # SQL 组装脚本（CLI）
└── query_example.json   # 查询请求示例
```
