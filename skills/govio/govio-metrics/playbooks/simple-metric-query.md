# Playbook: 单指标 + 维度过滤取数

适用于"查某指标在某维度下的值"这类简单取数场景。

> 示例 trace: 查询"外滩业务中心的预计销售额"

## 前置条件

- 用户问题包含一个明确的业务指标名和一个维度筛选条件
- 指标为原子指标（非派生），或派生指标的依赖原子指标均可直接查到

## 流程

```
Step 0  读 schema（仅首次）
  ↓
Step 1  匹配指标编码
  ↓
Step 2  查询指标元数据（来源表、实际列名、维度）
  ↓
Step 3  探查数据范围 + 维度值
  ↓
Step 4  组装 SQL
  ↓
Step 5  执行查询
  ↓
Step 6  格式化输出
```

---

### Step 0: 读取 schema（仅首次）

每次会话仅需读取一次，了解图数据库节点/边结构。

```bash
# 无需操作，SKILL.md 加载时已自动读取 assets/schema.md
```

### Step 1: 匹配指标编码

读取 `assets/metrics_index.md`，根据用户问题中的业务关键词匹配指标编码。

| 用户表述 | 匹配指标编码 | 说明 |
|---------|-------------|------|
| 预计销售额 | `forecast_income_amt` | 预计当月会产生的账单收入 |
| 账单收入 | `bill_income_amt` | 当月账单收入 |
| 签约额 | `signed_amt` | 当月销售签约额 |
| 签约覆盖率 | `book_to_bill` | 派生指标，需组合查询 |

**歧义处理**: 若无法确定唯一指标，向用户列出候选指标及其定义，让用户确认。

### Step 2: 查询指标元数据

用 `govio-cli query -c` 并行查询以下 4 项信息：

```bash
# 2a. 指标基本信息（类型、公式、来源层、单位）
govio-cli query -c 'MATCH (m:Metric {code: "FORECAST_CODE"}) RETURN m.code, m.name, m.type, m.formula, m.source_layer, m.unit'

# 2b. 来源表
govio-cli query -c 'MATCH (m:Metric {code: "FORECAST_CODE"})-[:USES_TABLE]->(t:PhysicalTable) RETURN t.full_table_name, t.name'

# 2c. 引用列（获取实际列名，与 code 不同时必须记录）
govio-cli query -c 'MATCH (m:Metric {code: "FORECAST_CODE"})-[:REFERS_COLUMN]->(c:Col) RETURN c.column_name, c.data_type'

# 2d. 可用维度
govio-cli query -c 'MATCH (m:Metric {code: "FORECAST_CODE"})-[d:DIMENSION_USED]->(dim:Dimension) RETURN dim.code, dim.name, d.usage_type'
```

**记录要点**:
- `report_ym` 拉链字段：维度包含 `report_ym` 时该字段为**必须过滤条件**。`report_ym` 表示数据生成时间而非业务时间，每月生成快照形成拉链；不指定会导致不同时期数据被合并，产生无意义结果。一般查询直接取最新年月，仅追溯历史时才用范围条件
- `actual_column`: 引用列名（如 `bill_income_amt_lastyear`），与 metric code 不同时后续 SQL 必须用此列名
- `time_column`: 时间字段名（`ym`），来自 Step 2d 的维度列表或 REFERS_COLUMN
- 可用维度列表：确认用户指定的维度在其中

### Step 3: 探查数据范围 + 维度值

在组装最终 SQL 前，确认两件事：

```bash
# 3a. 最新可用时间周期（不要假设当前月份有数据）
govio-cli observe load --name range_check --datasource <ds> \
  --sql "SELECT DISTINCT {time_column} FROM {source_table} ORDER BY {time_column} DESC LIMIT 10" \
  -o /tmp/range.json

# 3b. 维度值是否存在（确认用户指定的维度值在数据中）
govio-cli observe load --name dim_check --datasource <ds> \
  --sql "SELECT DISTINCT {dim_column} FROM {source_table} ORDER BY {dim_column}" \
  -o /tmp/dim_check.json
```

**必须带 `-o` 输出到文件**，否则只能拿到行数/列数，看不到实际数据值。

**注意事项**:
- 若最新周期超过当前月份，向用户说明为预测/计划值并确认
- 用户未指定时间时，默认使用最新可用月份
- `report_ym` 为必须过滤条件，即使用户未指定时间也必须在 filters 中填写最新值

### Step 4: 组装 SQL

创建 JSON 请求文件，用 `sql_builder.py` 生成 SQL：

```json
{
  "metrics": [
    {
      "code": "forecast_income_amt",
      "name": "预计当月会产生的账单收入",
      "type": "原子",
      "source_table": "dws.income_lastyear_bill_monthly",
      "actual_column": "bill_income_amt_lastyear",
      "time_column": "ym"
    }
  ],
  "dimensions": ["sales_dept"],
  "filters": {
    "ym": "202606",
    "sales_dept": "外滩业务中心"
  },
  "order_by": null,
  "limit": 100
}
```

```bash
uv run python scripts/sql_builder.py /tmp/query.json
```

**关键字段**:
- `actual_column`: 来自 Step 2c，当实际列名与 metric code 不同时必须填写
- `time_column`: 来自 Step 2c，与实际表的时间列名一致
- `filters`: 必须包含时间条件 + 用户指定的维度筛选。若表含 `report_ym`，该字段为必须项（拉链表不同时期数据不可合并）

### Step 5: 执行查询

> 执行前必须获得用户许可。未确认时先展示 SQL 问询。

```bash
govio-cli observe load --name metric_{code}_{period} --datasource <ds> \
  --sql "<generated_sql>" -o /tmp/result.json
```

然后读取结果：

```bash
cat /tmp/result.json
```

### Step 6: 格式化输出

```
📊 查询结果: {指标名称} ({时间})

| {维度列} | 指标值 |
|---------|-------|
| {值}    | {格式化数值} |

共 N 条记录
```

数值格式化：金额类指标加千分位分隔符，大额值换算为"亿元"等可读单位。

---

## 完整示例

**用户问题**: "外滩业务中心的预计销售额"

| 步骤 | 操作 | 结果 |
|------|------|------|
| Step 1 | 读 metrics_index.md | 匹配 `forecast_income_amt` |
| Step 2a | 查询 Metric 基本信息 | atomic, DWS, 元 |
| Step 2b | 查询来源表 | `dws.income_lastyear_bill_monthly` |
| Step 2c | 查询引用列 | `bill_income_amt_lastyear` (DOUBLE) |
| Step 2d | 查询维度 | ym, sales_unit, sales_dept, ... |
| Step 3a | 探查时间范围 | 最新: 202612 (超过当前月，需说明) |
| Step 3b | 探查维度值 | "外滩业务中心" 存在 |
| Step 4 | 组装 SQL | sql_builder.py 生成 CTE SQL |
| Step 5 | 执行查询 | 1 行结果 |
| Step 6 | 格式化输出 | 145,647,336.15 元 ≈ 1.46 亿 |

## 常见变体

### 用户未指定时间

Step 3a 探查到最新周期后直接填入 filters，无需询问。注意：`report_ym` 是必须条件，即使用户未指定也必须填写。

### 派生指标

Step 2 额外查询 `DERIVED_FROM` 血缘，拆解为多个原子指标分别查询后组合。

### 多维度分组

`dimensions` 数组添加多个维度字段，如 `["sales_unit", "sales_dept"]`。
