---
name: govio-observe
description: 数据探查与比对命令组。当需要查看数据源、加载数据、探索表关系、比对数据差异、生成图表时触发。包含 info（查看数据源/DataFrame）、load（加载数据）、release（释放资源）、explore（探索关系）、compare（比对数据）、chart（生成图表）等子命令。
---

# Govio Observe 数据探查与比对

本 Skill 对应 `govio-cli observe` 命令组，提供数据加载、探查、比对、可视化等能力。

## 子命令总览

| 子命令 | 用途 | 典型场景 |
|--------|------|----------|
| `govio-cli observe info --datasource` | 查看可用数据源 | 开始任务前确认数据源 |
| `govio-cli observe info --df` | 查看已加载的 DataFrame | 监控已加载数据的状态 |
| `govio-cli observe info --name <df_name>` | 查看 DataFrame 详情 | 查看结构和样本数据 |
| `govio-cli observe load` | 加载数据 | 从数据库抽取或二次加工 |
| `govio-cli observe release` | 释放资源 | 清理不需要的 DataFrame |
| `govio-cli observe explore` | 探索关系 | 发现表之间的关联 |
| `govio-cli observe compare` | 比对数据 | 验证数据一致性 |
| `govio-cli observe chart` | 生成图表 | 可视化数据趋势 |

## 前置条件

1. 已运行 `govio-cli onboard` 完成初始化配置
2. 通过 `govio-cli observe info --datasource` 确认数据源已配置
3. DataFrame 持久化在 `.govio/observe/dataframes/` 目录下

## 核心原则

### 1. 澄清先于执行

**永远不要直接开始执行。** 先通过提问澄清：

- **数据源**: 从哪个数据源抽取？目标是什么？
- **比对对象**: 要比对哪些表/数据？
- **成功标准**: 什么样的结果算"完成"？
- **输出要求**: 需要什么格式的报告？

### 2. 计划驱动

复杂任务（3 步以上）必须编写 Plan：

```markdown
# 数据治理计划: [任务名称]

**目标**: [一句话描述]
**数据源**: [源系统] → [目标系统]
**成功标准**: [可验证的条件]

## Task 1: [步骤名称]
- [ ] 子步骤...

## Task 2: [步骤名称]
- [ ] 子步骤...
```

Plan 保存位置: `docs/govio/plans/YYYY-MM-DD-[task-name].md`

---

## 子命令详解

### info - 查看信息

#### 查看数据源

```bash
govio-cli observe info --datasource
```

返回可用的数据库连接列表。

#### 查看已加载 DataFrame

```bash
govio-cli observe info --df
```

返回当前已加载的 DataFrame 列表，包含行数、列数、字段类型。

#### 查看 DataFrame 详情

```bash
govio-cli observe info --name <df_name> [--rows N]
```

查看已加载 DataFrame 的结构（列名+类型）和样本数据，**只读操作**，不会产生新的 DataFrame。

参数：
- `--name`: ObserveStore 中已加载的 DataFrame 名称
- `--rows`: 显示的样本行数，默认 10

返回 JSON 包含：`name`、`rows`（总行数）、`columns`（总列数）、`schema`（列名+dtype 数组）、`sample`（前 N 行数据）。

---

### load - 加载数据

从数据库抽取数据或在已加载 DataFrame 上二次加工。

#### 参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| --name | keyword | 是 | DataFrame 名称（小写字母+下划线） |
| --datasource | keyword | 二选一 | 数据源名称（从 info --datasource 获取），从数据库抽取 |
| --memory | flag | 二选一 | 不连数据库，改为在已加载的 DataFrame 上跑 SQL |
| --sql | keyword | 是 | 查询 SQL 语句 |
| -o, --output | keyword | 否 | 将数据内容输出到指定 JSON 文件 |

> `--datasource` 与 `--memory` 互斥，必须给出其一。

> **重要**: 使用 `-o` 输出数据集内容前，必须先征得用户确认。如果用户未明确表示可以查看数据内容，应先问询用户是否允许输出。

#### 从数据库抽取

```bash
govio-cli observe load --name customers --datasource prod_db --sql "SELECT customer_id, name, email FROM customers WHERE created_at > '2024-01-01'"
```

#### 输出数据内容到 JSON 文件

```bash
govio-cli observe load --name customers --datasource prod_db --sql "SELECT * FROM customers" -o customers.json
```

#### 从已加载的 DataFrame 查询（--memory）

`--memory` 模式不连接数据库，而是把 ObserveStore 里所有已注册的 DataFrame 注入到一个内存 DuckDB 中（按 DataFrame 名注册成同名表），然后执行 `--sql`，结果以 `--name` 存回 store。适合在已抽取的 DataFrame 之上做过滤、连接、聚合等二次加工，避免反复访问数据库。

```bash
# 先从数据库加载基础数据
govio-cli observe load --name orders --datasource prod_db --sql "SELECT order_id, customer_id, amount FROM orders"

# 在已加载的 orders 之上再做查询（不连数据库）
govio-cli observe load --name high_value_orders --memory --sql "SELECT customer_id, SUM(amount) AS total FROM orders GROUP BY customer_id HAVING SUM(amount) > 10000"
```

返回结果会额外带 `source_tables` 字段，列出本次查询实际用到的已加载 DataFrame。

注意：`--memory` 要求 store 里至少有一个已加载的 DataFrame，否则返回 `{"success": false, "error": "没有已加载的 DataFrame"}`。SQL 中的表名必须与已加载的 DataFrame 名一致。

#### 命名规范

DataFrame 名称使用小写+下划线，如 `customers_2024_q1`。

---

### release - 释放资源

#### 参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| --name | keyword | 否 | DataFrame 名称（与 --all 二选一） |
| --all | flag | 否 | 释放所有已加载的 DataFrame |

#### 释放单个 DataFrame

```bash
govio-cli observe release --name customers
```

#### 释放所有已加载的 DataFrame

```bash
govio-cli observe release --all
```

---

### explore - 探索关系

探查多个 DataFrame 之间的潜在关系，包括外键关联和列名相似度。

#### 参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| --dataframes | keyword | 否 | DataFrame 名称列表。省略时探查所有已加载的 DataFrame |

#### 调用方式

```bash
# 探查指定的 DataFrame
govio-cli observe explore --dataframes customers orders products

# 探查所有已加载的 DataFrame
govio-cli observe explore
```

#### 返回结果

```json
{
  "success": true,
  "relations": [
    {
      "source_table": "orders",
      "source_column": "customer_id",
      "target_table": "customers",
      "target_column": "customer_id",
      "confidence": 0.95
    },
    {
      "type": "column_similarity",
      "table1": "customers",
      "column1": "email",
      "table2": "orders",
      "column2": "customer_email",
      "similarity": 0.85
    }
  ]
}
```

#### 关系类型

**外键关系 (foreign_key)**：
- 检测方式: 基于列名模式（`*_id`, `*id`）和值重叠率
- confidence: 重叠值比例（0-1）
- 阈值: confidence > 0.5 时报告
- 大小写不敏感: `Customer_ID` 与 `customer_id` 视为匹配

**列名相似 (column_similarity)**：
- 检测方式: 字符串相似度（SequenceMatcher）
- similarity: 相似度分数（0-1）
- 阈值: similarity > 0.7 时报告

#### 可视化关系图谱

```bash
# 先探查关系，保存 JSON
govio-cli observe explore --dataframes customers orders > relations.json

# 生成图谱数据
govio-cli observe visualize-relations --relations "$(cat relations.json)"
```

返回结果包含 `nodes` 和 `edges`，可用于关系图谱渲染。

---

### compare - 比对数据

比对两个 DataFrame 的结构和数据差异。

#### 参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| --source | keyword | 是 | 源 DataFrame 名称 |
| --target | keyword | 是 | 目标 DataFrame 名称 |
| --join-columns | keyword | 是 | 用于比对的列，逗号分隔（通常是主键） |

#### join_columns 列名匹配规则

两个数据集的 join 列名必须完全相同。如果存在名称差异，需要先进行列名转换：

1. **大小写差异**: 如 `a.COMP_NO` 与 `b.comp_no`，需统一转换为小写
2. **主键引用差异**: 如 `t_order.cust_id` 与 `t_cust.id`，需将 `t_cust.id` 重命名为 `t_cust.cust_id`

#### 调用方式

```bash
govio-cli observe compare --source legacy_customers --target new_customers --join-columns customer_id
```

多个 join 列用逗号分隔：

```bash
govio-cli observe compare --source source_table --target target_table --join-columns customer_id,order_id
```

#### 返回结果

```json
{
  "success": true,
  "schema": {
    "match": false,
    "source_columns": ["customer_id", "name", "email", "phone"],
    "target_columns": ["customer_id", "name", "email", "mobile"],
    "common_columns": ["customer_id", "name", "email"],
    "source_only": ["phone"],
    "target_only": ["mobile"]
  },
  "data": {
    "report": "datacompy 比对报告文本..."
  }
}
```

#### 结果解读

**结构差异 (schema)**：

| 字段 | 含义 |
|------|------|
| match | 结构是否完全一致 |
| source_columns | 源表所有列 |
| target_columns | 目标表所有列 |
| common_columns | 共有的列 |
| source_only | 仅在源系统存在的列 |
| target_only | 仅在目标系统存在的列 |

**数据差异 (data)**：

| 字段 | 含义 |
|------|------|
| report | datacompy 生成的完整比对报告，包含匹配行数、独有行、值差异等详细信息 |

---

### chart - 生成图表

从已加载的 DataFrame 生成 PNG 图表，支持柱状图（`bar`）和折线图（`line`）。

#### 参数

所有参数必填：

| 参数 | 类型 | 说明 |
|------|------|------|
| --name | keyword | ObserveStore 中已加载的 DataFrame 名称 |
| --type | keyword | `bar` 或 `line` |
| --x | keyword | X 轴列名（分类轴/时序轴） |
| --y | keyword | Y 轴列名（数值轴，单列） |
| -o, --output | keyword | 输出 PNG 路径 |

#### 调用方式

```bash
# 画柱状图
govio-cli observe chart --name sales --type bar --x region --y revenue -o /tmp/sales_bar.png

# 画折线图（时序数据）
govio-cli observe chart --name monthly --type line --x month --y revenue -o /tmp/monthly_trend.png
```

成功返回 `{"success": true, "output": "<abs_path>"}`；DataFrame 或列不存在时返回 `{"success": false, "error": "..."}`。

内置中文字体回退链（Noto Sans CJK SC / WenQuanYi Zen Hei / SimHei / Microsoft YaHei / Arial Unicode MS），自动适配主流系统；同时修复负号显示问题。第一版单系列，不支持分组。

---

## 使用模式

### 模式 A: 简单任务（1-2 步）

直接执行，无需编写 Plan。

**示例**：
> 用户: "帮我看一下有哪些数据源"
>
> 执行: `govio-cli observe info --datasource`

### 模式 B: 数据探查任务

探查数据源和表结构，通常需要多步操作。

**示例**：
> 用户: "帮我分析一下客户相关表的结构和关系"
>
> 1. `govio-cli observe info --datasource` → 获得可用数据源
> 2. `govio-cli observe load` (多次) → 加载 customers, orders 等表
> 3. `govio-cli observe explore` → 发现外键关系和相似列

### 模式 C: 复杂任务（3 步以上）

必须编写 Plan，然后逐步执行。

**示例**：
> 用户: "帮我验证客户数据迁移"
>
> 1. 澄清：源系统、目标系统、比对表、匹配率预期
> 2. 编写 Plan
> 3. 逐步执行

---

## 完整工作流示例

### 数据迁移验证流程

```bash
# 1. 查看数据源
govio-cli observe info --datasource

# 2. 加载源数据
govio-cli observe load --name legacy_customers --datasource legacy_db --sql "SELECT customer_id, name, email FROM customers"

# 3. 加载目标数据
govio-cli observe load --name new_customers --datasource new_db --sql "SELECT customer_id, name, email FROM customers"

# 4. 查看加载状态
govio-cli observe info --df

# 5. 比对数据
govio-cli observe compare --source legacy_customers --target new_customers --join-columns customer_id

# 6. 释放资源
govio-cli observe release --all
```

---

## 与其他 Skill 的协作

| 操作 | 关联 Skill |
|------|-----------|
| 查询元数据 | `govio-query` |
| 知识图谱维护 | `govio-meta` |
| EDA 分析 | `govio-eda` |

## 注意事项

1. **不要假设**: 总是先澄清需求，不要猜测用户的意图
2. **小步快跑**: 复杂任务分解为小步骤，每步完成后确认
3. **及时反馈**: 每个 Task 完成后告知用户进展
4. **资源清理**: 确保最后释放所有 DataFrame
5. **记录日志**: 重要操作记录到 `docs/govio/logs/`
6. **禁止直接读取底层文件**: 查看已加载 DataFrame 的结构和数据时，必须使用 `govio-cli observe info --name` 或 `govio-cli observe info --df`，**不要**直接读取 `.govio/observe/manifest.json` 或 `.govio/observe/dataframes/*.parquet` 文件

## 排除场景

以下场景**不要**触发本技能：
- 查询元数据（应用、表、字段） → 使用 `govio-query`
- 知识图谱维护（同步、导出、推荐） → 使用 `govio-meta`
- 数据探查、EDA → 使用 `govio-eda`
