# Phase 1: 数据集画像

**目标**：回答"我拿到的是什么数据？"

## 流程

```
info --datasource -> 确认数据源 -> load 加载 -> info --df 查看结构 -> 画像分析（字段级 + 维度级） -> 可视化
```

## Step 1: 确认数据源

```bash
govio-cli observe info --datasource
```

确认可用数据源，选择目标数据源。

## Step 2: 加载数据集

对每个目标表执行加载：

```bash
govio-cli observe load --name <df_name> --datasource <ds> --sql "SELECT * FROM <table>"
```

**命名规范**：`<应用缩写>_<表名>`，如 `crm_customers`、`erp_orders`。

**大数据量处理**：先用 `LIMIT 1000` 采样确认结构，再全量加载。

## Step 3: 查看结构概览

```bash
govio-cli observe info --df
```

确认所有数据集已加载，检查行数、列数、字段类型。

## Step 4: 画像分析

通过 `load --memory` 在已加载 DataFrame 上执行画像 SQL（DuckDB 方言）。

### 4a. 字段级画像

```sql
SELECT
  COUNT(*) AS total_rows,
  COUNT(DISTINCT col1) AS col1_cardinality,
  SUM(CASE WHEN col1 IS NULL THEN 1 ELSE 0 END) AS col1_null_count,
  MIN(col1) AS col1_min,
  MAX(col1) AS col1_max
FROM df_name
```

### 4b. 数值列统计

```sql
SELECT
  AVG(numeric_col) AS mean_val,
  STDDEV(numeric_col) AS std_val,
  MIN(numeric_col) AS min_val,
  MAX(numeric_col) AS max_val,
  PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY numeric_col) AS median_val
FROM df_name
```

### 4c. 分类列 Top 值

```sql
SELECT category_col, COUNT(*) AS cnt
FROM df_name
GROUP BY category_col
ORDER BY cnt DESC
LIMIT 20
```

### 4d. 日期列范围

```sql
SELECT
  MIN(date_col) AS earliest,
  MAX(date_col) AS latest,
  COUNT(DISTINCT DATE_TRUNC('month', date_col)) AS month_span
FROM df_name
```

### 4e. 维度聚合画像（业务输入驱动）

当用户提供了维度定义（收集模板见 [business-inputs.md](business-inputs.md)）时执行。字段级画像回答"数据长什么样"，维度画像回答"数据在业务维度上如何分布"。

**核心思路**：对 k 个维度只做一次全维 GROUP BY，产出"维度组合 -> 计数"长表并存为新 df；边际分布、层级汇总都是对小体量长表的再聚合，代价极小。

```sql
-- L1: 全维度聚合长表（一次扫描，存为 {df}_dim_profile）
SELECT
  {dim_1_expr} AS d1,
  {dim_2_expr} AS d2,
  -- …
  {dim_k_expr} AS dk,
  COUNT(DISTINCT {id_col}) AS cnt
FROM {df}
WHERE {口径谓词}            -- 可选，来自维度定义或覆盖口径
GROUP BY ALL
```

```sql
-- L2: 单维度边际（对长表再聚合）
SELECT d1, SUM(cnt) AS cnt
FROM {df}_dim_profile
GROUP BY d1
ORDER BY cnt DESC
```

```sql
-- L3: 层级汇总（前 j 层，任意层级可视化均可直接消费）
SELECT d1, d2, SUM(cnt) AS cnt
FROM {df}_dim_profile
GROUP BY d1, d2
ORDER BY cnt DESC
LIMIT 20
```

#### 可视化数据契约

图表如何渲染（treemap、sunburst 等）是图表能力提供方的实现细节，EDA 侧只保证产出符合以下契约的 DataFrame，供其消费：

| 契约项 | 内容 |
|--------|------|
| 数据结构 | 长表：`d1, d2, …, dk, cnt`（每行一个维度组合及其计数） |
| 层级语义 | 列序即层级路径（d1 为根，逐层下钻） |
| 取值 | `cnt >= 1`，无空行；NULL 已在维度表达式中替换为展示值 |
| 典型消费方 | treemap（path = d1..dk, values = cnt）、逐层 bar、markdown 表 |

降级形态：图表能力不可用时，用 markdown 表（按 cnt 降序 Top-N）呈现，不影响方法完整性。

#### 输出: 维度画像卡片

```markdown
### {数据集} 维度画像

**口径**: {口径谓词的自然语言描述}
**长表**: {df}_dim_profile（{行数} 个维度组合）

| 维度 | 基数 | Top 3 |
|------|------|-------|
| 合同状态 | 6 | 待关闭 45%、草稿 20%、审批中 15% |
| 合同主体 | 2 | 外服本部 82%、其他 18% |

**层级分布**（状态 × 主体 × 机构）: 长表符合可视化数据契约，由图表能力渲染或降级为 markdown 表
```

## Step 5: 可视化分布

```bash
govio-cli observe chart --name df_name --type bar --x category_col --y count_col -o /tmp/eda_profile.png
```

## 产出: 数据集卡片

每个数据集产出一张画像卡片：

```markdown
### [数据集名称]

| 维度 | 值 |
|------|-----|
| 来源 | [数据源] |
| 行数 | [N] |
| 列数 | [M] |
| 时间范围 | [最早 ~ 最晚] |

**字段详情**:

| 字段 | 类型 | 非空率 | 基数 | 说明 |
|------|------|--------|------|------|
| id | int64 | 100% | N | 主键 |
| name | object | 98.5% | M | 客户名称 |

**数值字段统计**:

| 字段 | 均值 | 标准差 | 最小值 | 中位数 | 最大值 |
|------|------|--------|--------|--------|--------|
| amount | ... | ... | ... | ... | ... |

**分类字段 Top 5**:

| 字段 | 值 | 占比 |
|------|-----|------|
| status | active | 75% |
```

## 画像 SQL 快速参考

| 分析项 | SQL 模式 |
|--------|---------|
| 总行数 | `SELECT COUNT(*) FROM df` |
| 非空率 | `SUM(CASE WHEN col IS NOT NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)` |
| 基数 | `COUNT(DISTINCT col)` |
| 数值统计 | `AVG`, `STDDEV`, `MIN`, `MAX`, `PERCENTILE_CONT` |
| Top 值 | `GROUP BY col ORDER BY COUNT(*) DESC LIMIT N` |
| 日期范围 | `MIN(date_col)`, `MAX(date_col)` |
| 维度长表 | 一次全维 `GROUP BY ALL`，边际/层级由长表 `SUM(cnt)` 再聚合 |
