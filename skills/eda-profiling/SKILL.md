---
name: eda-profiling
description: EDA Phase 1 - 数据集画像。加载数据集后自动执行结构概览、字段统计、缺失分析、分布可视化，回答"我拿到的是什么数据"。依赖 observe-dataset-ops 完成数据加载和管理。
---

# EDA Phase 1: 数据集画像

加载数据集并产出画像卡片，了解每个数据集的结构、质量和特征。

## 目标

回答："我拿到的是什么数据？"

## 依赖 Skill

| 操作 | Skill | CLI 命令 |
|------|-------|---------|
| 查看数据源 | `observe-dataset-ops` | `govio-cli observe show-datasource` |
| 加载数据 | `observe-dataset-ops` | `govio-cli observe load` |
| 查看已加载 | `observe-dataset-ops` | `govio-cli observe list` |
| 二次加工查询 | `observe-dataset-ops` | `govio-cli observe load --memory` |
| 可视化 | (inline) | `govio-cli observe chart` |
| 释放资源 | `observe-dataset-ops` | `govio-cli observe release` |

## 流程

```
show-datasource → 确认数据源 → load 加载 → list 查看结构 → load --memory 画像 SQL → chart 可视化
```

### Step 1: 确认数据源

```bash
govio-cli observe show-datasource
```

确认可用数据源，选择目标数据源。

### Step 2: 加载数据集

对每个目标表执行加载：

```bash
govio-cli observe load --name <df_name> --datasource <ds> --sql "SELECT * FROM <table>"
```

**命名规范**：`<应用缩写>_<表名>`，如 `crm_customers`、`erp_orders`。

**大数据量处理**：先用 `LIMIT 1000` 采样确认结构，再全量加载。

### Step 3: 查看结构概览

```bash
govio-cli observe list
```

确认所有数据集已加载，检查行数、列数、字段类型。

### Step 4: 执行画像分析

通过 `load --memory` 在已加载 DataFrame 上执行画像 SQL。

#### 4a. 字段级画像

```sql
SELECT
  COUNT(*) AS total_rows,
  COUNT(DISTINCT col1) AS col1_cardinality,
  SUM(CASE WHEN col1 IS NULL THEN 1 ELSE 0 END) AS col1_null_count,
  MIN(col1) AS col1_min,
  MAX(col1) AS col1_max
FROM df_name
```

对每个字段重复，或用 SQL 一次性统计多个字段。

#### 4b. 数值列统计

```sql
SELECT
  AVG(numeric_col) AS mean_val,
  STDDEV(numeric_col) AS std_val,
  MIN(numeric_col) AS min_val,
  MAX(numeric_col) AS max_val,
  PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY numeric_col) AS median_val
FROM df_name
```

#### 4c. 分类列 Top 值

```sql
SELECT category_col, COUNT(*) AS cnt
FROM df_name
GROUP BY category_col
ORDER BY cnt DESC
LIMIT 20
```

#### 4d. 日期列范围

```sql
SELECT
  MIN(date_col) AS earliest,
  MAX(date_col) AS latest,
  COUNT(DISTINCT DATE_TRUNC('month', date_col)) AS month_span
FROM df_name
```

### Step 5: 可视化分布

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
| ... | ... | ... | ... | ... |

**数值字段统计**:

| 字段 | 均值 | 标准差 | 最小值 | 中位数 | 最大值 |
|------|------|--------|--------|--------|--------|
| amount | ... | ... | ... | ... | ... |

**分类字段 Top 5**:

| 字段 | 值 | 占比 |
|------|-----|------|
| status | active | 75% |
| status | inactive | 25% |
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
| 模式检测 | `LENGTH(col)`, `REGEXP_MATCHES` |

## 与其他阶段的衔接

- 画像结果传递给 Phase 2（推断关联）：字段名、类型、基数用于推断外键
- 画像发现的质量问题（高 NULL 率、低基数）记录到报告，在 Phase 4 深入核查
