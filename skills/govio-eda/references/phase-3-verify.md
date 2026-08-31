# Phase 3: 核查关联性

**目标**：回答"这些关联真实存在吗？数据支撑吗？覆盖了多少？"

**前置条件**: Phase 2 完成，已产出候选关联清单；或业务输入直接给出键映射。

## 流程

对每个候选关联执行：

```
JOIN 匹配率检查 -> 缺失键分析 -> 重复键检查 -> 记录验证结果
（业务输入给出覆盖口径时，追加覆盖漏斗）
```

## Step 1: JOIN 匹配率检查

通过 `load --memory` 执行 JOIN 查询：

```sql
SELECT
  COUNT(*) AS source_total,
  COUNT(t.join_key) AS matched,
  COUNT(*) - COUNT(t.join_key) AS unmatched,
  ROUND(COUNT(t.join_key) * 100.0 / COUNT(*), 2) AS match_rate_pct
FROM source_df s
LEFT JOIN target_df t ON s.source_key = t.target_key
```

**判定标准**：

| 匹配率 | 判定 | 说明 |
|--------|------|------|
| >= 95% | 强关联 | 关联可靠 |
| 80% - 95% | 弱关联 | 存在缺失，需分析原因 |
| < 80% | 非关联 | 可能是误匹配 |

## Step 2: 缺失键分析

找出源表中存在但目标表中无匹配的键：

```sql
SELECT s.source_key, COUNT(*) AS cnt
FROM source_df s
LEFT JOIN target_df t ON s.source_key = t.target_key
WHERE t.target_key IS NULL
GROUP BY s.source_key
ORDER BY cnt DESC
LIMIT 50
```

**分析要点**：
- 缺失键是否有规律（如特定前缀、特定范围）
- 是否为 NULL 值导致
- 是否为数据延迟（新增未同步）

## Step 3: 重复键检查

检查目标表的关联键是否有重复：

```sql
SELECT target_key, COUNT(*) AS cnt
FROM target_df
GROUP BY target_key
HAVING COUNT(*) > 1
ORDER BY cnt DESC
LIMIT 50
```

**影响**：
- 重复键会导致 JOIN 膨胀（1:N 变 1:M）
- 需要确认是数据问题还是业务设计（如 1 项目对多商机）

## Step 4: 值域重叠分析

检查关联字段的值域重叠情况：

```sql
SELECT
  '仅源表' AS scope, COUNT(DISTINCT s.key) AS cnt
FROM source_df s LEFT JOIN target_df t ON s.key = t.key WHERE t.key IS NULL
UNION ALL
SELECT
  '仅目标表', COUNT(DISTINCT t.key)
FROM target_df t LEFT JOIN source_df s ON t.key = s.key WHERE s.key IS NULL
UNION ALL
SELECT
  '交集', COUNT(DISTINCT s.key)
FROM source_df s JOIN target_df t ON s.key = t.key
```

## Step 5: 覆盖漏斗（业务输入驱动）

当用户提供了覆盖口径（收集模板见 [business-inputs.md](business-inputs.md)）时执行。与 Step 1 的区别：Step 1 验证"关联是否成立"（判定语义），覆盖漏斗陈述"业务口径下覆盖了多少、漏在哪里"（发现语义）。

**核心思路**：以业务口径圈定**全集**，对覆盖目标键（**先去重防 1:N 膨胀**）做 LEFT JOIN 统计已覆盖/未覆盖；未覆盖部分按有序归因规则分层。

### F1: 覆盖统计

```sql
WITH target_keys AS (
  SELECT DISTINCT {target_key} AS join_key
  FROM {target_df}
  WHERE {target_key} IS NOT NULL
),
universe AS (
  SELECT
    {source_key} AS join_key,
    {归因所需列及其额外 LEFT JOIN，全部在此带出}
  FROM {source_df}
  WHERE {口径谓词}            -- 业务口径：什么记录算进全集
)
SELECT
  COUNT(*)                        AS universe_cnt,
  COUNT(t.join_key)               AS covered_cnt,
  COUNT(*) - COUNT(t.join_key)    AS uncovered_cnt,
  ROUND(COUNT(t.join_key) * 100.0 / COUNT(*), 1) AS coverage_pct
FROM universe u
LEFT JOIN target_keys t ON u.join_key = t.join_key
```

### F2: 未覆盖分层归因

```sql
SELECT
  {归因表达式} AS reason,       -- 有序 CASE WHEN，只引用 universe 内的列
  COUNT(*) AS cnt,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS pct
FROM universe u
LEFT JOIN target_keys t ON u.join_key = t.join_key
WHERE t.join_key IS NULL
GROUP BY 1
ORDER BY cnt DESC
```

归因表达式示例（业务输入）：

```sql
CASE
  WHEN comp 关联不存在 THEN '无关联客户'
  ELSE '未归集'
END
```

### F3: 反向检查（目标侧多余键，可选）

```sql
SELECT COUNT(*) AS target_only_cnt
FROM target_keys t
LEFT JOIN universe u ON u.join_key = t.join_key
WHERE u.join_key IS NULL
```

### 输出: 覆盖漏斗卡片

```markdown
### 覆盖漏斗: {全集名} -> {覆盖目标名}

**关联键**: {source}.{key} = {target}.{key}

```
{全集名}（{universe_cnt:,}）
├── 已覆盖：{covered_cnt:,} 条（{coverage_pct}%）
└── 未覆盖：{uncovered_cnt:,} 条（{100-coverage_pct}%）
    ├── {原因1}：{n:,}（{pct}%）
    └── {原因2}：{n:,}（{pct}%）
```

| 层 | 数量 | 占全集比 | 占未覆盖比 |
|----|------|---------|-----------|
```

### 语义约定

1. 覆盖率是**发现**（业务事实），不做强/弱/非关联判定
2. 可选：与用户给定的关注阈值比较，超过时加"需关注"标记
3. 归因分层建议 ≤ 3 层，更深改用表格呈现
4. F1 与 Step 1 是同一 SQL 骨架；本机制的增量在**口径谓词圈定全集 + 分层归因 + 漏斗叙事**

## 产出: 关联验证报告

```markdown
## 关联验证结果

### 关联 #1: orders.customer_id -> customers.id

| 指标 | 值 |
|------|-----|
| 源表总行数 | 50,000 |
| 匹配行数 | 49,200 |
| 匹配率 | 98.4% |
| 缺失键数 | 800 (1.6%) |
| 目标表重复键 | 0 |
| **判定** | 强关联 |

**缺失键分析**：
- 800 条未匹配记录中，780 条 customer_id 为 NULL
- 20 条为已删除客户（已从 customers 表移除）
```

## 核查 SQL 速查

| 检查项 | SQL 模式 |
|--------|---------|
| 匹配率 | `LEFT JOIN + COUNT(key) / COUNT(*)` |
| 缺失键 | `LEFT JOIN ... WHERE t.key IS NULL` |
| 重复键 | `GROUP BY key HAVING COUNT(*) > 1` |
| 值域交集 | `INTERSECT` / `JOIN` + `COUNT(DISTINCT)` |
| NULL 比例 | `SUM(CASE WHEN key IS NULL THEN 1 END)` |
| 覆盖统计 | 目标键 `DISTINCT` + 口径谓词全集 + `LEFT JOIN`（F1） |
| 分层归因 | 未覆盖记录按有序 CASE WHEN 分桶 + 窗口算占比（F2） |
