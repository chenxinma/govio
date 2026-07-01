---
name: eda-verify-relations
description: EDA Phase 3 - 核查关联性。对 Phase 2 产出的候选关联逐一验证，检查 JOIN 匹配率、缺失键、重复键，确认或否定每个关联。依赖 observe-compare-dfs 和 observe-dataset-ops。
---

# EDA Phase 3: 核查关联性

对候选关联逐一验证，用数据证据确认或否定每个关联。

## 目标

回答："这些关联真实存在吗？数据支撑吗？"

## 依赖 Skill

| 操作 | Skill | CLI 命令 |
|------|-------|---------|
| 二次加工查询 | `observe-dataset-ops` | `govio-cli observe load --memory` |
| 数据比对 | `observe-compare-dfs` | `govio-cli observe compare` |
| 释放资源 | `observe-dataset-ops` | `govio-cli observe release` |

## 前置条件

Phase 2 完成，已产出候选关联清单。

## 流程

对每个候选关联执行：

```
JOIN 匹配率检查 → 缺失键分析 → 重复键检查 → 记录验证结果
```

### Step 1: JOIN 匹配率检查

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
| >= 95% | ✅ 强关联 | 关联可靠 |
| 80% - 95% | ⚠️ 弱关联 | 存在缺失，需分析原因 |
| < 80% | ❌ 非关联 | 可能是误匹配 |

### Step 2: 缺失键分析

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

### Step 3: 重复键检查

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

### Step 4: 值域重叠分析

检查关联字段的值域重叠情况：

```sql
SELECT
  '仅源表' AS scope, COUNT(DISTINCT s.key) AS cnt
FROM source_df s LEFT JOIN target_df t ON s.key = t.key WHERE t.key IS NULL
UNION ALL
SELECT
  '仅目标表', COUNT(DISTINCT t.key)
FROM target_df t LEFT JOIN source_df s ON s.key = t.key WHERE s.key IS NULL
UNION ALL
SELECT
  '交集', COUNT(DISTINCT s.key)
FROM source_df s JOIN target_df t ON s.key = t.key
```

## 产出: 关联验证报告

```markdown
## 关联验证结果

### 关联 #1: orders.customer_id → customers.id

| 指标 | 值 |
|------|-----|
| 源表总行数 | 50,000 |
| 匹配行数 | 49,200 |
| 匹配率 | 98.4% |
| 缺失键数 | 800 (1.6%) |
| 目标表重复键 | 0 |
| **判定** | ✅ 强关联 |

**缺失键分析**：
- 800 条未匹配记录中，780 条 customer_id 为 NULL
- 20 条为已删除客户（已从 customers 表移除）

### 关联 #2: projects.bpext → customers.comp_no

| 指标 | 值 |
|------|-----|
| 源表总行数 | 1,200 |
| 匹配行数 | 1,050 |
| 匹配率 | 87.5% |
| 缺失键数 | 150 (12.5%) |
| 目标表重复键 | 5 |
| **判定** | ⚠️ 弱关联 |

**缺失键分析**：
- 150 条未匹配中，100 条为历史项目（客户已注销）
- 50 条为录入错误（bpext 格式异常）
```

## 核查 SQL 速查

| 检查项 | SQL 模式 |
|--------|---------|
| 匹配率 | `LEFT JOIN + COUNT(key) / COUNT(*)` |
| 缺失键 | `LEFT JOIN ... WHERE t.key IS NULL` |
| 重复键 | `GROUP BY key HAVING COUNT(*) > 1` |
| 值域交集 | `INTERSECT` / `JOIN` + `COUNT(DISTINCT)` |
| NULL 比例 | `SUM(CASE WHEN key IS NULL THEN 1 END)` |

## 与其他阶段的衔接

- 确认的关联（✅）进入 Phase 4（一致性核查）
- 弱关联（⚠️）记录到报告，视情况选择性进入 Phase 4
- 否定的关联（❌）记录到报告，说明否定原因
- 验证过程中发现的数据质量问题（缺失、重复）记录到报告
