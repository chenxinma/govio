---
name: govio-reconcile
description: 多源数据对账分析。当需要跨系统比对数据一致性、分析实体关联覆盖、做多维度统计画像时触发。典型场景：订单与客户主数据对账、跨系统数据质量检查、多数据源关联分析。
---

# Govio Reconcile 多源数据对账

跨多个异构数据源做实体对账、一致性比对和维度画像的交互式分析流程。

**核心抽象**：分析范围 = **实体集合** + **实体间关联** + **筛选口径**

```
┌──────────┐   join_key   ┌──────────┐   join_key   ┌──────────┐
│ Entity A │ ──────────── │ Entity B │ ──────────── │ Entity C │
│ (主实体)  │              │ (关联实体) │              │ (维度实体) │
└──────────┘              └──────────┘              └──────────┘
  口径: WHERE ...           口径: WHERE ...           (外部数据)
```

## 触发场景

| 场景 | 典型请求 |
|------|---------|
| 跨系统对账 | "CRM 的客户和 ERP 客户对比" |
| 数据质量 | "CRM 和 ERP 的客户数据是否一致" |
| 关联覆盖 | "这批客户在 ERP 系统里的覆盖情况" |
| 维度画像 | "按区域/渠道统计订单分布" |
| 多源分析 | "从多个业务系统拉数据做综合分析" |

## 五阶段流程

```
Phase 1 定义     Phase 2 加载     Phase 3 维度     Phase 4 对比     Phase 5 拓展
─────────── → ─────────── → ─────────── → ─────────── → ───────────
  分析图         数据加载         聚合画像         跨系统比对       关联拓展
  (交互确认)     (逐步加载)       (chart)         (report)        (可选)
```

每阶段产出画布节点，**阶段间需用户确认再继续**。

**渐进披露**：本文件只保留流程骨架与核心 SQL。进入某个阶段时读取 `references/` 下对应文档获取详细步骤、交互话术和 SQL 模板，不要预载全部。

| 阶段 | 何时读取 reference |
|------|-------------------|
| Phase 1 定义 | **进入 Phase 1 时**读取 [references/phase-1-define.md](references/phase-1-define.md) |
| Phase 2 加载 | **进入 Phase 2 时**读取 [references/phase-2-load.md](references/phase-2-load.md) |
| Phase 3 维度聚合 | **进入 Phase 3 时**读取 [references/phase-3-aggregate.md](references/phase-3-aggregate.md) |
| Phase 4 跨系统对比 | **进入 Phase 4 时**读取 [references/phase-4-compare.md](references/phase-4-compare.md) |
| Phase 5 关联拓展 | **进入 Phase 5 时**读取 [references/phase-5-enrich.md](references/phase-5-enrich.md) |

---

## Phase 1: 定义分析图

**目标**：和用户对齐"分析什么、用什么数据、怎么关联"

**交互流程**：

1. 用户描述分析目标（自然语言）
2. Agent 提取并提议分析图：
   - 实体列表（名称、数据来源、预估粒度）
   - 实体间关联（join key）
   - 主实体筛选口径
3. 用户确认或修正
4. 输出分析图摘要（Markdown）

**分析图模板**：

```markdown
## 分析图

| # | 实体 | 数据来源 | 粒度 | 筛选口径 |
|---|------|---------|------|---------|
| A | {名称} | {数据源}.{表} | {主键} | {WHERE 条件} |
| B | {名称} | {数据源}.{表} | {主键} | {WHERE 条件} |

### 关联链
A.[{key}] → B.[{key}] → C.[{key}]

### 分析口径
- 主实体 A 筛选: {条件}
- 排除范围: {条件}
- 最终分析范围: {描述}
```

**确认问题**（逐个问）：
1. "实体 A 的筛选口径是否正确？"
2. "A 和 B 的关联键是 `{key}`，对吗？"
3. "还有其他需要加入的实体吗？"

分析图确定后写入 Plan：`docs/govio/plans/YYYY-MM-DD-reconcile-[名称].md`

---

## Phase 2: 逐步加载

**目标**：按分析图逐个加载数据源，每步产出一个 DataFrame 节点

**执行顺序**：按关联链从左到右（先主实体，再关联实体）

**每步操作**：

1. 告知用户："现在加载实体 {X}，从 {数据源}，SQL 如下："
2. 执行 `govio-cli observe load --name {df_name} --datasource {ds} --sql "..."`
3. 展示 DataFrame 节点（行数、列数、字段摘要）
4. **询问用户**："数据加载完成，继续加载下一个实体？还是先对这个实体做筛选/聚合？"

**df 命名规范**：`{实体名}_{来源缩写}`，如 `order_crm`、`customer_erp`

**筛选变体**：如果主实体需要复杂筛选，分两步：
1. 先加载全量：`df_order_all`
2. 再 `--memory` 筛选：`df_order`（带 WHERE 条件）

```bash
# Step 1: 全量加载
govio-cli observe load --name order_all --datasource crm --sql "SELECT ... FROM ..."

# Step 2: 业务筛选
govio-cli observe load --name order --memory --sql "SELECT ... FROM order_all WHERE {口径条件}"
```

---

## Phase 3: 维度聚合与画像

**目标**：对主实体做 GROUP BY 聚合，产出统计表和图表

**时机**：主实体加载完成后，对比之前。先了解数据分布，再做精细对比。

**标准聚合模式**：

1. **层级统计（treemap）**：适合多层级维度（如：订单状态 → 客户等级 → 区域）

```sql
-- 先用 CASE WHEN 打标，再 GROUP BY
SELECT category1, category2, ..., COUNT(*) AS cnt
FROM (
  SELECT
    CASE WHEN status IN ('active','pending') THEN '进行中'
         WHEN status = 'closed' THEN '已完成'
         ELSE '其他' END AS category1,
    CASE WHEN amount >= 100000 THEN '大额'
         WHEN amount >= 10000 THEN '中额'
         ELSE '小额' END AS category2
  FROM df_main
)
GROUP BY category1, category2, ...
```

```bash
govio-cli observe load --name df_cat_stats --memory --sql "<上述 SQL>"
```

然后用 `govio_show_chart` 出 treemap 节点。

2. **排名统计（bar）**：适合单维度 Top N

```sql
SELECT dimension, COUNT(*) AS cnt
FROM df_main
GROUP BY dimension
ORDER BY cnt DESC
LIMIT 20
```

用 `govio_show_chart` 出 bar chart 节点。

3. **明细检查**：加载前确认用户是否需要看明细数据

**产出**：chart 节点 + 可选的 DataFrame 节点（聚合结果）

---

## Phase 4: 跨系统对比

**目标**：对有关联的实体对做匹配分析和字段级比对

**两层对比**：

### 4.1 匹配概览

先看两个实体通过 join key 的匹配情况：

```sql
SELECT
  COUNT(*) AS source_total,
  COUNT(b.key) AS matched,
  COUNT(*) - COUNT(b.key) AS unmatched_only_source,
  ROUND(COUNT(b.key) * 100.0 / COUNT(*), 1) AS match_rate_pct
FROM df_source a
LEFT JOIN df_target b ON a.key = b.key
```

产出 DataFrame 节点（匹配统计）。

再看反向：

```sql
SELECT
  COUNT(*) AS target_only
FROM df_target b
LEFT JOIN df_source a ON b.key = a.key
WHERE a.key IS NULL
```

**向用户汇报匹配率，询问**："匹配率 {X}%，要深入看不匹配的记录吗？"

### 4.2 不匹配明细

```sql
-- 仅在源表
SELECT a.key, a.name, ...
FROM df_source a
LEFT JOIN df_target b ON a.key = b.key
WHERE b.key IS NULL

-- 仅在目标表
SELECT b.key, b.name, ...
FROM df_target b
LEFT JOIN df_source a ON b.key = a.key
WHERE a.key IS NULL
```

### 4.3 字段级比对（匹配记录）

用 `govio-cli observe compare`：

```bash
govio-cli observe compare --source df_source --target df_target --join-columns key
```

产出 report 节点（datacompy 结果）。

或用 `--memory` SQL 做特定字段比对：

```sql
SELECT a.key, a.field1 AS source_val, b.field1 AS target_val
FROM df_source a
JOIN df_target b ON a.key = b.key
WHERE a.field1 != b.field1
```

**产出**：report 节点（比对报告）+ DataFrame 节点（差异明细，可选导出）

---

## Phase 5: 关联拓展（可选）

**目标**：用外部维度数据 enrich 主实体，做更深层分析

**典型场景**：主实体 A 通过 B 关联到 C（维度实体），分析 A 在 C 维度上的分布

**操作**：

1. 加载维度数据（如果还没加载）
2. `--memory` JOIN enrich：

```sql
SELECT a.*, c.dimension_col1, c.dimension_col2
FROM df_main a
LEFT JOIN df_dimension c ON a.key = c.key
```

3. 对 enrich 后的数据做维度聚合（复用 Phase 3 模式）

**多归属分析变体**：当同一实体关联到多个维度值

```sql
SELECT key, COUNT(DISTINCT category) AS cat_cnt
FROM df_enriched
GROUP BY key
HAVING COUNT(DISTINCT category) > 1
```

**产出**：DataFrame 节点（enriched 数据）+ chart 节点（维度分布）

---

## 产出物汇总

每阶段完成后，画布上应有对应的节点链：

```
sourceTable → sqlQuery → dataFrame(主实体)
                          ↓
              dataFrame(关联实体) → dataFrame(聚合) → chart(treemap/bar)
                                    ↓
                          report(比对结果) → dataFrame(差异明细)
                                              ↓
                                    dataFrame(enriched) → chart(维度分布)
```

最后可生成 Markdown 汇总报告（参考 govio-eda 的报告格式）。

---

## 与其他 Skill 的协作

| 操作 | Skill |
|------|-------|
| 查看数据源/加载数据 | `govio-observe`（load, compare, chart 命令） |
| 探查表关系 | `govio-observe`（explore 命令） |
| 查询元数据 | `govio-query` |
| 完整 EDA 探查 | `govio-eda`（本 skill 是其子集/特化） |

## 注意事项

1. **每步交互确认**：不要一口气执行完所有阶段，每个阶段结束后等用户确认
2. **画布优先**：统计结果用 chart 节点展示，不要在聊天里贴大表格
3. **资源管理**：中间 DataFrame 及时释放，避免内存堆积
4. **渐进披露**：进入某阶段时读取对应 reference 文档，不要预载全部
5. **命名一致**：df_name 全程保持一致，避免后续 SQL 引用错误
