# EDA 维度聚合画像与覆盖漏斗设计

**创建日期**: 2026-08-26
**状态**: 设计阶段
**来源**: `contract_explore.py`（合同-开账覆盖探查）分析过程提炼

## 1. 背景与边界

从 marimo 探查脚本提炼 EDA 过程时，需明确切分两类内容：

| 类别 | 内容 | 处理方式 |
|------|------|---------|
| **业务输入** | 码值语义（如 doc_status '30'->待关闭）、维度含义、口径定义（什么算有效记录）、归因规则 | **运行时由用户提供**，skill 不内置。在澄清阶段捕获，记入 Plan 文档 |
| **通用机制** | 聚合 SQL 骨架、长表派生、覆盖统计、漏斗渲染 | 本设计定义，落到 SKILL.md 方法论 |

本文只设计两个通用机制：

1. **维度聚合画像**：给定业务维度定义后，产出多维度分布画像
2. **覆盖漏斗**：给定业务口径后，统计跨数据集覆盖情况并渲染漏斗

## 2. 范围

### 2.1 非目标

- 实体归一/代表记录解析（主子合同 fallback 链）
- 枢纽实体对账（A、B 经 C 中转的间接关联发现）
- 多源合并口径（UNION + src 溯源）
- 快照水位（可重复性）
- Excel 多 sheet 导出
- 新增 CLI 命令（复用 `observe load --memory` / `chart`）
- 图表实现（treemap 等层级可视化）--由图表能力提供方另行实现，本设计只约定数据契约（见 4.2）

### 2.2 技术基线

- `load --memory` 将 store 中**所有**已加载 DataFrame 注册进 DuckDB 内存库，按 df 名称执行 SQL
- 每次执行产出新的具名 DataFrame，可作为后续查询输入
- SQL 模板按 DuckDB 方言书写（`GROUP BY ALL`、窗口函数、`COALESCE` 均可用）

## 3. 业务输入的捕获：维度与口径清单

澄清阶段（"澄清先于执行"原则的扩展）向用户收集两张输入表，存入 Plan 文档，随探索迭代修订。

### 3.1 维度定义表

```markdown
### 维度定义（数据集: {df_name}）

| 维度列 | 展示名 | 码值/归并规则 | NULL 展示值 | 度量 |
|--------|--------|--------------|------------|------|
| doc_status | 合同状态 | 见下方 CASE | - | COUNT(DISTINCT doc_number) |
| cb_name | 合同主体 | = '上海外服…' → 外服本部，else 其他 | - | 同上 |
| bu_name | 所属机构 | 无 | 空白 | 同上 |
```

维度表达式的四种形态（业务输入的落点）：

| 形态 | 表达式模式 | 示例 |
|------|----------|------|
| 直接列 | `col` | `org_name` |
| 码值解码 | `CASE col WHEN 'v' THEN '语义' … ELSE '其他' END` | doc_status '30'/'14' -> 待关闭 |
| 归并分组 | `CASE WHEN col IN (…) THEN '组A' ELSE '其他' END` | 主体归并为本部/其他 |
| NULL 展示 | `COALESCE(col, '展示值')` | bu_name 空白 |

### 3.2 覆盖口径定义表

```markdown
### 覆盖口径定义

| 项 | 值 |
|----|-----|
| 全集 | df_a，口径谓词: `delete_flag IS NULL AND comp 关联存在` |
| 覆盖目标 | df_b（目标键列: doc_number） |
| 关联键 | df_a.center_id = df_b.doc_number |
| 归因分层（有序） | 1) 无 comp 关联 → 无关联客户；2) else → 未归集 |
```

### 3.3 兜底路径

用户给不出输入时的降级：

- **无维度定义** -> 执行现有 Phase 1 字段级画像（4a-4d），从 4c Top 值结果中挑低基数分类列，**提议为候选维度**请用户确认语义后，再进入维度画像
- **无覆盖口径** -> 只做现有 Phase 3 匹配率检查，不做漏斗

## 4. 机制一：维度聚合画像

### 4.1 核心思路：一次全维聚合成长表，其余全部由长表再聚合

对 k 个维度只做**一次** GROUP BY，产出"维度组合 -> 计数"长表并存为新 df。边际分布、层级汇总都是对小体量长表的再聚合，代价极小。

### 4.2 SQL 模板

```sql
-- L1: 全维度聚合长表（一次扫描，存为 {df}_dim_profile）
SELECT
  {dim_1_expr} AS d1,
  {dim_2_expr} AS d2,
  -- …
  {dim_k_expr} AS dk,
  COUNT(DISTINCT {id_col}) AS cnt
FROM {df}
WHERE {口径谓词}            -- 可选，来自覆盖口径或维度定义
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

### 4.2.1 可视化数据契约

图表如何渲染（treemap、sunburst 等）是图表能力提供方的实现细节，EDA 侧只保证产出符合以下契约的 DataFrame，供其消费：

| 契约项 | 内容 |
|--------|------|
| 数据结构 | 长表：`d1, d2, …, dk, cnt`（每行一个维度组合及其计数） |
| 层级语义 | 列序即层级路径（d1 为根，逐层下钻） |
| 取值 | `cnt >= 1`，无空行；NULL 已在维度表达式中替换为展示值 |
| 典型消费方 | treemap（path = d1..dk, values = cnt）、逐层 bar、markdown 表 |

降级形态：图表能力不可用时，用 markdown 表（按 cnt 降序 Top-N）呈现，不影响方法完整性。

### 4.3 输出：维度画像卡片

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

## 5. 机制二：覆盖漏斗

### 5.1 核心思路

以业务口径圈定**全集**，对覆盖目标键（**先去重防 1:N 膨胀**）做 LEFT JOIN，统计已覆盖/未覆盖；未覆盖部分按有序归因规则分层。覆盖率是**发现**（业务事实），不是判定。

### 5.2 SQL 模板

```sql
-- F1: 覆盖统计
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

```sql
-- F2: 未覆盖分层归因（归因表达式只引用 universe 内的列）
SELECT
  {归因表达式} AS reason,
  COUNT(*) AS cnt,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS pct
FROM universe u
LEFT JOIN target_keys t ON u.join_key = t.join_key
WHERE t.join_key IS NULL
GROUP BY 1
ORDER BY cnt DESC
```

归因表达式为有序 CASE WHEN（业务输入），例：

```sql
CASE
  WHEN comp 关联不存在 THEN '无关联客户'
  ELSE '未归集'
END
```

```sql
-- F3: 反向检查（目标侧多余键，可选）
SELECT COUNT(*) AS target_only_cnt
FROM target_keys t
LEFT JOIN universe u ON u.join_key = t.join_key
WHERE u.join_key IS NULL
```

### 5.3 输出：覆盖漏斗卡片

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

### 5.4 语义约定

1. 覆盖率是发现，**不做**强/弱/非关联判定（区别于 Phase 3 的匹配率阈值）
2. 可选：与用户给定的关注阈值比较，超过时加"需关注"标记
3. 归因分层建议 ≤ 3 层，更深改用表格呈现
4. F1 与 Phase 3 Step 1 是同一 SQL 骨架；本机制的增量在**口径谓词圈定全集 + 分层归因 + 漏斗叙事**

## 6. 与 govio-eda SKILL.md 的整合（最小改动）

**已实施（2026-08-27）**：skill 已按本设计更新为渐进披露结构，内容自包含（不回引本文档）：`SKILL.md` 保留流程骨架与产出契约，详细步骤与 SQL 模板拆至 `references/`（phase-1-profile / phase-2-relations / phase-3-verify / phase-4-consistency / business-inputs）。

| 改动点 | 内容 |
|--------|------|
| 触发场景表 | +2 行：维度画像（"按业务维度看看分布"）、覆盖漏斗（"A 在 B 的覆盖情况/归集了多少"） |
| Phase 1 | +Step 4e 维度聚合画像（引用本设计第 4 节） |
| Phase 3 | +Step 5 覆盖漏斗（引用本设计第 5 节） |
| Plan 模板 | +维度与口径清单小节（本设计第 3 节模板） |
| 产出规范 | +维度画像卡片、覆盖漏斗卡片模板 |
| 澄清原则 | +两项澄清内容：维度定义、覆盖口径；含兜底路径 |

## 7. 示例映射（contract_explore.py -> 本设计）

| 脚本片段 | 机制 |
|---------|------|
| `df_contract_detail`（状态×主体×机构×部门聚合 + treemap） | 维度聚合画像 L1/L3 |
| `df_all_wf_cnt`（速创 -> 合同中心 已进/历史 + ASCII 树） | 覆盖漏斗 F1 + 渲染 |
| `df_detail`（有效性归因：已删除/无关联客户/未归集） | 覆盖漏斗 F2 分层归因 |
| `df_cntr_comp_cnt`（关联表 ↔ 合同双向孤儿） | F3 反向检查 |
| 业务值（待关闭、外服本部等中文语义） | 运行时业务输入，仅作样例 |

## 8. 开放问题

1. 可视化数据契约是否需要独立的交付格式（当前为“长表即契约”，若图表能力提供方需要固定 schema 再固化）
2. 链式漏斗（A->B->C 多级串联覆盖）是否需要：当前按单级设计，链式 = 多次执行 F1 后手工拼接
3. 长表 df 的生命周期管理（`{df}_dim_profile` 类中间产物何时 release，沿用现有资源管理注意事项即可）
