# Phase 2: 推断关联

**目标**：回答"这些数据集之间有什么关系？"

**前置条件**: Phase 1 完成，目标数据集已全部加载到 ObserveStore。

## 流程

```
info --df 确认已加载 -> explore 自动发现 -> visualize-relations 生成图谱 -> 人工审查筛选
```

## Step 1: 确认已加载数据集

```bash
govio-cli observe info --df
```

确认要探查的数据集已全部加载。如有遗漏，返回 Phase 1 补充加载。

## Step 2: 运行关系探索

```bash
govio-cli observe explore --dataframes df1 df2 df3
```

或省略参数探查所有已加载：

```bash
govio-cli observe explore
```

**返回结果包含两类关系**：

1. **外键关系 (foreign_key)**：基于列名模式（`*_id`, `*id`）和值重叠率，confidence > 0.5 时报告
2. **列名相似 (column_similarity)**：基于字符串相似度，similarity > 0.7 时报告

## Step 3: 生成关系图谱

先将 `explore` 结果保存为 JSON 文件，再通过文件路径传入：

```bash
govio-cli observe explore --dataframes df1 df2 df3 > relations.json
govio-cli observe visualize-relations --relations-file relations.json
```

产出 nodes 和 edges 数据，用于渲染关系图谱。输入文件格式参见 `govio-observe` Skill 的 `assets/visualize-relations-schema.json`。

## Step 4: 人工审查筛选

自动发现的结果需要结合业务知识筛选：

| 审查项 | 判断标准 |
|--------|---------|
| 候选外键是否合理 | 字段语义是否匹配（如 `customer_id` -> `customers.id`） |
| 相似列是否同义 | 如 `email` ~ `customer_email` 可能是同一数据 |
| 是否遗漏关联 | 业务上应有关联但自动发现未覆盖的（如非标准命名） |
| 是否误报 | 如 `order_status_id` -> `products.id` 可能是误匹配 |

## 产出: 候选关联清单

```markdown
## 候选关联清单

| # | 类型 | 源表.字段 | 目标表.字段 | 置信度 | 审查结果 |
|---|------|----------|------------|--------|---------|
| 1 | 外键 | orders.customer_id | customers.id | 95% | 确认 |
| 2 | 相似列 | customers.email | orders.customer_email | 85% | 同义 |
| 3 | 外键 | orders.product_id | products.id | 90% | 确认 |
| 4 | 相似列 | orders.status | orders.order_status | 75% | 同表不同列 |
| 5 | 业务关联 | projects.bpext | customers.comp_no | - | 业务补充 |
```

**审查结果分类**：
- 确认：进入 Phase 3 核查
- 排除：记录排除原因
- 补充：业务知识补充的关联

## 候选关联的补充来源

自动发现覆盖不了所有关联，以下场景需要人工补充：

1. **语义关联但命名不同**：如 `bpext` -> `comp_no`（业务编码对齐）
2. **多跳关联**：A->B->C 的间接关系
3. **业务规则关联**：如"项目结算客户 ⊆ 商机客户"（非 FK 关系）
4. **跨库关联**：不同数据源中的同义字段

当关联已由业务输入明确给出（键映射已知）时，可跳过本阶段的自动发现，直接进入 Phase 3 核查。
