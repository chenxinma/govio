---
name: govio-eda
description: EDA 探索性数据分析流程。当用户需要对数据集进行全面探查、了解数据全貌、发现数据关联、核查数据质量时触发。定义标准 4 阶段探查流程（画像→推断关联→核查关联→一致性检查），产出 Markdown 报告 + 异常数据集（JSON 导出，后续合并为 Excel）。
---

# Govio EDA 探索性数据分析

本 Skill 定义标准的 EDA 探查流程，通过 4 个阶段从数据加载到一致性核查，产出完整的探查报告。

## 触发场景

| 场景 | 典型请求示例 |
|------|------------|
| 数据探查 | "帮我探查一下这批数据"、"分析这些表的结构和关系" |
| 数据质量 | "检查数据一致性"、"发现跨系统的数据差异" |
| 关联分析 | "这些表之间有什么关系"、"验证外键是否正确" |
| 业务核查 | "检查项目和商机的状态是否一致"、"客户主数据是否对齐" |

## 核心原则

### 1. 澄清先于执行

开始前确认：
- **探查目标**：要解决什么业务问题？
- **数据范围**：涉及哪些数据源和表？
- **探查深度**：快速概览还是完整 4 阶段？
- **产出要求**：报告给谁看？需要哪些异常数据集？

### 2. 计划驱动

EDA 任务必须编写 Plan，保存到 `docs/govio/plans/YYYY-MM-DD-eda-[项目名].md`。

### 3. 阶段化执行

4 个阶段按序执行，每阶段产出明确物，下一阶段依赖上一阶段结果。

## 标准流程

```
Phase 1 画像 → Phase 2 推断关联 → Phase 3 核查关联 → Phase 4 一致性核查
    ↓              ↓                   ↓                   ↓
 数据集卡片      候选关联清单        确认关联报告        一致性报告 + 异常数据(JSON)
    ↓──────────────↓───────────────────↓───────────────────↓
                    汇总为 EDA Markdown 报告
```

## 使用模式

### 模式 A: 快速探查（Phase 1 + 2）

了解数据全貌，不深入核查。

```
用户: "帮我看看这些表的情况"
→ Phase 1: 加载 + 画像
→ Phase 2: 推断关联
→ 产出: 数据集卡片 + 关系图谱
```

### 模式 B: 标准 EDA（Phase 1-4）

完整探查流程，产出完整报告。

```
用户: "对销售系统做一次完整的数据探查"
→ Phase 1-4 全部执行
→ 产出: 完整 EDA 报告 + 异常数据 JSON（后续合并为 Excel）
```

### 模式 C: 专项核查（直接 Phase 4）

已有数据，直接做一致性检查。

```
用户: "检查这两个系统的客户数据是否一致"
→ 跳过 Phase 1-2（数据已加载或快速加载）
→ Phase 3: 验证关联
→ Phase 4: 执行"字段一致性"规则
→ 产出: 一致性报告 + 异常数据 JSON（后续合并为 Excel）
```

---

## Phase 1: 数据集画像

**目标**：回答"我拿到的是什么数据？"

### 流程

```
info --datasource → 确认数据源 → load 加载 → info --df 查看结构 → load --memory 画像 SQL → chart 可视化
```

### Step 1: 确认数据源

```bash
govio-cli observe info --datasource
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
govio-cli observe info --df
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

### 产出: 数据集卡片

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

### 画像 SQL 快速参考

| 分析项 | SQL 模式 |
|--------|---------|
| 总行数 | `SELECT COUNT(*) FROM df` |
| 非空率 | `SUM(CASE WHEN col IS NOT NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)` |
| 基数 | `COUNT(DISTINCT col)` |
| 数值统计 | `AVG`, `STDDEV`, `MIN`, `MAX`, `PERCENTILE_CONT` |
| Top 值 | `GROUP BY col ORDER BY COUNT(*) DESC LIMIT N` |
| 日期范围 | `MIN(date_col)`, `MAX(date_col)` |

---

## Phase 2: 推断关联

**目标**：回答"这些数据集之间有什么关系？"

### 前置条件

Phase 1 完成，目标数据集已全部加载到 ObserveStore。

### 流程

```
info --df 确认已加载 → explore 自动发现 → visualize-relations 生成图谱 → 人工审查筛选
```

### Step 1: 确认已加载数据集

```bash
govio-cli observe info --df
```

确认要探查的数据集已全部加载。如有遗漏，返回 Phase 1 补充加载。

### Step 2: 运行关系探索

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

### Step 3: 生成关系图谱

```bash
govio-cli observe visualize-relations --relations '<explore_result_json>'
```

产出 nodes 和 edges 数据，用于渲染关系图谱。

### Step 4: 人工审查筛选

自动发现的结果需要结合业务知识筛选：

| 审查项 | 判断标准 |
|--------|---------|
| 候选外键是否合理 | 字段语义是否匹配（如 `customer_id` → `customers.id`） |
| 相似列是否同义 | 如 `email` ~ `customer_email` 可能是同一数据 |
| 是否遗漏关联 | 业务上应有关联但自动发现未覆盖的（如非标准命名） |
| 是否误报 | 如 `order_status_id` → `products.id` 可能是误匹配 |

### 产出: 候选关联清单

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

### 候选关联的补充来源

自动发现覆盖不了所有关联，以下场景需要人工补充：

1. **语义关联但命名不同**：如 `bpext` → `comp_no`（业务编码对齐）
2. **多跳关联**：A→B→C 的间接关系
3. **业务规则关联**：如"项目结算客户 ⊆ 商机客户"（非 FK 关系）
4. **跨库关联**：不同数据源中的同义字段

---

## Phase 3: 核查关联性

**目标**：回答"这些关联真实存在吗？数据支撑吗？"

### 前置条件

Phase 2 完成，已产出候选关联清单。

### 流程

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
| >= 95% | 强关联 | 关联可靠 |
| 80% - 95% | 弱关联 | 存在缺失，需分析原因 |
| < 80% | 非关联 | 可能是误匹配 |

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

### 产出: 关联验证报告

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
| **判定** | 强关联 |

**缺失键分析**：
- 800 条未匹配记录中，780 条 customer_id 为 NULL
- 20 条为已删除客户（已从 customers 表移除）
```

### 核查 SQL 速查

| 检查项 | SQL 模式 |
|--------|---------|
| 匹配率 | `LEFT JOIN + COUNT(key) / COUNT(*)` |
| 缺失键 | `LEFT JOIN ... WHERE t.key IS NULL` |
| 重复键 | `GROUP BY key HAVING COUNT(*) > 1` |
| 值域交集 | `INTERSECT` / `JOIN` + `COUNT(DISTINCT)` |
| NULL 比例 | `SUM(CASE WHEN key IS NULL THEN 1 END)` |

---

## Phase 4: 一致性核查

**目标**：回答"关联的数据之间是否一致？有没有业务逻辑违规？"

### 前置条件

Phase 3 完成，已产出确认的关联清单。

### 流程

```
选择规则模式 → 实例化规则（指定表/字段/JOIN 键） → 执行规则 SQL → 导出异常数据（JSON）
```

### 规则模板库

#### 模式 1: 字段一致性

**适用场景**：两个系统存储相同实体，需要比对关键字段是否一致。

##### 1a. 多字段逐列比对

```sql
SELECT
  s.join_key,
  s.field1 AS src_field1, t.field1 AS tgt_field1,
  s.field2 AS src_field2, t.field2 AS tgt_field2,
  CASE
    WHEN s.field1 = t.field1 AND s.field2 = t.field2 THEN '一致'
    WHEN s.field1 IS NULL AND t.field1 IS NULL THEN '双方都空'
    WHEN s.field1 IS NOT NULL AND t.field1 IS NULL THEN '源有值目标空'
    WHEN s.field1 IS NULL AND t.field1 IS NOT NULL THEN '源空目标有值'
    ELSE '不一致'
  END AS status
FROM source_df s
JOIN target_df t ON s.join_key = t.join_key
```

##### 1b. 名称差异分类

```sql
SELECT
  s.join_key,
  s.name AS src_name, t.name AS tgt_name,
  CASE
    WHEN REPLACE(REPLACE(s.name, ' ', ''), '-', '') = REPLACE(REPLACE(t.name, ' ', ''), '-', '') THEN '标点差异'
    WHEN s.name LIKE CONCAT('%', t.name, '%') OR t.name LIKE CONCAT('%', s.name, '%') THEN '表述差异(子串)'
    ELSE '其他差异'
  END AS diff_type
FROM source_df s
JOIN target_df t ON s.join_key = t.join_key
WHERE s.name <> t.name
```

**异常判定**：status = '不一致' 或 diff_type = '其他差异' 的记录。

---

#### 模式 2: 子集检查

**适用场景**：验证 A 表的某字段值集合是否 ⊆ B 表的对应字段值集合。

##### 2a. A ⊆ B 存在性检查

```sql
SELECT s.field AS missing_value, COUNT(*) AS cnt
FROM source_df s
LEFT JOIN target_df t ON s.field = t.field
WHERE t.field IS NULL
GROUP BY s.field
ORDER BY cnt DESC
```

##### 2b. 反向检查 B 中多余的值

```sql
SELECT t.field AS extra_value, COUNT(*) AS cnt
FROM target_df t
LEFT JOIN source_df s ON t.field = s.field
WHERE s.field IS NULL
GROUP BY t.field
ORDER BY cnt DESC
```

**异常判定**：missing_value 不为空则子集关系不成立。

---

#### 模式 3: 状态机校验

**适用场景**：关联实体的状态组合是否符合业务规则。

##### 3a. 状态组合统计

```sql
SELECT
  s.status AS src_status,
  t.status AS tgt_status,
  COUNT(*) AS cnt
FROM source_df s
JOIN target_df t ON s.join_key = t.join_key
GROUP BY s.status, t.status
ORDER BY cnt DESC
```

##### 3b. 定义非法状态组合

```sql
SELECT s.join_key, s.status AS src_status, t.status AS tgt_status
FROM source_df s
JOIN target_df t ON s.join_key = t.join_key
WHERE (s.status, t.status) IN (
  ('已撤销', '赢单'),
  ('立项中', '赢单'),
  ('已结束', '进行中'),
  ('不通过', '赢单')
)
```

##### 3c. 1:N 状态分裂检测

```sql
SELECT join_key, COUNT(DISTINCT tgt_status) AS status_count,
       STRING_AGG(DISTINCT tgt_status, ', ') AS statuses
FROM source_df s JOIN target_df t ON s.join_key = t.join_key
GROUP BY join_key
HAVING COUNT(DISTINCT tgt_status) > 1
```

**异常判定**：非法状态组合的记录。

---

#### 模式 4: 结构完整性

**适用场景**：验证 1:N 层级关系是否异常。

##### 4a. 1:N 关系统计

```sql
SELECT parent_key, COUNT(DISTINCT child_key) AS child_count
FROM relation_df
GROUP BY parent_key
HAVING COUNT(DISTINCT child_key) > 1
ORDER BY child_count DESC
```

##### 4b. 异常阈值检测

```sql
SELECT parent_key, COUNT(DISTINCT child_key) AS child_count
FROM relation_df
GROUP BY parent_key
HAVING COUNT(DISTINCT child_key) > <expected_max>
```

**异常判定**：child_count 超过业务预期阈值。

---

#### 模式 5: 孤儿检测

**适用场景**：检测无父记录的子记录。

##### 5a. 孤儿记录检测

```sql
SELECT c.*
FROM child_df c
LEFT JOIN parent_df p ON c.parent_key = p.parent_key
WHERE p.parent_key IS NULL
```

##### 5b. 反向孤儿（无子记录的父记录）

```sql
SELECT p.*
FROM parent_df p
LEFT JOIN child_df c ON p.parent_key = c.parent_key
WHERE c.parent_key IS NULL
```

**异常判定**：存在孤儿记录。

---

### 规则实例化示例

以 R1（IHRO 项目结算客户 vs MDM 主数据）为例：

```markdown
**规则**: IHRO 项目结算客户 vs MDM 主数据字段一致性
**模式**: 字段一致性 (1a)
**源表**: ihro_settle_cust（已加载）
**目标表**: mdm_customer（已加载）
**JOIN 键**: settle_cust.comp_no = mdm.comp_no
**比对字段**: uni_soc_cret_code, cust_no, pot_cust_no, cust_name

**SQL**:
SELECT
  s.comp_no,
  s.uni_soc_cret_code AS src_usci, t.uni_soc_cret_code AS tgt_usci,
  s.cust_no AS src_cust_no, t.cust_no AS tgt_cust_no,
  CASE
    WHEN s.uni_soc_cret_code = t.uni_soc_cret_code
     AND s.cust_no = t.cust_no
     AND s.pot_cust_no = t.pot_cust_no
     AND s.cust_name = t.cust_name THEN '一致'
    WHEN s.uni_soc_cret_code IS NULL AND t.uni_soc_cret_code IS NULL
     AND s.cust_no IS NULL AND t.cust_no IS NULL THEN '双方都空'
    ELSE '不一致'
  END AS status
FROM ihro_settle_cust s
JOIN mdm_customer t ON s.comp_no = t.comp_no
```

### 产出: 一致性核查报告

```markdown
## 一致性核查结果

### 规则 1: IHRO vs MDM 字段一致性

| 状态 | 数量 | 占比 |
|------|------|------|
| 一致 | 8,500 | 85% |
| 不一致 | 1,200 | 12% |
| 源有值目标空 | 200 | 2% |
| 源空目标有值 | 100 | 1% |
| 双方都空 | 0 | 0% |

**异常**: 1,200 条不一致记录 → `eda_项目_IHRO_vs_MDM_异常.json`（后续合并为 Excel）

### 规则 2: 项目结算客户 ⊆ 商机客户

| 检查项 | 结果 |
|--------|------|
| 项目客户总数 | 1,000 |
| 在商机中找到 | 950 |
| 缺失数 | 50 |
| **判定** | 50 个项目客户不在商机集合中 |

**异常**: 50 条缺失记录 → `eda_项目_客户子集_异常.json`（后续合并为 Excel）
```

### 导出异常数据

每条规则执行后，异常记录需要导出为结构化文件。

**当前能力**：`govio-cli observe load -o` 仅支持将单个 DataFrame 导出为 JSON 格式。

**需求**：将多条规则的异常数据合并导出为一个 Excel 文件（每个 sheet 对应一条规则），该能力需要另外实现。EDA skill 仅描述此需求，不涉及具体导出方案。

**当前可行的做法**：
1. 用 `load --memory -o` 将各规则的异常记录分别导出为 JSON
2. 后续由独立的 Excel 合并工具将多个 JSON 整合为一个多 sheet 的 Excel 文件

**注意**：导出数据内容前需征得用户确认。

---

## Plan 模板

```markdown
# EDA 探查计划: [项目名称]

**目标**: [探查目的]
**数据源**: [数据源列表]
**阶段**: [快速/标准/专项]
**创建时间**: YYYY-MM-DD

---

## Phase 1: 数据集画像

- [ ] 加载数据集: [列表]
- [ ] 执行画像分析
- [ ] 产出数据集卡片

## Phase 2: 推断关联

- [ ] 运行关系探索
- [ ] 生成关系图谱
- [ ] 筛选候选关联

## Phase 3: 核查关联

- [ ] 验证每个候选关联
- [ ] 记录匹配率和异常

## Phase 4: 一致性核查

- [ ] 选择适用规则模式
- [ ] 实例化并执行规则
- [ ] 导出异常数据（JSON，后续合并为 Excel）

## 汇总

- [ ] 生成 EDA 报告 (Markdown)
- [ ] 整理异常数据集（JSON → 合并为 Excel，需另外实现）
```

## 产出规范

### Markdown 报告结构

```markdown
# EDA 探查报告: [项目名称]

**探查时间**: YYYY-MM-DD
**数据源**: [列表]
**探查范围**: [表列表]

## 1. 数据集画像
[每个数据集的画像卡片]

## 2. 关联分析
[候选关联清单 + 关系图谱]

## 3. 关联验证
[每个关联的验证结果]

## 4. 一致性核查
[每条规则的通过/异常情况]

## 5. 发现汇总
### 5.1 关键发现
### 5.2 异常清单
### 5.3 建议

## 附录: 异常数据集
[JSON 文件清单及说明，后续合并为 Excel]
```

### 异常数据产出

异常数据集统一导出为 JSON，命名规范：
- `eda_[项目]_[规则名]_异常.json`
- 每个文件对应一条规则的异常记录
- 后续合并为 Excel（每个 sheet 一条规则），该合并能力需另外实现

---

## 与其他 Skill 的协作

| 操作 | 关联 Skill |
|------|-----------|
| 查询元数据 | `govio-query` |
| 知识图谱维护 | `govio-meta` |
| 数据探查 | `govio-observe` |

## 注意事项

1. **资源管理**: 每个阶段结束检查 `govio-cli observe info --df`，及时释放不再需要的 DataFrame
2. **大数据量**: 画像阶段先用 `LIMIT` 采样，确认结构后再全量加载
3. **用户确认**: 导出数据内容前，征得用户同意
4. **阶段回顾**: 每阶段完成后向用户展示产出，确认后再进入下一阶段
