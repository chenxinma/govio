---
name: eda-check-consistency
description: EDA Phase 4 - 一致性核查。对已确认的关联执行深度一致性检查，包含 5 类通用规则模板（字段一致性、子集检查、状态机校验、结构完整性、孤儿检测）。异常记录导出为 JSON，后续合并为 Excel（需另外实现）。依赖 observe-dataset-ops。
---

# EDA Phase 4: 一致性核查

对已确认的关联执行深度一致性检查，发现数据质量问题。

## 目标

回答："关联的数据之间是否一致？有没有业务逻辑违规？"

## 依赖 Skill

| 操作 | Skill | CLI 命令 |
|------|-------|---------|
| 二次加工查询 | `observe-dataset-ops` | `govio-cli observe load --memory` |
| 导出数据 | `observe-dataset-ops` | `govio-cli observe load --memory -o <file>` |
| 释放资源 | `observe-dataset-ops` | `govio-cli observe release` |

## 前置条件

Phase 3 完成，已产出确认的关联清单。

## 流程

```
选择规则模式 → 实例化规则（指定表/字段/JOIN 键） → 执行规则 SQL → 导出异常数据（JSON）
```

## 规则模板库

### 模式 1: 字段一致性

**适用场景**：两个系统存储相同实体，需要比对关键字段是否一致。
**参考**：R1（IHRO vs MDM 4 字段一致性）、R15（ERP vs MDM 名称差异）

#### 1a. 多字段逐列比对

```sql
-- 将 source_df, target_df, join_key, field1, field2... 替换为实际值
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

#### 1b. 名称差异分类（参考 R15）

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
**产出**：异常记录 → JSON（后续合并为 Excel）。

---

### 模式 2: 子集检查

**适用场景**：验证 A 表的某字段值集合是否 ⊆ B 表的对应字段值集合。
**参考**：R2（项目结算客户 ⊆ 商机客户）、R8（外包商机 ⊆ CRM 商机）、R13/R14（MDM 客户完整性）

#### 2a. A ⊆ B 存在性检查

```sql
-- 检查 source_df.field 是否都在 target_df.field 中
SELECT s.field AS missing_value, COUNT(*) AS cnt
FROM source_df s
LEFT JOIN target_df t ON s.field = t.field
WHERE t.field IS NULL
GROUP BY s.field
ORDER BY cnt DESC
```

#### 2b. 反向检查 B 中多余的值

```sql
SELECT t.field AS extra_value, COUNT(*) AS cnt
FROM target_df t
LEFT JOIN source_df s ON t.field = s.field
WHERE s.field IS NULL
GROUP BY t.field
ORDER BY cnt DESC
```

**异常判定**：missing_value 不为空则子集关系不成立。
**产出**：缺失值清单 → JSON（后续合并为 Excel）。

---

### 模式 3: 状态机校验

**适用场景**：关联实体的状态组合是否符合业务规则。
**参考**：R3（商机状态 vs 项目状态）、R4（1 项目多商机状态不一致）、R5（1 商机多项目）

#### 3a. 状态组合统计

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

#### 3b. 定义非法状态组合

```sql
-- 基于 3a 的结果，筛选非法组合
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

#### 3c. 1:N 状态分裂检测（参考 R4）

```sql
SELECT join_key, COUNT(DISTINCT tgt_status) AS status_count,
       STRING_AGG(DISTINCT tgt_status, ', ') AS statuses
FROM source_df s JOIN target_df t ON s.join_key = t.join_key
GROUP BY join_key
HAVING COUNT(DISTINCT tgt_status) > 1
```

**异常判定**：非法状态组合的记录。
**产出**：异常状态记录 → JSON（后续合并为 Excel）。

---

### 模式 4: 结构完整性

**适用场景**：验证 1:N 层级关系是否异常。
**参考**：R6（主合同-子合同-项目 1 对多）

#### 4a. 1:N 关系统计

```sql
SELECT parent_key, COUNT(DISTINCT child_key) AS child_count
FROM relation_df
GROUP BY parent_key
HAVING COUNT(DISTINCT child_key) > 1
ORDER BY child_count DESC
```

#### 4b. 异常阈值检测

```sql
-- 超过预期 N 倍的异常
SELECT parent_key, COUNT(DISTINCT child_key) AS child_count
FROM relation_df
GROUP BY parent_key
HAVING COUNT(DISTINCT child_key) > <expected_max>
```

**异常判定**：child_count 超过业务预期阈值。
**产出**：异常层级关系 → JSON（后续合并为 Excel）。

---

### 模式 5: 孤儿检测

**适用场景**：检测无父记录的子记录。
**参考**：R10（孤儿商机客户）

#### 5a. 孤儿记录检测

```sql
SELECT c.*
FROM child_df c
LEFT JOIN parent_df p ON c.parent_key = p.parent_key
WHERE p.parent_key IS NULL
```

#### 5b. 反向孤儿（无子记录的父记录）

```sql
SELECT p.*
FROM parent_df p
LEFT JOIN child_df c ON p.parent_key = c.parent_key
WHERE c.parent_key IS NULL
```

**异常判定**：存在孤儿记录。
**产出**：孤儿记录清单 → JSON（后续合并为 Excel）。

---

## 规则实例化示例

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

## 产出: 一致性核查报告

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
| **判定** | ⚠️ 50 个项目客户不在商机集合中 |

**异常**: 50 条缺失记录 → `eda_项目_客户子集_异常.json`（后续合并为 Excel）
```

## 导出异常数据

每条规则执行后，异常记录需要导出为结构化文件。

**当前能力**：`govio-cli observe load -o` 仅支持将单个 DataFrame 导出为 JSON 格式。

**需求**：将多条规则的异常数据合并导出为一个 Excel 文件（每个 sheet 对应一条规则），该能力需要另外实现。EDA skill 仅描述此需求，不涉及具体导出方案。

**当前可行的做法**：
1. 用 `load --memory -o` 将各规则的异常记录分别导出为 JSON
2. 后续由独立的 Excel 合并工具将多个 JSON 整合为一个多 sheet 的 Excel 文件

**注意**：导出数据内容前需征得用户确认。

## 与其他阶段的衔接

- Phase 3 的验证结果决定哪些关联进入一致性核查
- 核查结果汇总到 EDA 报告的"一致性核查"章节
- 异常数据集 Excel 清单汇总到报告的"附录"
