---
name: observe-compare-dfs
description: 比对两个 DataFrame 的结构和数据差异。当需要验证数据一致性、检查数据迁移结果、或发现数据差异时使用。通过 govio-cli observe compare 命令执行，使用 datacompy 进行数据比对。
---

# Observe Compare DataFrames

比对两个 DataFrame 的结构和数据差异。

## CLI 命令

```bash
govio-cli observe compare <source> <target> --join-columns col1,col2
```

## 参数

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| source | positional | 是 | 源 DataFrame 名称 |
| target | positional | 是 | 目标 DataFrame 名称 |
| --join-columns | option | 是 | 用于比对的列，逗号分隔（通常是主键） |

## 使用场景

- 验证数据迁移的一致性
- 检查数据同步结果
- 发现源系统和目标系统的差异
- 数据质量检查

## 前置条件

必须先使用 `govio-cli observe load` 加载两个 DataFrame：

```bash
govio-cli observe load legacy_customers legacy_db "SELECT customer_id, name, email FROM customers"
govio-cli observe load new_customers new_db "SELECT customer_id, name, email FROM customers"
```

**join_columns 列名匹配规则**:

两个数据集的 join 列名必须完全相同。如果存在名称差异，需要先进行列名转换：

1. **大小写差异**: 如 `a.COMP_NO` 与 `b.comp_no`，需统一转换为小写
2. **主键引用差异**: 如 `t_order.cust_id` 与 `t_cust.id`，需将 `t_cust.id` 重命名为 `t_cust.cust_id`

## 调用方式

```bash
govio-cli observe compare legacy_customers new_customers --join-columns customer_id
```

多个 join 列用逗号分隔：

```bash
govio-cli observe compare source_table target_table --join-columns customer_id,order_id
```

## 返回结果

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

## 结果解读

### 结构差异 (schema)

| 字段 | 含义 |
|---|---|
| match | 结构是否完全一致 |
| source_columns | 源表所有列 |
| target_columns | 目标表所有列 |
| common_columns | 共有的列 |
| source_only | 仅在源系统存在的列 |
| target_only | 仅在目标系统存在的列 |

### 数据差异 (data)

| 字段 | 含义 |
|---|---|
| report | datacompy 生成的完整比对报告，包含匹配行数、独有行、值差异等详细信息 |

## 示例

**场景**: 用户说"比对 legacy_customers 和 new_customers"

**执行**:
```bash
govio-cli observe compare legacy_customers new_customers --join-columns customer_id
```

**返回解读**:
```
比对结果:

结构差异:
- 共有列: customer_id, name, email
- 源系统独有: phone
- 目标系统独有: mobile
- 结构匹配: ❌

数据差异:
- 详见 datacompy 报告
```

## 最佳实践

1. **选择合适的 join_columns**: 使用能唯一标识记录的列（通常是主键）
2. **先检查结构**: 查看 schema 差异，了解列的变化
3. **关注匹配率**: 如果匹配率低，需要深入调查原因
4. **记录差异**: 保存比对结果用于分析

## 后续操作

- 使用 `govio-cli observe release <name>` 释放 DataFrame
- 根据差异结果决定是否需要修复数据

## 与其他 Skill 的协作

| 前置操作 | 关联 Skill |
|---|---|
| 加载数据 | `observe-dataset-ops` |
| 主控协调 | `govio-observe` |
