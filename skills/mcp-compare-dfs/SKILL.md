---
name: mcp-compare-dfs
description: 比对 Govio MCP 中的两个 DataFrame。当需要验证数据一致性、检查数据迁移结果、或发现数据差异时使用。比对内容包括结构差异（列的增减）和数据差异（匹配率、独有行）。需要指定源 DataFrame、目标 DataFrame 和 join 列。
---

# MCP Compare DataFrames

比对两个 DataFrame 的结构和数据差异。

## 使用场景

- 验证数据迁移的一致性
- 检查数据同步结果
- 发现源系统和目标系统的差异
- 数据质量检查

## 参数

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| source_df | string | 是 | 源 DataFrame 名称 |
| target_df | string | 是 | 目标 DataFrame 名称 |
| join_columns | list[string] | 是 | 用于比对的列（通常是主键） |

## 调用方式

```python
compare_dfs(
    source_df="legacy_customers",
    target_df="new_customers",
    join_columns=["customer_id"]
)
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
    "match_rate": 0.92,
    "rows_matched": 920,
    "rows_in_source": 1000,
    "rows_in_target": 1000,
    "rows_only_in_source": 80,
    "rows_only_in_target": 80
  }
}
```

## 结果解读

### 结构差异 (schema)

| 字段 | 含义 |
|---|---|
| match | 结构是否完全一致 |
| common_columns | 共有的列 |
| source_only | 仅在源系统存在的列 |
| target_only | 仅在目标系统存在的列 |

### 数据差异 (data)

| 字段 | 含义 |
|---|---|
| match_rate | 数据匹配率（0-1） |
| rows_matched | 匹配的行数 |
| rows_only_in_source | 仅在源系统存在的行 |
| rows_only_in_target | 仅在目标系统存在的行 |

## 示例

**场景**: 用户说"比对 legacy_customers 和 new_customers"

**执行**:
```
[调用 mcp-compare-dfs]
参数:
- source_df: legacy_customers
- target_df: new_customers
- join_columns: ["customer_id"]
```

**返回**:
```
比对结果:

结构差异:
- 共有列: customer_id, name, email
- 源系统独有: phone
- 目标系统独有: mobile
- 结构匹配: ❌

数据差异:
- 匹配率: 92%
- 源系统行数: 1,000
- 目标系统行数: 1,000
- 匹配行数: 920
- 源系统独有: 80 行
- 目标系统独有: 80 行
```

## 前置条件

必须先使用 `mcp-dataset-ops.load_df` 加载两个 DataFrame：
```python
load_df("legacy_db", "legacy_customers", "SELECT * FROM customers")
load_df("new_db", "new_customers", "SELECT * FROM customers")
compare_dfs("legacy_customers", "new_customers", ["customer_id"])
```

## 最佳实践

1. **选择合适的 join_columns**: 使用能唯一标识记录的列（通常是主键）
2. **先检查结构**: 查看 schema 差异，了解列的变化
3. **关注 match_rate**: match_rate < 0.95 时需要深入调查
4. **记录差异**: 保存 rows_only_in_source/target 用于分析

## 后续操作

- 使用 `mcp-dataset-ops.release_df` 释放 DataFrame
- 根据差异结果决定是否需要修复数据