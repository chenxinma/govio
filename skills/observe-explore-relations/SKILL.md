---
name: observe-explore-relations
description: 探查 DataFrame 之间的关系。当需要发现表之间的关联、分析数据血缘、或寻找潜在的外键关系时使用。通过 govio-cli observe explore 命令执行，基于列名相似度和值重叠率推断关系。
---

# Observe Explore Relations

探查多个 DataFrame 之间的潜在关系，包括外键关联和列名相似度。

## CLI 命令

```bash
govio-cli observe explore [dataframe ...]
```

## 参数

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| dataframes | positional | 否 | DataFrame 名称列表。省略时探查所有已加载的 DataFrame |

## 使用场景

- 发现表之间的关联关系
- 分析数据血缘
- 验证外键关系
- 理解数据结构

## 前置条件

必须先使用 `govio-cli observe load` 加载数据：

```bash
govio-cli observe load customers prod_db "SELECT * FROM customers"
govio-cli observe load orders prod_db "SELECT * FROM orders"
```

## 调用方式

探查指定的 DataFrame：

```bash
govio-cli observe explore customers orders products
```

探查所有已加载的 DataFrame（省略参数）：

```bash
govio-cli observe explore
```

## 返回结果

```json
{
  "success": true,
  "relations": [
    {
      "source_table": "orders",
      "source_column": "customer_id",
      "target_table": "customers",
      "target_column": "customer_id",
      "confidence": 0.95
    },
    {
      "type": "column_similarity",
      "table1": "customers",
      "column1": "email",
      "table2": "orders",
      "column2": "customer_email",
      "similarity": 0.85
    }
  ]
}
```

## 关系类型

### 外键关系 (foreign_key)

- **检测方式**: 基于列名模式（`*_id`, `*id`）和值重叠率
- **confidence**: 重叠值比例（0-1）
- **阈值**: confidence > 0.5 时报告
- **大小写不敏感**: `Customer_ID` 与 `customer_id` 视为匹配

### 列名相似 (column_similarity)

- **检测方式**: 字符串相似度（SequenceMatcher）
- **similarity**: 相似度分数（0-1）
- **阈值**: similarity > 0.7 时报告

## 示例

**场景**: 用户说"分析 customers 和 orders 表之间的关系"

**执行**:
```bash
govio-cli observe explore customers orders
```

**返回**:
```
发现 2 个关系:

1. [外键] orders.customer_id → customers.customer_id
   置信度: 95%
   说明: orders 表中的 customer_id 与 customers 表匹配

2. [相似列] customers.email ~ orders.customer_email
   相似度: 85%
   说明: 列名相似，可能存储相同含义的数据
```

## 可视化关系图谱

探查结果可通过 `govio-cli observe visualize-relations` 生成图谱数据：

```bash
# 先探查关系，保存 JSON
govio-cli observe explore customers orders > relations.json

# 生成图谱数据
govio-cli observe visualize-relations "$(cat relations.json)"
```

返回结果包含 `nodes` 和 `edges`，可用于关系图谱渲染。

## 最佳实践

1. **先加载数据**: 确保要探查的 DataFrame 已通过 load 命令加载
2. **指定范围**: 明确指定要探查的 DataFrame 名称，避免不必要的计算
3. **检查 confidence**: confidence > 0.8 的关系可靠性较高
4. **结合可视化**: 使用 visualize-relations 生成图谱便于分析

## 后续操作

- 使用 `observe-compare-dfs` 验证发现的关系
- 使用 `govio-cli observe visualize-relations` 生成关系图谱

## 与其他 Skill 的协作

| 前置操作 | 关联 Skill |
|---|---|
| 加载数据 | `observe-dataset-ops` |
| 比对验证 | `observe-compare-dfs` |
| 主控协调 | `govio-observe` |
