---
name: mcp-explore-relations
description: 探查 Govio MCP 中 DataFrame 之间的关系。当需要发现表之间的关联、分析数据血缘、或寻找潜在的外键关系时使用。基于列名相似度和值重叠率推断关系。需要指定要探查的 DataFrame 名称列表。
---

# MCP Explore Relations

探查多个 DataFrame 之间的潜在关系，包括外键关联和列名相似度。

## 使用场景

- 发现表之间的关联关系
- 分析数据血缘
- 验证外键关系
- 理解数据结构

## 参数

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| dataframes | list[string] | 是 | DataFrame 名称列表（至少 2 个） |

## 调用方式

```python
explore_df_relations(
    dataframes=["customers", "orders", "products"]
)
```

## 返回结果

```json
{
  "success": true,
  "relations": [
    {
      "type": "foreign_key",
      "source_table": "orders",
      "source_column": "customer_id",
      "target_table": "customers",
      "target_column": "customer_id",
      "confidence": 0.95
    },
    {
      "type": "foreign_key",
      "source_table": "orders",
      "source_column": "product_id",
      "target_table": "products",
      "target_column": "product_id",
      "confidence": 0.88
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

### foreign_key (外键关系)
- **检测方式**: 基于列名模式（*_id）和值重叠率
- **confidence**: 重叠值比例（0-1）
- **阈值**: confidence > 0.5 时报告

### column_similarity (列名相似)
- **检测方式**: 字符串相似度（SequenceMatcher）
- **similarity**: 相似度分数（0-1）
- **阈值**: similarity > 0.7 时报告

## 示例

**场景**: 用户说"分析 customers 和 orders 表之间的关系"

**执行**:
```
[调用 mcp-explore-relations]
参数:
- dataframes: ["customers", "orders"]
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

## 前置条件

必须先使用 `mcp-dataset-ops.load_df` 加载数据：
```python
load_df("prod_db", "customers", "SELECT * FROM customers")
load_df("prod_db", "orders", "SELECT * FROM orders")
explore_df_relations(["customers", "orders"])
```

## 后续操作

- 使用 `mcp-compare-dfs` 验证发现的关系
- 使用可视化工具生成关系图谱
