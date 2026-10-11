# LadybugGraph 使用说明

## 简介

LadybugGraph 用于连接和查询 Ladybug 嵌入式图数据库（本地 `.lbdb` 文件），提供以下核心功能：
- 打开本地 Ladybug 数据库文件
- 执行 Cypher 查询（与 FalkorDB 语法基本一致）
- 自动构建图模式信息
- 返回查询结果


## Cypher 方言坑（生成/手写查询必看）

1. **`column` 是保留字 + 存全限定名**（双重坑）：`column` 属性必须反引号，否则 Parser exception；
   且它的值是 `<datasource>.<schema>.<table>.<column>` 全限定名，按列名过滤请用 `column_name`：

​```cypher
MATCH (c:Col) WHERE c.`column` = 'ITS.ITS_USER.T_INVOICE.INV_NO' RETURN c.column_name  // ✓
MATCH (c:Col) WHERE c.column_name = 'INV_NO' RETURN c.name                               // ✓
MATCH (c:Col) WHERE c.column = 'INV_NO' RETURN c.name                                    // ✗ 两重都错
​```

2. **`WITH ... ORDER BY` 必须跟 `SKIP`/`LIMIT`**，否则 Binder exception：

​```cypher
MATCH (t:PhysicalTable)-[:HAS_COLUMN]->(c:Col)
WITH t, c ORDER BY c.order_no LIMIT 10     // ✓ 补 LIMIT
​```

3. **无窗口函数**：`sum(cnt) OVER ()`、`count(*) OVER ()` 解析报错。需要"每组 + 总计"时：

​```cypher
MATCH (d:Datasource) WITH collect(d) AS ds UNWIND ds AS d
RETURN d.datasource_name, size(ds) AS total   // ✓ collect/size 技巧
​```

4. **`round(x, n)` 必须带小数位参数**：`round(x*100)/100.0` 这类手写四舍五入会报类型签名错误，
   直接返回原值并由展示层取整，或用 `round(x, 2)`。

5. **聚合后变量出作用域**：`RETURN t.schema AS s, count(t) AS n` 之后不能再写 `ORDER BY t.table_name`，
   要排序的表达式必须进分组键，或对别名/聚合结果排序。

6. **没有 `toFloat()`**（`Catalog exception: function TOFLOAT does not exist`）；要浮点除法写
   `x * 1.0 / y`，整数除法会截断。

7. **变量/别名不能用保留字**：`WITH collect(d) AS all` 会 Parser exception，用 `all_nodes`、`ds_list`
   这类非保留名。

## 执行案例

### 元数据查询

#### 查询所有应用

​```cypher
MATCH (app:Application)
RETURN app.name, app.app_name_en, app.business_domain
LIMIT 300
​```

#### 查询某应用使用的所有物理表
​```cypher
MATCH (app:Application {name: 'AEP'})-[:USE]->(t:PhysicalTable)
RETURN t.name AS table_name, t.full_table_name
LIMIT 300
​```

#### 查询某表的所有字段
​```cypher
MATCH (t:PhysicalTable {name: 'T_INVOICE'})-[:HAS_COLUMN]->(c:Col)
RETURN c.column_name, c.data_type
ORDER BY c.order_no
LIMIT 300
​```

### 聚合查询

#### 统计每个应用使用的表数量
​```cypher
MATCH (app:Application)-[:USE]->(t:PhysicalTable)
RETURN app.name AS app_name, count(t) AS table_count
ORDER BY table_count DESC
LIMIT 300
​```

#### 统计每种节点类型的数量
​```cypher
MATCH (n)
RETURN label(n) AS node_type, count(n) AS count
LIMIT 300
​```

## 通过 CLI 查询

```bash
# 查看当前后端
govio-cli backend

# 执行 Cypher 查询（必须以 MATCH 开头）
govio-cli query -c 'MATCH (app:Application) RETURN app.name, app.app_name_en LIMIT 300'
```
