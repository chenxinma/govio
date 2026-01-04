# FalkorDBGraph 使用说明

## 简介

FalkorDBGraph 是用于与 FalkorDB 图数据库交互的 Python 类，提供以下核心功能：
- 图数据库连接管理
- 模式信息动态加载
- Cypher 查询执行
- 元数据发现（节点标签、属性、关系类型）


## 初始化连接

检查`assets/schema.md`是否存在，如果不存在就执行`uv run scripts/load_graph.py`初始化。

```python
from ontology import FalkorDBGraph

# 创建连接实例
db_graph = FalkorDBGraph()
```

## 核心方法

### `query(query: str, params: dict = {})`
执行 Cypher 查询
```python
results = db_graph.query("MATCH (n:Label) RETURN n LIMIT 5")
```

## 错误处理

查询失败时抛出 ValueError，包含原始错误信息：
```python
try:
    db_graph.query("INVALID CYPHER")
except ValueError as e:
    print(f"查询失败: {e}")
```

## 使用建议

使用参数化查询防止 Cypher 注入：
```python
db_graph.query("MATCH (n:User {id: $id}) RETURN n", {"id": 123})
```
# 执行案例

## 元数据查询

### 特定数据表的元数据查询

- 用户问题: 供应商相关的表有哪些？
- 思考: 查询供应商相关的节点，再根据相关节点获得表的信息。
- 用 Grep, Glob 查询 `assets/names/*.md` 获得供应商相关的实体在图数据库中的标准名称  
- 根据`assets/schema.md`编写cypher执行查询
    ```python
    from ontology import FalkorDBGraph

    g = FalkorDBGraph()
    cypher = """
    MATCH (a:Application)-[:USES]->(t:PhysicalTable)
    WHERE 
        t.name IN ['supplier', 'supplier_contact']
        or a.name = '供应商管理系统'
    RETURN t.full_table_name AS table
    """
    print(g.query(cypher)) # 数据量较小时直接输出
    ```
- 获得的输出结果
    [{'table': 'supplier'}, {'table': 'supplier_contact'}, {'table': '供应商管理系统'}]

### 特定数据表的字段差异比较

- 用户问题: 比较CRM应用的customer表与ERP应用中的customer_info表的字段差异。
- 思考: 先分别拿到CRM应用的customer表和ERP应用中的customer_info表的字段。
- 用 Grep 查询 `assets/names/CRM.md` 中customer表标准名称 和 `assets/names/ERP.md`中customer_info表标准名称
- 根据`assets/schema.md`编写cypher执行查询
    ```python
    from ontology import FalkorDBGraph
    import pandas as pd

    g = FalkorDBGraph()
    cypher = """
    MATCH (a:Application)-[:USES]->(t:PhysicalTable)-[:HAS_COLUMN]->(c:Col)
    WHERE a.name IN ['CRM','ERP']
        and t.name IN ['customer','customer_info']
    RETURN a.name AS application,
        c.column_name AS column_name,
        c.name AS description,
        c.data_type AS data_type,
        c.size AS size,
        c.precision AS precision,
        c.scale AS scale
    ORDER BY application, c.order_no
    """
    df = pd.DataFrame(g.query(cypher)) 
    df.to_csv("output.csv", index=False) # 数据量较大时存储为文件输出
    print(df.info())
    ```
- 读取输出结果output.csv

### 字段数据类型查询

- 用户问题: 哪种类型的更新时间字段占多数？
- 思考: 为了回答这个问题，我们需要找到所有包含'更新时间','update_time', 或 'updated_at'关键字的列，并按它们的数据类型进行分组计数。
- 根据`assets/schema.md`编写cypher执行查询
    ```python
    from ontology import FalkorDBGraph
    import pandas as pd

    g = FalkorDBGraph()
    cypher = """
    MATCH (c:Column) WHERE c.name =~ '.*更新时间|update_time|updated_at.*' 
    WITH c, c.data_type AS type 
    RETURN type, count(*) 
    ORDER BY count(*) DESC
    """
    df = pd.DataFrame(g.query(cypher)) 
    df.to_csv("output.csv", index=False) # 数据量较大时存储为文件输出
    print(df.info())
    ```
- 读取输出结果output.csv

## SQL生成

### 按科目、客户及财务期间汇总余额

- 用户问题: 请帮我写一个SQL查询，用于从会计引擎系统中的金蝶分录表里按科目、客户及财务期间汇总余额。
- 思考: 首先使用relevant_nodes工具确定与'会计引擎'和'金蝶分录'相关的物理表；接着通过datagov_query工具获取该表的完整列信息；基于此信息并结合业务常识构造出符合要求的SQL查询。
- 根据`assets/schema.md`编写cypher执行查询
    ```python
    from ontology import FalkorDBGraph
    import pandas as pd

    g = FalkorDBGraph()
    cypher = """
    MATCH (t:PhysicalTable) 
    WHERE t.name = 'AEP_USER.AEP_EAS_ENTRY_LINE' 
    RETURN t.full_table_name AS table, t.columns
    """
    print(g.query(cypher)) 
    ```
- 获得的输出结果生成SQL
