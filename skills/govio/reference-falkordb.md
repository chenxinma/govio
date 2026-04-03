# FalkorDBGraph 使用说明

## 简介

FalkorDBGraph 是用于连接和查询 FalkorDB 图数据库的 Python 类，提供以下核心功能：
- 连接远程 FalkorDB 图数据库
- 执行 Cypher 查询
- 自动构建图模式信息
- 返回查询结果

## 初始化连接

```python
from govio import FalkorDBGraph

g = FalkorDBGraph(graph='ontology', host='localhost', port=6379)
```

## 核心属性

### `schema` 属性
获取图的模式信息：
```python
print(g.schema)
```

### `query()` 方法
执行 Cypher 查询：
```python
data = g.query("MATCH (n) RETURN n LIMIT 10")
```

## 图结构

根据 `assets/schema.md`，图数据库包含以下节点和关系：

**节点类型**：
- `PhysicalTable`：物理表，包含 full_table_name, schema, table_name, name, data_entity_type, database_name
- `Col`：列/字段，包含 column, column_name, name, full_table_name, data_entity_type, dtype, size, precision, scale, order_no, data_type
- `Application`：应用，包含 app_id, name, app_name_en, app_type, business_domain, manager, network_area, maintenance_level, external_vendor
- `Standard`：数据标准，包含 standard_id, name, adaptability, alias, basis, core_system, data_category, data_expression, data_length, data_type, definition, name_en, source, standard_status, ref_code_define, business_rule

**关系类型**：
- `HAS_COLUMN`：表包含列 `(:PhysicalTable)-[:HAS_COLUMN]->(:Col)`
- `RELATES_TO`：表间关联 `(:PhysicalTable)-[:RELATES_TO]->(:PhysicalTable)`
- `USE`：应用使用表 `(:Application)-[:USE]->(:PhysicalTable)`

## 执行案例

### 元数据查询

#### 查询图的节点类型分布
```bash
uv run python -c "
from govio import FalkorDBGraph
g = FalkorDBGraph(graph='ontology', host='localhost', port=6379)
print(g.schema)
"
```

#### 查询所有标签/节点类型
```bash
uv run python -c "
from govio import FalkorDBGraph
g = FalkorDBGraph(graph='ontology', host='localhost', port=6379)
result = g.query('CALL db.labels()')
print(result)
"
```

#### 查询特定标签的属性
```bash
uv run python -c "
from govio import FalkorDBGraph
g = FalkorDBGraph(graph='ontology', host='localhost', port=6379)
result = g.query('MATCH (n:Application) WITH n LIMIT 1 UNWIND keys(n) AS k RETURN DISTINCT k')
print(result)
"
```

### 业务数据查询

#### 查询CRM应用使用的所有物理表
```bash
uv run python -c "
from govio import FalkorDBGraph
g = FalkorDBGraph(graph='ontology', host='localhost', port=6379)
cypher = '''
MATCH (app:Application {name: 'CRM'})-[:USE]->(t:PhysicalTable)
RETURN t.name AS table_name, t.full_table_name
'''
result = g.query(cypher)
"
```

#### 查询CUSTOMER表的所有字段
```bash
uv run python -c "
from govio import FalkorDBGraph
g = FalkorDBGraph(graph='ontology', host='localhost', port=6379)
cypher = '''
MATCH (t:PhysicalTable {name: 'CUSTOMER'})-[:HAS_COLUMN]->(c:Col)
RETURN c.name AS column_name, c.data_type, c.column
ORDER BY c.order_no
'''
result = g.query(cypher)
"
```

#### 查询两个表之间的关联关系
```bash
uv run python -c "
from govio import FalkorDBGraph
g = FalkorDBGraph(graph='ontology', host='localhost', port=6379)
cypher = '''
MATCH (t1:PhysicalTable)-[r:RELATES_TO]->(t2:PhysicalTable)
WHERE t1.name = 'CUSTOMER' AND t2.name = 'ORDER'
RETURN r.relationship_type AS rel_type, r.source_columns AS src_cols, r.target_columns AS tgt_cols
'''
result = g.query(cypher)
"
```

### 聚合查询

#### 统计每个应用使用的表数量
```bash
uv run python -c "
from govio import FalkorDBGraph
g = FalkorDBGraph(graph='ontology', host='localhost', port=6379)
cypher = '''
MATCH (app:Application)-[:USE]->(t:PhysicalTable)
RETURN app.name AS app_name, count(t) AS table_count
ORDER BY table_count DESC
'''
result = g.query(cypher)
"
```

#### 统计每种节点类型的数量
```bash
uv run python -c "
from govio import FalkorDBGraph
g = FalkorDBGraph(graph='ontology', host='localhost', port=6379)
cypher = '''
MATCH (n)
RETURN labels(n)[0] AS node_type, count(*) AS count
'''
result = g.query(cypher)
"
```

### 路径查询

#### 查询CRM到ERP之间的路径
```bash
uv run python -c "
from govio import FalkorDBGraph
g = FalkorDBGraph(graph='ontology', host='localhost', port=6379)
cypher = '''
MATCH path = shortestPath((crm:Application {name: 'CRM'})-[*]->(erp:Application {name: 'ERP'}))
RETURN path
'''
result = g.query(cypher)
"
```
