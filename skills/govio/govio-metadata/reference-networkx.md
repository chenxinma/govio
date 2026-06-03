# NetworkXGraph 使用说明

## 简介

NetworkXGraph 是用于加载和分析本地 GML 格式图数据的 Python 类，提供以下核心功能：
- 从 GML 文件加载图数据
- 自动构建图模式信息
- 节点类型和边关系发现
- 直接访问 NetworkX 图对象进行自定义分析

## 初始化连接

```python
from govio import NetworkXGraph

g = NetworkXGraph('assets/ontology.gml')
```

## 核心属性

### `schema` 属性
获取图的模式信息：
```python
print(g.schema)
```

### `G` 属性
获取原始 NetworkX 图对象：
```python
nx_graph = g.G
nodes = nx_graph.nodes()
edges = nx_graph.edges()
```

## 使用建议

结合 NetworkX 丰富的图算法库进行分析：

```python
import networkx as nx
from govio import NetworkXGraph

g = NetworkXGraph('assets/ontology.gml')
degrees = dict(g.G.degree())
path = nx.shortest_path(g.G, source="node1", target="node2")
```

## 执行案例

### 元数据查询

#### 查询图的节点类型分布
```bash
uv run python -c "from govio import NetworkXGraph
g = NetworkXGraph('assets/ontology.gml')
print(g.schema)
"
```

#### 查询特定类型节点的属性
```bash 
uv run python -c "from govio import NetworkXGraph
g = NetworkXGraph('assets/ontology.gml')
app_attrs = set()
for node, data in g.G.nodes(data=True):
    if data.get('node_type') == 'Application':
        app_attrs.update(data.keys())
print(sorted(app_attrs))
"
```

#### 查询节点间关系
```bash
uv run python -c "from govio import NetworkXGraph
g = NetworkXGraph('assets/ontology.gml')
related_types = set()
for u, v in g.G.edges():
    u_type = g.G.nodes[u].get('node_type')
    v_type = g.G.nodes[v].get('node_type')
    if u_type == 'Application':
        related_types.add(v_type)
    elif v_type == 'Application':
        related_types.add(u_type)
print(related_types)
"
```

#### 按边类型查询关系
```bash
uv run python -c "from govio import NetworkXGraph
g = NetworkXGraph('assets/ontology.gml')
uses_relations = []
for u, v, data in g.G.edges(data=True):
    if data.get('edge_type') == 'USES':
        u_type = g.G.nodes[u].get('node_type')
        v_type = g.G.nodes[v].get('node_type')
        uses_relations.append({
            'source': u,
            'source_type': u_type,
            'target': v,
            'target_type': v_type
        })
for rel in uses_relations[:10]:
    print(rel)
"
```

### 业务数据查询

#### 查询CUSTOMER表结构
```bash
grep -i "CUSTOMER" assets/names/node_names.md
```
output: `{"id": "111", "name": "CUSTOMER", "node_type": "PhysicalTable"}`

```bash
uv run python -c "
from govio import NetworkXGraph
g = NetworkXGraph('assets/ontology.gml')

table_node = '111'
columns = []
for u, v, edge_data in g.G.edges(data=True):
    if u == table_node and edge_data.get('edge_type') == 'HAS_COLUMN':
        node_data = g.G.nodes[v]
        columns.append({
            'id': v,
            'name': node_data.get('name', ''),
            'data_type': node_data.get('data_type', ''),
            'column_name': node_data.get('column_name', '')
        })
result = sorted(columns, key=lambda x: x['id'])
"
```

## 图分析

### 计算节点重要性
```bash
uv run python -c "from govio import NetworkXGraph
import pandas as pd

g = NetworkXGraph('assets/ontology.gml')
degree_centrality = nx.degree_centrality(g.G)

top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
for node, cent in top_nodes:
    node_type = g.G.nodes[node].get('node_type')
    print(f'{node} ({node_type}): {cent:.4f}')
"
```

### 查找特定路径
```bash
uv run python -c "from govio import NetworkXGraph
import networkx as nx

g = NetworkXGraph('assets/ontology.gml')

source = 'CRM'
target = 'ERP'

crm_nodes = [n for n, attr in g.G.nodes(data=True) if attr.get('node_type') == 'Application' and attr.get('name') == 'CRM']
if len(crm_nodes) != 1:
    print('未能找到name=CRM的节点')
    exit()
source_id = crm_nodes[0]

erp_nodes = [n for n, attr in g.G.nodes(data=True) if attr.get('node_type') == 'Application' and attr.get('name') == 'ERP']
if len(erp_nodes) != 1:
    print('未能找到name=ERP的节点')
    exit()
target_id = erp_nodes[0]

try:
    path = nx.shortest_path(g.G, source=source_id, target=target_id)
    print(f'路径: {\" -> \".join(path)}')
except nx.NetworkXNoPath:
    print('两节点间无路径')
"
```
