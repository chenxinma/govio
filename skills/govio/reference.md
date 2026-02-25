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

# 使用默认图文件 ontology.gml
g = NetworkXGraph('assets/ontology.gml')

# 或指定图文件路径
g = NetworkXGraph(graph="path/to/your/graph.gml")
```

## 核心属性

### `schema` 属性
获取图的模式信息，包含节点类型、边类型和边关系：
```python
print(g.schema)
```

### `G` 属性
获取原始 NetworkX 图对象，用于自定义分析：
```python
nx_graph = g.G
# 使用 NetworkX 的各种算法
nodes = nx_graph.nodes()
edges = nx_graph.edges()
```

## 使用建议

结合 NetworkX 丰富的图算法库进行分析：

```python
import networkx as nx
from govio import NetworkXGraph

g = NetworkXGraph('assets/ontology.gml')
# 计算节点度数
degrees = dict(g.G.degree())
# 查找最短路径
path = nx.shortest_path(g.G, source="node1", target="node2")
```

# 执行案例

## 元数据查询

### 查询图的节点类型分布

- 用户问题: 图中有哪些类型的节点？
- 思考: 使用 schema 属性获取节点类型信息，或直接遍历图节点。
    ```bash
    uv run python -c "from govio import NetworkXGraph
    g = NetworkXGraph('assets/ontology.gml')
    print(g.schema)
    "
    ```

### 查询特定类型节点的属性

- 用户问题: 查看 Application 类型节点有哪些属性？
- 思考: 遍历图节点，筛选 node_type 为 Application 的节点，收集其属性。
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

### 查询节点间关系

- 用户问题: 查找所有与 Application 相连的节点类型？
- 思考: 遍历边，找出与 Application 类型节点相连的其他节点类型。
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

### 按边类型查询关系

- 用户问题: 图中有哪些 USES 类型的关系？
- 思考: 遍历边，筛选 edge_type 为 USES 的边。
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
    for rel in uses_relations[:10]:  # 显示前10条
        print(rel)
    "
    ```
### 业务数据查询

- 用户问题: 查询CUSTOMER表结构
- 思考: 遍历节点获得CUSTOMER表的ID -> 根据CUSTOMER表的ID获得edge_type为HAS_COLUMN 的所有字段定义。
    ```bash
    grep -i "CUSTOMER" assets/names/node_names.md
    ```
    output: `{"id": "111", "name": "CUSTOMER", "node_type": "PhysicalTable"}`

    ```bash
    uv run python -c "
    from govio import NetworkXGraph
    g = NetworkXGraph('assets/ontology.gml')

    table_node = '111'  # CUSTOMER
    columns = []
    for u, v, edge_data in g.edges(data=True):
        # (PhysicalTable)-[HAS_COLUMN]->(Col)，所以u是表，v是列
        if u == table_node and edge_data.get('edge_type') == 'HAS_COLUMN':
            node_data = g.nodes[v]
            columns.append({
                'id': v,
                'name': node_data.get('name', ''),
                'data_type': node_data.get('data_type', ''),
                'column_name': node_data.get('column_name', '')
            })
    result = sorted(columns, key=lambda x: x['id'])
    ```
    output: 

## 图分析

### 计算节点重要性

- 用户问题: 哪些节点最重要（连接最多）？
- 思考: 使用 NetworkX 的度中心性算法。
    ```bash
    uv run python -c "from govio import NetworkXGraph
    import pandas as pd

    g = NetworkXGraph('assets/ontology.gml')
    degree_centrality = nx.degree_centrality(g.G)
    
    # 排序并取前10
    top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    for node, cent in top_nodes:
        node_type = g.G.nodes[node].get('node_type')
        print(f'{node} ({node_type}): {cent:.4f}')
    "
    ```

### 查找特定路径

- 用户问题: 找CRM和ERP出两个应用之间的关联？
- 思考: 使用 NetworkX 的最短路径算法。
    ```bash
    uv run python -c "from govio import NetworkXGraph
    import networkx as nx

    g = NetworkXGraph('assets/ontology.gml')
    
    # 假设知道节点名称
    source = 'CRM'
    target = 'ERP'

    # 找出所有 name 属性为 'CRM' 的节点 ID
    crm_nodes = [n for n, attr in G.nodes(data=True) if attr.get('node_type') == 'Application' and attr.get('name') == 'CRM']
    if len(crm_nodes) != 1:
        print('未能找到name=CRM的节点')
        return
    source_id = crm_nodes[0]

     找出所有 name 属性为 'ERP' 的节点 ID
    erp_nodes = [n for n, attr in G.nodes(data=True) if attr.get('node_type') == 'Application' and attr.get('name') == 'ERP']
    if len(erp_nodes) != 1:
        print('未能找到name=ERP的节点')
        return
    target_id = erp_nodes[0]
    
    try:
        path = nx.shortest_path(g.G, source=source_id, target=target_id)
        print(f'路径: { ' -> '.join(path) }')
    except nx.NetworkXNoPath:
        print('两节点间无路径')
    "
    ```
