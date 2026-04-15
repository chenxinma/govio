---
name: govio
description: 数据治理知识图谱,当需要做"元数据查询、表字段比较、SQL 生成"等数据治理相关工作时运行。
allowed-tools: Read, Grep, Glob
---

# Data Governance Knowledge Graph

作为一名数据治理专家，根据知识图谱中的信息提供数据治理的支持。

## Purpose

**必须先读取 `assets/schema.md` 确认当前图结构。**

元数据查询：用户可以询问关于数据资产的元数据信息，如数据资产的名称、描述、来源、状态等。
表字段比较：用户可以比较不同表之间的字段差异，如字段名称、数据类型、是否必填等。
SQL 生成：用户可以根据需求描述，自动生成符合要求的 SQL 语句。

## Backend Selection

根据 `assets/backend.txt` 文件选择后端图库：

```bash
cat assets/backend.txt
# 输出: networkx 或 falkordb
```

**不同后端的查询方式不同**：

| 后端 | 查询语言 | 参考文档 |
|------|---------|---------|
| `networkx` | Python 代码，操作 `g` 对象 | [reference-networkx.md](reference-networkx.md) |
| `falkordb` | Cypher 查询 | [reference-falkordb.md](reference-falkordb.md) |

## Best Practices

1. 使用 Grep 查询`assets/names/node_names.md`获得被记载的标准名称
2. **必须先阅读 `assets/schema.md`** 了解当前图结构（节点、属性、关联关系），schema.md 内容会随数据变化
3. 根据配置的 backend 选择对应的查询方式
4. `uv run python -c ...` 脚本取数应该控制输出行数一次获取应小于300行
5. **注意：遵守有限读取原则，仅在必要时读取 schema.md**
6. **列名属性**：Col 节点使用 `column_name` 属性表示列名，不要直接使用 `name`

## Resource Resolution

When this skill is loaded, the base directory is provided:

```
govio/
├── SKILL.md                   # 技能定义
├── reference-networkx.md      # NetworkX 后端参考
├── reference-falkordb.md      # FalkorDB 后端参考
├── assets/                    # 资源文件
│   ├── backend.txt            # 后端配置 (networkx 或 falkordb)
│   ├── schema.md              # 图数据库模式文件
│   ├── ontology.gml           # 数据治理元模型数据文件(GML格式)
│   └── names/
│        └── node_names.md     # 已知节点的名称，作为标准名称备参考
└── scripts/
     └── query.py              # 统一查询入口 (自动根据 backend.txt 选择后端)
```

## Usage

### 直接使用 query.py（推荐）

query.py 会自动读取 `assets/backend.txt` 选择后端：

```bash
# NetworkX 后端：传入 Python 代码
uv run python skills/govio/scripts/query.py "print(g.schema)"

# FalkorDB 后端：传入 Cypher 查询
uv run python skills/govio/scripts/query.py "MATCH (n) RETURN n LIMIT 10"
```

### NetworkX 后端查询示例

Q: 查询CRM应用有几张表

```bash
uv run python -c "
from govio import NetworkXGraph
g = NetworkXGraph('assets/ontology.gml')

app_node = next((n for n, d in g.G.nodes(data=True) if d.get('name')=='CRM' and d.get('node_type')=='Application'), None)
if app_node:
    count = sum(
        1 for neighbor in g.G.successors(app_node)
        if g.G.get_edge_data(app_node, neighbor).get('edge_type') == 'USE'
        and g.G.nodes[neighbor].get('node_type') == 'PhysicalTable'
    )
    result = count
else:
    result = 0
"
```

### FalkorDB 后端查询示例

Q: 查询CRM应用有几张表

```bash
uv run python -c "
from govio import FalkorDBGraph
g = FalkorDBGraph(graph='ontology', host='localhost', port=6379)

cypher = '''
MATCH (app:Application {name: 'CRM'})-[:USE]->(t:PhysicalTable)
RETURN count(t) AS table_count
'''
result = g.query(cypher)
"
```

