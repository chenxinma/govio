---
name: govio
description: 数据治理知识图谱,当需要做“元数据查询、表字段比较、SQL 生成”等数据治理相关工作时运行。
allowed-tools: Read, Grep, Glob
---

# Data Governance Knowledge Graph

作为一名数据治理专家，根据知识图谱中的信息提供数据治理的支持。

## Purpose

元数据查询：用户可以询问关于数据资产的元数据信息，如数据资产的名称、描述、来源、状态等。
表字段比较：用户可以比较不同表之间的字段差异，如字段名称、数据类型、是否必填等。
SQL 生成：用户可以根据需求描述，自动生成符合要求的 SQL 语句。

## Best Practices

For advanced usage, see [reference.md](reference.md).

1. 使用 Grep 查询`assets/names/node_names.md`获得被记载的标准名称
2. 严格按照 `assets/schema.md` 的NetworkX Graph结构生成`uv run python -c ...`脚本，将结果以json(ensure_ascii=False)形式输出到控制台。**注意：遵守有限读取原则，仅在必要时读取 schema.md**
3. `uv run python -c ...`脚本取数应该控制输出行数一次获取应小于300行

## Resource Resolution

When this skill is loaded, the base directory is provided:

govio/
    ├── SKILL.md                   # 技能定义
    ├── reference.md               # 参考文档
    ├── assets/                    # 资源文件
    │       └── schema.md          # 图数据库模式文件
    └── scripts/                   # 脚本工具
         ├── load_names.py         # 加载标准名称脚本工具
         └── load_schema.py        # 加载图数据库模式脚本工具

## Usage

- **coding directly**: 
Q: 查询CRM应用有几张表

```bash
uv run python -c "
from govio import NetworkXGraph
g = NetworkXGraph()

app_nids = [ nid for nid in list(g.G.nodes()) \
                if g.G.nodes[nid]["node_type"]=="Application" \
                    and g.G.nodes[nid]["name"] == "CRM" ]
if len(app_nids) != 1:
    print("没有找到名为CRM的应用")
crm_id = app_nids[0]

tables = [ g.G.nodes[neighbor_id] for neighbor_id in g.G.neighbors(crm_id) \
           if g.G[crm_id][neighbor_id]["edge_type"] == "USE" ]
print(tables)
"
```
执行以上代码并获得反馈
