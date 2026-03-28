"""
环境初始化, 获得层级结构的名称索引

生成的名称索引按应用存储在 `{name}_{app_name_en}.md` 的文件中
`{application_name}.md` 文件格式：

```markdown
# {full_table_name} {name}
- {column_name} {name}

```

---

输出样例：
assets
└── names
├── app1.md
├── app2.md
└── app3.md

app1.md
```markdown
# MDM.COLUMN_DEFINE 字段定义表
- ACC_YM 结算期间
- ADDRESS 地址
...
```
"""

import json
from pathlib import Path
from govio import NetworkXGraph
from networkx import Graph

ASSETS_DIR = Path(__file__).parent.parent / "assets"
ASSETS_NAMES_DIR = ASSETS_DIR / "names"


def generate_names(g: Graph):
    """节点名称索引文件
    格式：
    {"id":123, "name":"ABC", "node_type": "Application"}
    """

    if not ASSETS_DIR.exists():
        ASSETS_DIR.mkdir()
    if not ASSETS_NAMES_DIR.exists():
        ASSETS_NAMES_DIR.mkdir()

    # 遍历所有节点，查找Application类型的节点
    nodes = [ dict(id=node_id, name=g.nodes[node_id]['name'], node_type=g.nodes[node_id]['node_type']) \
              for node_id in g.nodes() \
                if g.nodes[node_id]['name'] \
                   and g.nodes[node_id]['name'] != "0" \
                   and isinstance(g.nodes[node_id]['name'], str)]
    
    if nodes and len(nodes) > 0:  # 只有在有内容时才写入文件
        file_path = ASSETS_NAMES_DIR / f"node_names.md"
        with open(file_path, "w", encoding="utf-8") as f:
            for node in nodes:
                f.write(json.dumps(node, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    # Use the provided graph file path relative to assets directory
    g = NetworkXGraph(ASSETS_DIR / "ontology.gml")
    generate_names(g.G)
