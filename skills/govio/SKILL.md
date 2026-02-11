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

1. 使用 Read, Grep, Glob 查询`assets/names/`获得被记载的标准名称
2. 严格按照 `assets/schema.md` 的图数据库结构生成`cypher`脚本。**注意：遵守有限读取原则，仅在必要时读取 schema.md**
3. `cypher`脚本取数应该控制输出行数一次获取应小于300行

## Resource Resolution

When this skill is loaded, the base directory is provided:

govio/
    ├── SKILL.md                   # 技能定义
    ├── reference.md               # 参考文档
    ├── assets/                    # 资源文件
    │       ├── names/             # 标准名称目录
    │       │       ├── app1.md    # 应用1的标准名称列表
    │       │       ├── app2.md    # 应用2标准的标准名称列表
    │       │       └── ...        # ... 其他应用标准名称列表
    │       └── schema.md          # 图数据库模式文件
    └── scripts/                   # 脚本工具
         ├── query.py              # Cypher 查询脚本工具
         ├── load_names.py         # 加载标准名称脚本工具
         └── load_schema.py        # 加载图数据库模式脚本工具

## Usage

- **coding directly**: `uv run python -c "from govio import FalkorDBGraph; g = FalkorDBGraph(); results = g.query(\"MATCH ...\"); ..."`
- **query cypher**: `uv run scripts/query.py --cypher "MATCH ..."`
