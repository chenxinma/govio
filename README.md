# Govio

✅ 核心构词：Gov（Governance，数据治理核心）+ io（Data IO，数据交互 / 数据流转，数据治理的核心载体）
✅ 深层内涵：以「数据治理」为核心内核，以「数据交互」为落地链路，一站式承载元数据管理、数据标准统一、数据质量校验全能力，赋能数据全生命周期的合规治理与高效流转。

基于 FalkorDB 的数据治理知识图谱项目，提供元数据查询、表字段比较、SQL 生成等数据治理支持功能。

## 功能特性

- **元数据查询**：查询数据资产的名称、描述、来源、状态等元数据信息
- **表字段比较**：比较不同表之间的字段差异（字段名称、数据类型、是否必填等）
- **SQL 生成**：根据需求描述自动生成符合要求的 SQL 语句
- **图数据库集成**：使用 FalkorDB 构建和管理知识图谱

## 技术栈

- **Python**: >= 3.13
- **图数据库**: FalkorDB >= 1.4.0
- **数据处理**: pandas >= 2.3.3, openpyxl >= 3.1.5
- **数据库连接**: SQLAlchemy >= 2.0.45, PyMySQL >= 1.1.2

## 项目结构

```
govio/
├── .agent/
│   └── skills/
│       └── data-gov-knowledge-graph/  # 数据治理知识图谱技能
│           ├── SKILL.md               # 技能定义
│           ├── reference.md           # 参考文档
│           ├── assets/                # 资源文件
│           │   ├── names/             # 标准名称
│           │   └── schema.md          # 图数据库结构
│           └── scripts/               # 脚本工具
│               ├── load_schema.py     # 加载模式
│               ├── load_names.py      # 加载名称
│               └── query.py           # 查询工具
├── src/govio/
│   ├── __init__.py
│   ├── graph/
│   │   └── falkordb_graph.py         # FalkorDB 图数据库封装
│   └── metadata/
│       ├── application.py             # 应用信息加载
│       ├── database.py                # 数据库元数据加载
│       ├── type.py                    # 类型定义
│       └── utility.py                 # 工具函数
├── tests/                             # 测试文件
├── pyproject.toml                     # 项目配置
└── README.md                          # 项目说明
```

## 安装

```bash
# 使用 uv 安装依赖
uv sync
```

## 快速开始

### 命令行工具

项目提供了两个命令行工具：

```bash
# 生成元数据 CSV
metadata-csv
```

### 使用图数据库

```python
from govio import FalkorDBGraph

# 连接到 FalkorDB
graph = FalkorDBGraph(graph="ontology", host='localhost', port=6379)

# 查看图模式
print(graph.schema)

# 执行 Cypher 查询
result = graph.query("MATCH (n) RETURN n LIMIT 10")
```

### 加载元数据

```python
from govio.metadata.application import AppInfoLoader
from govio.metadata.database import DatabseLoader

# 加载应用信息
app_loader = AppInfoLoader(app_list_file="path/to/app_list.xlsx")
apps = app_loader.Application

# 加载数据库元数据
db_loader = DatabseLoader(
    db="mysql+pymysql://user:pass@host/db",
    workspace_uuid="your-uuid",
    schema_limits=["schema1", "schema2"]
)
tables = db_loader.PhysicalTable
columns = db_loader.Col
```

## 使用数据治理技能

当需要进行元数据查询、表字段比较或 SQL 生成时，使用内置的数据治理知识图谱技能：

```bash
# 初始化知识图谱
uv run scripts/load_schema.py
uv run scripts/load_names.py

# 执行 Cypher 查询
uv run scripts/query.py --cypher "MATCH (n) RETURN n LIMIT 10"
```

## 开发

```bash
# 安装开发依赖
uv sync --group dev

# 运行测试
uv run -m unittest tests/
```

## 许可证

[Apache 2.0]