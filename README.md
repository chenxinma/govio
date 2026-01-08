# Govio

✅ 核心构词：Gov（Governance，数据治理核心）+ io（Data IO，数据交互 / 数据流转，数据治理的核心载体）
✅ 深层内涵：以「数据治理」为核心内核，以「数据交互」为落地链路，一站式承载元数据管理、数据标准统一、数据质量校验全能力，赋能数据全生命周期的合规治理与高效流转。

基于 FalkorDB 的数据治理知识图谱项目，提供元数据查询、表字段比较、SQL 生成、数据标准推荐等数据治理支持功能。

## 功能特性

- **元数据查询**：查询数据资产的名称、描述、来源、状态等元数据信息
- **表字段比较**：比较不同表之间的字段差异（字段名称、数据类型、是否必填等）
- **SQL 生成**：根据需求描述自动生成符合要求的 SQL 语句
- **数据标准推荐**：基于协同过滤算法，为未贯标列推荐合适的数据标准
- **图数据库集成**：使用 FalkorDB 构建和管理知识图谱

## 技术栈

- **Python**: >= 3.13
- **图数据库**: FalkorDB >= 1.4.0
- **数据处理**: pandas >= 2.3.3, openpyxl >= 3.1.5, scikit-learn >= 1.8.0
- **数据库连接**: SQLAlchemy >= 2.0.45, PyMySQL >= 1.1.2
- **其他**: tqdm >= 4.67.1, python-dotenv

## 项目结构

```
govio/
├── pyproject.toml                     # 项目配置
├── README.md                          # 项目说明
├── data/                              # 数据文件
├── docs/                              # 文档
│   └── specs/
│       └── data_standard/
│           ├── data_standard_recommendation.md
│           └── recommender_usage.md
├── skills/
│   └── govio/
│       ├── SKILL.md                   # 技能定义
│       ├── reference.md               # 参考文档
│       ├── assets/                    # 资源文件
│       ├── logs/                      # 日志文件
│       └── scripts/                   # 脚本工具
├── src/
│   └── govio/
│       ├── __init__.py
│       ├── graph/
│       │   ├── __init__.py
│       │   └── falkordb_graph.py      # FalkorDB 图数据库封装
│       └── metadata/
│           ├── __init__.py
│           ├── application.py         # 应用信息加载
│           ├── database.py            # 数据库元数据加载
│           ├── standard.py            # 数据标准加载
│           ├── recommender.py         # 数据标准推荐器
│           ├── type.py                # 类型定义
│           └── utility.py             # 工具函数（包含命令行入口）
└── dist/                              # 构建输出目录
```

## 安装

```bash
# 安装 uv (如果尚未安装)
pip install uv

# 安装项目依赖
uv sync

# 安装开发依赖
uv sync --group dev
```

## 快速开始

### 命令行工具

项目提供了一个命令行工具来生成元数据 CSV 文件：

```bash
# 生成元数据 CSV（用于图数据库导入）
metadata --kundb "mysql+pymysql://user:pass@host/db" --app-list "path/to/app_list.xlsx" -o ./output

# 生成数据标准推荐（可选模式）
metadata --kundb "mysql+pymysql://user:pass@host/db" --app-list "path/to/app_list.xlsx" -m recommend -o ./output
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
from govio.metadata.standard import StandardLoader

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

# 加载数据标准
std_loader = StandardLoader(
    db="mysql+pymysql://user:pass@host/db",
    workspace_uuid="your-uuid"
)
standards = std_loader.Standard
std_compliance = std_loader.StdCompliance
```

### 使用数据标准推荐器

```python
from govio.metadata.recommender import create_recommender

# 加载数据
std_compliance = std_loader.StdCompliance  # 已贯标列

# 创建推荐器
WEIGHTS = {
    'table': 0.25,     # 表名权重（仅使用从 full_table_name 提取的 table_name）
    'name': 0.35,      # 列名权重
    'comment': 0.25,   # 列注释权重
    'type': 0.05,      # 数据类型权重
    'numeric': 0.10    # 数值特征权重
}
recommender = create_recommender(
    std_compliance=std_compliance,
    weights=WEIGHTS,
    k_neighbors=5,  # 使用5个最近邻
    top_n=3  # 返回Top 3推荐
)

# 批量推荐
db_loader = DatabseLoader(db, workspace_uuid, ["schema_name"])
all_columns = db_loader.Col  # 所有列
recommendations = recommender.batch_recommend(all_columns)
```

## 环境配置

创建 `.env` 文件以配置数据库连接：

```env
KUNDB_URL=mysql+pymysql://username:password@host:port/database_name
APP_LIST_FILE=path/to/app_list.xlsx
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