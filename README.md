# Govio

✅ 核心构词：Gov（Governance，数据治理核心）+ io（Data IO，数据交互 / 数据流转，数据治理的核心载体）
✅ 深层内涵：以「数据治理」为核心内核，以「数据交互」为落地链路，一站式承载元数据管理、数据标准统一、数据质量校验全能力，赋能数据全生命周期的合规治理与高效流转。

基于 NetworkX 的数据治理知识图谱项目，提供元数据查询、表字段比较、SQL 生成、数据标准推荐等数据治理支持功能。

## 功能特性

- **元数据查询**：查询数据资产的名称、描述、来源、状态等元数据信息
- **表字段比较**：比较不同表之间的字段差异（字段名称、数据类型、是否必填等）
- **SQL 生成**：根据需求描述自动生成符合要求的 SQL 语句
- **数据标准推荐**：基于协同过滤算法，为未贯标列推荐合适的数据标准
- **图数据库集成**：使用 NetworkX 构建本地知识图谱，通过 Python 脚本执行查询

## 安装

```bash
# 安装 uv (如果尚未安装)
https://docs.astral.sh/uv/getting-started/installation/

# 安装项目依赖
uv sync

# 安装开发依赖
uv sync --group dev
```

## 从元数据到知识图谱

Govio 将企业元数据（图结构）转化为知识图谱，支持两种图数据库后端：

```
┌─────────────────┐                        ┌─────────────────┐
│   企业数据库      │                        │   CSV 文件       │
│ (KunDB/MySQL)   │                        │ (节点 + 边)     │
└────────┬────────┘                        └────────┬────────┘
         │                                          │
         │   govio-cli onboard                     │
         │   (交互式引导配置)                         │
         ▼                                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    onboard 向导                               │
│  • 配置元数据连接 (KunDB, app_list, app_map, relationship)  │
│  • 选择后端 (NetworkX / FalkorDB)                           │
│  • 生成 GML 或连接 FalkorDB                                 │
│  • 生成 schema.md 和 names/                                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                ┌─────────────────┐                        ┌─────────────────┐
                │   NetworkX       │                        │   FalkorDB      │
                │   GML 文件        │                        │   图数据库       │
                └────────┬─────────┘                        └────────┬────────┘
                         │                                          │
                         ▼                                          ▼
                ┌─────────────────────────────────────────────────────────┐
                │                 govio-cli std-recommend                 │
                │  • 批量推荐未贯标列的数据标准                             │
                │  • 输出 COMPLIES_WITH.csv                               │
                └─────────────────────────────────────────────────────────┘
```

## 快速开始

Govio 提供交互式 CLI 工具 `govio-cli`，支持两个主命令：

### 第一步：运行 Onboard 向导初始化

```bash
govio-cli onboard
```

向导会引导你完成以下步骤：

1. **配置元数据连接**
   - 输入元数据库 URL (KunDB)
   - 输入应用清单 Excel 文件路径
   - 输入应用数据库映射 JSON 文件路径
   - 输入表关系 JSON 文件路径（可选）
   - 输入 CSV 输出目录
   - 输入工作区 UUID

2. **选择图数据库后端**
   - NetworkX: 使用本地 GML 文件
   - FalkorDB: 连接 FalkorDB 图数据库

3. **NetworkX 模式**
   - 选择是否从 CSV 文件生成新的 GML 文件
   - 如果选择生成，输入 CSV 目录路径
   - 如果不生成，输入已有的 GML 文件路径

4. **FalkorDB 模式**
   - 输入 FalkorDB 连接信息（host, port, graph name）

5. **自动生成 Assets**
   - 配置文件保存到 `~/.govio/config.yaml`
   - Assets 生成到 `skills/govio/assets/`
      - `schema.md`: 图结构定义
      - `names/`: 节点名称索引

### 第二步：运行数据标准推荐

```bash
govio-cli std-recommend
```

推荐器会基于已有配置和 CSV 文件，为未贯标列推荐合适的数据标准，结果输出到配置的 `output_dir`。

### 查询工具 (Skills)

配置完成后，可使用 Skills 工具集进行图数据库查询。配置从 `~/.govio/config.yaml` 自动读取。

**目录结构：**

```
skills/govio/
├── SKILL.md              # 技能定义（AI Agent 使用）
├── reference.md          # 参考文档
├── assets/               # 资源文件
│   ├── schema.md         # 图数据库模式文件
│   ├── ontology.gml      # NetworkX GML 数据文件
│   └── names/
│       └── node_names.md # 节点名称索引
└── scripts/
    ├── query.py          # 统一查询入口（自动根据配置选择后端）
    ├── load_names.py     # 加载标准名称
    └── load_schema.py    # 加载图模式
```

**查询示例：**

```bash
# FalkorDB 模式（根据 config.yaml 自动选择）
uv run python skills/govio/scripts/query.py "MATCH (n:PhysicalTable) RETURN n.name LIMIT 5"

# 支持 stdin 输入
echo "MATCH (n:Application) RETURN n.name" | uv run python skills/govio/scripts/query.py

# NetworkX 模式（查询代码作为 Python 代码执行，结果赋给 result 变量）
uv run python skills/govio/scripts/query.py "result = [n for n, d in g.nodes(data=True) if d.get('node_type') == 'PhysicalTable'][:5]"
```

**查询规则：**

- FalkorDB：使用 Cypher 查询语言，必须以 `MATCH` 开头
- NetworkX：使用 Python 代码操作 `g`（NetworkX 图对象），结果赋值给 `result`

**AI Agent 使用：**

当 AI Agent 加载 `SKILL.md` 后，可直接使用自然语言查询，例如：
- "查询 CRM 应用有几张表"
- "查找所有包含 '用户' 的表名"

### CSV 文件格式要求

**节点文件：**
- `PhysicalTable.csv`: 物理表节点
- `Col.csv`: 字段节点
- `Application.csv`: 应用节点
- `Standard.csv`: 数据标准节点

**边文件：**
- `HAS_COLUMN.csv`: 表包含字段的关系
- `USE.csv`: 应用使用表的关系
- `COMPLIES_WITH.csv`: 字段贯标的关系

**CSV 格式示例：**

```csv
# PhysicalTable.csv
:ID(PhysicalTable),name,full_table_name
table1,用户表,DB.SCHEMA.TABLE1

# Col.csv
:ID(Col),name,column_name,full_table_name
col1,用户ID,USER_ID,DB.SCHEMA.TABLE1

# HAS_COLUMN.csv
:START_ID(PhysicalTable),:END_ID(Col)
table1,col1
```

### 使用图数据库

**NetworkX 模式：**

```python
from govio import NetworkXGraph

# 加载 NetworkX 图
graph = NetworkXGraph(graph="./output/ontology.gml")

# 查看图模式
print(graph.schema)

# 使用 Python 进行图查询
nodes = list(graph.G.nodes(data=True))
cols = [n for n, data in graph.G.nodes(data=True) if data.get("node_type") == "Col"]

for u, v, data in graph.G.edges(data=True):
    print(f"{u} --[{data.get('edge_type')}]--> {v}")
```

**FalkorDB 模式：**

```python
from govio import FalkorDBGraph

# 连接 FalkorDB 图数据库
graph = FalkorDBGraph(host="localhost", port=6379, graph="ontology")

# 查看图模式
print(graph.schema)

# 使用 Cypher 查询
result = graph.query("MATCH (n:Application) RETURN n.name LIMIT 10")
```

### 加载元数据

```python
from govio.metadata.application import AppInfoLoader
from govio.metadata.database import DatabaseLoader
from govio.metadata.standard import StandardLoader

# 加载应用信息
app_loader = AppInfoLoader(app_list_file="path/to/app_list.xlsx")
apps = app_loader.Application

# 加载数据库元数据
db_loader = DatabaseLoader(
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