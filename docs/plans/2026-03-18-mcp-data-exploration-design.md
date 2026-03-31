# MCP 数据探查服务设计文档

## 概述

创建一个基于 Streamable HTTP 的 MCP 服务，提供数据表比对和数据表关系探查功能。服务基于 pandas + duckdb + networkx + datacompy 构建。

## 项目结构

```
mcp/
├── __init__.py
├── server.py              # MCP 服务入口
├── config.py              # 数据源配置加载
├── tools/
│   ├── __init__.py
│   ├── load_dataframe.py
│   ├── list_dataframes.py
│   ├── release_dataframe.py
│   ├── compare_tables.py
│   ├── explore_relations.py
│   └── visualize_relations.py
└── core/
    ├── __init__.py
    ├── database.py        # SQLAlchemy 连接管理
    ├── dataframe_store.py # DataFrame 内存存储
    ├── comparator.py      # 表比对核心逻辑
    ├── explorer.py        # 关系探查核心逻辑
    └── visualizer.py      # 图谱可视化
```

## MCP 工具定义

### 1. load_dataframe

执行 SQL 并将结果存入内存 DataFrame。

**输入参数：**
- `datasource`（string）：数据源名称
- `name`（string）：DataFrame 名称（用于后续引用）
- `sql`（string）：查询 SQL

**输出：**
- 加载结果（行数、列数、列信息）

### 2. list_dataframes

列出已加载的 DataFrame 清单。

**输入参数：** 无

**输出：**
- DataFrame 清单（名称、行数、列数、列信息）

### 3. release_dataframe

从内存中移除指定 DataFrame。

**输入参数：**
- `name`（string）：DataFrame 名称

**输出：**
- 释放结果

### 4. compare_tables

比对两个 DataFrame 的结构和约束。

**输入参数：**
- `source_df`（string）：源 DataFrame 名称
- `target_df`（string）：目标 DataFrame 名称

**输出：**
- 比对报告（结构差异、约束差异）

### 5. explore_relations

探查 DataFrame 之间的关系。

**输入参数：**
- `dataframes`（array of strings，可选）：DataFrame 名称列表，默认全部

**输出：**
- 发现的关系列表（基于列名相似性和数据内容推断）

### 6. visualize_relations

生成关系图谱。

**输入参数：**
- `relations`（array）：关系列表

**输出：**
- 关系图谱（JSON 格式，兼容 NetworkX）

## 数据源配置格式

配置文件通过命令行参数传入：

```bash
uv run mcp-server --datasource-config /path/to/datasources.json --port 8000
```

配置文件格式：

```json
{
  "datasources": {
    "mydb": {
      "driver": "mysql+pymysql",
      "host": "localhost",
      "port": 3306,
      "database": "mydb",
      "username": "user",
      "password": "pass"
    },
    "postgres": {
      "driver": "postgresql",
      "host": "localhost",
      "port": 5432,
      "database": "mydb",
      "username": "user",
      "password": "pass"
    }
  }
}
```

## 技术选型

- **MCP 协议版本**：2025-03-26（最新版本，支持 Streamable HTTP）
- **传输方式**：Streamable HTTP
- **数据处理**：pandas + duckdb
- **数据比对**：datacompy
- **图谱处理**：networkx
- **数据库连接**：SQLAlchemy

## 核心模块设计

### dataframe_store.py

管理内存中的 DataFrame 存储：

```python
class DataFrameStore:
    _instance = None
    _dataframes: dict[str, DataFrameInfo]
    
    def store(name: str, df: pd.DataFrame) -> DataFrameInfo
    def get(name: str) -> pd.DataFrame | None
    def list() -> list[DataFrameInfo]
    def release(name: str) -> bool
```

### database.py

管理数据库连接：

```python
class DatabaseManager:
    def __init__(config_path: str)
    def get_engine(datasource: str) -> Engine
    def execute_sql(datasource: str, sql: str) -> pd.DataFrame
```

### comparator.py

表比对逻辑：

```python
class TableComparator:
    def compare(source: pd.DataFrame, target: pd.DataFrame) -> CompareResult
    def compare_schema(source: pd.DataFrame, target: pd.DataFrame) -> SchemaDiff
    def compare_constraints(source: pd.DataFrame, target: pd.DataFrame) -> ConstraintDiff
```

### explorer.py

关系探查逻辑：

```python
class RelationExplorer:
    def explore(dataframes: dict[str, pd.DataFrame]) -> list[Relation]
    def find_column_similarity(df1: pd.DataFrame, df2: pd.DataFrame) -> list[ColumnMatch]
    def infer_foreign_key(df1: pd.DataFrame, df2: pd.DataFrame) -> list[ForeignKeyRelation]
```

### visualizer.py

图谱可视化：

```python
class RelationVisualizer:
    def visualize(relations: list[Relation]) -> dict
    def to_networkx(relations: list[Relation]) -> nx.Graph
    def to_json(graph: nx.Graph) -> dict
```

## 依赖

需要在 pyproject.toml 中添加：

```toml
dependencies = [
    # 现有依赖...
    "mcp>=1.0.0",
    "duckdb>=0.10.0",
    "datacompy>=0.10.0",
]
```

## 启动入口

在 pyproject.toml 中添加脚本入口：

```toml
[project.scripts]
mcp-server = "govio.mcp.server:main"
```