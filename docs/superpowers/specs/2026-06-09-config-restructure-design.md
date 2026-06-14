# Config.yaml 结构重构设计

## 目标

将 `~/.govio/config.yaml` 从扁平结构重构为三段式嵌套结构，分离元数据生成、图构建、查询分析三类配置。

## 当前结构

```yaml
kundb: "..."
app_list: "..."
app_map: "..."
relationship: "..."
metric: "..."
csv_dir: "..."
workspace_uuid: "..."
backend: "networkx"
networkx:
  gml_path: "..."
falkordb:
  host: "..."
  port: 6379
  graph: "..."
datasources:
  my_mysql:
    url: "..."
```

问题：三类配置（metadata source / graph backend / query datasources）混在同一层级，语义不清晰。

## 新结构

```yaml
metadata:
  kundb: "mysql+pymysql://user:pass@host/db"
  workspace_uuid: "82ee37374b314a938bf28170ab4db7cf"
  app_list: "/path/to/app_list.xlsx"
  app_map: "/path/to/app_map.json"
  relationship: "/path/to/relationships.json"   # optional
  metric: "/path/to/metrics.json"               # optional
  csv_dir: "/path/to/csv_output"

graph:
  backend: "networkx"                           # or "falkordb"
  networkx:
    gml_path: "skills/govio/assets/ontology.gml"
  falkordb:
    host: "localhost"
    port: 6379
    graph: "ontology"

datasources:                                     # optional
  my_mysql:
    url: "mysql+pymysql://user:pass@host/db"
    connect_args:
      ssl: true
      timeout: 30
  my_duckdb:
    url: "duckdb:///path/to/file.duckdb"
    connect_args: {}
```

## 字段归属

| Section | 字段 | 说明 |
|---|---|---|
| metadata | kundb | TDS 元数据库 SQLAlchemy URL |
| metadata | workspace_uuid | TDS 工作空间 UUID |
| metadata | app_list | 应用信息 Excel 文件路径 |
| metadata | app_map | schema→应用名映射 JSON 路径 |
| metadata | relationship | 表关系 JSON 路径 (optional) |
| metadata | metric | 指标定义 JSON 路径 (optional) |
| metadata | csv_dir | CSV 输出目录 |
| graph | backend | "networkx" 或 "falkordb" |
| graph | networkx.gml_path | GML 文件路径 |
| graph | falkordb.host | FalkorDB 主机 |
| graph | falkordb.port | FalkorDB 端口 |
| graph | falkordb.graph | FalkorDB 图名 |
| datasources | * | observe 命令使用的数据源 |

## 向后兼容：自动迁移

`ConfigManager.load()` 检测到旧格式（顶层存在 `kundb` 或 `backend` 字段）时，自动转换为新格式并保存。

迁移规则：
- `kundb`, `workspace_uuid`, `app_list`, `app_map`, `relationship`, `metric`, `csv_dir` → 移入 `metadata` section
- `backend`, `networkx`, `falkordb` → 移入 `graph` section
- `datasources` → 保持不变

迁移后保留旧文件备份为 `config.yaml.bak`。

## 影响范围

### config.py — ConfigManager

- `validate()`: 验证 `metadata` / `graph` / `datasources` 三个 section 各自的必填字段
- `load()`: 加载时检测旧格式并自动迁移
- `save()`: 按新结构序列化

### onboard.py

- `prompt_csv_config()`: 读写 `config["metadata"]`
- `prompt_backend_choice()` / `prompt_backend_*()`: 读写 `config["graph"]`
- `prompt_datasource_config()`: 读写 `config["datasources"]`（不变）

### query.py

- `config["backend"]` → `config["graph"]["backend"]`
- `config["networkx"]` → `config["graph"]["networkx"]`
- `config["falkordb"]` → `config["graph"]["falkordb"]`

### meta_export.py

- metadata 字段从 `config["metadata"]` 读取
- graph 字段从 `config["graph"]` 读取

### std_recommend.py

- `config["kundb"]` → `config["metadata"]["kundb"]`
- `config["workspace_uuid"]` → `config["metadata"]["workspace_uuid"]`
- `config["app_map"]` → `config["metadata"]["app_map"]`
- `config["csv_dir"]` → `config["metadata"]["csv_dir"]`

### graph_factory.py

- 从 `config["graph"]` 读取 backend 配置

### main.py

- `backend` 子命令：`config["backend"]` → `config["graph"]["backend"]`

### observe.py

- `config["datasources"]` 路径不变
