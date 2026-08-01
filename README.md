# Govio

✅ 核心构词：Gov（Governance，数据治理核心）+ io（Data IO，数据交互 / 数据流转，数据治理的核心载体）
✅ 深层内涵：以「数据治理」为核心内核，以「数据交互」为落地链路，一站式承载元数据管理、数据标准统一、数据质量校验全能力，赋能数据全生命周期的合规治理与高效流转。

数据治理知识图谱平台，提供元数据查询、表字段比较、指标 SQL 组装、数据标准推荐、指标血缘分析、数据探查比对与 EDA 分析等数据治理支持功能。

## 功能特性

- **元数据查询**：查询数据资产的名称、描述、来源、状态等元数据信息
- **表字段比较**：比较不同表之间的字段差异（字段名称、数据类型、是否必填等）
- **指标 SQL 组装**：根据指标 JSON 规格自动组装查询 SQL 语句
- **数据标准推荐**：基于协同过滤算法，为未贯标列推荐合适的数据标准
- **指标管理**：将业务指标纳入知识图谱，支持指标血缘溯源、影响分析、数据溯源和维度发现
- **数据探查比对**：加载数据源数据、探索表间关系、比对数据差异、生成可视化图表
- **EDA 分析**：标准 4 阶段探索性数据分析流程（画像 → 推断关联 → 核查关联 → 一致性检查）
- **多图数据库后端**：支持 NetworkX（本地 GML）、FalkorDB（Redis 图数据库）、Ladybug（嵌入式 `.lbdb`）三种后端
- **多数据源接入**：元数据可从 TDS（KunDB/MySQL 元数据库）、DuckDB 读取，或两者合并

## 安装

```bash
# 安装 uv (如果尚未安装)
# https://docs.astral.sh/uv/getting-started/installation/

# 构建 wheel 并一键安装 govio-cli（持久化工具）
uv build
./start.sh
```

`start.sh` 内部执行 `uv tool install --from "$WHL" govio --force`，将 `govio-cli` 安装为持久化命令行工具。

**开发模式：**

```bash
# 安装项目依赖
uv sync

# 安装开发依赖（含 falkordb-bulk-loader、pytest、ruff）
uv sync --group dev

# 运行测试
uv run pytest tests/
```

## 从元数据到知识图谱

Govio 将企业元数据转化为知识图谱，支持三种图数据库后端与多种元数据源：

```
┌───────────────────────────────────┐    ┌─────────────────────────────┐
│  元数据源                          │    │  数据源 (observe 用)          │
│  TDS(KunDB/MySQL) / DuckDB / Both │    │  MySQL / DuckDB ...          │
└────────────────┬──────────────────┘    └──────────────┬──────────────┘
                 │                                      │
                 │ govio-cli meta config                │ govio-cli onboard
                 │ (元数据连接配置)                       │ (图后端 + 数据源)
                 ▼                                      ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       govio-cli meta sync                             │
│       读取元数据 -> 生成 CSV -> 更新图库 -> 生成 assets                │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│                  图数据库后端 (govio-cli onboard 选择)                 │
│      NetworkX (GML)   │   FalkorDB   │   Ladybug (.lbdb 嵌入式)       │
└────────────────────────────────────────┬─────────────────────────────┘
                                         │
                                         ▼
                    ┌────────────────────────────────────┐
                    │  skills/govio*/assets              │
                    │  schema.md / metrics_index.md      │
                    │  ontology.gml / names/             │
                    └────────────────────────────────────┘
                                         │
                                         ▼
                  govio-cli query / observe / sql  +  Skills 工具集
```

## 快速开始

Govio 提供交互式 CLI 工具 `govio-cli`，典型使用流程分三步：

### 第一步：安装 govio-cli

```bash
uv build && ./start.sh
```

### 第二步：运行 Onboard 向导初始化

```bash
govio-cli onboard
```

向导会引导你完成：

1. **选择图数据库后端**
   - `networkx`：本地 GML 文件
   - `falkordb`：连接 FalkorDB 图数据库（host, port, graph name）
   - `ladybug`：Ladybug 嵌入式图数据库（默认 `~/.govio/ontology.lbdb`）

2. **配置数据源（可选，供 `observe` 命令使用）**
   - 添加 MySQL、DuckDB 等数据源
   - 密码自动加密存储，配置文件可安全分享

配置保存到 `~/.govio/config.yaml`。

### 第三步：同步元数据到知识图谱

```bash
# 配置元数据连接（首次使用需先运行），保存到 ~/.govio/meta_config.yaml
govio-cli meta config

# 导入元数据与语义层定义（元数据 -> CSV -> 图库 -> assets）
govio-cli meta sync
```

`meta sync` 交互流程：

1. 显示当前 meta 配置（元数据库、应用清单、应用映射、表关系、指标定义、CSV 输出目录等）
2. 选择数据来源：`TDS` / `DuckDB` / `Both`
3. 根据选择引导填写参数（TDS 自动从 `app_map` 获取 schemas；DuckDB 需指定 db 路径与 schema）
4. 选择执行模式：仅生成 CSV / 增量更新图库 / 重建图库
5. 自动生成 assets 到 `skills/govio/assets/`

完成后，将 `skills/govio*` 复制到 Agent 的 skills 目录即可通过自然语言查询。

<details>
<summary>meta sync 运行示例</summary>

```bash
$ govio-cli meta sync
当前 meta 配置:
  元数据库: mysql+pymysql://vt_app:***@172.18.240.2:15307/catalog_catalog1
  工作区 UUID: 82ee37374b314a938bf28170ab4db7cf
  应用清单: .\data\应用系统清单20251114.xlsx
  应用映射: .\data\app_map.json
  表关系: .\data\chocolate_relationships.json
  指标定义: .\data\chocolate_metrics.json
  CSV 输出目录: .\data\meta

? 数据来源: DuckDB - 仅从 DuckDB 读取
? DuckDB 数据库文件路径: .\data\chocolate.db
? 要导出的 schema 列表（逗号分隔，留空跳过）: chocolate
? 单库模式 app 名称（留空使用全量模式）:
? CSV 输出目录: .\data\meta
? 执行模式: 生成 CSV 并重建图库（删除后重新插入）
从 DuckDB 读取元数据...
提示: DuckDB 模式跳过 Standard 数据标准的读取
成功生成 RELATES_TO.csv，包含 3 个关系 来自[.\data\chocolate_relationships.json]
成功生成指标数据：6 个指标, 5 个维度
成功导出: 4 张表, 17 个字段, 17 个应用, 0 个标准, 3个数据关系, 6个指标

正在重建 Ladybug 图 (C:\Users\Administrator\.govio\ontology.lbdb)...
  PhysicalTable.csv: 4 行 -> PhysicalTable
  ...
✓ Ladybug 数据已重建

正在生成 assets...
✓ Assets 已生成到: skills\govio\assets

✅ meta-export 完成！
```

</details>

## CLI 命令总览

| 命令 | 用途 |
|------|------|
| `govio-cli onboard` | 初始化配置向导（图后端 + 数据源） |
| `govio-cli backend` | 显示当前图后端类型 |
| `govio-cli meta` | 知识图库维护（`sync` / `recommend` / `config`） |
| `govio-cli query` | 知识图谱查询（Cypher 或 Python） |
| `govio-cli observe` | 数据表探查（加载 / 比对 / 探索 / 图表） |
| `govio-cli sql` | 指标 SQL 组装（`build`） |
| `govio-cli -V` | 查看版本 |

### meta 命令组

| 子命令 | 用途 |
|--------|------|
| `govio-cli meta sync` | 完整/增量同步管线：读取元数据源 -> 生成 CSV -> 更新图数据 -> 生成 assets |
| `govio-cli meta recommend` | 数据标准推荐：为非标字段推荐匹配的数据标准 |
| `govio-cli meta config` | 交互式查看/修改元数据连接配置 |

`meta sync` 支持命令行模式（跳过交互）：

```bash
# 从 DuckDB 读取
govio-cli meta sync --db /path/to/meta.duckdb --schemas dbo,public --output ./output/

# 单库模式：按 app 名导出单个数据库的相关子图
govio-cli meta sync --db /path/to/meta.duckdb --db-name sales --output ./output/

# 仅生成 CSV，不更新图数据和 assets
govio-cli meta sync --dry-run --db /path/to/meta.duckdb --schemas dbo
```

**数据来源说明：**

| 来源 | 说明 |
|------|------|
| `TDS` | 仅从元数据库读取，schemas 从 `app_map.json` 自动获取 |
| `DuckDB` | 仅从 DuckDB 读取，需指定 schemas 或 db-name；**跳过 Standard 数据标准读取** |
| `Both` | TDS + DuckDB 合并，DuckDB 数据覆盖同名 TDS 数据 |

**执行模式说明：**

| 模式 | 说明 |
|------|------|
| 仅生成 CSV | 不更新图数据（dry-run） |
| 增量更新 | 生成 CSV 并 MERGE 更新图库 |
| 重建 | 生成 CSV 并删除后重新插入图库 |

### 查询工具

```bash
# 自动根据 config.yaml 选择后端（FalkorDB/Ladybug 用 Cypher，NetworkX 用 Python）
govio-cli query -c "MATCH (n:PhysicalTable) RETURN n.name LIMIT 5"
```

**查询规则：**

- FalkorDB / Ladybug：使用 Cypher 查询语言，必须以 `MATCH` 开头
- NetworkX：使用 Python 代码操作 `g`（NetworkX 图对象），结果赋值给 `result`

```bash
# NetworkX 模式（Python 代码，结果赋值给 result）
govio-cli query -c "result = [n for n, d in g.nodes(data=True) if d.get('node_type') == 'PhysicalTable'][:5]"
```

> 结果超过 20 行时自动写入 `.govio/output-<timestamp>.json`，否则直接打印 JSON。查询日志记录在 `~/.govio/logs/query_<date>.log`。

### Skills 工具集

配置完成后，将 `skills/govio*` 复制到 Agent 目录，即可通过自然语言查询。配置从 `~/.govio/config.yaml` 自动读取。

**Skill 列表：**

| Skill | 用途 |
|-------|------|
| `govio` | 主控 Skill，识别需求类型并路由到子 Skill |
| `govio-meta` | 知识图谱维护（同步、推荐、配置） |
| `govio-query` | 元数据/指标查询（应用、表、字段、指标问数） |
| `govio-observe` | 数据探查与比对（加载、探索、比对、图表） |
| `govio-eda` | EDA 探索性数据分析（4 阶段标准流程） |

**目录结构：**

```
skills/govio/
├── SKILL.md              # 技能定义（AI Agent 使用）
├── assets/               # 资源文件（meta sync 自动生成）
│   ├── schema.md         # 图数据库模式
│   ├── metrics_index.md  # 指标索引（原子/派生分组）
│   ├── ontology.gml      # NetworkX GML 数据文件
│   └── names/            # 节点名称索引
│       └── *.md          # 按应用分文件（Cypher 后端）
└── ...
```

**AI Agent 使用：** 加载 `SKILL.md` 后可直接用自然语言查询，例如：
- "查询 CRM 应用有几张表"
- "查找所有包含 '用户' 的表名"
- "本月账单收入是多少"

### CSV 文件格式要求

`meta sync` 生成的 CSV 使用 FalkorDB bulk-import 头约定（`:ID(Type)`、`:START_ID(Type)`、`:END_ID(Type)`）。

**节点文件：**
- `PhysicalTable.csv`: 物理表节点
- `Col.csv`: 字段节点
- `Application.csv`: 应用节点
- `Standard.csv`: 数据标准节点
- `Metric.csv`: 指标节点（可选，由指标定义 JSON 生成）
- `Dimension.csv`: 维度节点（可选，由指标定义 JSON 生成）

**边文件：**
- `HAS_COLUMN.csv`: 表包含字段的关系
- `USE.csv`: 应用使用表的关系
- `COMPLIES_WITH.csv`: 字段贯标的关系（由 `meta recommend` 生成）
- `RELATES_TO.csv`: 表间关系
- `USES_TABLE.csv`: 指标数据来源表的关系（可选）
- `REFERS_COLUMN.csv`: 指标引用列的关系（可选）
- `DERIVED_FROM.csv`: 派生指标依赖上游指标的关系（可选）
- `DIMENSION_USED.csv`: 指标维度关系（可选）
- `SUPERSEDES.csv`: 指标版本演进关系（可选）

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

## Python API

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

**Ladybug 模式（嵌入式，Cypher 与 FalkorDB 兼容）：**

```python
from govio import LadybugGraph

# 打开本地 .lbdb 文件（无需额外服务）
graph = LadybugGraph("~/.govio/ontology.lbdb")

# 查看图模式
print(graph.schema)

# 使用 Cypher 查询
result = graph.query("MATCH (n:Application) RETURN n.name LIMIT 10")
```

### 加载元数据

```python
from govio.metadata.application import AppInfoLoader
from govio.metadata.database import TDSLoader
from govio.metadata.standard import StandardLoader
from govio.metadata.duckdb_loader import DuckDBLoader
from govio.metadata.metric import MetricLoader

# 加载应用信息
app_loader = AppInfoLoader(app_list_file="path/to/app_list.xlsx")
apps = app_loader.Application

# 从 TDS（元数据库）加载表/字段元数据
tds_loader = TDSLoader(
    db="mysql+pymysql://user:pass@host/db",
    workspace_uuid="your-uuid",
    schema_limits=["schema1", "schema2"],
)
tables = tds_loader.PhysicalTable
columns = tds_loader.Col

# 从 DuckDB 加载表/字段元数据
duck_loader = DuckDBLoader(
    db_path="xxx.duckdb",
    schemas=["schema1", "schema2"],
)
tables = duck_loader.PhysicalTable
columns = duck_loader.Col

# 加载数据标准（仅 TDS 有）
std_loader = StandardLoader(
    db="mysql+pymysql://user:pass@host/db",
    workspace_uuid="your-uuid",
)
standards = std_loader.Standard
std_compliance = std_loader.StdCompliance

# 加载指标定义
metric_loader = MetricLoader(
    metric_file="path/to/metrics.json",
    df_tables=tables,
    df_columns=columns,
)
metrics = metric_loader.Metric          # 指标节点 DataFrame
dimensions = metric_loader.Dimension    # 维度节点 DataFrame
```

### 指标定义 JSON 格式

指标定义使用 JSON 文件，通过 JSON Schema 校验：

```json
{
  "version": "1.0",
  "shared_dimensions": [
    { "code": "ym", "name": "年月", "granularity": "月" },
    { "code": "business_unit", "name": "事业部" }
  ],
  "metrics": [
    {
      "code": "bill_income_amt",
      "name": "当月账单收入",
      "business_definition": "当月实际确认的账单收入金额",
      "type": "atomic",
      "unit": "万元",
      "data_type": "decimal(18,2)",
      "source_layer": "DWS",
      "source_tables": [
        {
          "full_table_name": "db.schema.t_bill",
          "columns": [
            { "column_name": "bill_income_amt", "role": "measure" }
          ]
        }
      ],
      "dimensions": [
        { "code": "ym", "usage_type": "group" },
        { "code": "business_unit", "usage_type": "slice" }
      ]
    },
    {
      "code": "burndown_amt",
      "name": "存量消耗额",
      "business_definition": "预计当月产生的账单收入减去危机金额",
      "type": "derived",
      "formula": "forecast_income_amt - risk_amt",
      "unit": "万元",
      "data_type": "decimal(18,2)",
      "source_layer": "DM",
      "derived_from": ["forecast_income_amt", "risk_amt"],
      "dimensions": [
        { "code": "ym", "usage_type": "group" }
      ]
    }
  ]
}
```

### 指标查询示例

```python
# FalkorDB / Ladybug: 指标血缘溯源
result = graph.query(
    "MATCH p=(m:Metric {code: 'burndown_amt'})-[:DERIVED_FROM*1..5]->(up:Metric) "
    "RETURN up.code, up.name, up.type"
)

# FalkorDB / Ladybug: 影响分析（某指标变更会影响哪些下游）
result = graph.query(
    "MATCH (m:Metric {code: 'bill_income_amt'})<-[:DERIVED_FROM*1..5]-(dep:Metric) "
    "RETURN dep.code, dep.name, dep.formula"
)

# FalkorDB / Ladybug: 数据溯源（指标的来源表）
result = graph.query(
    "MATCH (m:Metric {code: 'bill_income_amt'})-[:USES_TABLE]->(t:PhysicalTable) "
    "RETURN t.full_table_name, t.name"
)

# FalkorDB / Ladybug: 维度发现（指标可按哪些维度拆分）
result = graph.query(
    "MATCH (m:Metric {code: 'burndown_amt'})-[d:DIMENSION_USED]->(dim:Dimension) "
    "RETURN dim.code, dim.name, d.usage_type"
)

# NetworkX: 指标血缘溯源
import networkx as nx
metric_node = [n for n, d in graph.G.nodes(data=True)
               if d.get('node_type') == 'Metric' and d.get('code') == 'burndown_amt'][0]
ancestors = nx.ancestors(graph.G, metric_node)
lineage = [(n, graph.G.nodes[n].get('code')) for n in ancestors
           if graph.G.nodes[n].get('node_type') == 'Metric']
```

### 指标 SQL 组装

```python
from govio import build_metric_sql

sql_text = build_metric_sql(
    metrics=[
        {"code": "bill_income_amt", "name": "当月账单收入", "type": "原子",
         "source_table": "db.schema.t_bill", "actual_column": "bill_income_amt"},
    ],
    dimensions=["ym"],
    filters={"report_ym": "202501"},
    order_by="metric_value DESC",
    limit=100,
)
print(sql_text)
```

或通过 CLI：

```bash
# 从 JSON 规格文件组装
govio-cli sql build -f spec.json -o output.sql

# 从 stdin 传入（metrics 为指标 dict 列表，type 取值"原子"/"派生"）
echo '{"metrics": [{"code": "bill_income_amt", "name": "当月账单收入", "type": "原子", "source_table": "db.schema.t_bill", "actual_column": "bill_income_amt"}], "dimensions": ["ym"], "filters": {"report_ym": "202501"}, "limit": 100}' | govio-cli sql build
```

### 使用数据标准推荐器

```python
from govio.metadata.recommender import create_recommender
from govio.metadata.database import TDSLoader

# 加载已贯标列
std_loader = StandardLoader(db="mysql+pymysql://user:pass@host/db", workspace_uuid="your-uuid")
std_compliance = std_loader.StdCompliance

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
    top_n=3,        # 返回Top 3推荐
)

# 批量推荐
tds_loader = TDSLoader(db, workspace_uuid, ["schema_name"])
all_columns = tds_loader.Col  # 所有列
recommendations = recommender.batch_recommend(all_columns)
```

## 图模型

**节点类型：** `PhysicalTable`、`Col`、`Application`、`Standard`、`Metric`、`Dimension`

**边类型：**

| 边类型 | 方向 | 含义 |
|--------|------|------|
| `HAS_COLUMN` | table -> col | 表包含字段 |
| `USE` | app -> table | 应用使用表 |
| `COMPLIES_WITH` | col -> standard | 字段贯标 |
| `RELATES_TO` | table -> table | 表间关系 |
| `USES_TABLE` | metric -> table | 指标数据来源表 |
| `REFERS_COLUMN` | metric -> col | 指标引用列 |
| `DERIVED_FROM` | metric -> metric | 派生指标依赖上游指标 |
| `DIMENSION_USED` | metric -> dimension | 指标维度关系 |
| `SUPERSEDES` | metric -> metric | 指标版本演进 |

> `Calculation` 节点类型和 `CALCULATED_BY`/`BASED_ON` 边为预留的共享计算模板，暂未启用。

节点身份使用点分格式：`db.schema.table.column`。

## 许可证

[MIT License]
