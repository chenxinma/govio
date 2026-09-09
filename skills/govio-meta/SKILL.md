---
name: govio-meta
description: 知识图谱维护命令组。当需要导入元数据、推荐数据标准、或管理图数据库时触发。包含独立的导入子命令（meta/app/std/compliance/rel/metric）、graph（图数据库更新/清空）、recommend（数据标准推荐）。
---

# Govio Meta 知识图谱维护

本 Skill 对应 `govio-cli meta` 命令组，负责知识图谱的维护和管理。

## 子命令

| 子命令 | 用途 | 典型场景 |
|--------|------|----------|
| `govio-cli meta meta` | 导入 TDS/DuckDB 元数据 | PhysicalTable, Col, HAS_COLUMN |
| `govio-cli meta app` | 导入应用清单 | Application 节点 + USE 边 |
| `govio-cli meta std` | 导入数据标准 | Standard 节点（仅 TDS） |
| `govio-cli meta compliance` | 导出已有标准关联 | COMPLIES_WITH 边（仅 TDS） |
| `govio-cli meta rel` | 导入表关系 | RELATES_TO 边 |
| `govio-cli meta metric` | 导入指标维度 | Metric, Dimension + 5 种边 |
| `govio-cli meta graph` | 图数据库管理 | 更新/重建/清空图数据库 + 生成 assets |
| `govio-cli meta recommend` | 数据标准推荐 | 为非标字段推荐匹配的数据标准 |

## 前置条件

1. 已运行 `govio-cli onboard` 完成初始化配置（图数据库后端）
2. 图数据库后端配置在 `~/.govio/config.yaml` 的 `graph` section

## 推荐导入顺序

各子命令完全独立，所有输入通过 CLI 参数显式传入，不依赖配置文件。推荐按以下顺序执行：

```
meta → app → std → compliance → rel → metric → graph
```

各步骤可以按需独立运行，只要依赖的 CSV 文件已存在于 output 目录中。

| 步骤 | 依赖 | 说明 |
|------|------|------|
| `meta` | 无 | 必须最先运行，生成 PhysicalTable.csv, Col.csv |
| `app` | PhysicalTable.csv | 生成 Application.csv, USE.csv |
| `std` | 无 | 生成 Standard.csv（仅 TDS 模式） |
| `compliance` | Col.csv, Standard.csv | 生成 COMPLIES_WITH.csv（仅 TDS 模式） |
| `rel` | PhysicalTable.csv, Col.csv | 生成 RELATES_TO.csv |
| `metric` | PhysicalTable.csv, Col.csv | 生成 Metric.csv, Dimension.csv + 5 种边 CSV |
| `graph` | 所有 CSV | 更新图数据库 + 生成 assets |

## 使用示例

以下是一次完整的导入流程示例（敏感信息已脱敏）：

```bash
# 步骤 1: 导入元数据（TDS，--kundb / --workspace-uuid / --schemas 为必填）
govio-cli meta meta --source tds \
  --kundb "mysql+pymysql://user:pass@host:port/catalog" \
  --workspace-uuid 00000000-0000-0000-0000-000000000000 \
  --schemas "schema_a,schema_b,schema_c,schema_d" \
  --output ./data/meta

# 步骤 2: 导入应用清单
govio-cli meta app \
  --app-list ./ref/app_list.xlsx \
  --app-map ./ref/app_map.json \
  --output ./data/meta

# 步骤 3: 导入数据标准（--kundb / --workspace-uuid 为必填）
govio-cli meta std \
  --kundb "mysql+pymysql://user:pass@host:port/catalog" \
  --workspace-uuid 00000000-0000-0000-0000-000000000000 \
  --output ./data/meta

# 步骤 4: 导出已有标准关联（--kundb / --workspace-uuid 为必填）
govio-cli meta compliance \
  --kundb "mysql+pymysql://user:pass@host:port/catalog" \
  --workspace-uuid 00000000-0000-0000-0000-000000000000 \
  --output ./data/meta

# 步骤 5: 导入表关系
govio-cli meta rel --file ./ref/relationships.json --output ./data/meta

# 步骤 6: 导入指标维度
govio-cli meta metric --file ./ref/metrics.json --output ./data/meta

# 步骤 7: 更新图数据库 + 生成 assets
govio-cli meta graph --output ./data/meta --mode update
```

其他常用操作：

```bash
# 从 DuckDB 导入元数据
govio-cli meta meta --source duckdb --db /path/to/meta.duckdb --schemas schema_a --output ./data/meta

# TDS + DuckDB 合并
govio-cli meta meta --source both \
  --kundb "mysql+pymysql://user:pass@host:port/catalog" \
  --db /path/to/meta.duckdb --schemas schema_a --output ./data/meta

# 全量重建图数据库
govio-cli meta graph --output ./data/meta --mode rebuild

# 清空图数据库
govio-cli meta graph --mode clear

# 数据标准推荐
govio-cli meta recommend \
  --kundb "mysql+pymysql://user:pass@host:port/catalog" \
  --app-map ./ref/app_map.json \
  --csv-dir ./data/meta --output-dir ./data/meta
```

## 子命令详解

### meta — 元数据导入

从 TDS/DuckDB 提取元数据，生成 CSV。

**数据源（--source）**：
- **tds**：仅从元数据库读取；`--kundb`、`--workspace-uuid`、`--schemas` 为必填
- **duckdb**：仅从 DuckDB 读取，需 `--db`；跳过 Standard 数据标准
- **both**：TDS + DuckDB 合并，DuckDB 覆盖同名 TDS 数据；TDS 侧参数同 tds 模式

**输出 CSV**：`PhysicalTable.csv`, `Col.csv`, `HAS_COLUMN.csv`

### app — 应用清单导入

**必需参数**：`--app-list`（Excel）, `--app-map`（JSON）

**输出 CSV**：`Application.csv`, `USE.csv`

### std — 数据标准导入

**必需参数**：`--kundb`

**输出 CSV**：`Standard.csv`

### compliance — 已有标准关联

**必需参数**：`--kundb`

**前置**：Col.csv 和 Standard.csv 已存在

**输出 CSV**：`COMPLIES_WITH.csv`

### rel — 表关系导入

**必需参数**：`--file`（JSON）

**前置**：PhysicalTable.csv 和 Col.csv 已存在

**输出 CSV**：`RELATES_TO.csv`

### metric — 指标维度导入

**必需参数**：`--file`（JSON）

**前置**：PhysicalTable.csv 和 Col.csv 已存在

**输出 CSV**：`Metric.csv`, `Dimension.csv`, `USES_TABLE.csv`, `REFERS_COLUMN.csv`, `DERIVED_FROM.csv`, `DIMENSION_USED.csv`, `SUPERSEDES.csv`

### graph — 图数据库管理

**模式（--mode）**：
- **update**：增量 MERGE（默认）
- **rebuild**：全量重建（删除后重新插入）
- **clear**：清空图数据库（不重新导入）

图后端配置从 `~/.govio/config.yaml` 的 `graph` section 读取。

### recommend — 数据标准推荐

**必需参数**：`--kundb`, `--app-map`

**可选参数**：`--csv-dir`（已导入的 CSV 目录）, `--output-dir`（推荐结果输出目录）

## 节点与边类型

**节点类型**：`PhysicalTable`, `Col`, `Application`, `Standard`, `Metric`, `Dimension`

**边类型**：`HAS_COLUMN`, `USE`, `COMPLIES_WITH`, `RELATES_TO`, `USES_TABLE`, `REFERS_COLUMN`, `DERIVED_FROM`, `DIMENSION_USED`, `SUPERSEDES`

## 与其他 Skill 的协作

| 操作 | 关联 Skill |
|------|-----------|
| 查询图数据 | `govio-query` |
| 数据探查 | `govio-observe` |
| EDA 分析 | `govio-eda` |

## 排除场景

以下场景**不要**触发本技能：
- 查询元数据（应用、表、字段） → 使用 `govio-query`
- 数据探查、比对 → 使用 `govio-observe`
- 数据分析 → 使用 `govio-eda`
