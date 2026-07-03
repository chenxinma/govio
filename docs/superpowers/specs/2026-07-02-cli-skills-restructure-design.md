# CLI 命令与 Skills 目录重构设计

## 概述

重构 `govio-cli` 命令树和 `skills/` 目录层级，解决当前命令命名混乱、skill 层级不一致的问题。

**核心变更**：
1. 新增 `meta` 命令组（知识图库维护），吸收 `meta-export` 和 `std-recommend`
2. 合并 `observe show-datasource` / `list` / `show` 为 `observe info`
3. 精简 `onboard` 仅保留配置向导
4. Skills 目录按 CLI 命令组重新组织

---

## 一、CLI 命令树（目标状态）

```
govio-cli
├── onboard                              # 配置向导（图数据库 + observe 数据源）
├── backend                              # 显示图数据库类型（不变）
├── meta                                 # 知识图库维护（命令组）
│   ├── sync                             #   读取 meta 源 → CSV → 更新图库 → 资产生成
│   ├── export                           #   仅导出 CSV（不更新图库）
│   ├── recommend                        #   数据标准推荐（从 std-recommend 迁入）
│   └── config                           #   交互式配置 meta_config.yaml
├── query                                # 知识图谱查询（不变）
└── observe                              # 数据操作（命令组）
    ├── info                             #   查看信息（数据源/DF列表/DF结构）
    ├── load                             #   加载 DataFrame（不变）
    ├── release                          #   释放 DataFrame（不变）
    ├── compare                          #   比较 DataFrame（不变）
    ├── explore                          #   发现关系（不变）
    ├── visualize-relations              #   可视化关系（不变）
    └── chart                            #   生成图表（不变）
```

### 删除的命令

| 命令 | 替代方案 |
|------|---------|
| `govio-cli meta-export` | `govio-cli meta sync` / `govio-cli meta export` |
| `govio-cli std-recommend` | `govio-cli meta recommend` |
| `govio-cli observe show-datasource` | `govio-cli observe info --datasource` |
| `govio-cli observe show` | `govio-cli observe info --name <df>` |
| `govio-cli observe list` | `govio-cli observe info --df` |

---

## 二、命令详细设计

### 2.1 `govio-cli meta` — 知识图库维护

配置独立为 `~/.govio/meta_config.yaml`，从 `config.yaml` 的 `metadata` section 剥离。

#### `meta_config.yaml` 结构

```yaml
# 从 config.yaml metadata section 迁移
kundb: "mysql+pymysql://user:pass@host/db"
workspace_uuid: "82ee37374b314a938bf28170ab4db7cf"
app_list: "/path/to/app_list.xlsx"
app_map: "/path/to/app_map.json"
relationship: "/path/to/relationship.json"    # 可选
metric: "/path/to/metric.json"                # 可选
csv_dir: "/path/to/csv_output"

# 从 config.yaml graph section 复制（meta 需要知道图库目标）
graph:
  backend: falkordb                            # 或 networkx
  falkordb:
    host: localhost
    port: 6379
    graph: ontology
  networkx:
    gml_path: skills/govio/assets/ontology.gml
```

#### `govio-cli meta sync`

全量/增量同步流程：读取 meta 源 → 生成 CSV → 更新图库 → 生成 assets。

```
用法（CLI 参数模式）:
  govio-cli meta sync --db <duckdb_path> --schemas <schema_list> --output <csv_dir>
  govio-cli meta sync --db <duckdb_path> --db-name <app_name> --output <csv_dir>

用法（交互式模式）:
  govio-cli meta sync                          # 无参数，从 meta_config.yaml 读取默认值，逐步确认

参数:
  --db          DuckDB 数据库文件路径
  --schemas     schema 列表，逗号分隔（全量模式，与 --db-name 互斥）
  --db-name     单库模式：按 app 名导出（与 --schemas 互斥）
  --output      CSV 输出目录
  --dry-run     仅生成 CSV，不更新图库和 assets
```

行为：
- 无参数时进入交互式，从 `meta_config.yaml` 读取默认值
- 有参数时直接执行，参数覆盖配置文件
- `--dry-run` 等同于 `meta export`
- 执行链：DuckDB/TDS 读取 → CSV 生成 → 图库更新（FalkorDB MERGE 或 NetworkX GML rebuild）→ assets 生成

#### `govio-cli meta export`

等同于 `meta sync --dry-run`，仅导出 CSV。

```
用法:
  govio-cli meta export --db <duckdb_path> --schemas <schema_list> --output <csv_dir>
  govio-cli meta export --db <duckdb_path> --db-name <app_name> --output <csv_dir>

参数: 同 meta sync（不需要 --dry-run）
```

#### `govio-cli meta recommend`

从 `std-recommend` 迁入，读取 meta_config.yaml 中的配置。

```
用法:
  govio-cli meta recommend --output-dir <path>

参数:
  --output-dir   推荐结果输出目录（生成 COMPLIES_WITH.csv）
```

#### `govio-cli meta config`

交互式配置 `meta_config.yaml`。

```
用法:
  govio-cli meta config                        # 交互式配置向导
```

行为：
- 若 `meta_config.yaml` 不存在，引导创建
- 若已存在，显示当前配置并允许修改
- 从旧 `config.yaml` 的 `metadata` section 迁移（如存在）

### 2.2 `govio-cli observe info` — 统一信息查看

合并 `show-datasource`、`list`、`show` 三个命令。

```
用法:
  govio-cli observe info                       # 概览：数据源名称 + 已加载 DF 列表（分区段）
  govio-cli observe info --datasource          # 仅数据源名称列表
  govio-cli observe info --df                  # 仅已加载 DataFrame 列表
  govio-cli observe info --name <df_name>      # DataFrame 结构 + 样例数据
  govio-cli observe info --name <df> --rows 50 # 指定样例行数

参数:
  --datasource     显示数据源名称列表
  --df             显示已加载 DataFrame 列表
  --name           指定 DataFrame 名称，显示结构和样例数据
  --rows           样例行数（默认 10，仅配合 --name）
```

行为：
- 无参数：先输出数据源名称列表，再输出已加载 DF 列表，两段分开
- `--datasource`：仅输出数据源名称（JSON 数组）
- `--df`：仅输出已加载 DF 名称列表（JSON 数组），配合 `release` 使用
- `--name`：输出 DF 的 schema + 样例数据（JSON）
- `--datasource`、`--df`、`--name` 三者互斥

### 2.3 `govio-cli onboard` — 精简配置向导

剥离 meta 源配置和图库初始化操作，仅保留：
1. 图数据库选择（falkordb/networkx）+ 连接配置 → 写入 `config.yaml`
2. observe 数据源配置 → 写入 `config.yaml`

```
用法:
  govio-cli onboard                            # 交互式配置向导

删除的参数:
  --new-falkordb   （移至 meta sync 的图库更新逻辑）
  --new-networkx   （移至 meta sync 的图库更新逻辑）
```

行为：
- 引导选择图数据库后端 + 配置连接信息
- 引导配置 observe 数据源
- 保存到 `~/.govio/config.yaml`（仅 graph + datasources section）
- 不再包含 CSV 生成、图库导入、assets 生成逻辑

### 2.4 不变的命令

| 命令 | 说明 |
|------|------|
| `govio-cli backend` | 显示图数据库类型，不变 |
| `govio-cli query` | 知识图谱查询，不变 |
| `govio-cli observe load` | 加载 DataFrame，不变 |
| `govio-cli observe release` | 释放 DataFrame，不变 |
| `govio-cli observe compare` | 比较 DataFrame，不变 |
| `govio-cli observe explore` | 发现关系，不变 |
| `govio-cli observe visualize-relations` | 可视化关系，不变 |
| `govio-cli observe chart` | 生成图表，不变 |

---

## 三、配置文件拆分

### 当前状态

`~/.govio/config.yaml`：
```yaml
metadata:           # meta 源配置（kundb, app_list, app_map 等）
  kundb: ...
  workspace_uuid: ...
  app_list: ...
  app_map: ...
  csv_dir: ...
graph:              # 图数据库配置
  backend: ...
  falkordb/networkx: ...
datasources:        # observe 数据源
  my_db: ...
```

### 目标状态

`~/.govio/config.yaml`（onboard 管理）：
```yaml
graph:
  backend: falkordb
  falkordb:
    host: localhost
    port: 6379
    graph: ontology
datasources:
  my_db:
    url: mysql+pymysql://...
    connect_args: {}
```

`~/.govio/meta_config.yaml`（meta config 管理）：
```yaml
kundb: "mysql+pymysql://..."
workspace_uuid: "..."
app_list: "/path/to/app_list.xlsx"
app_map: "/path/to/app_map.json"
relationship: "/path/to/relationship.json"
metric: "/path/to/metric.json"
csv_dir: "/path/to/csv"
graph:                  # 图库目标（meta sync 需要知道写入哪里）
  backend: falkordb
  falkordb: ...
  networkx: ...
```

### 迁移策略

- `meta config` 首次运行时检测旧 `config.yaml` 中的 `metadata` section，自动迁移到 `meta_config.yaml`
- `onboard` 精简后不再写入 `metadata` section
- `meta sync` / `meta export` / `meta recommend` 优先读取 `meta_config.yaml`，兼容旧 `config.yaml` 的 `metadata` section

---

## 四、Skills 目录重组

### 目标结构

```
skills/govio/
├── SKILL.md                         # 根路由（query → metadata/metrics）
├── meta/
│   └── SKILL.md                     # 对应 govio-cli meta（sync/export/recommend/config）
├── query/
│   ├── SKILL.md                     # 图谱查询（合并原 govio-metadata + govio-metrics 查询能力）
│   ├── reference-falkordb.md        # 从 govio/govio-metadata/ 迁入
│   ├── reference-networkx.md        # 从 govio/govio-metadata/ 迁入
│   └── scripts/
│       └── sql_builder.py           # 从 govio/govio-metrics/ 迁入
├── observe/
│   └── SKILL.md                     # 对应 govio-cli observe 全部子命令
├── eda/
│   └── SKILL.md                     # EDA 4 阶段流程
└── assets/                          # 生成的资产（不变）
    ├── schema.md
    ├── ontology.gml
    ├── metrics_index.md
    └── names/
```

### 删除的目录

| 原目录 | 处理方式 |
|--------|---------|
| `govio-observe/` | 内容合并到 `govio/observe/SKILL.md` |
| `observe-dataset-ops/` | 内容合并到 `govio/observe/SKILL.md` |
| `observe-compare-dfs/` | 内容合并到 `govio/observe/SKILL.md` |
| `observe-explore-relations/` | 内容合并到 `govio/observe/SKILL.md` |
| `govio-eda/` | 内容合并到 `govio/eda/SKILL.md` |
| `eda-profiling/` | 内容合并到 `govio/eda/SKILL.md` |
| `eda-infer-relations/` | 内容合并到 `govio/eda/SKILL.md` |
| `eda-verify-relations/` | 内容合并到 `govio/eda/SKILL.md` |
| `eda-check-consistency/` | 内容合并到 `govio/eda/SKILL.md` |
| `govio/govio-metadata/` | 内容合并到 `govio/query/SKILL.md` |
| `govio/govio-metrics/` | 内容合并到 `govio/query/SKILL.md` |

### Skill 路由更新

`skills/govio/SKILL.md`（根路由）更新子 Skill 表：

| Skill | 用途 | 触发场景 |
|-------|------|----------|
| `meta` | 知识图库维护 | 元数据同步、数据标准推荐 |
| `query` | 图谱查询 | 元数据查询、指标问数 |
| `observe` | 数据操作 | 数据加载、比较、探查、可视化 |
| `eda` | 数据探查流程 | 4 阶段 EDA（profiling → infer → verify → consistency） |

---

## 五、实现要点

### 5.1 CLI 文件变更

| 文件 | 变更 |
|------|------|
| `src/govio/cli/main.py` | 删除 `meta-export`、`std-recommend`；新增 `meta` 命令组；修改 `onboard` 参数 |
| `src/govio/cli/meta.py` | **新建**，`meta` 命令组入口（sync/export/recommend/config 子命令） |
| `src/govio/cli/meta_export.py` | 核心逻辑保留，被 `meta.py` 调用；CLI 入口删除 |
| `src/govio/cli/std_recommend.py` | 核心逻辑保留，被 `meta.py` 调用；CLI 入口删除 |
| `src/govio/cli/observe.py` | 删除 `show-datasource`/`list`/`show` 子命令；新增 `info` 子命令 |
| `src/govio/cli/onboard.py` | 删除 `--new-falkordb`/`--new-networkx`；删除 meta 源配置逻辑；删除 CSV 生成/图库导入/assets 生成 |
| `src/govio/cli/config.py` | 新增 `MetaConfigManager` 类，管理 `meta_config.yaml` |

### 5.2 Skills 文件变更

| 操作 | 文件 |
|------|------|
| 新建 | `skills/govio/meta/SKILL.md` |
| 新建 | `skills/govio/query/SKILL.md` |
| 新建 | `skills/govio/observe/SKILL.md` |
| 新建 | `skills/govio/eda/SKILL.md` |
| 移动 | `skills/govio/govio-metadata/reference-*.md` → `skills/govio/query/` |
| 移动 | `skills/govio/govio-metrics/scripts/` → `skills/govio/query/` |
| 删除 | `skills/govio-observe/` |
| 删除 | `skills/observe-dataset-ops/` |
| 删除 | `skills/observe-compare-dfs/` |
| 删除 | `skills/observe-explore-relations/` |
| 删除 | `skills/govio-eda/` |
| 删除 | `skills/eda-profiling/` |
| 删除 | `skills/eda-infer-relations/` |
| 删除 | `skills/eda-verify-relations/` |
| 删除 | `skills/eda-check-consistency/` |
| 删除 | `skills/govio/govio-metadata/` |
| 删除 | `skills/govio/govio-metrics/` |
| 更新 | `skills/govio/SKILL.md`（根路由） |

### 5.3 向后兼容

- `config.yaml` 中旧的 `metadata` section 保留读取能力，`meta_config.yaml` 优先
- `meta sync` 无参数时从 `meta_config.yaml` 读取默认值，确保可重复运行
- 删除的 CLI 命令不保留别名，通过 skill 文档引导新命令

---

## 六、验证清单

- [ ] `govio-cli meta config` 能交互式创建/编辑 `meta_config.yaml`
- [ ] `govio-cli meta sync --db ... --schemas ... --output ...` 完整执行 CSV → 图库 → assets 链路
- [ ] `govio-cli meta sync`（无参数）从 `meta_config.yaml` 读取默认值并交互确认
- [ ] `govio-cli meta export --db ... --schemas ... --output ...` 仅生成 CSV
- [ ] `govio-cli meta recommend --output-dir ...` 生成 COMPLIES_WITH.csv
- [ ] `govio-cli observe info` 显示数据源 + DF 列表（分区段）
- [ ] `govio-cli observe info --datasource` 仅显示数据源
- [ ] `govio-cli observe info --df` 仅显示已加载 DF
- [ ] `govio-cli observe info --name <df>` 显示 DF 结构
- [ ] `govio-cli onboard` 仅配置图数据库 + 数据源
- [ ] `govio-cli backend` 不变
- [ ] `govio-cli query` 不变
- [ ] Skills 目录结构正确，所有 SKILL.md 中的 CLI 命令引用已更新
- [ ] 旧 `config.yaml` 中的 `metadata` section 能被 `meta_config.yaml` 正确迁移
