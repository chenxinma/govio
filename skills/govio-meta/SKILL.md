---
name: govio-meta
description: 知识图谱维护命令组。当需要导入元数据、推荐数据标准、或管理图数据库时触发。包含独立的导入子命令（meta/app/std/compliance/rel/metric）、graph（图数据库更新/清空）、recommend（数据标准推荐）。
---

# Govio Meta 知识图谱维护

对应 `govio-cli meta` 命令组：各子命令把输入写成 CSV，再由 `graph` 把 CSV 导入图库。

## 硬性约束（优先级最高）

1. **一切操作只通过 `govio-cli` 完成**。未经用户明确授权，禁止读写源数据库——包括 `python -c`、duckdb / sqlalchemy 直连、`ATTACH`、`CREATE SCHEMA`、`CREATE TABLE AS SELECT`、复制或重建库文件等任何形式。确有必要时先说明理由并取得授权。
2. **禁止手写或编辑 `--output` 目录下的 CSV**。CSV 只能由子命令生成；要修正数据就修正输入（JSON / Excel / 源库 schema 参数）后重跑子命令。
3. **禁止读 govio 源码、git 历史或 `docs/specs` 来反推 CLI 语义**。语义以本 Skill 和 `govio-cli meta <子命令> --help` 为准；两者都没有的，直接问用户。
4. **禁止导入后二次校验**（重算 node_id、用 pandas 复核 CSV、用 `query` 计数比对）。以 CLI stdout 的 `✓ / ✅` 与行数输出为完成依据，**只有报错时才排查**。
5. **无内容即停止**。CLI 报 `❌ 未发现任何表` 时，把错误原文（含它列出的可用 schema）转告用户并停止；不得改名、造临时库或手改 CSV 绕过。
6. **导入结果不上画布**。元数据导入只做文字汇总。源库可能非常庞大，全量展示代价过高；画布展示表结构属于用户主动查询场景（`govio-query`），不在导入流程内。
7. **禁止建 junction / 软链接**。assets 目录与应用目录不一致时，把绝对路径告知用户，由用户自行合并。
8. **先验输入文件类型，再选参数**。`--db` 指向的文件用魔数确认，不靠后缀名猜（`.db` 既可能是 SQLite 也可能是 DuckDB，DuckDB 文件头偏移 8 起是 `DUCK`）：
   `head -c 12 <file> | grep -qa DUCK && echo DUCKDB || echo 非DuckDB`
   不是 DuckDB 文件就不能用 `--source duckdb`；对不上时告知用户并停止，不要自行转库（见约束 1）。

## 标准工作循环

零星内容导入永远是两步：

```
① 有新输入的那个子命令 → 写入/合并 CSV（--output 复用既有 CSV 目录，增量幂等）
② govio-cli meta graph --output <同一目录> --mode update
```

- **只跑本次有新输入的子命令**，不需要走完整 7 步
- `--output` 必须指向既有 CSV 目录：指错目录等于丢掉历史增量
- CSV 目录是子命令之间唯一的共享状态；`graph` 只认目录里的固定文件名
- `--mode update` 是增量 upsert，可反复执行

| 本次新增的输入 | 跑哪条 | 产出 CSV |
|---|---|---|
| 元数据库（TDS）/ DuckDB 文件 | `meta meta` | PhysicalTable, Col, HAS_COLUMN |
| 已配置的 DuckDB 数据源 | `meta import-schema` | PhysicalTable, Col, HAS_COLUMN（一步完成 meta + graph） |
| 应用清单 Excel + app_map JSON | `meta app` | Application, USE |
| 数据标准（仅 TDS） | `meta std` | Standard |
| 已有贯标关系（仅 TDS） | `meta compliance` | COMPLIES_WITH |
| 表关系 JSON | `meta rel` | RELATES_TO |
| 指标定义 JSON | `meta metric` | Metric, Dimension + 5 种边 |
| 上面任何一项跑完 | `meta graph --mode update` | 图库 + assets |

## 命名语义（节点名从哪来）

- `full_table_name = <schema>.<table>`；节点名与 node_id 都由它决定（node_id = 类型前缀 + SHA256(业务键) 前 8 位，自动生成，无需干预）
- `--schemas` **必填**，必须写源库里真实存在的 schema 名；DuckDB 文件的默认 schema 是 `main`
- 用户给的名字与源库真实 schema 不一致时（如源库 schema 是 `main` 但用户给了 `orders`）：**告知并停止**，请用户确认按哪个 schema 导入。当前版本不支持改名导入，**不得为此改动源库**
- schema 写错或源库为空时，CLI 会失败并列出该库可导入的 schema，直接转告用户即可

## 子命令

| 子命令 | 用途 |
|--------|------|
| `meta meta` | 导入 TDS/DuckDB 元数据（PhysicalTable, Col, HAS_COLUMN） |
| `meta app` | 导入应用清单（Application 节点 + USE 边） |
| `meta std` | 导入数据标准（Standard 节点，仅 TDS） |
| `meta compliance` | 导出已有标准关联（COMPLIES_WITH 边，仅 TDS） |
| `meta rel` | 导入表关系（RELATES_TO 边） |
| `meta metric` | 导入指标维度（Metric, Dimension + 5 种边） |
| `meta graph` | 更新/重建/清空图数据库 + 生成 assets |
| `meta recommend` | 为非标字段推荐匹配的数据标准 |
| `meta import-schema` | 从已配置的 DuckDB 数据源导入元数据到图库（meta + graph 一步完成） |

## 前置条件

1. 已运行 `govio-cli onboard` 完成初始化（图后端与数据源由 onboard 写入配置）
2. 导入子命令不读任何配置文件，输入全部由 CLI 参数显式给出；只有 `meta graph` 读 `graph` section

## 命令速查（最小可运行）

```bash
# 元数据：DuckDB（--schemas 必填，DuckDB 默认 schema 为 main）
govio-cli meta meta --source duckdb --db /path/to/meta.duckdb --schemas main --output ./data/meta

# 元数据：TDS
govio-cli meta meta --source tds --kundb "mysql+pymysql://user:pass@host:port/catalog" \
  --workspace-uuid <uuid> --schemas "schema_a,schema_b" --output ./data/meta

# 元数据：TDS + DuckDB 合并（DuckDB 覆盖同名 TDS 数据）
govio-cli meta meta --source both --kundb "mysql+pymysql://..." --workspace-uuid <uuid> \
  --db /path/to/meta.duckdb --schemas schema_a --output ./data/meta

# 应用清单
govio-cli meta app --app-list ./ref/app_list.xlsx --app-map ./ref/app_map.json --output ./data/meta

# 数据标准 / 已有贯标关系（仅 TDS；compliance 需 Col.csv 与 Standard.csv 已存在）
govio-cli meta std --kundb "mysql+pymysql://..." --workspace-uuid <uuid> --output ./data/meta
govio-cli meta compliance --kundb "mysql+pymysql://..." --workspace-uuid <uuid> --output ./data/meta

# 表关系 / 指标维度（需 PhysicalTable.csv 与 Col.csv 已存在）
govio-cli meta rel --file ./ref/relationships.json --output ./data/meta
govio-cli meta metric --file ./ref/metrics.json --output ./data/meta

# 导入图库 + 生成 assets（默认 assets 目录 .agent/skills/govio/assets）
govio-cli meta graph --output ./data/meta --mode update

# 指定 assets 输出目录
govio-cli meta graph --output ./data/meta --assets-dir ./my/assets --mode update

# 数据标准推荐
govio-cli meta recommend --kundb "mysql+pymysql://..." --app-map ./ref/app_map.json \
  --csv-dir ./data/meta --output-dir ./data/meta

# 从已配置的 DuckDB 数据源导入元数据到图库（一步完成 meta + graph）
govio-cli meta import-schema --datasource mydb --schemas main --output ./data/meta
```

`meta graph --mode`：`update` 增量 MERGE（默认）· `rebuild` 全量重建 · `clear` 只清空不导入。

## 首次全量导入

各子命令相互独立，首次建库按依赖顺序执行一遍即可（`meta` 必须最先跑）：

```
meta → app → std → compliance → rel → metric → graph
```

| 步骤 | 依赖 |
|------|------|
| `meta` | 无 |
| `app` | PhysicalTable.csv |
| `std` | 无 |
| `compliance` | Col.csv, Standard.csv |
| `rel` / `metric` | PhysicalTable.csv, Col.csv |
| `graph` | 目录内所有 CSV |

## 结果说明（唯一汇报口径）

以 CLI 输出为准，向用户汇报以下几项，不做额外验证：

- `meta meta` 等子命令：`✓ 元数据已导出: N 张表, M 个字段` + CSV 目录
- `meta graph`：各 CSV 导入行数、图后端与库文件路径
- `meta graph` 末尾打印的 **assets 绝对路径**；若与应用读取的 assets 目录不一致，提示用户自行合并
- 失败时：原样转述 `❌` 错误信息与其中的可用 schema 列表，然后停止



## 节点与边类型

**节点**：`PhysicalTable`, `Col`, `Application`, `Standard`, `Metric`, `Dimension`

**边**：`HAS_COLUMN`, `USE`, `COMPLIES_WITH`, `RELATES_TO`, `USES_TABLE`, `REFERS_COLUMN`, `DERIVED_FROM`, `DIMENSION_USED`, `SUPERSEDES`

## 与其他 Skill 的协作

| 操作 | 关联 Skill |
|------|-----------|
| 查询图数据、查看表结构 | `govio-query` |
| 数据探查、比对 | `govio-observe` |
| EDA 分析 | `govio-eda` |

## 排除场景

以下场景**不要**触发本技能：
- 查询元数据（应用、表、字段）、在画布上展示表结构 → `govio-query`
- 数据探查、比对 → `govio-observe`
- 数据分析 → `govio-eda`
