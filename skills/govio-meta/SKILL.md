---
name: govio-meta
description: 知识图谱维护命令组。当需要同步元数据、推荐数据标准、或管理配置时触发。包含 sync（同步元数据到图数据库）、recommend（数据标准推荐）、config（查看/修改配置）三个子命令。
---

# Govio Meta 知识图谱维护

本 Skill 对应 `govio-cli meta` 命令组，负责知识图谱的维护和管理。

## 子命令

| 子命令 | 用途 | 典型场景 |
|--------|------|----------|
| `govio-cli meta sync` | 完整/增量同步管线 | 读取元数据源 → 生成 CSV → 更新图数据 → 生成 assets |
| `govio-cli meta recommend` | 数据标准推荐 | 为非标字段推荐匹配的数据标准 |
| `govio-cli meta config` | 查看/修改配置 | 管理元数据源和输出配置 |

## 前置条件

1. 已运行 `govio-cli onboard` 完成初始化配置（图数据库后端 + 数据源）
2. meta 配置已设置（通过 `govio-cli meta config` 查看/修改）

## 配置

通过 `govio-cli meta config` 管理，字段包括：kundb（元数据库 URL）、workspace_uuid、app_list、app_map、relationship、metric、csv_dir。

## 使用模式

### 模式 A: 交互模式

启动交互式向导，显示配置后选择数据源：

```bash
govio-cli meta sync
```

交互流程：
1. 显示当前配置
2. 选择数据源：TDS / DuckDB / Both
3. 根据选择引导填写参数（TDS 自动从 app_map 获取 schemas）
4. 选择 CSV 输出目录
5. 选择执行模式：仅生成 CSV / 生成 CSV 并更新图库（增量 MERGE）/ 生成 CSV 并重建图库（删除后重新插入）

### 模式 B: 命令行模式

```bash
# 从 DuckDB 读取
govio-cli meta sync --db /path/to/meta.duckdb --schemas dbo,public --output ./output/

# 从 DuckDB 单库模式
govio-cli meta sync --db /path/to/meta.duckdb --db-name sales --output ./output/

# 仅生成 CSV（不更新图数据和 assets）
govio-cli meta sync --dry-run --db /path/to/meta.duckdb --schemas dbo
```

### 模式 C: 数据标准推荐

为不符合标准的字段推荐匹配的数据标准：

```bash
govio-cli meta recommend
govio-cli meta recommend --output-dir ./output/
```

### 模式 D: 配置管理

查看或修改 meta 配置：

```bash
govio-cli meta config
```

交互式引导填写所有配置项，已有的配置会显示当前值作为默认值。

## 子命令详解

### sync - 同步元数据

从元数据库（DuckDB/TDS）提取元数据，生成 CSV，更新图数据库（FalkorDB / Ladybug MERGE 或 NetworkX GML 重建），最后生成 assets。

**数据流**：
```
DuckDB/TDS → meta_export() in meta.py → CSV → FalkorDB / Ladybug MERGE / NetworkX GML rebuild → assets
```

**数据源（交互模式可选）**：
- **TDS**：仅从元数据库读取，schemas 从 `app_map.json` 自动获取
- **DuckDB**：仅从 DuckDB 读取，需指定 schemas 或 db-name；**跳过 Standard 数据标准读取**（数据标准仅存于 TDS，DuckDB 无对应来源），产出的 `Standard.csv` 只有表头
- **Both**：TDS + DuckDB 合并，DuckDB 数据覆盖同名 TDS 数据

命令行模式通过 `--db` 有无自动推断（有则 DuckDB，无则 TDS）。

> 注：DuckDB 模式不访问 TDS，因此配置中的 `kundb` 在该模式下非必须（TDS/Both 模式仍必须）。数据标准节点需后续从 TDS 补齐或在 Both 模式下获取。

**交互模式行为**：
1. 加载 meta 配置
2. 显示当前配置项及值
3. 选择数据源（TDS / DuckDB / Both）
4. 根据选择引导填写参数
5. 用配置值作为交互提示的默认值

**节点类型**：`PhysicalTable`, `Col`, `Application`, `Standard`, `Metric`, `Dimension`

**边类型**：`HAS_COLUMN`, `USE`, `COMPLIES_WITH`, `RELATES_TO`, `USES_TABLE`, `REFERS_COLUMN`, `DERIVED_FROM`, `DIMENSION_USED`, `SUPERSEDES`

### recommend - 数据标准推荐

基于 k-NN 协同过滤算法，为不符合标准的字段推荐匹配的数据标准。输出目录默认使用配置中的 `csv_dir`。

### config - 配置管理

交互式管理 meta 配置。已存在配置时先显示当前值，确认后逐项引导修改。

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
