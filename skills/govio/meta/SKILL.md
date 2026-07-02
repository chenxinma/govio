---
name: govio-meta
description: 知识图谱维护命令组。当需要同步元数据、导出图数据、推荐数据标准、或管理配置时触发。包含 sync（同步元数据到图数据库）、export（导出图数据为 GML）、recommend（数据标准推荐）、config（查看/修改配置）四个子命令。
---

# Govio Meta 知识图谱维护

本 Skill 对应 `govio-cli meta` 命令组，负责知识图谱的维护和管理。

## 子命令

| 子命令 | 用途 | 典型场景 |
|--------|------|----------|
| `govio-cli meta sync` | 同步元数据到图数据库 | 从 MySQL 提取元数据，写入 FalkorDB |
| `govio-cli meta export` | 导出图数据为 GML | 生成 NetworkX 可加载的 GML 文件 |
| `govio-cli meta recommend` | 数据标准推荐 | 为非标字段推荐匹配的数据标准 |
| `govio-cli meta config` | 查看/修改配置 | 查看当前后端、数据源等配置 |

## 前置条件

1. 已运行 `govio-cli onboard` 完成初始化配置
2. 数据源连接信息已配置（通过 `govio-cli meta config` 查看）

## 使用模式

### 模式 A: 全量同步

首次或全量更新知识图谱：

```bash
# 1. 同步元数据（表、字段、应用、标准）
govio-cli meta sync

# 2. 导出为 GML（供 NetworkX 后端使用）
govio-cli meta export
```

### 模式 B: 增量同步

日常更新，仅同步变更部分：

```bash
govio-cli meta sync --incremental
```

### 模式 C: 数据标准推荐

为不符合标准的字段推荐匹配的数据标准：

```bash
govio-cli meta recommend
```

### 模式 D: 配置管理

查看或修改配置：

```bash
# 查看当前配置
govio-cli meta config

# 修改后端类型
govio-cli meta config --backend falkordb
govio-cli meta config --backend networkx
```

## 子命令详解

### sync - 同步元数据

从 MySQL 数据库提取元数据（表、字段、应用、标准），写入 FalkorDB 图数据库。

**数据流**：
```
MySQL (源数据库) → DatabaseLoader + AppInfoLoader + StandardLoader → CSV → FalkorDB
```

**节点类型**：
- `PhysicalTable`：物理表
- `Col`：字段
- `Application`：应用系统
- `Standard`：数据标准
- `Metric`：指标
- `Dimension`：维度

**边类型**：
- `HAS_COLUMN`：表→字段
- `USE`：应用→表
- `COMPLIES_WITH`：字段→标准
- `RELATES_TO`：表→表
- `USES_TABLE`：指标→表
- `REFERS_COLUMN`：指标→字段
- `DERIVED_FROM`：指标→指标
- `DIMENSION_USED`：指标→维度

### export - 导出图数据

将 FalkorDB 中的图数据导出为 GML 格式，供 NetworkX 后端加载。

```bash
govio-cli meta export -o ./output/
```

产出文件：`ontology.gml`

### recommend - 数据标准推荐

基于 k-NN 协同过滤算法，为不符合标准的字段推荐匹配的数据标准。

```bash
govio-cli meta recommend -o ./output/
```

**算法**：使用字段名、注释、数据类型等特征的加权相似度，找出最匹配的数据标准。

### config - 配置管理

查看或修改 Govio 配置。

```bash
# 查看配置
govio-cli meta config

# 设置后端
govio-cli meta config --backend falkordb
```

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
