---
name: govio
description: 数据治理知识图谱,当需要做"元数据查询、表字段比较、SQL 生成"等数据治理相关工作时运行。
allowed-tools: Read, Grep, Glob
---

# Data Governance Knowledge Graph

作为一名数据治理专家，根据知识图谱中的信息提供数据治理的支持。

## Purpose

**必须先读取 `assets/schema.md` 确认当前图结构。**

元数据查询：用户可以询问关于数据资产的元数据信息，如数据资产的名称、描述、来源、状态等。
表字段比较：用户可以比较不同表之间的字段差异，如字段名称、数据类型、是否必填等。
SQL 生成：用户可以根据需求描述，自动生成符合要求的 SQL 语句。

## Backend Selection

根据 `assets/backend.txt` 文件选择后端图库：

```bash
cat assets/backend.txt
# 输出: networkx 或 falkordb
```

**不同后端的查询方式不同**：

| 后端 | 查询语言 | 参考文档 |
|------|---------|---------|
| `networkx` | Python 代码，操作 `g` 对象 | [reference-networkx.md](reference-networkx.md) |
| `falkordb` | Cypher 查询 | [reference-falkordb.md](reference-falkordb.md) |

## Best Practices

1. **优先使用 `govio-query` 查询**，自动适配后端，无需手动导入
2. 使用 Grep 查询 `assets/names/` 下对应文件获得被记载的标准名称（见下方"节点名称文件"说明）
3. **必须先阅读 `assets/schema.md`** 了解当前图结构（节点、属性、关联关系），schema.md 内容会随数据变化
4. 查询取数应控制输出行数，一次获取小于 300 行
5. **注意：遵守有限读取原则，仅在必要时读取 schema.md**
6. **列名属性**：Col 节点使用 `column_name` 属性表示列名，不要直接使用 `name`

## Resource Resolution

When this skill is loaded, the base directory is provided:

```
govio/
├── SKILL.md                   # 技能定义
├── reference-networkx.md      # NetworkX 后端参考
├── reference-falkordb.md      # FalkorDB 后端参考
├── assets/                    # 资源文件
│   ├── backend.txt            # 后端配置 (networkx 或 falkordb)
│   ├── schema.md              # 图数据库模式文件
│   ├── ontology.gml           # 数据治理元模型数据文件(GML格式)
│   ├── govio-*.whl            # govio 包 (uvx 自动解析依赖)
│   └── names/                 # 已知节点的名称，作为标准名称备参考
│        ├── node_names.md     #   (networkx 后端) 所有节点名称汇总
│        └── *.md              #   (falkordb 后端) 按应用系统分文件，格式: {应用名}_{缩写}.md
```

## Usage

**优先使用 `govio-query` 命令进行查询**，它会自动根据 `assets/backend.txt` 选择后端，无需手动导入。

### govio-query 基本用法

```bash
# 通用格式（uvx 自动从 whl 解析 govio 及其所有依赖，无需手动安装）
uvx --from skills/govio/assets/govio-*.whl govio-query --assets skills/govio/assets "查询语句"
# NetworkX 后端传 Python 代码，FalkorDB 后端传 Cypher
```

### govio-query 查询示例

#### 查询图模式（通用）

```bash
uvx --from skills/govio/assets/govio-*.whl govio-query --assets skills/govio/assets "print(g.schema)"
```

#### 查询 CRM 应用有几张表（FalkorDB）

```bash
uvx --from skills/govio/assets/govio-*.whl govio-query --assets skills/govio/assets "MATCH (app:Application {name: 'CRM'})-[:USE]->(t:PhysicalTable) RETURN count(t) AS table_count"
```

#### 查询 CRM 应用使用的所有表（FalkorDB）

```bash
uvx --from skills/govio/assets/govio-*.whl govio-query --assets skills/govio/assets "MATCH (app:Application {name: 'CRM'})-[:USE]->(t:PhysicalTable) RETURN t.name, t.full_table_name"
```

#### 查询 CUSTOMER 表的所有字段（FalkorDB）

```bash
uvx --from skills/govio/assets/govio-*.whl govio-query --assets skills/govio/assets "MATCH (t:PhysicalTable {name: 'CUSTOMER'})-[:HAS_COLUMN]->(c:Col) RETURN c.column_name, c.dtype ORDER BY c.order_no"
```

#### 查询所有应用及其表数量（FalkorDB）

```bash
uvx --from skills/govio/assets/govio-*.whl govio-query --assets skills/govio/assets "MATCH (app:Application)-[:USE]->(t:PhysicalTable) RETURN app.name, count(t) AS table_count ORDER BY table_count DESC"
```

#### 查询 NetworkX 图节点数

```bash
uvx --from skills/govio/assets/govio-*.whl govio-query --assets skills/govio/assets "result = g.G.number_of_nodes()"
```

### 查询终止策略

当查询目标可能不存在于知识图谱中时，必须遵守**有限尝试原则**，避免无限循环查询：

1. **最多 3 次尝试**：对同一语义目标的查询（换关键词、换查询方式都算同一目标），最多尝试 3 次
2. **先查节点名称**：确认目标关键词是否存在于已记载的节点名称中。若不存在，可直接判定无结果。查询方式因后端而异：
   - **networkx**：Grep 搜索 `assets/names/node_names.md`（单一汇总文件）
   - **falkordb**：Grep 搜索 `assets/names/` 目录下所有 `*.md` 文件，或先用 Glob 列出文件名定位到具体应用（文件名格式：`{应用名}_{缩写}.md`，如 `薪税生产系统_PAYPRO.md`），再针对性搜索
3. **逐步放宽策略**：
   - 第 1 次：精确匹配（如 `name: '银行'`）
   - 第 2 次：模糊/包含匹配（如 `name CONTAINS '银行'` 或搜索描述字段）
   - 第 3 次：同义词扩展（如 `金融`、`bank` 等）
4. **及时终止并告知**：3 次尝试后仍无结果，**必须**停止查询，明确告知用户"知识图谱中未找到相关数据"，而非继续尝试其他查询方式

### 注意事项

- Cypher 查询必须用**双引号**包裹参数
- 结果超过 10 行时自动输出到 `assets/output-*.json` 文件
- Cypher 查询必须以 `MATCH` 开头

