---
name: govio-metadata
description: 查询数据治理元数据（应用、表、字段、数据标准）。当需要了解数据资产结构、表字段信息、应用归属时触发。不适用于指标查询、数据分析、SQL生成。
---

# Govio 元数据查询

查询知识图谱中的元数据信息，包括应用、物理表、字段、数据标准。

## 步骤（强制顺序）

**Step 0** ⚠️ 读取 `assets/schema.md`（**仅此一次，不得重复读取**）

**Step 1** 如果 prompt 含中文系统名，用 Grep 搜索 `assets/names/` 获取标准英文代码（见"名称解析"）

**Step 2** 使用 `govio-cli query --code "..."` 执行查询（自动适配后端）

**Step 3** 格式化输出：中文回答，应用名/表名/字段名等技术术语保留英文原文

## 后端

后端类型通过 `govio-cli backend` 获取（由 `govio-cli onboard` 设置）。

| 后端 | 查询语言 | 深度参考 |
|------|---------|---------|
| `falkordb` | Cypher | [reference-falkordb.md](reference-falkordb.md) |
| `networkx` | Python，操作 `g` 对象 | [reference-networkx.md](reference-networkx.md) |

## Cypher 语法规范

- 属性值**必须用双引号**：`{name: "AEP"}` 而非 `{name: 'AEP'}`
- 必须以 `MATCH` 开头
- **必须包含 `LIMIT 300`**（明确需全量除外）
- Col 节点用 `column_name` 属性表示列名，不要用 `name`

## 常用查询模板

| 场景 | FalkorDB (Cypher) | NetworkX (Python) |
|------|-------------------|--------------------|
| 所有应用 | `MATCH (app:Application) RETURN app.name, app.app_name_en, app.business_domain LIMIT 300` | `apps = [d for _,d in g.G.nodes(data=True) if d.get("node_type")=="Application"]` |
| 应用下的表 | `MATCH (app:Application {name: "AEP"})-[:USE]->(t:PhysicalTable) RETURN t.name, t.full_table_name LIMIT 300` | `tables = [g.G.nodes[v] for u,v,e in g.G.edges(data=True) if g.G.nodes[u].get("name")=="AEP" and e.get("edge_type")=="USE"]` |
| 表的字段 | `MATCH (t:PhysicalTable {name: "T1"})-[:HAS_COLUMN]->(c:Col) RETURN c.column_name, c.dtype ORDER BY c.order_no LIMIT 300` | `cols = sorted([g.G.nodes[v] for u,v,e in g.G.edges(data=True) if g.G.nodes[u].get("name")=="T1" and e.get("edge_type")=="HAS_COLUMN"], key=lambda x: x.get("order_no",0))` |
| 两应用同名表 | `MATCH (app1:Application {name: "A"})-[:USE]->(t1:PhysicalTable), (app2:Application {name: "B"})-[:USE]->(t2:PhysicalTable) WHERE t1.name = t2.name RETURN t1.name LIMIT 300` | 复杂查询参见 reference-networkx.md |
| 聚合排序 | `MATCH (app:Application)-[:USE]->(t:PhysicalTable) RETURN app.name, count(t) AS cnt ORDER BY cnt DESC LIMIT 300` | `from collections import Counter; cnt = Counter(g.G.nodes[u].get("name") for u,v,e in g.G.edges(data=True) if g.G.nodes[u].get("node_type")=="Application" and e.get("edge_type")=="USE")` |
| 按业务领域筛选 | `MATCH (app:Application {business_domain: "财务管理"}) RETURN app.name, app.app_name_en LIMIT 300` | `apps = [d for _,d in g.G.nodes(data=True) if d.get("node_type")=="Application" and d.get("business_domain")=="财务管理"]` |

## 名称解析

当 prompt 包含中文系统名（如"报价单中心系统""薪税系统"），**必须先 Grep 确认标准英文代码**：

- **networkx 后端**：Grep 搜索 `assets/names/node_names.md`
- **falkordb 后端**：Grep 搜索 `assets/names/` 下所有 `*.md` 文件，或先用 Glob 列出文件名定位（格式：`{应用名}_{缩写}.md`，如 `薪税生产系统_PAYPRO.md`）

## 查询终止策略

对同一语义目标最多 **3 次尝试**，逐步放宽：

1. 精确匹配（`name: "银行"`）
2. 模糊/包含匹配（`name CONTAINS "银行"`）
3. 同义词扩展（`金融`、`bank` 等）

3 次后仍无结果，**必须停止**并告知"知识图谱中未找到相关数据"。

## 排除场景

以下场景**不要**触发本技能：
- 指标查询、指标血缘分析 → 使用 `govio-metrics`
- 数据迁移脚本编写
- 代码调试/修复
- 功能模块开发

## 资源文件

```
assets/
├── schema.md          # 图模式（Step 0 必读，仅读一次）
├── ontology.gml       # GML 元模型数据（NetworkX 后端）
└── names/             # 标准名称
     ├── node_names.md #   (networkx) 全部节点名称汇总
     └── *.md          #   (falkordb) 按应用分文件
```
