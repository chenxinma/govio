---
name: govio-query
description: 数据治理查询能力。当需要查询元数据（应用、表、字段、数据标准）或指标数据（指标值、维度分析、趋势统计）时触发。通过 govio-cli query 命令执行，支持 Cypher（FalkorDB）和 Python（NetworkX）两种后端。
---

# Govio 查询

本 Skill 合并了元数据查询和指标问数能力，通过 `govio-cli query` 命令统一执行。

## 查询类型

| 类型 | 触发场景 | 示例 |
|------|----------|------|
| **元数据查询** | 查询应用、表、字段、数据标准 | "AEP 有哪些表"、"T_INVOICE 的字段" |
| **指标问数** | 查询指标数据、维度分析、趋势统计 | "本月账单收入"、"按部门统计签约额" |

## 步骤（强制顺序）

**Step 0** ⚠️ 读取 `../govio/assets/schema.md`（**仅此一次，不得重复读取**）

**Step 1** 如果 prompt 含中文系统名，用 Grep 搜索 `../govio/assets/names/` 获取标准英文代码（见"名称解析"）

**Step 2** 判断查询类型：
- 包含"指标"、"收入"、"签约"、"金额"、"YTD"等业务指标关键词 → 指标问数流程
- 包含"应用"、"表"、"字段"、"结构"等元数据关键词 → 元数据查询流程
- 不确定 → 先询问用户意图

**Step 3** 使用 `govio-cli query -c "..."` 执行查询（自动适配后端）

**Step 4** 格式化输出：中文回答，应用名/表名/字段名等技术术语保留英文原文

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

---

## 元数据查询

查询知识图谱中的元数据信息，包括应用、物理表、字段、数据标准。

### 常用查询模板

| 场景 | FalkorDB (Cypher) | NetworkX (Python) |
|------|-------------------|--------------------|
| 所有应用 | `MATCH (app:Application) RETURN app.name, app.app_name_en, app.business_domain LIMIT 300` | `apps = [d for _,d in g.G.nodes(data=True) if d.get("node_type")=="Application"]` |
| 应用下的表 | `MATCH (app:Application {name: "AEP"})-[:USE]->(t:PhysicalTable) RETURN t.name, t.full_table_name LIMIT 300` | `tables = [g.G.nodes[v] for u,v,e in g.G.edges(data=True) if g.G.nodes[u].get("name")=="AEP" and e.get("edge_type")=="USE"]` |
| 表的字段 | `MATCH (t:PhysicalTable {name: "T1"})-[:HAS_COLUMN]->(c:Col) RETURN c.column_name, c.dtype ORDER BY c.order_no LIMIT 300` | `cols = sorted([g.G.nodes[v] for u,v,e in g.G.edges(data=True) if g.G.nodes[u].get("name")=="T1" and e.get("edge_type")=="HAS_COLUMN"], key=lambda x: x.get("order_no",0))` |
| 两应用同名表 | `MATCH (app1:Application {name: "A"})-[:USE]->(t1:PhysicalTable), (app2:Application {name: "B"})-[:USE]->(t2:PhysicalTable) WHERE t1.name = t2.name RETURN t1.name LIMIT 300` | 复杂查询参见 reference-networkx.md |
| 聚合排序 | `MATCH (app:Application)-[:USE]->(t:PhysicalTable) RETURN app.name, count(t) AS cnt ORDER BY cnt DESC LIMIT 300` | `from collections import Counter; cnt = Counter(g.G.nodes[u].get("name") for u,v,e in g.G.edges(data=True) if g.G.nodes[u].get("node_type")=="Application" and e.get("edge_type")=="USE")` |
| 按业务领域筛选 | `MATCH (app:Application {business_domain: "财务管理"}) RETURN app.name, app.app_name_en LIMIT 300` | `apps = [d for _,d in g.G.nodes(data=True) if d.get("node_type")=="Application" and d.get("business_domain")=="财务管理"]` |

---

## 指标问数

基于知识图谱中的指标元数据，生成分析 SQL 并执行查询。

### 工作流程

```
用户问题 → 解析指标 → 查询元数据 → 组装 SQL → 执行查询 → 返回结果
                              ↓
                     来源表 / 维度 / 公式
```

### 指标类型

| 类型 | 说明 | 查询方式 |
|------|------|----------|
| 原子指标 | 直接从来源表获取 | `SELECT metric_col FROM source_table` |
| 派生指标 | 由原子指标计算得出 | 通过 CTE 组合多个原子指标后计算 |

### 通用数据定义

#### 时间字段

- `report_ym`（报告年月）：数据按月更新的拉链字段，格式 `YYYYMM`（如 `202605`）
  - **含义**：`report_ym` 表示数据生成/更新的时间，**不是业务发生时间**。每月系统会生成当月的快照数据，形成数据变化的拉链（同一业务可能在不同 `report_ym` 下有不同的值）
  - **必须条件**：指标维度包含 `report_ym` 时，该字段为**必须条件**。不指定 `report_ym` 会导致 GROUP BY 将不同时期生成的拉链数据合并，产生无意义的汇总结果。一般查询直接指定 `report_ym` 为最新年月即可获取最新数据
  - **历史追溯**：只有在需要查看数据变化历史时（如"该指标过去半年的趋势"），才将 `report_ym` 放入 `dimensions` 或使用范围条件
  - 用户未指定时间时，通过探查确认最新可用月份，**不要假设当前月份有数据**
- 部分表的时间列可能命名为 `ym`，以查询到的 `time_column` 或实际表结构为准

#### 维度字段

| 字段 | 含义 | 示例 |
|------|------|------|
| `sales_unit` | 事业部 | 华东区、华北区 |
| `sales_dept` | 业务中心 | 外滩业务中心、南京路业务中心 |
| `biz_mode` | 业务模式 | — |
| `product_catalog` | 产品目录 | — |
| `customr_group` | 客户组合 | — |

### 查询指标元数据

使用 `govio-cli query -c` 查询图数据库（**必须带 `-c` 标志**）。

> **歧义处理**：当用户表述与指标名称不完全对应时（如"销售额"对应"账单收入"还是"签约额"），**必须停下来向用户列出候选指标及其定义，让用户确认后再继续**。不要自行猜测语义。

#### 查询指标基本信息

```bash
govio-cli query -c 'MATCH (m:Metric {code: "bill_income_amt"}) RETURN m.code, m.name, m.type, m.formula, m.source_layer, m.unit'
```

#### 查询指标来源表

```bash
govio-cli query -c 'MATCH (m:Metric {code: "bill_income_amt"})-[:USES_TABLE]->(t:PhysicalTable) RETURN t.full_table_name, t.name'
```

#### 查询指标维度

```bash
govio-cli query -c 'MATCH (m:Metric {code: "bill_income_amt"})-[d:DIMENSION_USED]->(dim:Dimension) RETURN dim.code, dim.name, d.usage_type'
```

#### 查询指标引用列

```bash
govio-cli query -c 'MATCH (m:Metric {code: "bill_income_amt"})-[:REFERS_COLUMN]->(c:Col) RETURN c.column_name, c.data_type'
```

#### 查询派生指标血缘

```bash
govio-cli query -c 'MATCH (m:Metric {code: "book_to_bill"})-[:DERIVED_FROM]->(up:Metric) RETURN up.code, up.name, up.type'
```

### 组装 SQL

使用 `govio-cli sql build` 命令组装 SQL。接受 JSON 文件作为输入。

#### 调用方式

```bash
# 打印到 stdout
govio-cli sql build -f query.json

# 输出到文件
govio-cli sql build -f query.json -o output.sql

# 从 stdin 读取
cat query.json | govio-cli sql build
```

#### JSON 请求格式

```json
{
  "metrics": [
    {
      "code": "bill_income_amt",
      "name": "当月账单收入",
      "type": "原子",
      "source_table": "dws.income_bill_monthly",
      "time_column": "report_ym"
    }
  ],
  "dimensions": ["sales_unit", "sales_dept"],
  "filters": {
    "report_ym": "2026-05",
    "ym": "2026-04"
  },
  "order_by": null,
  "limit": 100,
  "cte_refs": {}
}
```

#### 字段说明

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `metrics` | list[dict] | 是 | 指标列表，包含 code, name, type, source_table, formula |
| `dimensions` | list[str] | 否 | 分组维度字段，如 `["sales_unit", "sales_dept"]` |
| `filters` | dict[str, str] | 否 | 过滤条件，如 `{"report_ym": "2026-05"}` |
| `order_by` | str | 否 | 排序字段，如 `"metric_value DESC"` |
| `limit` | int | 否 | 返回行数限制，默认 100 |
| `cte_refs` | dict[str, str] | 否 | 已加载 DataFrame 的 CTE 引用 |

#### 指标对象字段

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `code` | str | 是 | 指标编码 |
| `name` | str | 是 | 指标名称 |
| `type` | str | 是 | `"原子"` 或 `"派生"` |
| `source_table` | str | 原子必填 | 来源表，如 `"dws.income_bill_monthly"` |
| `actual_column` | str | 否 | 表中实际列名（来自 REFERS_COLUMN），与 `code` 不同时**必须填写** |
| `formula` | str | 派生必填 | 计算公式，如 `"signed_amt / bill_income_amt"` |
| `time_column` | str | 否 | 时间字段名，默认 `"report_ym"` |

### 确认数据范围

当 filters 包含时间条件（`report_ym`、`ym` 等）且无法从 `metrics_index.md` 确认最新周期时，先探查数据范围：

```bash
govio-cli observe load --name range_check --datasource <ds> --sql "SELECT DISTINCT {time_column} FROM {source_table} ORDER BY {time_column} DESC LIMIT 10" -o /tmp/range.json
```

**必须带 `-o` 输出到文件**，否则只能拿到行数/列数，看不到实际数据值。

确认最新可用周期后再组装最终 SQL。**不要假设当前月份有数据。**

> **未来月份注意**：如果最新周期超过当前月份（如当前 6 月但数据到 12 月），应向用户说明该数据为预测/计划值，并确认是否使用。

### 执行查询

> **重要**: 执行查询前必须获得用户许可。如果用户未明确表示可以执行，应先将组装好的 SQL 展示给用户，问询确认后再执行。

使用 `govio-cli observe load` 执行 SQL 并加载为 DataFrame：

```bash
# 最终结果加载（输出到 JSON 文件）
govio-cli observe load --name <df_name> --datasource <ds_name> --sql "<sql>" -o <output.json>

# 前置辅助数据集加载（仅持久化，不输出文件）
govio-cli observe load --name <df_name> --datasource <ds_name> --sql "<sql>"

# 汇总统计数据，复用之前加载的dataframe进行加工，生成新的datafame
govio-cli observe load --name <df_name> --memory --sql "<sql>"
```

最终结果的加载使用 `-o` 参数输出数据内容到 JSON 文件；前置的辅助数据集（如 CTE 场景中的中间数据）仅加载持久化，不使用 `-o`。

> **DataFrame 机制说明**：`observe load` 将查询结果保存在本地缓存，**不会注册为 DuckDB 表**。可以通过`govio-cli observe load --name <df_name> --memory `对已加载的 DataFrame进行二次加工。如需查看数据，必须使用 `-o` 输出到 JSON 文件后读取。

#### 命名规范

DataFrame 名称格式：`metric_{指标编码}_{时间}`

示例：
- `metric_bill_income_202605`
- `metric_book_to_bill_202605`

#### 数据源

通过 `govio-cli observe info --datasource` 获取可用数据源，或询问用户指定。

### CTE 组合查询

当需要组合多次查询结果时，使用 CTE (Common Table Expression) 引用已加载的 DataFrame：

#### 场景：环比分析

```bash
# 1. 查询当月数据并加载
govio-cli sql build -f current.json -o current.sql
govio-cli observe load --name metric_current --datasource dw --sql "$(cat current.sql)"

# 2. 查询上月数据，通过 cte_refs 引用当月结果
# compare.json 中 cte_refs 包含: {"metric_current": "<当月SQL>"}
govio-cli sql build -f compare.json -o compare.sql
```

#### 场景：多指标组合

先加载基础指标，再通过 CTE 组合计算派生指标：

```bash
# 1. 加载签约数据
govio-cli observe load --name signed_data --datasource dw --sql "SELECT sales_unit, SUM(signed_amt) as signed_amt FROM dws.signed_monthly WHERE report_ym = '2026-05' GROUP BY sales_unit"

# 2. 加载账单数据
govio-cli observe load --name bill_data --datasource dw --sql "SELECT sales_unit, SUM(bill_income_amt) as bill_income_amt FROM dws.income_bill_monthly WHERE report_ym = '2026-05' GROUP BY sales_unit"

# 3. 组合计算签约覆盖率（通过 cte_refs 引用已加载的 DataFrame）
```

### 常见指标查询模板

#### 按时间趋势

```json
{
  "metrics": [
    {"code": "bill_income_amt", "name": "当月账单收入", "type": "原子", "source_table": "dws.income_bill_monthly"}
  ],
  "dimensions": ["report_ym"],
  "order_by": "report_ym"
}
```

#### 按部门排名

```json
{
  "metrics": [
    {"code": "bill_income_amt", "name": "当月账单收入", "type": "原子", "source_table": "dws.income_bill_monthly"}
  ],
  "dimensions": ["sales_unit", "sales_dept"],
  "filters": {"report_ym": "2026-05"},
  "order_by": "bill_income_amt DESC",
  "limit": 10
}
```

#### 派生指标（签约覆盖率）

```json
{
  "metrics": [
    {"code": "signed_amt", "name": "签约额", "type": "原子", "source_table": "dws.signed_monthly"},
    {"code": "bill_income_amt", "name": "账单收入", "type": "原子", "source_table": "dws.income_bill_monthly"},
    {"code": "book_to_bill", "name": "签约覆盖率", "type": "派生", "formula": "signed_amt / bill_income_amt"}
  ],
  "dimensions": ["sales_unit"],
  "filters": {"report_ym": "2026-05"}
}
```

### 输出格式

#### 单指标结果

```
查询结果: 当月账单收入 (2026-05)

| 销售单元 | 销售部门 | 指标值 |
|---------|---------|-------|
| 华东区   | 上海分部 | 1,234 |
| 华北区   | 北京分部 | 2,345 |

共 2 条记录
```

#### 多指标结果

```
查询结果: 指标对比 (2026-05)

| 销售单元 | 指标名称 | 指标值 |
|---------|---------|-------|
| 华东区   | 当月签约额 | 1,000 |
| 华东区   | 当月账单收入 | 1,234 |
| 华北区   | 当月签约额 | 2,000 |
| 华北区   | 当月账单收入 | 2,345 |

共 4 条记录
```

### 指标概览

`../govio/assets/metrics_index.md` 中包含所有指标的索引：

#### 原子指标

| 编码 | 名称 | 来源层 | 单位 |
|------|------|--------|------|
| bill_income_amt | 当月账单收入 | DWS | 万元 |
| signed_amt | 当月销售签约额 | DWS | 万元 |
| forecast_income_amt | 预计当月账单收入 | DWS | 万元 |
| ... | ... | ... | ... |

#### 派生指标

| 编码 | 名称 | 公式 | 单位 |
|------|------|------|------|
| book_to_bill | 签约覆盖率 | signed_amt / bill_income_amt | 倍 |
| burndown_amt | 存量消耗额 | forecast_income_amt - risk_amt | 万元 |
| ... | ... | ... | ... |

---

## 名称解析

当 prompt 包含中文系统名（如"报价单中心系统""薪税系统"），**必须先 Grep 确认标准英文代码**：

- **networkx 后端**：Grep 搜索 `../govio/assets/names/node_names.md`
- **falkordb 后端**：Grep 搜索 `../govio/assets/names/` 下所有 `*.md` 文件，或先用 Glob 列出文件名定位（格式：`{应用名}_{缩写}.md`，如 `薪税生产系统_PAYPRO.md`）

## 查询终止策略

对同一语义目标最多 **3 次尝试**，逐步放宽：

1. 精确匹配（`name: "银行"`）
2. 模糊/包含匹配（`name CONTAINS "银行"`）
3. 同义词扩展（`金融`、`bank` 等）

3 次后仍无结果，**必须停止**并告知"知识图谱中未找到相关数据"。

## 输出纪律

- 中间步骤**不要输出思考过程**，直接执行工具调用
- 只在最终结果时输出格式化的表格
- 出错时简要说明原因和修正动作，不要展开分析推理

## 排除场景

以下场景**不要**触发本技能：
- 数据比对、迁移验证 → 使用 `govio-observe`
- 知识图谱维护（同步、导出、推荐） → 使用 `govio-meta`
- 数据探查、EDA → 使用 `govio-eda`
- 数据迁移脚本编写
- 代码调试/修复
- 功能模块开发

## 资源文件

```
../govio/assets/            # 共享资源（位于父 govio skill 目录下）
├── schema.md               # 图模式（Step 0 必读，仅读一次）
├── metrics_index.md        # 指标索引
├── ontology.gml            # GML 元模型数据（NetworkX 后端）
└── names/                  # 名称映射
     ├── node_names.md      #   (networkx) 全部节点名称汇总
     └── *.md               #   (falkordb) 按应用分文件

本 Skill 目录下:
├── reference-falkordb.md   # FalkorDB Cypher 深度参考
└── reference-networkx.md   # NetworkX Python 深度参考
```
