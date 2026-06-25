---
name: govio-observe
description: Govio 数据治理主控 Skill。当用户提出数据治理需求时触发，包括：数据探查（查看数据源、探索表结构、发现表间关系）、数据比对、数据迁移验证、数据质量检查等。负责澄清需求、编写执行计划、协调子 Skill 完成数据治理目标。不直接执行操作，而是通过计划和子 Skill 组合完成任务。
---

# Govio 数据治理主控

本 Skill 是 Govio 数据治理的入口点，负责将用户的自然语言需求转化为结构化的执行计划，并协调各个子 Skill 完成目标。

## 触发场景

| 场景 | 典型请求示例 |
|---|---|
| **数据探查** | "查看有哪些数据源"、"分析表结构"、"发现表之间的关系" |
| **数据比对** | "比对两个表的数据"、"验证数据迁移结果" |
| **关系探测** | "分析表之间的关联"、"找出外键关系" |
| **数据质量** | "检查数据一致性"、"发现数据差异" |
| **数据可视化** | "画个柱状图看看销售趋势"、"用折线图展示数据变化" |

## 前置条件

所有数据探查操作通过 `govio-cli observe` 命令组执行。确保：

1. 已运行 `govio-cli onboard` 完成初始化配置
2. 通过 `govio-cli observe show-datasource` 确认数据源已配置
3. 通过 `uvx --from skills/govio/assets/govio-*.whl` 自动解析依赖，无需手动安装

## 工作流程

```
用户请求 → 澄清需求 → 编写计划 → 执行步骤 → 完成目标
                ↓           ↓
         多问问题      分解为子 Skill 调用
```

## 核心原则

### 1. 澄清先于执行

**永远不要直接开始执行。** 先通过提问澄清：

- **数据源**: 从哪个数据源抽取？目标是什么？
- **比对对象**: 要比对哪些表/数据？
- **成功标准**: 什么样的结果算"完成"？
- **输出要求**: 需要什么格式的报告？

### 2. 计划驱动

复杂任务（3 步以上）必须编写 Plan：

```markdown
# 数据治理计划: [任务名称]

**目标**: [一句话描述]
**数据源**: [源系统] → [目标系统]
**成功标准**: [可验证的条件]

## Task 1: [步骤名称]
- [ ] 子步骤...

## Task 2: [步骤名称]
- [ ] 子步骤...
```

Plan 保存位置: `docs/govio/plans/YYYY-MM-DD-[task-name].md`

### 3. 子 Skill 组合

每个数据治理步骤对应一个子 Skill：

| 步骤 | 子 Skill | CLI 命令 |
|---|---|---|
| 数据集操作 | `observe-dataset-ops` | `govio-cli observe show-datasource / list / load / release` |
| 探查关系 | `observe-explore-relations` | `govio-cli observe explore` |
| 比对数据 | `observe-compare-dfs` | `govio-cli observe compare` |
| 数据可视化 | `observe-chart` | `govio-cli observe chart` |

### chart 命令说明

从已加载的 DataFrame 生成 PNG 图表,支持柱状图(`bar`)和折线图(`line`)。

```bash
govio-cli observe chart --name <df_name> --type {bar|line} --x <x_col> --y <y_col> -o <output.png>
```

所有参数必填:
- `--name`: ObserveStore 中已加载的 DataFrame 名称
- `--type`: `bar` 或 `line`
- `--x`: X 轴列名(分类轴/时序轴)
- `--y`: Y 轴列名(数值轴,单列)
- `-o/--output`: 输出 PNG 路径

成功返回 `{"success": true, "output": "<abs_path>"}`;DataFrame 或列不存在时返回 `{"success": false, "error": "..."}`。

内置中文字体回退链(Noto Sans CJK SC / WenQuanYi Zen Hei / SimHei / Microsoft YaHei / Arial Unicode MS),自动适配主流系统;同时修复负号显示问题。第一版单系列,不支持分组。

**示例:**

```bash
# 先加载数据
govio-cli observe load --name sales --datasource mysql --sql "SELECT region, revenue FROM sales_summary"

# 画柱状图
govio-cli observe chart --name sales --type bar --x region --y revenue -o /tmp/sales_bar.png

# 画折线图(时序数据)
govio-cli observe load --name monthly --datasource mysql --sql "SELECT month, revenue FROM monthly_revenue"
govio-cli observe chart --name monthly --type line --x month --y revenue -o /tmp/monthly_trend.png
```

`observe-dataset-ops` 包含四个操作，对应 CLI 命令：
- `list_ds()` → `govio-cli observe show-datasource`
- `list_dfs()` → `govio-cli observe list`
- `load_df()` → `govio-cli observe load --name <name> --datasource <ds> --sql <sql>`
- `release_df()` → `govio-cli observe release --name <name>`

### CLI命令说明

所有参数均为关键字参数，顺序无关：
- `govio-cli observe load --name <name> --datasource <ds> --sql <sql>`
- `govio-cli observe release --name <name>`

### load 加载的两种方式

**方式 1：直接指定 SQL**

当用户已明确知道要查询的表和字段时，直接传入 SQL：

```bash
govio-cli observe load --name customers --datasource mysql --sql "SELECT customer_id, name, email FROM customers"
```

**方式 2：通过 govio skill 查询元数据后生成 SQL**

当用户不确定表结构时，先用 `govio` skill 查询知识图谱获取元数据，再生成 SQL：

```
用户: "加载 CRM 系统的客户表数据"

Agent: "我先查询知识图谱获取 CRM 的表结构..."

[调用 govio skill 查询]
govio-cli query --query "MATCH (app:Application {name: 'CRM'})-[:USE]->(t:PhysicalTable)-[:HAS_COLUMN]->(c:Col) RETURN t.full_table_name, c.column_name, c.dtype ORDER BY t.full_table_name, c.order_no"

→ 获得表名和字段信息

Agent: "根据元数据，生成加载 SQL..."

[执行 load 命令]
govio-cli observe load --name crm_customers --datasource crm_db --sql "SELECT customer_id, name, email, phone FROM crm.customers"
```

## 使用模式

### 模式 A: 简单任务（1-2 步）

直接调用子 Skill，无需编写 Plan。

**示例对话：**
> 用户: "帮我看一下有哪些数据源"

> Agent: "我来帮你查看可用的数据源。"

> [执行 govio-cli observe show-datasource]

### 模式 B: 数据探查任务

探查数据源和表结构，通常需要多步操作。

**示例对话：**
> 用户: "帮我分析一下客户相关表的结构和关系"

> Agent: "好的，我来帮你进行数据探查：
> 1. 先查看可用数据源
> 2. 加载相关表数据
> 3. 分析表间关系"

> [执行 govio-cli observe show-datasource]
> → 返回数据源列表

> [执行 govio-cli observe load] (多次)
> → 加载 customers, orders, addresses 等表

> [执行 govio-cli observe explore]
> → 发现外键关系和相似列

### 模式 C: 复杂任务（3 步以上）

必须编写 Plan，然后逐步执行。

**示例对话：**
> 用户: "帮我验证客户数据迁移"

> Agent: "好的，我需要先澄清几个细节：
> 1. 源系统和目标系统分别是什么？
> 2. 要比对哪些表？
> 3. 用什么字段作为比对键？
> 4. 匹配率达到多少算通过？"

> [用户回答后]

> Agent: "明白了，我来编写执行计划..."

> [生成 Plan 文件]

> Agent: "计划已生成，现在开始执行 Task 1..."

## 需求澄清问题清单

### 数据探查类
- 需要探查哪些数据源？
- 需要分析哪些表？
- 是否需要发现表间关系？
- 需要了解表结构还是数据内容？
- 是否需要采样数据查看？

### 数据抽取类
- 数据源名称是什么？
- 需要抽取哪些表？
- 需要过滤条件吗？
- DataFrame 命名有什么偏好？
- **SQL 生成方式**：直接指定 SQL，还是通过 `govio` skill 查询元数据后生成？

### 数据比对类
- 源系统和目标系统分别是什么？
- 要比对哪些表？
- 用什么字段作为 join key？
- 匹配率预期是多少？
- 需要输出差异详情吗？

### 关系探测类
- 要分析哪些表之间的关系？
- 需要输出什么格式的关系图谱？

### 通用问题
- 有时间和资源限制吗？
- 需要什么格式的报告？
- 谁需要查看结果？

## Plan 编写规范

### 头部格式

```markdown
# 数据治理计划: [任务名称]

**目标**: [一句话描述]
**数据源**: [源] → [目标]
**成功标准**: [可验证的条件]
**创建时间**: YYYY-MM-DD

---
```

### Task 格式

```markdown
## Task N: [步骤名称]

**使用 Skill**: [子 Skill 名称]
**CLI 命令**: [对应的 govio-cli 命令]
**输入**: [参数]
**预期输出**: [结果]

- [ ] [具体执行步骤]
```

### 示例

```markdown
# 数据治理计划: 客户数据迁移验证

**目标**: 验证从 legacy_db 到 new_db 的客户数据迁移完整性
**数据源**: legacy_db → new_db
**成功标准**: 匹配率 >= 99%，无结构差异
**创建时间**: 2026-03-26

---

## Task 1: 加载源数据

**使用 Skill**: observe-dataset-ops.load_df
**CLI 命令**: `govio-cli observe load --name legacy_customers --datasource legacy_db --sql "SELECT customer_id, name, email FROM customers"`

- [ ] 执行加载命令
- [ ] 验证加载成功（rows > 0）

## Task 2: 加载目标数据

**使用 Skill**: observe-dataset-ops.load_df
**CLI 命令**: `govio-cli observe load --name new_customers --datasource new_db --sql "SELECT customer_id, name, email FROM customers"`

- [ ] 执行加载命令
- [ ] 验证加载成功

## Task 3: 比对数据

**使用 Skill**: observe-compare-dfs
**CLI 命令**: `govio-cli observe compare --source legacy_customers --target new_customers --join-columns customer_id`

- [ ] 执行比对命令
- [ ] 验证 match_rate >= 99%
- [ ] 记录差异详情

## Task 4: 释放资源

**使用 Skill**: observe-dataset-ops.release_df

- [ ] `govio-cli observe release --name legacy_customers`
- [ ] `govio-cli observe release --name new_customers`
```

## 与 Govio Skill 的协作

当需要查询知识图谱中的元数据时，可以结合使用 `govio` skill：

### 场景 1：查询表结构生成 SQL

```
用户: "帮我比对 CRM 和 ERP 的客户表"

Agent: "我先查询一下知识图谱，看看这两个系统的表结构..."

[调用 govio skill 查询表结构]
govio-cli query --query "MATCH (app:Application)-[:USE]->(t:PhysicalTable)-[:HAS_COLUMN]->(c:Col) WHERE app.name IN ['CRM', 'ERP'] RETURN app.name, t.full_table_name, c.column_name, c.dtype ORDER BY app.name, t.full_table_name, c.order_no"

→ 获得表名、字段信息

Agent: "根据查询结果，CRM 有 customers 表，ERP 有 customer_master 表。
现在生成加载 SQL..."

[生成并执行 load 命令]
govio-cli observe load --name crm_customers --datasource crm --sql "SELECT customer_id, name, email FROM customers"
govio-cli observe load --name erp_customers --datasource erp --sql "SELECT id, full_name, email_addr FROM customer_master"
```

### 场景 2：发现可用表后选择加载

```
用户: "帮我看看销售系统有哪些表可以用"

Agent: "我先查询知识图谱中销售系统的数据资产..."

[调用 govio skill 查询]
govio-cli query --query "MATCH (app:Application {name: '销售系统'})-[:USE]->(t:PhysicalTable) RETURN t.name, t.full_table_name"

→ 返回: orders, order_items, products, customers

Agent: "销售系统有以下表：orders, order_items, products, customers
你需要加载哪些表？"

用户: "加载 orders 和 customers"

[执行 load 命令]
govio-cli observe load --name sales_orders --datasource sales_db --sql "SELECT * FROM orders"
govio-cli observe load --name sales_customers --datasource sales_db --sql "SELECT * FROM customers"
```

## 注意事项

1. **不要假设**: 总是先澄清需求，不要猜测用户的意图
2. **小步快跑**: 复杂任务分解为小步骤，每步完成后确认
3. **及时反馈**: 每个 Task 完成后告知用户进展
4. **资源清理**: 确保最后释放所有 DataFrame
5. **记录日志**: 重要操作记录到 `docs/govio/logs/`
