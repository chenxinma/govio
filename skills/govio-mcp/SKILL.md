---
name: govio-mcp
description: Govio MCP 数据治理主控 Skill。当用户提出数据治理需求时触发，包括：数据探查（查看数据源、探索表结构、发现表间关系）、数据比对、数据迁移验证、数据质量检查等。负责澄清需求、编写执行计划、协调子 Skill 完成数据治理目标。不直接执行操作，而是通过计划和子 Skill 组合完成任务。
---

# Govio MCP 数据治理主控

本 Skill 是 Govio MCP 数据治理的入口点，负责将用户的自然语言需求转化为结构化的执行计划，并协调各个子 Skill 完成目标。

## 触发场景

| 场景 | 典型请求示例 |
|---|---|
| **数据探查** | "查看有哪些数据源"、"分析表结构"、"发现表之间的关系" |
| **数据比对** | "比对两个表的数据"、"验证数据迁移结果" |
| **关系探测** | "分析表之间的关联"、"找出外键关系" |
| **数据质量** | "检查数据一致性"、"发现数据差异" |

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

| 步骤 | 子 Skill | 功能 |
|---|---|---|
| 数据集操作 | `mcp-dataset-ops` | 数据源查看、数据加载、DataFrame 管理、资源释放 |
| 探查关系 | `mcp-explore-relations` | 发现表间关系 |
| 比对数据 | `mcp-compare-dfs` | 比对两个数据集 |

`mcp-dataset-ops` 包含四个操作：
- `list_ds()` - 列出可用数据源
- `list_dfs()` - 列出已加载的 DataFrame
- `load_df()` - 执行 SQL 加载数据
- `release_df()` - 清理内存

## 使用模式

### 模式 A: 简单任务（1-2 步）

直接调用子 Skill，无需编写 Plan。

**示例对话：**
> 用户: "帮我看一下有哪些数据源"
> 
> Agent: "我来帮你查看可用的数据源。"
> 
> [调用 mcp-dataset-ops.list_ds]

### 模式 B: 数据探查任务

探查数据源和表结构，通常需要多步操作。

**示例对话：**
> 用户: "帮我分析一下客户相关表的结构和关系"
> 
> Agent: "好的，我来帮你进行数据探查：
> 1. 先查看可用数据源
> 2. 加载相关表数据
> 3. 分析表间关系"
> 
> [调用 mcp-dataset-ops.list_ds]
> → 返回数据源列表
> 
> [调用 mcp-dataset-ops.load_df] (多次)
> → 加载 customers, orders, addresses 等表
> 
> [调用 mcp-explore-relations]
> → 发现外键关系和相似列

### 模式 C: 复杂任务（3 步以上）

必须编写 Plan，然后逐步执行。

**示例对话：**
> 用户: "帮我验证客户数据迁移"
> 
> Agent: "好的，我需要先澄清几个细节：
> 1. 源系统和目标系统分别是什么？
> 2. 要比对哪些表？
> 3. 用什么字段作为比对键？
> 4. 匹配率达到多少算通过？"
> 
> [用户回答后]
> 
> Agent: "明白了，我来编写执行计划..."
> 
> [生成 Plan 文件]
> 
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

**使用 Skill**: mcp-dataset-ops.load_df
**输入**: 
- datasource: legacy_db
- name: legacy_customers
- sql: SELECT customer_id, name, email FROM customers

- [ ] 调用 mcp-dataset-ops.load_df 加载数据
- [ ] 验证加载成功（rows > 0）

## Task 2: 加载目标数据

**使用 Skill**: mcp-dataset-ops.load_df
**输入**: 
- datasource: new_db
- name: new_customers
- sql: SELECT customer_id, name, email FROM customers

- [ ] 调用 mcp-dataset-ops.load_df 加载数据
- [ ] 验证加载成功

## Task 3: 比对数据

**使用 Skill**: mcp-compare-dfs
**输入**:
- source_df: legacy_customers
- target_df: new_customers
- join_columns: [customer_id]

- [ ] 调用 mcp-compare-dfs 执行比对
- [ ] 验证 match_rate >= 99%
- [ ] 记录差异详情

## Task 4: 释放资源

**使用 Skill**: mcp-dataset-ops.release_df
**输入**: [legacy_customers, new_customers]

- [ ] 释放 legacy_customers
- [ ] 释放 new_customers
```

### 数据探查计划示例

```markdown
# 数据治理计划: 客户模块数据探查

**目标**: 探查客户相关表的结构和关系
**数据源**: prod_db
**成功标准**: 获得完整的表结构和关系图谱
**创建时间**: 2026-03-28

---

## Task 1: 查看数据源

**使用 Skill**: mcp-dataset-ops.list_ds

- [ ] 调用 mcp-dataset-ops.list_ds
- [ ] 确认 prod_db 数据源可用

## Task 2: 加载相关表数据

**使用 Skill**: mcp-dataset-ops.load_df

- [ ] 加载 customers 表
- [ ] 加载 orders 表
- [ ] 加载 addresses 表
- [ ] 验证加载成功

## Task 3: 分析表结构

**使用 Skill**: mcp-dataset-ops.list_dfs

- [ ] 查看已加载 DataFrame 的结构
- [ ] 记录各表的行数、列数

## Task 4: 探查表间关系

**使用 Skill**: mcp-explore-relations
**输入**:
- dataframes: [customers, orders, addresses]

- [ ] 调用 mcp-explore-relations
- [ ] 记录发现的外键关系
- [ ] 记录相似列

## Task 5: 释放资源

**使用 Skill**: mcp-dataset-ops.release_df

- [ ] 释放所有加载的 DataFrame
```

## 子 Skill 调用格式

调用子 Skill 时，使用以下格式：

```
[调用 mcp-dataset-ops.load_df]
参数:
- datasource: prod_db
- name: customers
- sql: SELECT * FROM customers
```

## 与 Govio Skill 的协作

当需要查询知识图谱中的元数据时，可以结合使用 `govio` skill：

```
用户: "帮我比对 CRM 和 ERP 的客户表"

Agent: "我先查询一下知识图谱，看看这两个系统的表结构..."

[调用 govio skill 查询表结构]
→ 获得表名、字段信息

Agent: "好的，现在编写执行计划..."

[生成 Plan，包含 mcp-* skill 调用]
```

## 注意事项

1. **不要假设**: 总是先澄清需求，不要猜测用户的意图
2. **小步快跑**: 复杂任务分解为小步骤，每步完成后确认
3. **及时反馈**: 每个 Task 完成后告知用户进展
4. **资源清理**: 确保最后释放所有 DataFrame
5. **记录日志**: 重要操作记录到 `docs/govio/logs/`
