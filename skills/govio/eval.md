# Govio Skill Eval 评估测试用例

---

依据 [eval-skill.md](../../docs/eval-skill.md) 框架设计，将"感觉更好"转化为可量化分数。

## 1. 成功标准定义

### 1.1 结果目标（Outcome）

| 编号 | 检查项 | 量化方式 |
|------|--------|----------|
| O-1 | 查询返回正确结果 | 人工比对知识图谱原始数据，结果集完全匹配 = 1 分，部分匹配 = 0.5 分，错误/空 = 0 |
| O-2 | SQL 生成可执行且语义正确 | SQL 语法合法 = 0.5 分，执行后返回预期数据 = 1 分 |
| O-3 | 表字段比较结果完整 | 列出所有差异项 = 1 分，遗漏 > 20% = 0.5 分，遗漏 > 50% = 0 |
| O-4 | 回答与问题直接相关 | 回答直接回应提问 = 1 分，偏题但相关 = 0.5 分，无关 = 0 |

### 1.2 过程目标（Process）

| 编号 | 检查项 | 量化方式 |
|------|--------|----------|
| P-1 | 首次行动是读取 `assets/schema.md` | 是 = 1 分，否 = 0 |
| P-2 | 根据 `~/.govio/config.yaml` 选择后端 | 是 = 1 分，否 = 0 |
| P-3 | 优先使用 `govio-query` 而非手动 Python 代码 | 使用 govio-query = 1 分，手动导入 = 0.5 分，两者都没用 = 0 |
| P-4 | 使用 `Grep` 查询 `node_names.md` 获取标准名称 | 需要名称解析时执行 = 1 分，未执行 = 0 |
| P-5 | 查询结果行数 ≤ 300 | 是 = 1 分，否 = 0 |
| P-6 | 正确使用 `column_name` 而非 `name` 引用 Col 节点列名 | 正确 = 1 分，错误 = 0 |

### 1.3 风格目标（Style）

| 编号 | 检查项 | 量化方式 |
|------|--------|----------|
| S-1 | Cypher 查询使用双引号包裹参数 | 是 = 1 分，否 = 0 |
| S-2 | Cypher 查询以 `MATCH` 开头 | 是 = 1 分，否 = 0 |
| S-3 | 输出格式清晰（表格/列表），非原始 JSON 倾倒 | 结构化 = 1 分，半结构化 = 0.5 分，原始 = 0 |
| S-4 | 中文回答，技术术语保留英文 | 符合 = 1 分，全英文 = 0.5 分，全中文含翻译术语 = 0.5 分 |

### 1.4 效率目标（Efficiency）

| 编号 | 检查项 | 量化方式 |
|------|--------|----------|
| E-1 | 未重复读取 schema.md | 仅读 1 次 = 1 分，2 次 = 0.5 分，> 2 次 = 0 |
| E-2 | 未执行无关命令 | 无冗余命令 = 1 分，有 1-2 条冗余 = 0.5 分，> 2 条 = 0 |
| E-3 | 单次查询完成而非多次碎片查询 | 1 次查询 = 1 分，2 次 = 0.5 分，> 2 次 = 0（合理分步除外） |
| E-4 | 未不必要地读取 SKILL.md / reference-*.md | 未读 = 1 分，读取 = 0（除非后端切换场景） |

---

## 2. 测试提示词集

### 2.1 显式调用（直接提及数据治理 / 元数据 / 知识图谱）

| ID | 提示词 | 测试重点 |
|----|--------|----------|
| T-01 | 查询 CRM 应用使用了哪些物理表 | 基本元数据查询，P-1/P-2/P-3/O-1 |
| T-02 | CUSTOMER 表有哪些字段？列出字段名和数据类型 | Col 节点查询，P-6（column_name vs name） |
| T-03 | 比较 CUSTOMER 表和 ORDER 表的字段差异 | 表字段比较核心能力，O-3 |
| T-04 | 生成一个查询 CRM 客户及订单的 SQL | SQL 生成能力，O-2 |
| T-05 | 数据标准有多少条？列出名称和状态 | Standard 节点查询，跨节点类型 |
| T-06 | 查询所有应用及其使用的表数量，按表数量降序排列 | 聚合查询，O-1 |

### 2.2 隐式调用（描述场景但不提数据治理关键词）

| ID | 提示词 | 测试重点 |
|----|--------|----------|
| T-07 | 我想知道 CRM 系统涉及哪些数据表 | 隐式元数据查询，触发判定 |
| T-08 | 帮我看看 CUSTOMER 和 ORDER 这两张表有什么不同 | 隐式字段比较 |
| T-09 | 写一段 SQL，从 CRM 的客户表关联订单表查出客户名和订单金额 | 隐式 SQL 生成 |
| T-10 | 哪些应用的表最多？ | 隐式聚合分析 |

### 2.3 上下文调用（带噪声的真实场景）

| ID | 提示词 | 测试重点 |
|----|--------|----------|
| T-11 | 我们正在做数据治理专项，需要梳理 CRM 系统的数据资产，先帮我查一下 CRM 下面有哪些表，然后看看有没有关联到数据标准 | 多步复合任务，流程衔接 |
| T-12 | 业务方反馈 ORDER 表和 CUSTOMER 表的数据对不上，请帮我对比这两个表的结构差异，重点看关联字段 | 带业务上下文的比较，E-3（应合并查询） |
| T-13 | 我需要给领导汇报数据标准落标情况，哪些表的字段已经符合数据标准了？ | 复杂路径查询（Col→Standard），输出格式 S-3 |

### 2.4 负向控制（不应触发 govio skill 的请求）

| ID | 提示词 | 期望结果 |
|----|--------|----------|
| T-14 | 帮我写一个 Python 快速排序算法 | 不应触发 govio skill |
| T-15 | 解释一下什么是知识图谱 | 不应触发 govio skill（纯概念问题，非操作） |
| T-16 | 帮我查一下系统里有没有 Redis 的配置 | 不应触发 govio skill（基础设施问题，非数据治理） |

---

## 3. 确定性评分检查（基于 trace/产物）

运行 skill 后，从对话 trace 中提取以下信号进行自动评分：

```python
# 评分脚本伪代码
def score_run(trace: list[Event]) -> dict:
    scores = {}

    # P-1: 首次行动读取 schema.md
    first_read = next(e for e in trace if e.type == "tool_call" and e.tool in ("Read", "Grep", "Bash"))
    scores["P-1"] = 1 if "schema.md" in first_read.target else 0

    # P-2: 读取 config.yaml 获取后端配置
    scores["P-2"] = 1 if any("config.yaml" in e.target for e in trace if e.type == "tool_call") else 0

    # P-3: 优先使用 govio-query
    query_calls = [e for e in trace if "govio-query" in e.target]
    python_calls = [e for e in trace if "python -c" in e.target and "govio" in e.target]
    if query_calls:
        scores["P-3"] = 1
    elif python_calls:
        scores["P-3"] = 0.5
    else:
        scores["P-3"] = 0

    # P-5: 查询结果 ≤ 300 行
    outputs = [e for e in trace if e.type == "tool_result"]
    scores["P-5"] = 1 if all(len(e.content.splitlines()) <= 300 for e in outputs) else 0

    # P-6: Col 节点使用 column_name
    cypher_or_code = " ".join(e.target for e in trace if e.type == "tool_call")
    scores["P-6"] = 0 if "Col" in cypher_or_code and "c.name" in cypher_or_code and "column_name" not in cypher_or_code else 1

    # S-1: Cypher 双引号
    if any("MATCH" in e.target for e in trace):
        scores["S-1"] = 1 if all("{'" not in e.target for e in trace if "MATCH" in e.target) else 0

    # S-2: Cypher 以 MATCH 开头
    scores["S-2"] = 1 if all(
        e.target.strip().startswith("MATCH")
        for e in trace if "MATCH" in e.target and e.tool == "Bash"
    ) else 0

    # E-1: schema.md 读取次数
    schema_reads = sum(1 for e in trace if "schema.md" in getattr(e, "target", ""))
    scores["E-1"] = {1: 1, 2: 0.5}.get(schema_reads, 0)

    # E-4: 未不必要读取 SKILL.md / reference-*.md
    ref_reads = sum(1 for e in trace if any(x in getattr(e, "target", "") for x in ("SKILL.md", "reference-")))
    scores["E-4"] = 0 if ref_reads > 0 else 1

    return scores
```

---

## 4. 评分标准（Rubric）——确定性规则难以量化的质量维度

### 4.1 回答质量评分

| 维度 | 5 分 | 3 分 | 1 分 |
|------|------|------|------|
| **准确性** | 结果与知识图谱完全一致 | 有 1-2 处小偏差 | 重大错误或幻觉 |
| **完整性** | 包含所有相关信息，无遗漏 | 覆盖主要信息，少量次要遗漏 | 遗漏关键信息 |
| **可操作性** | 回答可直接用于下一步工作 | 需要少量补充查询 | 需要大量返工 |

### 4.2 SQL 生成质量评分

| 维度 | 5 分 | 3 分 | 1 分 |
|------|------|------|------|
| **语法正确性** | 可直接执行 | 小修可执行 | 语法错误 |
| **语义匹配** | 完全匹配需求 | 基本匹配但字段/条件偏差 | 与需求不符 |
| **表关联合理性** | 基于图谱中 RELATES_TO 关系 | 合理推断但未引用图谱关系 | 凭空猜测关联 |

### 4.3 表字段比较质量评分

| 维度 | 5 分 | 3 分 | 1 分 |
|------|------|------|------|
| **差异识别** | 列出所有字段级差异（名称、类型、长度） | 列出主要差异，遗漏次要属性 | 仅列出表名差异 |
| **呈现方式** | 对比表格，清晰标注差异类型 | 文字描述，可理解 | 原始数据倾倒 |
| **关联字段分析** | 指出关联字段及其类型匹配情况 | 提到关联字段但未分析 | 未分析关联字段 |

---

## 5. 按提示词的期望结果与评分映射

### T-01: 查询 CRM 应用使用了哪些物理表

**期望过程**：
1. 读取 `schema.md` → 2. 使用 `govio-cli query` 执行查询（自动从 `~/.govio/config.yaml` 读取后端配置） → 3. 格式化输出

**期望输出**：CRM 应用关联的所有 PhysicalTable 节点的 name / full_table_name 列表

**自动评分项**：P-1=1, P-2=1, P-3=1, P-5=1, S-3=1, E-1=1, E-4=1

**Rubric 项**：准确性 5, 完整性 5, 可操作性 5

---

### T-02: CUSTOMER 表有哪些字段

**期望过程**：
1. 读取 `schema.md` → 2. Grep `node_names.md` 确认 CUSTOMER 标准名 → 3. 使用 `govio-query` 查询 → 4. 使用 `column_name` 属性

**关键陷阱**：SKILL.md 明确要求 Col 节点使用 `column_name` 而非 `name`，此用例专门验证 P-6

**自动评分项**：P-1=1, P-4=1, P-6=1, S-1=1（FalkorDB 时）, S-2=1

**Rubric 项**：准确性 5, 完整性 5

---

### T-03: 比较 CUSTOMER 和 ORDER 的字段差异

**期望过程**：
1. 读取 `schema.md` → 2. 查询两表字段 → 3. 对比分析 → 4. 以表格形式展示差异

**关键陷阱**：是否在一次查询中获取两张表数据（E-3），是否使用 `column_name`（P-6）

**自动评分项**：P-6=1, S-3=1, E-3=1

**Rubric 项**：差异识别 5, 呈现方式 5, 关联字段分析 3+

---

### T-04: 生成查询 CRM 客户及订单的 SQL

**期望过程**：
1. 读取 `schema.md` → 2. 查询 CUSTOMER/ORDER 表结构和关系 → 3. 基于图谱关系生成 JOIN SQL

**关键陷阱**：是否引用 RELATES_TO 关系确定 JOIN 条件，而非凭空猜测

**自动评分项**：O-2=1, S-3=1

**Rubric 项**：语法正确性 5, 语义匹配 5, 表关联合理性 5

---

### T-05: 数据标准有多少条

**期望过程**：
1. 读取 `schema.md` → 2. 查询 Standard 节点 → 3. 聚合统计

**自动评分项**：P-1=1, P-2=1, O-1=1, S-3=1

**Rubric 项**：准确性 5, 完整性 5

---

### T-06: 所有应用及其表数量

**期望过程**：
1. 读取 `schema.md` → 2. 使用聚合 Cypher / NetworkX 查询 → 3. 降序排列

**自动评分项**：P-3=1, S-2=1, S-3=1, E-3=1

**Rubric 项**：准确性 5, 完整性 5

---

### T-07 ~ T-10: 隐式调用

**核心评估点**：skill 是否被正确触发。若未被触发则 O-4=0，整项得 0 分。触发后的评分标准与对应的显式用例一致。

---

### T-11: 多步复合任务

**期望过程**：
1. 读取 `schema.md` → 2. 查询 CRM 表 → 3. 查询关联标准 → 4. 综合输出

**额外评分项**：
| 编号 | 检查项 | 量化 |
|------|--------|------|
| M-1 | 两步查询逻辑衔接，上下文连贯 | 是=1, 否=0 |
| M-2 | 未因多步而重复读取 schema.md | E-1 依然适用 |

---

### T-12: 带业务上下文的比较

**期望过程**：
1. 读取 `schema.md` → 2. 一次性查询两表字段及 RELATES_TO 关系 → 3. 重点分析关联字段

**额外评分项**：
| 编号 | 检查项 | 量化 |
|------|--------|------|
| B-1 | 识别并突出关联字段 | 是=1, 否=0 |
| B-2 | 结合业务语境解读差异 | 是=1, 否=0 |

---

### T-13: 数据标准落标情况

**期望过程**：
1. 读取 `schema.md` → 2. 查询 COMPLIES_WITH 关系 → 3. 按表聚合统计 → 4. 格式化汇报

**额外评分项**：
| 编号 | 检查项 | 量化 |
|------|--------|------|
| R-1 | 输出适合领导汇报（摘要 + 明细） | 是=1, 否=0 |
| R-2 | 使用 COMPLIES_WITH 边查询落标 | 是=1, 否=0 |

---

### T-14 ~ T-16: 负向控制

**期望结果**：不应触发 govio skill。

| 编号 | 检查项 | 量化 |
|------|--------|------|
| N-1 | 未调用 govio-query | 未调用=1, 调用=0 |
| N-2 | 未读取 schema.md / config.yaml | 未读取=1, 读取=0 |
| N-3 | 未使用数据治理相关工具链 | 未使用=1, 使用=0 |

---

## 6. 总分计算

```
总分 = (自动评分项加权平均 × 60%) + (Rubric 评分加权平均 × 40%)

自动评分项权重：
  过程目标 (P-1~P-6): 每项 1 分，共 6 分
  风格目标 (S-1~S-4): 每项 1 分，共 4 分
  效率目标 (E-1~E-4): 每项 1 分，共 4 分
  负向控制 (N-1~N-3): 每项 1 分，共 3 分
  → 自动评分满分 = 17 分

Rubric 评分（每个提示词单独评）：
  回答质量 3 维度 × 5 分 = 15 分（T-01/02/05/06/07/10）
  SQL 质量 3 维度 × 5 分 = 15 分（T-04/09）
  比较质量 3 维度 × 5 分 = 15 分（T-03/08/12）
  复合任务 = 基础 Rubric + 额外项
  → 按 5 分制归一化
```

---

## 7. 扩展方向（随 skill 成熟逐步增加）

- [ ] 命令计数预算：单次交互工具调用 ≤ 8 次
- [ ] Token 预算监控：单次交互总 token ≤ 4000
- [ ] 构建检查：`govio-query` 命令是否能成功执行
- [ ] 运行时冒烟：`uvx --from skills/govio/assets/govio-*.whl govio-query` 退出码 = 0
- [ ] 权限回归：skill 仅使用 `allowed-tools: Read, Grep, Glob`，未尝试写入或执行其他命令
- [ ] 后端切换测试：修改 `~/.govio/config.yaml` 中的 `backend` 字段，验证查询逻辑自动适配
