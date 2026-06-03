# Govio Skill Eval 评估测试用例

---

依据 [eval-skill.md](../../docs/eval-skill.md) 框架设计，将"感觉更好"转化为可量化分数。

## 技能架构

```
govio (主控 - 需求路由)
├── govio-metadata (元数据查询)
│   └── 应用、表、字段、数据标准
└── govio-metrics (指标问数)
    └── 指标数据、维度分析、趋势统计
```

提示词数据源：[`data/eval-prompts.json`](../../data/eval-prompts.json)，覆盖 8 类场景。

---

## 1. 成功标准定义

### 1.1 结果目标（Outcome）

| 编号 | 检查项 | 量化方式 |
|------|--------|----------|
| O-1 | 查询返回正确结果 | 人工比对知识图谱原始数据，结果集完全匹配 = 1 分，部分匹配 = 0.5 分，错误/空 = 0 |
| O-2 | SQL 生成可执行且语义正确 | SQL 语法合法 = 0.5 分，执行后返回预期数据 = 1 分 |
| O-3 | 指标数据准确 | 指标值与预期一致 = 1 分，偏差 < 5% = 0.5 分，偏差 > 5% = 0 |
| O-4 | 回答与问题直接相关 | 回答直接回应提问 = 1 分，偏题但相关 = 0.5 分，无关 = 0 |
| O-5 | 技能正确路由 | 元数据问题→govio-metadata=1，指标问题→govio-metrics=1，错误路由=0 |

### 1.2 过程目标（Process）

| 编号 | 检查项 | 适用技能 | 量化方式 |
|------|--------|----------|----------|
| P-1 | 首次行动是读取 `assets/schema.md` | 全部 | 是 = 1 分，否 = 0 |
| P-2 | 使用 `govio-cli query` 查询元数据 | metadata | 使用 = 1 分，手动导入 = 0.5 分，未使用 = 0 |
| P-3 | 使用 `Grep` 查询 `names/` 目录获取标准名称 | metadata | 需要时执行 = 1 分，未执行 = 0 |
| P-4 | 查询结果行数 ≤ 300 | 全部 | 是 = 1 分，否 = 0 |
| P-5 | 正确使用 `column_name` 引用 Col 节点 | metadata | 正确 = 1 分，错误 = 0 |
| P-6 | 读取 `assets/metrics_index.md` 获取指标概览 | metrics | 读取 = 1 分，未读取 = 0 |
| P-7 | 使用 `scripts/sql_builder.py` 组装 SQL | metrics | 使用脚本 = 1 分，手写 SQL = 0.5 分，未生成 = 0 |
| P-8 | 使用 `govio-cli observe load` 执行查询 | metrics | 使用 = 1 分，其他方式 = 0.5 分，未执行 = 0 |

### 1.3 风格目标（Style）

| 编号 | 检查项 | 量化方式 |
|------|--------|----------|
| S-1 | Cypher 查询使用双引号包裹参数 | 是 = 1 分，否 = 0 |
| S-2 | Cypher 查询以 `MATCH` 开头 | 是 = 1 分，否 = 0 |
| S-3 | 输出格式清晰（表格/列表），非原始 JSON 倾倒 | 结构化 = 1 分，半结构化 = 0.5 分，原始 = 0 |
| S-4 | 中文回答，技术术语保留英文 | 符合 = 1 分，全英文 = 0.5 分，全中文含翻译术语 = 0.5 分 |
| S-5 | 指标结果包含单位和时间范围 | 包含 = 1 分，部分包含 = 0.5 分，缺失 = 0 |

### 1.4 效率目标（Efficiency）

| 编号 | 检查项 | 量化方式 |
|------|--------|----------|
| E-1 | 未重复读取 schema.md | 仅读 1 次 = 1 分，2 次 = 0.5 分，> 2 次 = 0 |
| E-2 | 未执行无关命令 | 无冗余命令 = 1 分，有 1-2 条冗余 = 0.5 分，> 2 条 = 0 |
| E-3 | 单次查询完成而非多次碎片查询 | 1 次查询 = 1 分，2 次 = 0.5 分，> 2 次 = 0（合理分步除外） |
| E-4 | 未不必要地读取 SKILL.md / reference-*.md | 未读 = 1 分，读取 = 0（除非后端切换场景） |
| E-5 | 路由决策快速，无犹豫 | 直接路由 = 1 分，询问后路由 = 0.5 分，错误路由 = 0 |

---

## 2. 测试提示词集

### 2.1 路由测试（routing，6 条）

测试主控 `govio` 的路由能力。

| ID | 提示词 | 期望路由 | 测试重点 |
|----|--------|----------|----------|
| route-01 | 查询所有应用列表 | govio-metadata | 元数据查询路由 |
| route-02 | 本月账单收入是多少 | govio-metrics | 指标问数路由 |
| route-03 | 按部门统计签约覆盖率 | govio-metrics | 维度分析路由 |
| route-04 | AEP 有哪些表 | govio-metadata | 表结构查询路由 |
| route-05 | 签署覆盖率怎么计算的 | govio-metrics | 指标语义查询路由 |
| route-06 | 帮我比对两个表的数据差异 | 不触发 govio | 负向控制，应路由到 govio-observe |

### 2.2 元数据查询 - 显式调用（metadata_explicit，3 条）

| ID | 提示词 | 测试重点 |
|----|--------|----------|
| meta-explicit-01 | 使用 $govio-metadata 查询所有应用 | P-1/P-2/O-1，返回 15 个应用 |
| meta-explicit-02 | 用 $govio-metadata 查询会计引擎有哪些表 | P-1/P-2/O-1，AEP 下 50 张表 |
| meta-explicit-03 | 使用 $govio-metadata 查询 T_INVOICE 表字段 | P-1/P-5/O-1，Col 节点 column_name |

### 2.3 元数据查询 - 隐式调用（metadata_implicit，5 条）

| ID | 提示词 | 测试重点 |
|----|--------|----------|
| meta-implicit-01 | 查询元数据，列出所有应用 | O-5 路由判定 |
| meta-implicit-02 | 查询金额相关的字段 | O-5 路由 + 关键词匹配 |
| meta-implicit-03 | 报价单中心系统里有哪些数据表 | 中文应用名隐式匹配，SQC 下 29 张表 |
| meta-implicit-04 | 财务管理领域的应用有哪些 | 按业务域筛选，返回 AEP/ITS/CDPS |
| meta-implicit-05 | 外包雇员管理系统和外包项目管理系统各自用了多少张表 | 比较 IHRM(67) vs IHRO(403) |

### 2.4 指标问数 - 显式调用（metrics_explicit，4 条）

| ID | 提示词 | 测试重点 |
|----|--------|----------|
| metrics-explicit-01 | 使用 $govio-metrics 查询本月账单收入 | P-6/P-7/P-8/O-3，bill_income_amt |
| metrics-explicit-02 | 用 $govio-metrics 按销售单元统计签约覆盖率 | P-6/P-7/P-8/O-3，book_to_bill + 维度 |
| metrics-explicit-03 | 使用 $govio-metrics 查询 YTD 账单收入前 5 的部门 | P-6/P-7/P-8/O-3，排序 + limit |
| metrics-explicit-04 | 用 $govio-metrics 查询签约额的月度趋势 | P-6/P-7/P-8/S-5，时间维度 |

### 2.5 指标问数 - 隐式调用（metrics_implicit，5 条）

| ID | 提示词 | 测试重点 |
|----|--------|----------|
| metrics-implicit-01 | 本月账单收入是多少 | O-5 路由 + O-3 数据准确 |
| metrics-implicit-02 | 按部门统计当月签约额 | O-5 路由 + 维度分组 |
| metrics-implicit-03 | 华东区的签约覆盖率是多少 | O-5 路由 + 维度过滤 |
| metrics-implicit-04 | 最近 6 个月的账单收入趋势 | O-5 路由 + 时间范围 |
| metrics-implicit-05 | 签约覆盖率这个指标怎么算的 | O-5 路由 + 指标语义查询 |

### 2.6 指标问数 - 派生指标（metrics_derived，3 条）

| ID | 提示词 | 测试重点 |
|----|--------|----------|
| metrics-derived-01 | 计算本月签约覆盖率 | P-7，派生指标 SQL 组装 |
| metrics-derived-02 | 查询存量消耗额 | P-7，formula: forecast_income_amt - risk_amt |
| metrics-derived-03 | 账单收入全年预测是多少 | P-7，复合派生指标 |

### 2.7 负向控制（negative_control，4 条）

| ID | 提示词 | 期望结果 |
|----|--------|----------|
| negative-01 | 帮我写一个 Python 脚本把 MySQL 数据迁移到 PostgreSQL | 不触发任何 govio 技能 |
| negative-02 | 帮我修复代码里的一个 bug | 不触发任何 govio 技能 |
| negative-03 | 在应用里新增一个用户管理模块 | 不触发任何 govio 技能 |
| negative-04 | 帮我比对两个表的数据差异 | 不触发 govio，应触发 govio-observe |

### 2.8 边界情况（edge_case，4 条）

| ID | 提示词 | 测试重点 |
|----|--------|----------|
| edge-01 | 外服内部机构管理系统有几个表 | 最小应用（IOMS 2 张表） |
| edge-02 | 查询所有由埃森哲维护的应用 | 非主流属性筛选，返回 7 个应用 |
| edge-03 | 哪些表名是空的（没有中文名称） | 数据质量检查 |
| edge-04 | 列出所有应用及其表数量，按表数量从多到少排序 | 聚合 + 排序 |

---

## 3. 确定性评分检查（基于 trace/产物）

### 3.1 路由评分

```python
def score_routing(trace: list[Event], expected_skill: str) -> dict:
    """评分路由决策"""
    scores = {}

    # O-5: 技能正确路由
    invoked_skills = [e for e in trace if e.type == "skill_invoked"]
    if invoked_skills:
        actual_skill = invoked_skills[0].skill_name
        scores["O-5"] = 1 if actual_skill == expected_skill else 0
    else:
        scores["O-5"] = 0

    # E-5: 路由决策快速
    routing_events = [e for e in trace if e.type == "routing_decision"]
    if len(routing_events) == 1:
        scores["E-5"] = 1
    elif len(routing_events) <= 2:
        scores["E-5"] = 0.5
    else:
        scores["E-5"] = 0

    return scores
```

### 3.2 元数据查询评分

```python
def score_metadata_query(trace: list[Event]) -> dict:
    """评分元数据查询过程"""
    scores = {}

    # P-1: 首次行动读取 schema.md
    first_read = next(
        (e for e in trace if e.type == "tool_call" and e.tool in ("Read", "Grep")),
        None
    )
    scores["P-1"] = 1 if first_read and "schema.md" in first_read.target else 0

    # P-2: 使用 govio-cli query
    query_calls = [e for e in trace if "govio-cli query" in e.target]
    python_calls = [e for e in trace if "python -c" in e.target and "govio" in e.target]
    if query_calls:
        scores["P-2"] = 1
    elif python_calls:
        scores["P-2"] = 0.5
    else:
        scores["P-2"] = 0

    # P-3: Grep names/
    names_grep = [e for e in trace if e.tool == "Grep" and "names/" in e.target]
    scores["P-3"] = 1 if names_grep else 0

    # P-5: Col 节点使用 column_name
    cypher_or_code = " ".join(e.target for e in trace if e.type == "tool_call")
    scores["P-5"] = 0 if "c.name" in cypher_or_code and "column_name" not in cypher_or_code else 1

    # S-1: Cypher 双引号
    if any("MATCH" in e.target for e in trace):
        scores["S-1"] = 1 if all("{'" not in e.target for e in trace if "MATCH" in e.target) else 0

    # S-2: Cypher 以 MATCH 开头
    scores["S-2"] = 1 if all(
        e.target.strip().startswith("MATCH")
        for e in trace if "MATCH" in e.target and e.tool == "Bash"
    ) else 0

    return scores
```

### 3.3 指标问数评分

```python
def score_metrics_query(trace: list[Event]) -> dict:
    """评分指标问数过程"""
    scores = {}

    # P-6: 读取 metrics_index.md
    metrics_index_reads = [e for e in trace if "metrics_index.md" in e.target]
    scores["P-6"] = 1 if metrics_index_reads else 0

    # P-7: 使用 sql_builder.py
    sql_builder_calls = [e for e in trace if "sql_builder" in e.target]
    scores["P-7"] = 1 if sql_builder_calls else 0

    # P-8: 使用 govio-cli observe load
    load_calls = [e for e in trace if "govio-cli observe load" in e.target]
    scores["P-8"] = 1 if load_calls else 0

    # S-5: 结果包含单位和时间范围
    # 需要人工检查输出

    return scores
```

### 3.4 通用评分

```python
def score_common(trace: list[Event]) -> dict:
    """通用评分项"""
    scores = {}

    # E-1: schema.md 读取次数
    schema_reads = sum(1 for e in trace if "schema.md" in e.target)
    scores["E-1"] = {1: 1, 2: 0.5}.get(schema_reads, 0)

    # E-2: 无关命令
    relevant_tools = {"Read", "Grep", "Glob", "Bash"}
    all_commands = [e for e in trace if e.type == "tool_call"]
    irrelevant = [e for e in all_commands if e.tool not in relevant_tools]
    scores["E-2"] = 1 if len(irrelevant) == 0 else (0.5 if len(irrelevant) <= 2 else 0)

    # E-4: 未不必要读取 SKILL.md / reference-*.md
    ref_reads = sum(1 for e in trace if any(x in e.target for x in ("SKILL.md", "reference-")))
    scores["E-4"] = 0 if ref_reads > 0 else 1

    return scores
```

---

## 4. 评分标准（Rubric）

### 4.1 路由质量评分

| 维度 | 5 分 | 3 分 | 1 分 |
|------|------|------|------|
| **路由准确性** | 直接路由到正确子技能 | 询问后路由到正确技能 | 路由到错误技能 |
| **路由效率** | 无犹豫，直接决策 | 1 次确认后决策 | 多次犹豫或错误路由 |

### 4.2 元数据查询质量评分

| 维度 | 5 分 | 3 分 | 1 分 |
|------|------|------|------|
| **准确性** | 结果与知识图谱完全一致 | 有 1-2 处小偏差 | 重大错误或幻觉 |
| **完整性** | 包含所有相关信息，无遗漏 | 覆盖主要信息，少量次要遗漏 | 遗漏关键信息 |
| **可操作性** | 回答可直接用于下一步工作 | 需要少量补充查询 | 需要大量返工 |

### 4.3 指标问数质量评分

| 维度 | 5 分 | 3 分 | 1 分 |
|------|------|------|------|
| **数据准确性** | 指标值与预期完全一致 | 偏差 < 5% | 偏差 > 5% |
| **维度正确性** | 分组/过滤维度完全正确 | 维度基本正确，有遗漏 | 维度错误 |
| **SQL 质量** | 可直接执行，语义正确 | 小修可执行 | 语法错误或语义不符 |
| **结果呈现** | 包含单位、时间范围、格式清晰 | 包含部分元信息 | 缺失元信息或格式混乱 |

---

## 5. 按提示词的期望结果与评分映射

### route-01: 查询所有应用列表

**期望路由**：`govio-metadata`

**期望过程**：
1. 主控识别为元数据查询
2. 路由到 `govio-metadata`
3. 读取 `schema.md`
4. 使用 `govio-cli query` 查询 Application 节点

**期望输出**：15 个应用列表

**自动评分项**：O-5=1, P-1=1, P-2=1, E-5=1

**Rubric 项**：路由准确性 5, 路由效率 5

---

### route-02: 本月账单收入是多少

**期望路由**：`govio-metrics`

**期望过程**：
1. 主控识别为指标问数
2. 路由到 `govio-metrics`
3. 读取 `metrics_index.md`
4. 查询 `bill_income_amt` 指标元数据
5. 使用 `sql_builder.py` 组装 SQL
6. 使用 `govio-cli observe load` 执行查询

**期望输出**：本月账单收入数值，单位：万元

**自动评分项**：O-5=1, P-6=1, P-7=1, P-8=1

**Rubric 项**：数据准确性 5, 结果呈现 5

---

### route-03: 按部门统计签约覆盖率

**期望路由**：`govio-metrics`

**期望过程**：
1. 主控识别为指标问数 + 维度分析
2. 路由到 `govio-metrics`
3. 读取 `metrics_index.md`，确认 `book_to_bill` 是派生指标
4. 查询指标元数据：`signed_amt / bill_income_amt`
5. 使用 `sql_builder.py` 组装带维度的 SQL
6. 执行查询

**期望输出**：按 sales_dept 分组的签约覆盖率列表

**自动评分项**：O-5=1, P-6=1, P-7=1, P-8=1, S-5=1

**Rubric 项**：数据准确性 5, 维度正确性 5, 结果呈现 5

---

### route-04: AEP 有哪些表

**期望路由**：`govio-metadata`

**期望过程**：
1. 主控识别为元数据查询（表结构）
2. 路由到 `govio-metadata`
3. 读取 `schema.md`
4. 使用 `govio-cli query` 查询 AEP 下的 PhysicalTable

**期望输出**：AEP 应用下 50 张表

**自动评分项**：O-5=1, P-1=1, P-2=1

**Rubric 项**：路由准确性 5, 准确性 5

---

### route-05: 签署覆盖率怎么计算的

**期望路由**：`govio-metrics`

**期望过程**：
1. 主控识别为指标语义查询
2. 路由到 `govio-metrics`
3. 查询 `book_to_bill` 指标元数据
4. 返回公式：`signed_amt / bill_income_amt`

**期望输出**：签约覆盖率 = 当月销售签约额 / 当月账单收入

**自动评分项**：O-5=1, P-6=1

**Rubric 项**：路由准确性 5, 准确性 5

---

### route-06: 帮我比对两个表的数据差异

**期望路由**：不触发 govio，应触发 `govio-observe`

**期望结果**：不调用 `govio-cli query` 或 `govio-cli observe load`

**自动评分项**：O-5=1（未触发 govio）

**Rubric 项**：路由准确性 5

---

### meta-explicit-01: 使用 $govio-metadata 查询所有应用

**期望过程**：
1. 读取 `schema.md`
2. 使用 `govio-cli query` 执行查询
3. 格式化输出

**期望输出**：15 个应用列表，含 PDM/IHRM/IHRO/HPM/PO/PAYPRO/SQC/AEP/SSOP/IOMS/SPRT/NHRS/CDPS/ITS/BILL

**自动评分项**：P-1=1, P-2=1, P-4=1, S-3=1, E-1=1, E-4=1

**Rubric 项**：准确性 5, 完整性 5, 可操作性 5

---

### meta-explicit-02: 用 $govio-metadata 查询会计引擎有哪些表

**期望过程**：读取 schema.md → govio-cli query 查询 AEP→PhysicalTable → 格式化输出

**期望输出**：AEP 应用下 50 张表

**自动评分项**：P-1=1, P-2=1, P-4=1, S-3=1

**Rubric 项**：准确性 5, 完整性 5

---

### meta-explicit-03: 使用 $govio-metadata 查询 T_INVOICE 表字段

**期望过程**：读取 schema.md → Grep names/ 确认标准名称 → govio-cli query 查询 PhysicalTable→Col → 使用 column_name 属性

**关键陷阱**：Col 节点必须使用 `column_name` 而非 `name`（P-5）

**期望输出**：T_INVOICE 表字段列表

**自动评分项**：P-1=1, P-3=1, P-5=1, S-1=1, S-2=1

**Rubric 项**：准确性 5, 完整性 5

---

### meta-implicit-01: 查询元数据，列出所有应用

**核心评估点**：不提技能名时 skill 是否被正确路由（O-5）

**触发后评分**：与 meta-explicit-01 一致

---

### meta-implicit-02: 查询金额相关的字段

**期望过程**：读取 schema.md → govio-cli query 查询 column_name 含金额关键词的 Col 节点

**核心评估点**：O-5 路由 + 关键词匹配能力

**自动评分项**：P-1=1, P-2=1, P-4=1, O-5=1

**Rubric 项**：准确性 5, 完整性 3+

---

### meta-implicit-03: 报价单中心系统里有哪些数据表

**期望过程**：Grep names/ 匹配"报价单中心"→ 确认对应 SQC → 查询 USE 边

**核心评估点**：中文应用名隐式匹配（P-3），skill 路由（O-5）

**期望输出**：SQC 下 29 张表

**自动评分项**：P-3=1, P-2=1, O-5=1

**Rubric 项**：准确性 5, 完整性 5

---

### meta-implicit-04: 财务管理领域的应用有哪些

**期望过程**：读取 schema.md → govio-cli query 按 business_domain 筛选

**期望输出**：3 个应用：AEP(会计引擎), ITS(发票管理), CDPS(收付费管理)

**自动评分项**：P-1=1, P-2=1, O-5=1, S-3=1

**Rubric 项**：准确性 5, 完整性 5

---

### meta-implicit-05: 外包雇员管理系统和外包项目管理系统各自用了多少张表

**期望过程**：Grep names/ 确认 IHRM/IHRO → 分别查询表数量

**期望输出**：IHRM 有 67 张表，IHRO 有 403 张表

**自动评分项**：P-3=1, P-2=1, E-3=1, O-5=1

**Rubric 项**：准确性 5, 完整性 5

---

### metrics-explicit-01: 使用 $govio-metrics 查询本月账单收入

**期望过程**：
1. 读取 `metrics_index.md`，确认 `bill_income_amt` 是原子指标
2. 使用 `govio-cli query` 查询指标元数据：来源表 = `dws.income_bill_monthly`
3. 使用 `sql_builder.py` 组装 SQL
4. 使用 `govio-cli observe load` 执行查询

**期望输出**：本月账单收入数值，单位：万元，时间范围：2026-05

**自动评分项**：P-6=1, P-7=1, P-8=1, O-3=1, S-5=1

**Rubric 项**：数据准确性 5, 结果呈现 5

---

### metrics-explicit-02: 用 $govio-metrics 按销售单元统计签约覆盖率

**期望过程**：
1. 读取 `metrics_index.md`，确认 `book_to_bill` 是派生指标
2. 查询指标元数据：公式 = `signed_amt / bill_income_amt`
3. 使用 `sql_builder.py` 组装带维度的 SQL
4. 执行查询

**期望输出**：按 sales_unit 分组的签约覆盖率列表

**自动评分项**：P-6=1, P-7=1, P-8=1, O-3=1, S-5=1

**Rubric 项**：数据准确性 5, 维度正确性 5, 结果呈现 5

---

### metrics-explicit-03: 使用 $govio-metrics 查询 YTD 账单收入前 5 的部门

**期望过程**：
1. 读取 `metrics_index.md`，确认 `bill_income_amt_ytd` 是原子指标
2. 使用 `sql_builder.py` 组装 SQL，带 `order_by` 和 `limit=5`
3. 执行查询

**期望输出**：按 bill_income_amt_ytd 降序排列的前 5 个部门

**自动评分项**：P-6=1, P-7=1, P-8=1, O-3=1

**Rubric 项**：数据准确性 5, 维度正确性 5

---

### metrics-explicit-04: 用 $govio-metrics 查询签约额的月度趋势

**期望过程**：
1. 读取 `metrics_index.md`，确认 `signed_amt` 是原子指标
2. 使用 `sql_builder.py` 组装 SQL，维度 = `report_ym`
3. 执行查询

**期望输出**：按 report_ym 分组的签约额列表

**自动评分项**：P-6=1, P-7=1, P-8=1, O-3=1, S-5=1

**Rubric 项**：数据准确性 5, 维度正确性 5, 结果呈现 5

---

### metrics-implicit-01: 本月账单收入是多少

**核心评估点**：不提技能名时是否正确路由到 `govio-metrics`（O-5）

**触发后评分**：与 metrics-explicit-01 一致

---

### metrics-implicit-02: 按部门统计当月签约额

**核心评估点**：O-5 路由 + 维度分组

**期望输出**：按 sales_dept 分组的 signed_amt 列表

**自动评分项**：O-5=1, P-6=1, P-7=1, P-8=1

**Rubric 项**：数据准确性 5, 维度正确性 5

---

### metrics-implicit-03: 华东区的签约覆盖率是多少

**核心评估点**：O-5 路由 + 维度过滤

**期望输出**：华东区的 book_to_bill 数值

**自动评分项**：O-5=1, P-6=1, P-7=1, P-8=1

**Rubric 项**：数据准确性 5, 维度正确性 5

---

### metrics-implicit-04: 最近 6 个月的账单收入趋势

**核心评估点**：O-5 路由 + 时间范围处理

**期望输出**：最近 6 个月的 bill_income_amt 趋势

**自动评分项**：O-5=1, P-6=1, P-7=1, P-8=1, S-5=1

**Rubric 项**：数据准确性 5, 维度正确性 5, 结果呈现 5

---

### metrics-implicit-05: 签约覆盖率这个指标怎么算的

**核心评估点**：O-5 路由 + 指标语义查询

**期望输出**：签约覆盖率 = 当月销售签约额 / 当月账单收入

**自动评分项**：O-5=1, P-6=1

**Rubric 项**：准确性 5

---

### metrics-derived-01: 计算本月签约覆盖率

**期望过程**：
1. 读取 `metrics_index.md`，确认 `book_to_bill` 是派生指标，公式 = `signed_amt / bill_income_amt`
2. 查询两个原子指标的元数据
3. 使用 `sql_builder.py` 组装派生指标 SQL
4. 执行查询

**关键陷阱**：派生指标 SQL 组装（P-7），需要正确处理 CTE

**期望输出**：本月签约覆盖率数值

**自动评分项**：P-6=1, P-7=1, P-8=1, O-3=1

**Rubric 项**：数据准确性 5, SQL 质量 5

---

### metrics-derived-02: 查询存量消耗额

**期望过程**：
1. 读取 `metrics_index.md`，确认 `burndown_amt` 是派生指标，公式 = `forecast_income_amt - risk_amt`
2. 使用 `sql_builder.py` 组装 SQL
3. 执行查询

**期望输出**：存量消耗额数值

**自动评分项**：P-6=1, P-7=1, P-8=1, O-3=1

**Rubric 项**：数据准确性 5, SQL 质量 5

---

### metrics-derived-03: 账单收入全年预测是多少

**期望过程**：
1. 读取 `metrics_index.md`，确认 `income_forecast_annual` 是派生指标
2. 公式 = `bill_income_amt_ytd + burndown_amt + leads_forecast_amt + opp_forecast_amt`
3. 使用 `sql_builder.py` 组装复合派生指标 SQL
4. 执行查询

**关键陷阱**：复合派生指标可能需要多层 CTE

**期望输出**：账单收入全年预测数值

**自动评分项**：P-6=1, P-7=1, P-8=1, O-3=1

**Rubric 项**：数据准确性 5, SQL 质量 5

---

### negative-01 ~ negative-04: 负向控制

**期望结果**：不应触发任何 govio 技能。

| 编号 | 检查项 | 量化 |
|------|--------|------|
| N-1 | 未调用 govio-cli query | 未调用=1, 调用=0 |
| N-2 | 未读取 schema.md | 未读取=1, 读取=0 |
| N-3 | 未使用数据治理相关工具链 | 未使用=1, 使用=0 |

---

### edge-01: IOMS 有几个表

**期望输出**：2 张表（最小应用）

**自动评分项**：P-1=1, P-2=1, O-1=1

**Rubric 项**：准确性 5

---

### edge-02: 由埃森哲维护的应用

**期望输出**：7 个应用：PDM, NHRS, AEP, HPM, SQC, SPRT, BILL

**自动评分项**：P-1=1, P-3=1, S-3=1

**Rubric 项**：准确性 5, 完整性 5

---

### edge-03: 哪些表名是空的（没有中文名称）

**期望过程**：读取 schema.md → 查询 PhysicalTable 中 name 为空或缺失的节点

**自动评分项**：P-1=1, P-2=1, O-1=1

**Rubric 项**：准确性 5, 完整性 3+

---

### edge-04: 所有应用及其表数量，按表数量降序

**期望过程**：读取 schema.md → govio-cli query 聚合查询 → 降序输出

**期望输出**：IHRO(403) > SSOP(383) > PDM(123) > IHRM(67) > HPM(62) > NHRS(58) > AEP(50) > PAYPRO(49) > CDPS(46) > PO(44) > BILL(35) > SQC(29) > SPRT(28) > ITS(9) > IOMS(2)

**自动评分项**：P-2=1, S-2=1, S-3=1, E-3=1

**Rubric 项**：准确性 5, 完整性 5

---

## 6. 总分计算

```
总分 = (自动评分项加权平均 × 60%) + (Rubric 评分加权平均 × 40%)

自动评分项权重：
  结果目标 (O-1~O-5): 每项 1 分，共 5 分
  过程目标 (P-1~P-8): 每项 1 分，共 8 分
  风格目标 (S-1~S-5): 每项 1 分，共 5 分
  效率目标 (E-1~E-5): 每项 1 分，共 5 分
  负向控制 (N-1~N-3): 每项 1 分，共 3 分
  → 自动评分满分 = 26 分

Rubric 评分（每个提示词单独评）：
  路由质量 2 维度 × 5 分 = 10 分（route-*）
  元数据查询 3 维度 × 5 分 = 15 分（meta-*）
  指标问数 4 维度 × 5 分 = 20 分（metrics-*）
  → 按 5 分制归一化
```

---

## 7. 扩展方向（随 skill 成熟逐步增加）

- [ ] 命令计数预算：单次交互工具调用 ≤ 8 次
- [ ] Token 预算监控：单次交互总 token ≤ 4000
- [ ] 构建检查：`govio-cli query` 和 `govio-cli observe load` 命令是否能成功执行
- [ ] 运行时冒烟：`govio-cli query --code "print(g.schema)"` 退出码 = 0
- [ ] 权限回归：skill 仅使用 `allowed-tools: Read, Grep, Glob`，未尝试写入或执行其他命令
- [ ] 后端切换测试：修改 `~/.govio/config.yaml` 中的 `backend` 字段，验证查询逻辑自动适配
- [ ] CTE 组合测试：多次 `govio-cli observe load` 后通过 CTE 引用组合查询
- [ ] 派生指标覆盖率：所有派生指标（book_to_bill, burndown_amt, income_forecast_annual, total_income_forecast_amt, total_sales_forecast_ytd）均有测试用例
