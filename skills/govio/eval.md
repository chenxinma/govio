# Govio Skill Eval 评估测试用例

---

依据 [eval-skill.md](../../docs/eval-skill.md) 框架设计，将"感觉更好"转化为可量化分数。

提示词数据源：[`data/eval-prompts.json`](../../data/eval-prompts.json)，共 24 条，覆盖 6 类场景。

## 1. 成功标准定义

### 1.1 结果目标（Outcome）

| 编号 | 检查项 | 量化方式 |
|------|--------|----------|
| O-1 | 查询返回正确结果 | 人工比对知识图谱原始数据，结果集完全匹配 = 1 分，部分匹配 = 0.5 分，错误/空 = 0 |
| O-2 | SQL 生成可执行且语义正确 | SQL 语法合法 = 0.5 分，执行后返回预期数据 = 1 分 |
| O-3 | 表字段比较结果完整 | 列出所有差异项 = 1 分，遗漏 > 20% = 0.5 分，遗漏 > 50% = 0 |
| O-4 | 回答与问题直接相关 | 回答直接回应提问 = 1 分，偏题但相关 = 0.5 分，无关 = 0 |
| O-5 | 技能正确触发/不触发 | should_trigger=true 时触发 = 1 分，should_trigger=false 时未触发 = 1 分，否则 = 0 |

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

> 完整数据见 [`data/eval-prompts.json`](../../data/eval-prompts.json)。以下为各分类摘要。

### 2.1 显式调用（explicit_invocation，3 条）

| ID | 提示词 | 测试重点 |
|----|--------|----------|
| explicit-01 | 使用 $govio 技能查询元数据，列出所有应用 | P-1/P-2/P-3/O-1/O-5，返回 15 个应用 |
| explicit-02 | 用 $govio 查询会计引擎应用有哪些表 | P-1/P-3/O-1，AEP 应用下 50 张表 |
| explicit-03 | 使用 $govio 查询 ITS_USER.T_INVOICE 表有哪些字段 | P-1/P-6/O-1，Col 节点 column_name 属性 |

### 2.2 隐式调用（implicit_invocation，5 条）

| ID | 提示词 | 测试重点 |
|----|--------|----------|
| implicit-01 | 查询元数据，列出所有应用 | O-5 触发判定，来源：真实对话样例 |
| implicit-02 | 查询元数据，列出金额相关的字段 | O-5 触发 + 关键词匹配，来源：真实对话样例 |
| implicit-03 | 我想知道报价单中心系统里有哪些数据表 | 中文应用名隐式匹配，SQC 下 29 张表 |
| implicit-04 | 财务管理领域的应用有哪些？ | 按业务域筛选，返回 AEP/ITS/CDPS |
| implicit-05 | 帮我看看外包雇员管理系统和外包项目管理系统各自用了多少张表 | 比较 IHRM(67) vs IHRO(403) |

### 2.3 上下文调用（contextual_invocation，5 条）

| ID | 提示词 | 测试重点 |
|----|--------|----------|
| contextual-01 | 我们需要对接发票管理系统的开票接口，先帮我了解一下发票表 T_INVOICE 的字段结构 | 业务上下文（接口对接）+ 元数据查询 |
| contextual-02 | 数据治理团队需要梳理外服内部机构管理系统的数据资产，请列出这个系统下的所有表和字段 | 数据治理场景，IOMS 仅 2 张表 |
| contextual-03 | 薪税系统的数据库里有没有跟银行相关的表？列出表名 | 关键词筛选 + 应用范围限定 |
| contextual-04 | 帮我比较一下收付费管理（CDPS）和发票管理（ITS）之间有没有名称相同的数据表 | 表字段比较核心能力 O-3 |
| contextual-05 | 我们计划给客户账单管理系统做数据标准治理，先帮我查一下 BILL 应用中所有表名包含 FEE 或 SRV 的字段 | 数据标准治理 + 关键词筛选 |

### 2.4 SQL 生成（sql_generation，3 条）

| ID | 提示词 | 测试重点 |
|----|--------|----------|
| sql-01 | 帮我生成一条 SQL，查询所有业务领域为'财务管理'的应用名称和代码 | Cypher 生成，S-1/S-2/O-2 |
| sql-02 | 写一个查询，找出哪些应用使用了超过 100 张表 | 聚合 Cypher，PDM(123)/IHRO(403)/SSOP(383) |
| sql-03 | 生成查询语句：查找企业法定福利服务系统中字段最多的前 5 张表 | Cypher 排序+限制，SSOP 下按 HAS_COLUMN 计数 |

### 2.5 负向控制（negative_control，3 条）

| ID | 提示词 | 期望结果 |
|----|--------|----------|
| negative-01 | 帮我写一个 Python 脚本把 MySQL 数据迁移到 PostgreSQL | 不应触发 govio skill |
| negative-02 | 帮我修复代码里的一个 bug，pandas DataFrame 合并后列名丢失 | 不应触发 govio skill |
| negative-03 | 在我们的应用里新增一个用户管理模块 | 不应触发 govio skill（功能开发，非数据治理） |

### 2.6 边界情况（edge_case，4 条）

| ID | 提示词 | 测试重点 |
|----|--------|----------|
| edge-01 | 外服内部机构管理系统有几个表？ | 最小应用（IOMS 2 张表），少量数据准确性 |
| edge-02 | 查询应用表中所有由埃森哲维护的应用 | 非主流属性筛选，返回 7 个应用 |
| edge-03 | 帮我查一下所有应用中，哪些表名是空的（没有中文名称） | 数据质量检查，空值/缺失值查询 |
| edge-04 | 列出所有应用及其表数量，按表数量从多到少排序 | 聚合+排序，来源：对话样例 agent 输出模式 |

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

### explicit-01: 使用 $govio 技能查询元数据，列出所有应用

**期望过程**：
1. 读取 `schema.md` → 2. 使用 `govio-cli query` 执行查询（自动从 `~/.govio/config.yaml` 读取后端配置） → 3. 格式化输出

**期望输出**：15 个应用列表，含 PDM/IHRM/IHRO/HPM/PO/PAYPRO/SQC/AEP/SSOP/IOMS/SPRT/NHRS/CDPS/ITS/BILL

**自动评分项**：P-1=1, P-2=1, P-3=1, P-5=1, S-3=1, E-1=1, E-4=1

**Rubric 项**：准确性 5, 完整性 5, 可操作性 5

---

### explicit-02: 用 $govio 查询会计引擎应用有哪些表

**期望过程**：读取 schema.md → 读取 backend.txt → govio-query 查询 AEP→PhysicalTable → 格式化输出

**期望输出**：AEP 应用下 50 张表，包括 AEP_ASSACT_DIM(辅助核算维度), AEP_BUSINESS_UNIT_MAPPING(事业部映射) 等

**自动评分项**：P-1=1, P-2=1, P-3=1, P-5=1, S-3=1

**Rubric 项**：准确性 5, 完整性 5

---

### explicit-03: 使用 $govio 查询 ITS_USER.T_INVOICE 表有哪些字段

**期望过程**：读取 schema.md → Grep node_names.md 确认标准名称 → govio-query 查询 PhysicalTable→Col → 使用 column_name 属性

**关键陷阱**：Col 节点必须使用 `column_name` 而非 `name`（P-6）

**期望输出**：T_INVOICE 表字段列表，含 INV_NO, SELLER_TAX_NO, BUYER_NAME, TAX_AMOUNT, TAX_INCLUDE_AMOUNT 等

**自动评分项**：P-1=1, P-4=1, P-6=1, S-1=1, S-2=1

**Rubric 项**：准确性 5, 完整性 5

---

### implicit-01: 查询元数据，列出所有应用

**核心评估点**：不提技能名时 skill 是否被正确触发（O-5）。来源：真实对话样例。

**触发后评分**：与 explicit-01 一致。

---

### implicit-02: 查询元数据，列出金额相关的字段

**期望过程**：读取 schema.md → govio-query 查询 column_name 含 Pay/Amount/Tax/Rate/Price/Fee 等关键词的 Col 节点

**核心评估点**：O-5 触发 + 关键词匹配能力。来源：真实对话样例。

**期望输出**：金额相关字段，涉及 14 个应用（SPRT 无金额字段），IHRO(784)/SSOP(768) 居多

**自动评分项**：P-1=1, P-3=1, P-5=1, O-5=1

**Rubric 项**：准确性 5, 完整性 3+

---

### implicit-03: 我想知道报价单中心系统里有哪些数据表

**期望过程**：Grep node_names.md 匹配"报价单中心"→ 确认对应 SQC → 查询 USE 边

**核心评估点**：中文应用名隐式匹配（P-4），skill 触发（O-5）

**期望输出**：SQC 下 29 张表，含 ofr_comb_quotation(组合报价单表), ofr_comb_quotation_dtl(组合报价单明细表) 等

**自动评分项**：P-4=1, P-3=1, O-5=1

**Rubric 项**：准确性 5, 完整性 5

---

### implicit-04: 财务管理领域的应用有哪些？

**期望过程**：读取 schema.md → govio-query 按 business_domain 筛选 Application 节点

**期望输出**：3 个应用：AEP(会计引擎), ITS(发票管理), CDPS(收付费管理)

**自动评分项**：P-1=1, P-3=1, O-5=1, S-3=1

**Rubric 项**：准确性 5, 完整性 5

---

### implicit-05: 外包雇员管理系统和外包项目管理系统各自用了多少张表

**期望过程**：Grep node_names.md 确认中文名对应 IHRM/IHRO → 分别查询表数量

**期望输出**：IHRM 有 67 张表，IHRO 有 403 张表

**关键陷阱**：是否合并为一次查询（E-3）

**自动评分项**：P-4=1, P-3=1, E-3=1, O-5=1

**Rubric 项**：准确性 5, 完整性 5

---

### contextual-01: 对接发票管理系统的开票接口，先了解 T_INVOICE 字段结构

**期望过程**：读取 schema.md → Grep node_names.md 确认"发票管理"= ITS → 查询 T_INVOICE 字段

**核心评估点**：从业务上下文提取元数据查询需求，O-5 触发

**期望输出**：T_INVOICE 表字段，含 INV_NO(发票号码), STATUS(开票状态), TAX_AMOUNT(税额) 等

**自动评分项**：P-1=1, P-4=1, P-6=1, O-5=1

**Rubric 项**：准确性 5, 可操作性 5

---

### contextual-02: 梳理外服内部机构管理系统的数据资产

**期望过程**：Grep node_names.md 确认 IOMS → 查询 USE + HAS_COLUMN → 列出表及字段

**期望输出**：IOMS 仅 2 张表：fsg_company_info(公司基本信息表), fsg_company_shareholder(公司股东信息表) 及其字段

**自动评分项**：P-4=1, P-3=1, P-5=1, O-5=1

**Rubric 项**：准确性 5, 完整性 5

---

### contextual-03: 薪税系统里有没有跟银行相关的表

**期望过程**：Grep node_names.md 确认 PAYPRO → 查询 PAYPRO 下表名/字段含 BANK 的 PhysicalTable

**核心评估点**：关键词筛选 + 应用范围限定

**自动评分项**：P-4=1, P-3=1, O-5=1

**Rubric 项**：准确性 5, 完整性 3+

---

### contextual-04: 比较 CDPS 和 ITS 有没有名称相同的数据表

**期望过程**：读取 schema.md → 查询两个应用下的 PhysicalTable → 对比 table_name 集合

**关键陷阱**：表字段比较核心能力 O-3，是否一次查询完成（E-3）

**期望输出**：共有的表如 FSG_DICT(字典表)

**自动评分项**：P-6=1, S-3=1, E-3=1, O-3=1

**Rubric 项**：差异识别 5, 呈现方式 5, 关联字段分析 3+

---

### contextual-05: BILL 应用中表名包含 FEE 或 SRV 的字段

**期望过程**：Grep node_names.md 确认 BILL → 查询 HAS_COLUMN 下 column_name 含 FEE/SRV 的 Col 节点

**核心评估点**：数据标准治理场景 + 关键词筛选，P-6 验证 column_name

**期望输出**：如 BIL_SRV_FEE_RCV_DTL 表中的服务费相关字段

**自动评分项**：P-4=1, P-6=1, P-3=1

**Rubric 项**：准确性 5, 完整性 3+

---

### sql-01: 生成查询财务管理应用的 SQL

**期望过程**：读取 schema.md → 确认 Application 节点属性 → 生成 Cypher

**期望输出**：`MATCH (app:Application {business_domain: "财务管理"}) RETURN app.name, app.app_name_en`

**关键陷阱**：S-1 双引号包裹，S-2 MATCH 开头

**自动评分项**：O-2=1, S-1=1, S-2=1, S-3=1

**Rubric 项**：语法正确性 5, 语义匹配 5, 表关联合理性 3+

---

### sql-02: 找出哪些应用使用了超过 100 张表

**期望过程**：读取 schema.md → 生成含 count + HAVING 的聚合 Cypher

**期望输出**：PDM(123), IHRO(403), SSOP(383)

**自动评分项**：O-2=1, S-1=1, S-2=1, E-3=1

**Rubric 项**：语法正确性 5, 语义匹配 5

---

### sql-03: 查找 SSOP 中字段最多的前 5 张表

**期望过程**：读取 schema.md → 生成按 HAS_COLUMN 边计数排序的 Cypher，LIMIT 5

**期望输出**：T_SI_EMPLOYEE_EXT_LOG(94), T_SI_EMPLOYEE_EXT(92), T_FILE_EXCEL_DATA_DETAIL(91) 等

**自动评分项**：O-2=1, S-1=1, S-2=1, S-3=1

**Rubric 项**：语法正确性 5, 语义匹配 5

---

### negative-01 ~ negative-03: 负向控制

**期望结果**：不应触发 govio skill。

| 编号 | 检查项 | 量化 |
|------|--------|------|
| N-1 | 未调用 govio-query | 未调用=1, 调用=0 |
| N-2 | 未读取 schema.md / config.yaml | 未读取=1, 读取=0 |
| N-3 | 未使用数据治理相关工具链 | 未使用=1, 使用=0 |

---

### edge-01: IOMS 有几个表

**期望输出**：2 张表（最小应用），边界情况验证少量数据返回的准确性

**自动评分项**：P-1=1, P-3=1, O-1=1

**Rubric 项**：准确性 5

---

### edge-02: 由埃森哲维护的应用

**期望过程**：读取 schema.md → govio-query 按 external_vendor 筛选

**期望输出**：7 个应用：PDM, NHRS, AEP, HPM, SQC, SPRT, BILL

**自动评分项**：P-1=1, P-3=1, S-3=1

**Rubric 项**：准确性 5, 完整性 5

---

### edge-03: 哪些表名是空的（没有中文名称）

**期望过程**：读取 schema.md → 查询 PhysicalTable 中 name 为空或缺失的节点

**核心评估点**：数据质量检查场景，空值/缺失值查询

**自动评分项**：P-1=1, P-3=1, O-1=1

**Rubric 项**：准确性 5, 完整性 3+

---

### edge-04: 所有应用及其表数量，按表数量降序

**期望过程**：读取 schema.md → govio-query 聚合查询 → 降序输出

**期望输出**：IHRO(403) > SSOP(383) > PDM(123) > IHRM(67) > AEP(50) > PAYPRO(49) > CDPS(46) > PO(44) > HPM(62) > NHRS(58) > SQC(29) > SPRT(28) > BILL(35) > ITS(9) > IOMS(2)

**来源**：对话样例中 agent 的实际输出模式

**自动评分项**：P-3=1, S-2=1, S-3=1, E-3=1

**Rubric 项**：准确性 5, 完整性 5

---

## 6. 总分计算

```
总分 = (自动评分项加权平均 × 60%) + (Rubric 评分加权平均 × 40%)

自动评分项权重：
  结果目标 (O-1~O-5): 每项 1 分，共 5 分
  过程目标 (P-1~P-6): 每项 1 分，共 6 分
  风格目标 (S-1~S-4): 每项 1 分，共 4 分
  效率目标 (E-1~E-4): 每项 1 分，共 4 分
  负向控制 (N-1~N-3): 每项 1 分，共 3 分
  → 自动评分满分 = 22 分

Rubric 评分（每个提示词单独评）：
  回答质量 3 维度 × 5 分 = 15 分（元数据查询类）
  SQL 质量 3 维度 × 5 分 = 15 分（sql-01/02/03）
  比较质量 3 维度 × 5 分 = 15 分（contextual-04/05）
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
