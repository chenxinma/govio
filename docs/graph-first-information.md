# 图结构查询：领域 Agent 的信息构建优势

> 为什么专业领域的 agent 需要把领域信息组织成图？本文以数据治理场景为例，讨论图结构在信息构建与检索上的真实优势，以及它和"存进 SQLite"到底差在哪。

---

## 一、领域 Agent = 专门信息 + 技能 + Runtime

现阶段构建专业领域 Agent，本质上是三件事的组合：

1. **专门的信息**：领域独有、通用模型不知道的事实；
2. **技能**：操作信息和完成任务的工具（CLI、skills）；
3. **Runtime**：Codex、Claude Code 等 agent 运行时。

Runtime 和技能都相对容易获取，真正需要精心设计的**是信息层**：领域信息以何种形状组织，直接决定 Agent 能以多高效率和质量使用它。

数据治理是典型样本。其信息有两类来源：

- **原生结构化**：表、字段、类型等，可直接从数据库元数据加载；
- **非结构化**：业务关系、指标口径、数据标准，散落在文档中。

后者常用 JSON Schema 约束 LLM 输出，人工审核后导入知识库。格式由 schema 保证，事实由人保证，边界清晰。

关键在于：两类信息汇合后，以什么形式存储和查询。

## 二、治理信息的本质是"关系"，不是"记录"

典型治理问题都是路径问题，且路径长度未知：

- 改这张表的某个字段，会影响哪些下游指标？
- 这个数据标准被哪些列遵循？还有哪些缺口？
- 这个派生指标的口径，经过了哪些原子指标和物理列？

用"记录"思维回答，每类问题都需要定制 JOIN，问题越开放，SQL 越复杂，路径深度变化就要重写查询。

用图的思维，同一个问题始终是**一次遍历**：

```cypher
MATCH (t:PhysicalTable {full_table_name: "hr_prod.ihrodb.employee"})
      -[:HAS_COLUMN]->(c:Col)
      <-[:REFERS_COLUMN]-(m:Metric)
      -[:DERIVED_FROM*0..3]->(upstream:Metric)
RETURN t, c, m, upstream
```

返回的不是扁平行集，而是一张**子图**：相关的表、列、指标及它们之间的关系一次性呈现。

这就是图结构查询的核心优势：

> **领域问题是路径式的，图查询也是遍历式的。问题的形状与查询的形状天然对齐。**

## 三、对 Agent 意味着什么：查询自然度 × 返回形状

从 Agent 视角看，图的优势体现在两个维度相乘的效果。

**查询自然度**：把“找出生效日期影响的下游指标”翻译成多层 JOIN 或递归 CTE，需要 Agent 做大量关系代数推导，容易出错；翻译成 `MATCH` 模式，几乎是把问题里的名词换成节点标签、动词换成边类型。映射距离越短，生成正确查询的概率越高。

**返回形状**：Agent 的工作记忆就是上下文窗口。返回扁平行集还需要额外代码重组；返回子图则**直接可用作 prompt 上下文**——schema 片段、相关定义、血缘关系恰好是回答所需的最小上下文。子图序列化成本极低，token 消耗随遍历深度线性可控。

信息层的输出形状，直接决定了每轮推理的上下文质量。

## 四、图结构的另外两个构建优势

**Schema-free 易演化**。治理需求经常变化：今天加计算口径，明天要记录血缘变更历史。Property Graph 中新增节点或边类型只需增加标签，无需迁移旧数据。对于“数据标准属性因标准而异”的动态属性，图的属性或 JSON 属性也能自然容纳。

**审核产物有沉淀层**。LLM 抽取 + 人工审核后的“事实”需要可靠的存放处。图适合承担这个角色：基于业务键哈希的确定性 10 字符节点 ID（`PT`/`CO`/`ME` 前缀 + SHA256[:8]）保证重复导入幂等；审核通过的定义成为节点，关系成为边，来源和审核记录可作为 provenance 属性挂载。后续查询“这个口径的依据是什么、何时被替代”时，答案就在图里。

本项目通过 `GraphFactory` 抽象了 NetworkX（内存/GML）、FalkorDB（服务端 Cypher）和 Ladybug（嵌入式 Cypher）三种后端，验证了“逻辑模型优先、存储引擎可替换”的架构。

## 五、认真回答：存进 SQLite 不行吗？

必须正视这个反方观点。

真正的 ER 模型是为每类实体建正经的表，用外键表达关系。落到治理场景会产生 6~8 张带约束的表。例如下面是简化的核心表结构（重点展示 `metric_ref_column`、`col`、`physical_table` 的关系）:

```sql
-- 核心实体表
CREATE TABLE physical_table (
    id          TEXT PRIMARY KEY,
    full_table_name TEXT UNIQUE
);

CREATE TABLE col (
    id       TEXT PRIMARY KEY,
    table_id TEXT REFERENCES physical_table(id),
    column_name TEXT
);

CREATE TABLE metric (
    id   TEXT PRIMARY KEY,
    code TEXT UNIQUE
);

-- 关联表（重点：metric_ref_column 是多对多关联）
CREATE TABLE metric_ref_column (
    metric_id TEXT REFERENCES metric(id),
    col_id    TEXT REFERENCES col(id),
    role      TEXT,
    PRIMARY KEY (metric_id, col_id)
);

CREATE TABLE metric_derived_from (
    metric_id   TEXT REFERENCES metric(id),
    upstream_id TEXT REFERENCES metric(id),
    PRIMARY KEY (metric_id, upstream_id)
);
```

这套方案有明显优势：

- 物理层一致性（外键约束）；
- 点查和固定一两跳查询极快且直白；
- 单文件、零依赖，SQL 生态成熟。

如果问题主要是一两跳固定查询（“表有哪些列”、“指标定义是什么”），ER 模型完全够用，甚至更好维护。

差距在**开放的多跳异质路径**上显现。同样是“派生指标经过哪些原子指标和物理列”，ER 模型需要递归 CTE + 多表 JOIN，且每增加一种实体类型就要大幅改写查询：

```sql
WITH RECURSIVE upstream AS (
    SELECT metric_id, upstream_id, 1 AS depth 
    FROM metric_derived_from WHERE metric_id = ?
    UNION ALL
    SELECT d.metric_id, d.upstream_id, u.depth + 1
    FROM metric_derived_from d 
    JOIN upstream u ON d.metric_id = u.upstream_id
    WHERE u.depth < 3
)
SELECT ... FROM upstream
JOIN metric_ref_column ON ...
JOIN col ON ...
JOIN physical_table ON ...;
```

而图查询始终保持统一形式，路径长度和穿越的实体种类都不影响写法：

```cypher
MATCH (m:Metric {code: "revenue_growth"})-[:DERIVED_FROM*0..3]->(up)
      -[:REFERS_COLUMN]->(c:Col)<-[:HAS_COLUMN]-(t:PhysicalTable)
RETURN m, up, c, t
```

核心差异在于**认知负荷分配**：

| 维度           | 图查询                  | SQLite ER 模型             |
|----------------|------------------------|---------------------------|
| 固定一两跳查询 | 顺手                   | 更顺手，SQL 更直白         |
| 异质多跳/深度不定 | 一行 MATCH，深度随手改 | 递归 CTE 需处理异质分支，或应用层拼接 |
| 返回形状       | 天然子图，直接进上下文 | 扁平行集，需应用层重组     |
| 模型演化       | 加标签即可，无 DDL     | 新实体=新表+迁移；动态属性需宽表/EAV |
| 一致性约束     | 依赖导入逻辑           | 物理外键保证               |
| Agent schema 对齐成本 | 只需知道标签和边类型   | 需理解多张表的外键拓扑     |

图的价值**不在图数据库本身**，而在遍历式查询原语和子图返回形状。

正确架构是：把**节点/边逻辑模型和遍历查询接口**当作核心资产，把存储后端当作可替换实现。今天用 Ladybug 嵌入式，明天换 FalkorDB，只需实现同一 `Graph` 接口，上层 Agent 技能无需修改。

## 六、把相似度也变成"边"

治理信息中还有一类隐含关系：语义相似（如 "emp_id" 与 "employee_code"，或口径相近的指标定义）。

若将高置信度的相似对作为**带 confidence 属性的 `SIMILAR_TO` 边**沉淀到图中（或至少作为查询前的过滤器），Agent 一次子图展开就能同时获得：

- **显式关系邻居**：外键、引用、派生关系；
- **语义邻居**：同义、疑似重复关系。

这类边与人工审核的 `COMPLIES_WITH`、`DERIVED_FROM` 等边应明确区分（查询时可按类型/置信度过滤），以保持“事实由人保证”的可信边界。

这是 ER 模型最难自然复刻的价值点——相似度一旦成为可遍历的边，就进入了 Agent 的推理射程。对治理场景而言，“发现本该有却缺失的关系”往往是最高价值的问题。

## 结语

领域 Agent = 专门信息 + 技能 + Runtime。信息层的组织方式是最有设计价值的一环。

图结构的真正优势在于三点形状对齐：

1. **问题形状对齐**：路径问题 ↔ 遍历查询；
2. **Agent 形状对齐**：自然查询 + 子图直接作为上下文；
3. **演化形状对齐**：Schema-free 承接不齐的需求，事实与 provenance 有可靠沉淀层。

守住节点/边的逻辑模型与遍历查询接口，存储引擎随时可换。信息一旦织成图，Agent 的每次检索就是在展开一张恰到好处的子图——这可能是“专门信息”能给领域 Agent 带来的最大杠杆。
