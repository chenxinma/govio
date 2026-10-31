# Roadmap

> 定位：govio 是**无 LLM、可插拔任意 Agent Runtime 的数据治理 skill + CLI**。
> 本路线图按 T7 架构审视（[docs/tasks/T7-architecture-roadmap/design.md](tasks/T7-architecture-roadmap/design.md)）
> 重排：先做工程精简与边界收敛（Phase 3），再做功能扩展（Phase 4/5）。

## Phase 1: 结构定义层 ✅ 已完成

### 元数据知识图谱
**目标：** 将企业元数据（表、字段、应用、数据标准）转化为可查询的知识图谱。

**已完成：**
- [x] 实现 `TDSLoader` 从 TDS（KunDB/MySQL 元数据库）读取表/字段/标准
- [x] 实现 `DuckDBLoader` 从 DuckDB 读取表/字段元数据
- [x] 实现 `MysqlLoader` 从 MySQL `information_schema` 读取表/字段元数据
- [x] 实现 `PostgresLoader` 从 PostgreSQL `information_schema` + `pg_catalog` 读取表/字段元数据
- [x] 实现 `TrinoLoader` 从 Trino `SHOW TABLES` / `SHOW COLUMNS` 读取表/字段元数据（并接入 CLI）
- [x] 实现 `AppInfoLoader` 从 Excel 读取应用清单
- [x] 实现 `StandardLoader` 读取数据标准与贯标关系
- [x] 实现 `RelationshipLoader` 从 JSON 读取表间关系
- [x] 实现 `MetricLoader` 从 JSON 读取指标/维度定义，生成血缘边
- [x] 实现 `StandardRecommender` 数据标准推荐（k-NN 协同过滤）
- [x] 实现 `NetworkXGraph`（GML 文件读写、schema 内省）
- [x] 实现 `FalkorDBGraph`（Cypher 查询、schema 内省）
- [x] 实现 `LadybugGraph`（嵌入式 `.lbdb`，Cypher 兼容）
- [x] 实现 CSV → 图数据库导入流水线（FalkorDB bulk-loader、Ladybug loader、GML builder）
- [x] 实现 `AssetsGenerator` 自动生成 schema.md、metrics_index.md、names/ 索引
- [x] 实现 10 字符字符串节点 ID（`assign_node_ids`、`make_id`）

### CLI 工具链
**目标：** 提供交互式 CLI 覆盖元数据管理全流程。

**已完成：**
- [x] `govio-cli onboard` 初始化向导（图后端选择 + 数据源配置，含非交互式增删数据源）
- [x] `govio-cli meta` 完整同步管线（元数据 → CSV → 图库 → assets）
- [x] `govio-cli meta` 分步子命令（`meta`/`app`/`rel`/`std`/`compliance`/`metric`/`graph`），已重构为 CLI-only 独立子命令（原 `meta sync` / `meta config` 交互式入口已移除）
- [x] `govio-cli meta meta --source` 多数据源接入：duckdb / mysql / postgres / trino（`--db` 为连接 URL，解析为各 Loader 构造参数）
- [x] `govio-cli meta recommend` 数据标准推荐
- [x] `govio-cli query` 知识图谱查询（Cypher / Python 自适应，结果落 UTF-8 JSONL 文件）
- [x] `govio-cli backend` 查看当前图后端
- [x] `govio-cli -V` 版本查看
- [x] 配置文件格式重构（嵌套 YAML，自动迁移旧格式）
- [x] 数据源密码 Fernet 加密存储

### 数据探查层
**目标：** 在受控安全边界内对数据内容进行探查、比对和可视化。

**已完成：**
- [x] `govio-cli observe load` 从数据源加载 DataFrame
- [x] `govio-cli observe explore` 探查表间关系（列名相似性 + 值重叠推断）
- [x] `govio-cli observe visualize-relations` 生成关系图谱（networkx/JSON）
- [x] `govio-cli observe compare` 比对两个 DataFrame（datacompy）
- [x] `govio-cli observe chart` 从 DataFrame 生成图表 PNG（bar/line）
- [x] `govio-cli observe info` 数据源/DataFrame 信息查询（合并原 show-datasource / list / show）
- [x] `govio-cli observe release` 释放 DataFrame
- [x] `govio-cli sql build` 指标 SQL 组装（从 JSON 规格生成标准 SQL）
- [x] DataFrameStore 内存存储管理
- [x] DatabaseManager 统一连接管理与查询执行（DuckDB / MySQL / PostgreSQL / Trino，SQLAlchemy scheme 归一到统一 `execute_sql` 路径）

---

## Phase 2: Agent 集成层 ✅ 已完成

### Skill 体系
**目标：** 通过 Skills 让 AI Agent 基于治理过的语义路径完成自助数据分析。

**已完成：**
- [x] `govio` 主控 Skill — 意图识别与路由分发
- [x] `govio-meta` — 知识图谱维护（同步、推荐、配置）
- [x] `govio-query` — 元数据/指标查询（应用、表、字段、指标问数）
- [x] `govio-observe` — 数据探查与比对（加载、探索、比对、图表）
- [x] `govio-eda` — EDA 探索性数据分析（4 阶段标准流程）
- [x] `govio-reconcile` — 多源数据对账（5 阶段流程）
- [x] 指标 Playbook — 单指标 + 维度过滤的标准查询流程
- [x] 名称映射机制 — `assets/names/` 中文系统名到标准代码映射
- [x] Skill 与 CLI 命令组对齐重构

### 分发形态
**目标：** skill + CLI 可插拔任意 Agent Runtime，形态差异收敛在宿主注册面。

**已完成：**
- [x] 形态 0：`govio-skills.zip` + `uv tool install govio`（通用基线，适用 Codex / Claude Code 等）
- [x] 形态 1：pi.agent 扩展包（`packaging/pi/`，注册 `govio_cli`/`govio_query` 工具与 `/govio` 诊断命令）
- [x] 形态 2：DeepSeek Harness bundle 插件（`packaging/deepseek/`，同一工具面契约）
- [x] 共享注册契约：skills 单一真源、assets 契约（`<skills-root>/govio/assets`）、govio-cli 适配层契约（argv 子进程、子命令白名单、UTF-8、超时/取消）
- [x] 可重复打包脚本：`package_skills.py`、`scripts/package_pi_extension.py`、`scripts/package_harness_plugin.py`

### Eval 体系
**目标：** 将"感觉更好"转化为可量化的回归检测。

**已完成：**
- [x] 四维评分框架（结果目标 / 过程目标 / 风格目标 / 效率目标）
- [x] eval.md 测试提示词集（路由、元数据查询、指标问数、负向控制、边界情况）

---

## Phase 3: 工程精简与边界收敛（进行中，下一优先）

**目标：** 执行 T7 架构审视的精简建议（详见 [design.md](tasks/T7-architecture-roadmap/design.md) §6），
先删冗余、收敛边界，再扩功能。

- [ ] 删除未使用的 `mcp` 依赖（`src/`/`tests/` 无引用）
- [ ] `_parse_type` 类型解析三份（trino/mysql/postgres loader）收敛为共享实现
- [ ] 打包排除规则与版本注入收敛到 `scripts/` 共享模块（三支打包脚本复用）
- [x] `test_package_harness_plugin.py` 从任务目录迁入 `tests/` 常规集（V3 已完成；同轮补齐 `tests/test_package_pi_extension.py`、`tests/test_package_harness_plugin.py` 打包脚本自动化测试）
- [x] 真实库集成测试固化：`tests/test_real_mysql.py` / `tests/test_real_postgres.py` / `tests/test_real_observe.py`（`@pytest.mark.integration`，默认 skip，`GOVIO_REAL_DB=1` 才执行；用 wslc 容器 mysql:8 / postgres:17 / trino:483 验证，用后 `wslc stop` + `wslc remove` 清理，保证无容器时 `uv run pytest tests/` 全绿）
- [ ] `cli/meta.py`（1240 行）按子命令域拆文件（切点 `_build_metadata_loader` / 各 `cmd_*`）
- [ ] `metadata/utility.py` 拆分「导出编排 / 推荐入口」两类职责
- [ ] `metadata/gen_networkx.py`（CSV→GML）归属迁移到 `graph/`，同步 spec
- [ ] observe_data 的 tools/core 分层说明补进 docs/specs/observe-data.md

已停止 / 暂缓的路线（T7 复核，如需重启请先讨论）：

- **MCP 托管数据集服务**（design-thoughts.md 方案一）：observe_data 本地受控执行已覆盖单 Agent 场景，无多 Agent 共享消费方，暂缓
- **Iceberg 文件存储**（方案二）：维持放弃
- **npm 发布通道**（`@govio/pi-package` / `@dsh-external/govio-tools`）：本地/git 分发已够，待外部需求再启
- **DSH developer preview 深度适配**：保持薄壳，不做 typed wrapper 扩展

---

## Phase 4: 可信数据分析（下一阶段）

### SQL 组装增强
**目标：** 覆盖更复杂的指标查询场景，提升 SQL 生成的准确率和表达力。

- [x] 多指标组合查询（一次请求多个指标共享维度过滤，UNION ALL 合并）
- [ ] 环比/同比分析（自动生成时间偏移 JOIN）
- [ ] 派生指标递归展开（`DERIVED_FROM` 链自动拆解为 CTE）——单层派生已实现（derived CTE），派生引用派生的递归展开未支持
- [x] 聚合排序与 Top-N（`ORDER BY metric_value DESC LIMIT N`，`sql build` 支持 order_by/limit）
- [ ] 维度枚举值自动补全（从图谱中查询维度可选值）

### 数据质量检核
**目标：** 基于数据标准和治理规则，自动执行数据质量检查。

- [ ] 贯标合规检查（`COMPLIES_WITH` 关系驱动的字段级合规验证）
- [ ] 表间一致性检查（基于 `RELATES_TO` 关系的跨表数据比对）
- [ ] 空值率/唯一性/范围等基础质量指标自动计算
- [ ] 检核结果回写图模型（为节点/边附加质量标签）

### Playbook 扩展
**目标：** 从单指标查询扩展到更多高频分析场景。

- [ ] 多指标对比（"本月账单收入 vs 签约额"）
- [ ] 时间趋势分析（"近 6 个月账单收入趋势"）
- [ ] 维度下钻（"按事业部拆分账单收入"）
- [ ] 异常标记（"哪些维度值的指标偏离均值超过 2 倍标准差"）

### CLI 工程化待办
**目标：** 消除元数据导入流程中的现场变通（目录联接、临时库、手工合并 assets）。

- [x] **assets 输出目录可配置**：`meta graph --assets-dir <path>` 参数已实现，默认 `.agents/skills/govio/assets`
- [x] **多份 assets 副本的归并策略**：`--assets-dir` 单一输出目录已消解多副本问题（三种分发形态统一落位 `<skills-root>/govio/assets`，见 install.md）

已评估、暂不实现（T7 复核：维持停止，如需重启请先讨论）：

- `meta meta --schema-alias main=sales`（改名导入）：0.5.5 要求 `--schemas` 写源库真实 schema，命名不一致时告知并停止；改名会牵动 node_id 稳定性与既有图谱迁移
- `meta meta --dry-run`（只读探查规模）：目前由空结果守卫在报错时列出可用 schema 代替

---

## Phase 5: 智能化增强（规划中）

### 自然语言到查询的端到端优化
**目标：** 减少 Agent 从自然语言到正确查询的步骤数和失败率。

- [ ] 查询终止策略优化（精确 → 模糊 → 同义词，最多 3 次尝试）
- [ ] 上下文记忆（跨轮次的查询意图跟踪，避免重复读取 schema）
- [ ] 错误自修复（SQL 执行失败时自动分析错误原因并重试）
- [ ] 指标消歧增强（同义词表、业务域上下文辅助消歧）

### 可信解释（第二阶段）
**目标：** 从"可信取数"演进到"可信解释"——异常解释、原因分析、策略建议。

- [ ] 异常检测与自动归因（指标偏离时关联相关维度变化）
- [ ] 业务事件关联（将数据异常与外部事件时间线对齐）
- [ ] 假设验证框架（支持"如果 X 变化 Y%，指标 Z 会如何变化"的反事实分析）
- [ ] 分析结论溯源（每个结论可追溯到具体的数据源、查询路径和时间窗口）

---

## 架构总览

```
┌──────────────────────────────────────────────────────────────┐
│                        用户（自然语言）                         │
└────────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│            Agent Runtime（LLM + Skills，可插拔）                │
│  Codex / Claude Code / pi.agent / DeepSeek harness / …        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ govio    │  │govio-meta│  │govio-query│  │govio-eda │    │
│  │ 路由分发  │  │ 图谱维护  │  │ 元数据查询 │  │ EDA 分析 │    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘    │
│       │             │             │             │            │
│       ▼             ▼             ▼             ▼            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │        分发壳层（薄壳：注册 + govio-cli 子进程转发）      │   │
│  │  skills.zip+uv tool · pi 扩展 · DeepSeek Harness 插件  │   │
│  └──────────────────────────┬───────────────────────────┘   │
└─────────────────────────────┼───────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌──────────────────┐ ┌──────────────┐ ┌──────────────────┐
│  图数据库后端      │ │  元数据源     │ │  数据源（observe） │
│  NetworkX (GML)  │ │  TDS (KunDB) │ │  MySQL           │
│  FalkorDB        │ │  DuckDB      │ │  PostgreSQL      │
│  Ladybug (.lbdb) │ │  MySQL       │ │  Trino           │
│                  │ │  PostgreSQL  │ │  DuckDB          │
│                  │ │  Trino       │ │                  │
└──────────────────┘ └──────────────┘ └──────────────────┘
```

---

## 关键设计原则

1. **约束优于自由**：Agent 只能沿被治理过的语义路径查数，不能自由发挥
2. **结构化优于非结构化**：知识存在图谱中而非文档中，让 Agent 搜索路径确定且可审计
3. **安全边界分离**：Skill 决定操作意图，CLI 决定数据可见性
4. **Runtime 薄壳适配**：分发形态只做工具注册与 govio-cli 转发，不绑定 harness 私有行为
5. **Eval 驱动迭代**：每次人工修正转化为测试用例，防止回归
