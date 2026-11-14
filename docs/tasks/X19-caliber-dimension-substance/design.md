# X19-W5a 设计：口径判定升级——维度限定条件的实质比对（govio 仓）

> 计划事实源：`govio-map/docs/task-plan.json → crossCutting[X19]`（只读，主会话记账）。
> 定调来源：`govio-map/docs/PENDING-DISCUSSION.md` PD-2（2026-11-14 用户裁决：口径判定以
> **计算表达式**和**维度限定条件**为实质判断依据）。本任务只做 govio 侧（W5a），map 侧消费（W5b）另派。

## 1. 问题与定位

旧实现里「维度限定条件」只有 `_compare_filter` 一条弱证据路径：把指标定义中的剔除类词
（剔除/不含/排除…）拿去做 **WHERE 段子串包含 + 否定算子检查**，docstring 自述「全部存疑：只给
unknown，不判 conflict」。它没有回答 PD-2 的真正缺口：SQL 到底把哪个维度限定成了什么取值，
与指标声明的维度范围是否实质矛盾。

本次升级把维度判定改成**结构化实质比对**：

- SQL 侧 → **维度限定三元组**（列 / 算子 / 值集合），来自 WHERE（含子查询）与 GROUP BY；
- 指标侧 → **维度限定声明**（`Dimension.values_example` 枚举码值 ∩ 指标定义文本，
  并按 `DIMENSION_USED.usage_type` 区分限定/分组，用 `REFERS_COLUMN role` 做列绑定）；
- 判定 → `consistent` / `conflict` / `unknown` 三态。

`_compare_filter` 的剔除词路径**原样保留**（仍是 unknown-only），作为「元数据没配维度枚举」时的
兜底提示；两条路径分工写进函数 docstring。

## 2. 实质判据的来源（全部来自已声明元数据，无启发式词典）

| 侧 | 事实 | 出处（已核实字段名） |
|---|---|---|
| SQL | 三元组列名 | WHERE 谓词左值，按 `_canonical_expr` 同口径去表/库限定、小写 |
| SQL | 算子 | `= != <> > >= < < IN NOT IN LIKE NOT LIKE IS NULL IS NOT NULL BETWEEN` |
| SQL | 值集合 | **仅**引号串/数字字面量（`IN` 允许括号列表）；子查询/参数占位/列间等值 → 空集（不猜测） |
| SQL | 分组列 | 顶层 `GROUP BY`（位置号 `GROUP BY 1` 取不到列名，自然落空） |
| 指标 | 枚举码值 | `Dimension.values_example`（字符串，按 `, ，、; ； / | 换行` 切，不按空格切） |
| 指标 | 声明限定/剔除 | 码值出现在 `business_definition + statistical_scope + time_scope` 文本内；落在 `_EXCLUSION_RE` 命中区间内 → 剔除，否则 → 限定 |
| 指标 | 维度角色 | `DIMENSION_USED.usage_type`：`filter`/`slice` 是显式限定；`group` 需文本另有正向线索词（仅/只/限定/限制/范围/针对/口径为/选取） |
| 指标 | 列绑定 | `REFERS_COLUMN.role ∈ {dimension_ref, filter}` 的列名 token ⊇ 维度 `code` token |

列绑定是**声明名之间的字面 token 匹配**，不是别名/同义词词典：只比较元数据里已有的名字，
不做任何跨语言或跨命名风格的翻译（详见 §6 与方案 C 的界限）。

## 3. 判定矩阵（三态边界）

对每条维度限定声明 `req`（`expected` = 声明限定取值，`excluded` = 声明剔除取值，
`enum` = 该维度声明的枚举码值全集，`cols` = 绑定列）：

| # | 声明侧 | SQL 侧证据 | 判定 | differences 写法 |
|---|---|---|---|---|
| 1 | `expected` 非空 | 等值/IN 命中 `expected`（同列或跨列） | **无差异**（consistent） | 不产条目 |
| 2 | `expected` 非空 | 命中 `expected` 但取值集合超出声明 | unknown | `filter`：取值超出指标声明 |
| 3 | `expected` 非空 | 同列被限定到**同一维度的其它声明枚举值**（`values ⊆ enum 且与 expected 不相交`） | **conflict** | `filter`：限定到指标未选取的声明取值（实质矛盾） |
| 4 | `expected` 非空 | 同列有正向限定但取值不可枚举（范围 / `IS NULL` / 参数 / 列间等值） | unknown | `filter`：取值无法与声明枚举比对 |
| 5 | `expected` 非空 | 该列只出现在 `GROUP BY` | unknown | `filter`：分组而非限定 |
| 6 | `expected` 非空 | SQL 另有归属不到声明维度的等值限定 | unknown | `filter`：等值限定无法归属到该维度，未确证 |
| 7 | `expected` 非空 | 整条 SQL（含子查询）无任何等价落点，也无 #6 那种存疑线索 | **conflict** | `filter`：声明限定在 SQL 未见任何等价取值限定（实质矛盾） |
| 8 | `excluded` 非空 | SQL 正向限定的取值**全部**来自 `excluded` | **conflict** | `filter`：结果被限定到指标声明剔除的取值（实质矛盾） |
| 9 | `excluded` 非空 | 否定算子实现剔除（`!=` / `NOT IN`） | **无差异** | 不产条目 |
| 10 | 无声明（无维度 / 无枚举 / 枚举未出现在文本） | 任意 | **不产条目** | 见 §4 降噪 |
| 11 | 有声明 | SQL 附加了指标未声明的等值/IN/LIKE 限定 | unknown | `filter`：附加未声明限定，可能收窄口径（存疑，不判冲突） |

时间粒度（同批核查产物，aspect 仍为 `time_scope`）：

| # | 声明侧 | SQL 侧 | 判定 |
|---|---|---|---|
| T1 | 声明粒度（按月/季度/日粒度…） | `DATE_TRUNC('month', x)` / `YEAR(x)` 等显式粒度且相交 | 无差异 |
| T2 | 声明粒度 | 有显式粒度但不相交 | **conflict** |
| T3 | 声明粒度 | 无显式粒度 | 不产条目（按时间分组属查询自由度） |

### 3.1 为什么 conflict 只有 #3/#7/#8/T2

判定纪律是「宁可 unknown 不误报」。conflict 必须能对应到一条**实质矛盾**，且该矛盾的两个
取值都来自同一份声明数据：

- #3 的对照双方都在 `values_example` 里（声明维度内的兄弟码值），不存在「翻译/别名」推断；
- #7 的结论是「声明限定在整条 SQL（含子查询、含跨列取值匹配）里完全无落点」，
  且已先排除 #6（有其他等值限定但无法归属）这类映射不确定情形；
- #8 的取值直接落在指标写明「剔除/不含」的枚举码值上；
- T2 的粒度双方都是显式声明（月度 vs 年度）。

误报代价最高的形态（取值写法与声明不同语种、限定用了元数据之外的列）一律落 unknown，
并作为遗留问题记录（§7）。

## 4. 降噪边界（PD-2 第 2 案的延续）

- **无维度声明的指标完全不产 `filter` 条目**：维度判定要求「枚举码值 ∩ 定义文本」非空。
  `dimension_ref` 列缺失同样不产条目（既有 measure-first 逻辑不变）。
- **`group` 维度默认不是限定**：「按国家分组」只声明查询自由度；只有文本另有正向限定线索词
  才升级为限定要求。
- **#11 只统计 `= / IN / LIKE` 且取值可枚举的附加限定**：范围与时间窗类谓词交给 `time_scope`，
  避免正常带时间过滤的 SQL 被维度维度刷成 unknown。
- **`expected` 命中即视为已实现，跨列也算**：限定可能落在元数据未列出的同域列上，命中取值
  比命中列名更可靠。

## 5. `_compare_time_scope` 同批核查结论

**时间窗逻辑维持现状**：它已是实质比对（双侧都把窗口归一到「天数」集合再判相交/不相交，
不是子串包含），符合 PD-2 定调，无需改造；「SQL 侧无窗口 → 存疑」也符合不误报纪律。

**新增时间粒度比对**（`_compare_time_grain`，aspect 仍为 `time_scope`）：任务书点名的缺口
「指标声明月度而 SQL 取年度」原实现完全看不到——窗口相同但聚合粒度不同同样是口径偏差。
按 §3 表 T1-T3 执行，SQL 无显式粒度时不产条目，因此不给既有 45 个 caliber 用例增加任何噪声
（实测零回归）。

## 6. 与方案 C 的界限（未获授权，本任务不做）

| 方案 C 的手段 | 本任务是否使用 | 说明 |
|---|---|---|
| 中英同义词典（revenue→sales、qty→boxes） | ❌ 不使用 | 取值比较只做「声明枚举码值 vs SQL 字面量」的字面相等 |
| 别名修饰剥离（`calc_/m_/computed_`） | ❌ 不使用 | 三元组列名只做 `_canonical_expr`（去表限定/压空白/小写） |
| 声称判定改为「别名归一匹配」 | ❌ 不使用 | 检测范围仍由调用方（map 侧 `caliberMetrics`）指定 |
| 表达式规范化比对 | ✅ 沿用既有（`33080fd`） | 属已落地的计算表达式判据，不是 C |
| 维度 code 与列名的字面 token 匹配 | ✅ 本任务新增 | 只比较元数据自带的名字，不含任何翻译表；匹配不到时按 §3 #6 落 unknown |

## 7. 遗留问题

1. **跨语言/跨码值取值映射缺失**：`values_example` 若是「中国」而 SQL 写 `'China'` 或 `country_id = 3`，
   本实现判不出「同一取值」→ 落 unknown（#4/#6）。彻底解决需要元数据侧补「码值 ↔ 展示名」映射
   （或标准码 `Standard` 关联），不建议用词典启发式。
2. **`values_example` 只是「示例」码值**：schema 允许它是不完整示例；若指标声明的取值没写进示例，
   限定要求就无法生成（漏判，不误报）。需要元数据侧把「限定取值」升级为显式字段（如
   `dimension_filter` 声明），map W5b 之前不动 schema。
3. **`statistical_scope` 纯中文描述（无枚举可对齐）不产 conflict**：例如「限华东区」但
   `values_example` 为空 → 只走既有 `_compare_filter` 弱证据路径。属 §2 数据依赖，非算法缺口。
4. **`role=filter` 列未出现在 SQL 不产存疑**：刻意维持降噪（PD-2 第 2 案取向）；若将来要在
   元数据侧显式声明过滤列的期望取值，可再升级为可比对项。
5. **既有剔除词子串路径仍只给 unknown**：即便 `WHERE order_type = 'refund'` 与「剔除退款订单」
   直观矛盾，本任务不把它升级为 conflict——它的证据是「词出现在 WHERE 段里」，非结构化取值，
   升级会引入误报面（结构化升级路径见 §3 #8，要求取值命中声明枚举）。
6. **`NOT (...)` 包裹与 `EXISTS` 子查询的谓词被丢弃**：解析不出来就不比对（unknown 方向），
   未来若接真正的 SQL 方言解析器（sqlglot 一类）可覆盖。
7. **CTE 限定覆盖到「取值命中」但不覆盖「列绑定」**：`WITH x AS (...)` 里的限定参与 #1 匹配，
   但 #3/#4 的列归属仍以命中的 WHERE 节点为准，复杂多层 CTE 可能只落 unknown。
8. **`GROUP BY` 表达式形态**（`GROUP BY DATE_TRUNC(...)`）取不到裸列名 → 不参与 #5 判定；
   保守方向是「不因此降级为 unknown」，因此可能保留 conflict——依赖 #7 的前置条件
   （整条 SQL 无任何等值限定）才触发，实际风险面很小，已用 `GROUP BY 1`/裸列两档用例锁定。

## 8. 输出契约与合并顺序

- **CLI 形状不变**：`govio-cli sql caliber --sql/--sql-file/-m` 参数与单行 JSON
  `{"results": [{"metric", "status", "differences"}]}` 完全保持；`differences` 条目键仍是
  `aspect/expected/found/detail`，`aspect` 取值集合未新增（维度差异复用已有 `filter`，
  粒度差异复用 `time_scope`），三态取值不变 → map 侧 `parseCaliberOutput` 无需先行改动，
  未触发「必须先 ask 主会话」的新字段流程。
- **内部结构新增不算契约变更**：`CaliberMetric.dimensions` / `SqlFacts.dim_predicates` 等
  只是模块内数据类，不进 JSON 输出（`_metric_summary` 仍只输出 id/code/name）。
- **合并顺序**：先合 govio（本任务）→ 再派 map W5b（补 `verify-caliber-conflict` 维度断言、
  `CALIBER_ASPECT_LABELS` 无需新增）。两仓改动互相独立可跑绿：map 在未合并本改动时行为不变
  （新逻辑只在有维度声明数据时才产条目）。

## 9. 实测：8 档样本逐档判定（改动前 = PD-2 补充实测，改动后 = 本次用例集）

| 档 | SQL 要点 | 改动前 | 改动后 | 依据 |
|---|---|---|---|---|
| s1 | `SUM(price*boxes) AS sales_amount` | conflict | **conflict** ✓ | aggregation + column |
| s2 | `ROUND(SUM(f.amount),2) AS revenue_amount` + 国家分组 | unknown | **consistent** ✓ | 无维度声明要求，#10/#1 |
| s3 | 不含 amount | conflict | **conflict** ✓ | aggregation + column |
| s4 | `AVG(amount)` | conflict | **conflict** ✓ | aggregation 结构 |
| t1 | `AVG(amount) AS revenue_calc` | conflict | **conflict** ✓ | aggregation 结构 |
| t2 | 同 s2（第二轮） | unknown | **consistent** ✓ | 同上 |
| t3 | 不含 amount（第二轮） | conflict | **conflict** ✓ | aggregation + column |
| t4 | `SUM(f.amount*(1-discount_pct/100))` | conflict | **conflict** ✓ | 聚合参数表达式 |

s2/t2 的 `unknown → consistent` 由已落地的 measure-first（`33080fd`）达成，本任务确认其
**不被新增维度判定重新稀释**（用例 `test_pd2_dimension_aspect_does_not_dilute_correct_caliber`）；
维度三态能力由 §3 矩阵的独立用例集覆盖（`tests/test_caliber_dimension_compare.py`，42 例）。