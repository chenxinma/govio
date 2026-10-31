# I2 补齐回归测试数据并对齐 CLI 行为契约 — 进度记录

分支 `fix/regression-tests`（基于 main @08fe8ee，全程无 git 写操作）。
结论：tests/regression 原 20 个失败（13 ERROR + 7 FAILED）全部清零；
全量 `uv run python -m pytest tests/ -q` **342 passed, 18 skipped**；
`uv run ruff check src/ tests/` 通过。

---

## 一、原 7 项 FAILED 偏差对照表（测试期望 vs 原实现 vs 处置与理由）

| # | 用例 | 测试期望 | 原实现 | 处置与理由 |
|---|---|---|---|---|
| 1 | `test_q01_networkx_named_file_inline_and_ascii` | 6 个 Col 节点；`rows[0] == {"0": "co00000001", "1": "customer_id"}`；小结果内联；stdout 全 ASCII | graph_csv/ 未入库 → conftest 建出**空 GML**（gen_networkx 对缺失 CSV 静默跳过，夹具不报错）→ 断言失败 | **数据根因，非行为偏差**：重建 `graph_csv/` 5 个 CSV 后 src 零改动即通过（输出契约本已正确）。处置 = 补数据 |
| 2 | `test_q05_networkx_missing_gml_file` | `code == 1` 且 `"GML file not found"` 出现在 **stdout** | `_fail()` 仅把 `❌ 文案` 打到 stderr，stdout 为空 | **src 修复**：`_fail()` 双路输出——原文（无前缀）落 stdout、`❌ ` 前缀落 stderr。与 `test_exit_codes.py`（断言 `❌` 不进 stdout、文案进 stderr）完全兼容 |
| 3 | `test_q11_ladybug_missing_db_path` | `code == 1` 且 `"配置文件缺少 graph.ladybug.db_path 字段"` 在 stdout | 同 #2，文案只进 stderr | 同 `_fail()` 双路修复 |
| 4 | `test_q12_missing_config` | `code == 1` 且 `"配置文件不存在，请先运行 govio-cli onboard"` 在 stdout | 同 #2 | 同 `_fail()` 双路修复 |
| 5 | `test_q13_missing_backend_field` | `code == 1` 且 `"配置文件缺少 'graph.backend' 字段"` 在 stdout | 同 #2 | 同 `_fail()` 双路修复 |
| 6 | `test_q14_unsupported_backend` | `code == 1` 且 `"不支持的 backend: neo4j"` 在 stdout | 同 #2 | 同 `_fail()` 双路修复 |
| 7 | `test_l06_memory_without_loaded_data` | `code == 0` + `success=false` + error 含 `"没有已加载的 DataFrame"` | I1 契约（def3501）：load 失败退出 **1** | **测试侧最小改**（`assert code == 0` → `== 1`）：退出 0 期望与 `docs/specs/cli.md`「退出码契约（observe load）」表（SQL 错误/内存加载失败 → 1）及 `tests/test_exit_codes.py` 的 14 个已钉用例直接冲突，三方不可能同时成立；按裁决「契约冲突才最小改测试并注明依据」处理，且父代理已确认：**用户新契约（I1）优先于旧测试期望**。`success`/`error` 文案断言原样保留，未弱化 |

> 复盘修正：#1 当初与 #2-#6 一并被归为"行为契约偏差"，实测证明是**测试数据缺失**的连带表现（空图上执行查询仍返回退出 0 与合法输出格式，仅数据内容不符断言）。行为偏差实为 6 项（#2-#7）。

## 二、数据补齐后新暴露的 4 项偏差（原为 ERROR，随数据到位才进入断言阶段）

| # | 用例 | 测试期望 | 原实现 | 处置与理由 |
|---|---|---|---|---|
| 8 | `test_l07_bad_sql_returns_error` | `code == 0` + `error.startswith("SQL 执行错误")` | 失败退出 1（I1 契约） | 测试侧最小改 `code == 0` → `== 1`（依据同 #7）；error 文案与不落盘断言全部保留 |
| 9 | `test_l08_unknown_datasource_returns_error` | `code == 0` + `"数据源不存在"` in error | 失败退出 1（I1 契约） | 同 #8 |
| 10 | `test_q09_ladybug_rejects_non_match` | `out.strip() == "Please write a MATCH query."`（stdout **精确匹配**）+ `code == 1` | 文案仅 stderr（`❌ ` 前缀），stdout 为空 | `_fail()` 双路修复后 stdout 为无前缀原文，精确匹配成立；退出码断言本就符合 |
| 11 | `test_q10_ladybug_invalid_cypher_raises` | `ValueError("Generated Cypher Statement is not valid...")` 从 `run_cli` 向上传播 | `query()`/`cmd_*` 把查询异常统一兜成 `_fail → SystemExit(1)`，ValueError 被吞 | **src 修复**：`cmd_falkordb`/`cmd_ladybug` 的 `query_with_columns()` 调用与 `query()` 兜底处放行 `ValueError`（图层 `query_with_columns` 的包装契约），其余异常仍走 `_fail`；`cmd_networkx` 的 exec 失败仍退出 1（test_exit_codes 钉住） |

## 三、测试数据文件 ↔ 断言映射

### `tests/regression/data/seed.sql`（DuckDB 种子，经 `sample_duckdb` 夹具执行）

| 数据 | 断言映射 |
|---|---|
| `customers` 列 `customer_id/name/region/credit`，5 行 | L02 records 全部逐字段断言：`records[0].name == "张三"`（中文）、`records[1].region is None`（NULL→null）、`records[2].credit == -20.25`（负数）、`records[3].name == "O'Brien 李"` + `records[3].credit is None`（引号转义 + NULL）、`records[4].name == ""`（空串保留） |
| `orders` 列 `order_id/customer_id/amount/status/created_at`，5 行 | L01/L04 的 `ORDER_COLUMNS` 列序与列数 5；L01 `order_id == [1001..1005]`、`amount[0] == 120.5` |
| `orders.amount` 在 1002 为 NULL | README「NULL 金额」；L02 NULL→null 契约的金额侧对照 |
| `orders.customer_id=9999`（1005，孤儿外键） | L05 `rows == 4`（INNER JOIN 丢弃孤儿行）+ `customer_name == ["张三","张三","李四","王五"]`（1001/1002→客户1、1003→客户2、1004→客户3） |
| `created_at` 为 DATE | L01 列结构（dtype 非空）；README「日期」边界 |

### `tests/regression/data/graph_csv/`（FalkorDB 批量导入格式，经 `nx_gml`/`ladybug_db` 夹具分别构建 ontology.gml / ontology.lbdb）

| 文件 | 内容 | 断言映射 |
|---|---|---|
| `PhysicalTable.csv` | `pt00000001 customers/客户信息表`、`pt00000002 orders/订单明细表` | Q06：`RETURN t.name AS table_name, t.comment AS comment` 期望 `[{"table_name":"customers","comment":"客户信息表"}, {"table_name":"orders","comment":"订单明细表"}]`；中文注释进 UTF-8 文件、stdout 全 ASCII |
| `Col.csv` | `co00000001..co00000006` 共 6 个 Col（customers: customer_id/name/region；orders: order_id/amount/status，order_no 1/2/3） | Q01：`rows == 6` 且 `rows[0] == {"0": "co00000001", "1": "customer_id"}`（节点序 = Col.csv 行序）；Q07：orders 三列按 `order_no` 1/2/3 排序返回 `col/ord` |
| `HAS_COLUMN.csv` | 6 条 `PhysicalTable→Col` 边 | Q07 跨节点联查 `(t:PhysicalTable)-[:HAS_COLUMN]->(c:Col)` |
| `Datasource.csv` | `ds00000001 local`（source_type=duckdb，filter JSON） | README 图结构约定（Datasource/OWNS 完整性）；Q01-Q10 无直接断言 |
| `OWNS.csv` | `local → customers / orders` 2 条边 | 同上 |

Q02（标量包装）/Q03（60 行超内联阈值）/Q04（空结果不写文件）不依赖图内容，仅要求 GML 可加载。

数据文件均为相对/逻辑标识（`local.main.customers` 等），不含机器相关路径；绝对路径由 conftest 夹具在 tmp_path 下动态拼接。

## 四、测试文件改动清单（共 3 处，全部为退出码对齐；无删断言、无放宽值）

`tests/regression/test_load_cli.py`：L06/L07/L08 的 `assert code == 0` → `assert code == 1`，各附一行依据注释
（`docs/specs/cli.md` 退出码契约 + test_exit_codes 冲突）。三处的 `success is False`、error 文案、不落盘断言全部原样保留并通过。
`conftest.py`、`test_query_cli.py`、`README.md` 零改动。

## 五、src 改动

`src/govio/cli/query.py`（3 处）：

1. `_fail()`：报错文案双路输出——原文（无前缀）落 stdout，`❌ ` 前缀落 stderr（stdout 不出现 ❌）。
2. `cmd_falkordb()`/`cmd_ladybug()`：`query_with_columns()` 抛出的 `ValueError` 放行重抛（非法 Cypher 包装契约），其余异常仍 `_fail` 退出 1。
3. `query()` 兜底：`except ValueError: raise` 置于兜底 `except Exception` 之前，保证图层包装的 ValueError 不被吞成 SystemExit。

`src/govio/cli/observe.py`、`load_dataframe.py` 无需改动（L06-L08 的 JSON 契约与退出 1 行为与 I1 契约一致）。

## 六、文档同步

- `docs/specs/cli.md`「退出码契约（query）」表：输出列改为双路描述（stdout 原文 + stderr ❌）；「Cypher 语句非法/执行错误」单列为 ValueError 异常传播行；约定行改写为双路输出语义。`_fail` 签名注释同步。
- `docs/specs/observe-data.md`：load 失败语义未变（实现未动），与 cli.md 契约表一致，无需修改。

## 七、验证记录

| 命令 | 结果 |
|---|---|
| `uv run python -m pytest tests/ -q` | 342 passed, 18 skipped（含 tests/regression 25 个、tests/test_exit_codes.py 14 个全绿） |
| `uv run python -m pytest tests/regression tests/test_exit_codes.py -q` | 39 passed |
| `uv run ruff check src/ tests/` | All checks passed! |

18 个 skipped 为真实外部库集成用例（mysql/postgres/trino 等），与本任务无关。
