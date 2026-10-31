# V4：MysqlLoader 核验 + 修复 + 真实库 mysql:8 验证

## 目标

1. 修复 P2-f：`_parse_type` 不接受 MySQL `COLUMN_TYPE` 的 `unsigned`/`zerofill` 后缀，
   `decimal(10,2) unsigned` 整串不匹配导致 precision/scale 静默丢失（与 docs/specs/metadata.md 矛盾）。
2. 修复 P2：BASE TABLE 过滤此前无测试保护，补桩测试钉住 `load_tables` 与 `load_columns` 两处谓词。
3. 真实库验证：wslc 启动 mysql:8，建样例库真实抽取，核验输出列契约、注释、类型解析、order_no 语义，
   并固化为 `tests/test_real_mysql.py`（integration 标记，默认 skip）。

## 改动文件清单

| 文件 | 改动 |
|---|---|
| `src/govio/metadata/mysql_loader.py` | 新增 `_MODIFIER_RE`，`_parse_type` 解析前剥离 `unsigned`/`zerofill` 后缀（可叠加，如 `int(10) unsigned zerofill`）；dtype 保留原始 COLUMN_TYPE 不变 |
| `tests/test_mysql_loader.py` | 桩按 `TABLE_TYPE` 分发（SQL 缺谓词则视图行泄漏、断言失败，谓词被钉住）；fixture 加视图 `user_view`；新增 `test_mysql_loader_excludes_views`、`test_mysql_loader_type_parsing_unsigned`；`test_parse_type` 增加 `decimal(10,2) unsigned` / `decimal(10,2) unsigned zerofill` / `int(10) unsigned zerofill` 用例 |
| `tests/test_real_mysql.py` | 新建：真实库 integration 用例（5 个），`@pytest.mark.integration` + `GOVIO_REAL_DB != "1"` 时 skip；样例库 `govio_it` 自建自删（幂等） |
| `docs/tasks/V4-mysql-verify/` | 本目录：progress.md、status.json |

未改动 `docs/specs/metadata.md`：该文档第 106 行描述的契约（`decimal(p,s)` → precision/scale、
`varchar(n)`/`char(n)` → size、其余 → (0,0,0)、`order_no` 取 `ORDINAL_POSITION`）在修复后与代码
**一致**（此前是代码不达标，修复即向文档对齐），无过时内容；且本任务所有权清单限定
只改上述三类文件，docs/specs/ 明确不在内。可选的后续润色：在该行补一句
"数值类型的 unsigned/zerofill 后缀解析前剥离"，属 docs/specs/metadata.md 所有者。

## 真实库验证记录

容器：`wslc run -d --name govio-mysql8 -e MYSQL_ROOT_PASSWORD=govio123 -p 13306:3306 mysql:8`，
就绪探测 `mysqladmin ping` 约 20 秒通过。验证结束已 `wslc stop` + `wslc remove` 清理。

样例库 `govio_it`（DDL 见 `tests/test_real_mysql.py::_SAMPLE_DDL`）：

- `users`（表注释"用户表"，列注释"用户ID"/"用户名"，`email` 无注释）
- `orders`（无表注释，`amount DECIMAL(10,2) UNSIGNED COMMENT '金额'`、
  `qty INT(10) UNSIGNED ZEROFILL`、`note VARCHAR(255)`、`marker VARCHAR(0)`）
- `audit_log`（先建 3 列后 `DROP COLUMN old_col`，验证 order_no）
- 视图 `user_view`（`CONCAT(name, '!') AS label`，验证 BASE TABLE 过滤）

真实抽取核验（`MysqlLoader(host="localhost", port=13306, user="root", password="govio123", schemas=["govio_it"])`）：

1. **列契约**：`load_tables` / `load_columns` 输出列与 `_TABLE_COLUMNS` / `_COLUMN_COLUMNS` 完全一致 ✅
2. **注释**：表注释"用户表"/"审计表" 进 `name`；`orders` 空注释回退 `orders`；列注释"用户ID"/"金额" 进 `name`，无注释回退列名 ✅
3. **类型解析**：`decimal(10,2) unsigned` → (size=0, precision=10, scale=2)（修复前丢为 0,0,0）；
   `int(10) unsigned zerofill` → (0,0,0)；`varchar(64)` → (64,0,0)；`varchar(0)` → (0,0,0) ✅
4. **BASE TABLE 过滤**：`user_view` 表与其列均不出现在输出中 ✅
5. **order_no 语义**：mysql:8 对 `ORDINAL_POSITION` **删列后自动重排**（含 `ALGORITHM=INSTANT`
   删首列，实测 `gap2` 删 a 后 b/c/d 为 1/2/3，再删 c 后 b/d 为 1/2），
   `audit_log` 删 `old_col` 后 order_no 为 [1, 2]，每表连续从 1 递增，
   `recommender.py:344` 依赖的首列 `order_no==1` 未被破坏，**无需按 PostgresLoader 方案重排** ✅

### 关于"varchar 无长度"

mysql:8 中不可达：基础表 DDL `VARCHAR` 不带长度是语法错误（`VARCHAR(0)` 合法，COLUMN_TYPE
为 `varchar(0)`）；视图表达式列（CONCAT/CAST/IFNULL/递归 CTE 等 20+ 种实测）均被解析为
带具体长度的 `varchar(n)`。故样例库用 `marker VARCHAR(0)` 覆盖零长度形态，
裸 `varchar` 的 `(0,0,0)` 防御分支由单测 `test_parse_type` 与集成测试内断言钉住。

## 验证命令与结果

| 命令 | 结果 |
|---|---|
| `uv run pytest tests/test_mysql_loader.py -q` | 30 passed |
| `GOVIO_REAL_DB=1 uv run pytest tests/test_real_mysql.py -v` | 5 passed（真实 mysql:8） |
| `uv run pytest tests/ -q`（无 GOVIO_REAL_DB，容器无关） | 291 passed, 6 skipped |
| `uv run ruff check src/govio/metadata/mysql_loader.py tests/test_mysql_loader.py tests/test_real_mysql.py` | All checks passed |
| `uv run ruff format --check`（同上三文件） | 3 files already formatted |

> 注：本机 shell 下 `uv run` 报 "Failed to canonicalize script path"，验证用
> `.venv/Scripts/python.exe -m ...` 等价执行（同一虚拟环境）。

## 遗留问题

1. `pytest.mark.integration` 标记未在 `pyproject.toml` 注册（所有权外文件），
   运行 `tests/test_real_mysql.py` 时有 `PytestUnknownMarkWarning`（非阻断，测试全绿）。
   建议后续在 `[tool.pytest.ini_options] markers` 注册 `integration: 真实库集成测试`。
2. docs/specs/metadata.md MysqlLoader 小节可选补一句 unsigned/zerofill 后缀说明（见上）。
3. `GOVIO_REAL_DB=1` 时集成测试假定 mysql:8 已在 localhost:13306 就绪（连接参数可由
   GOVIO_MYSQL_HOST/PORT/USER/PASSWORD 覆盖），容器生命周期由外部命令管理（见上文）。
