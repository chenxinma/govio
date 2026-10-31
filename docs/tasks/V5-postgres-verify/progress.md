# V5：PostgresLoader 核验 + 修复 + 真实库 postgres:17 验证

## 目标

1. 修复 P1-1（本轮最高优先级，已复证）：`_parse_type` 对 pandas NaN 崩溃——
   `_fetch` 用 `pd.DataFrame(rows)` 承接 int+NULL 混合列时 pandas 推断为 float64、
   SQL NULL 变 NaN，varchar/bit 分支的 `int(x) if x else 0` 与 numeric/decimal 分支的
   `int(x) if x is not None else 0` 遇 NaN 均抛 `ValueError: cannot convert float NaN to integer`。
2. 修复 P2：Postgres 连接关闭无断言——镜像 `tests/test_mysql_loader.py` 的 `closed`
   标志与 `test_mysql_loader_closes_connection`。
3. 真实库验证：wslc 启动 postgres:17，建样例库真实抽取，核验列契约/注释/类型解析/
   order_no（attnum 重排），固化为 `tests/test_real_postgres.py`（integration 标记，默认 skip）。

## 改动文件清单

| 文件 | 改动 |
|---|---|
| `src/govio/metadata/postgres_loader.py` | `_parse_type` 两个分支判空统一改用 `pd.notna`（原真值判断 / `is not None` 均不识别 NaN）；参数注解 `int \| None` → `float \| None`（DataFrame 推断产物含 NaN float，ruff PYI041 收敛为 `float \| None`）；docstring 补 NaN 语义说明 |
| `tests/test_postgres_loader.py` | `FakePostgresConn` 加 `closed` 标志、`make_loader` 记录 `_conns`（镜像 test_mysql_loader.py）；新增 `test_postgres_loader_closes_connection`（P2）、`test_postgres_loader_mixed_null_attrs_nan`（P1-1 回归：混合列 + NULL 属性经 `_fetch` → `pd.DataFrame` 推断成 NaN 的真实路径，含 NaN 前提断言，不只测 Python None）；`test_parse_type` 参数化补 5 个 NaN 用例（varchar/bit/numeric/decimal 各分支） |
| `tests/test_real_postgres.py` | 新建：真实库 integration 用例（6 个），`@pytest.mark.integration` + `GOVIO_REAL_DB != "1"` 时 skip；样例库 `govio_it`（数据库 + 同名 schema）自建（幂等 drop + create） |
| `docs/tasks/V5-postgres-verify/` | 本目录：progress.md、status.json |

未改动 `docs/specs/metadata.md`：该文档 PostgresLoader 小节描述的契约
（`character varying` / `character` / `bit(n)` → size、`numeric` / `decimal` →
precision/scale、其余 → (0,0,0)、order_no 重排语义）在修复后与代码**一致**
（此前是代码不达标，修复即向文档对齐），无过时内容；且本任务所有权清单限定
只改上述文件，docs/specs/ 明确不在内（同 V4 判断）。

## P1-1 复现与修复说明

机制：`_fetch` 的 `pd.DataFrame(rows, columns=names)` 承接 psycopg 返回的
int+NULL 混合属性列时推断为 float64，SQL NULL 变 NaN（实测 `np.float64(nan)`）。
NaN 既非 None 也非 falsy：

- varchar/bit 分支 `int(x) if x else 0`：NaN 为真值 → `int(NaN)` → ValueError；
- numeric/decimal 分支 `int(x) if x is not None else 0`：NaN is not None → 同崩。

场景：同批结果里存在**无长度 varchar 或无精度 numeric**（与带长度/精度列混批，
使属性列成为混合列）时几乎必现；全 NULL 属性列是 object dtype 保留 Python None，
不触发（这正是旧单测未发现的原因）。

修复：两个分支判空统一 `pd.notna`（对 Python None 与 NaN 均返回 False，
None 路径行为不变；`character_maximum_length=0` 结果也不变）。

## 真实库验证记录

容器：`wslc run -d --name govio-pg17 -e POSTGRES_PASSWORD=govio123 -p 15432:5432 postgres:17`，
就绪探测 `wslc exec govio-pg17 pg_isready -U postgres` 一次通过。验证结束已
`wslc stop govio-pg17` + `wslc remove govio-pg17` 清理，`wslc list` 无残留。

样例库 `govio_it`（DDL 见 `tests/test_real_postgres.py::_SAMPLE_DDL`）：

- `users`（表注释"用户表"，列注释"用户ID"/"用户名"，`email` 无注释）
- `orders`（无表注释，`amount numeric(10,2)` 列注释"金额"、`qty integer`、`order_id bigint`）
- `p1_case`（P1-1 复现表：`a integer` + `b numeric` 无精度 + `c varchar` 无长度）
- `audit_log`（先建 3 列后 `DROP COLUMN old_col`，attnum 空洞 1,3 → order_no 重排）
- 视图 `user_view`（`name || '!' AS label`）
- 外表 `ext_users`（postgres_fdw，指向 `govio_it.users`，表注释"外表样本"）

真实抽取核验（`PostgresLoader(host="localhost", port=15432, user="postgres",
password="govio123", dbname="govio_it", schemas=["govio_it"])`）：

1. **列契约**：`load_tables` / `load_columns` 输出列与 `_TABLE_OUTPUT_COLUMNS` /
   `_COLUMN_OUTPUT_COLUMNS` 完全一致 ✅
2. **注释**：表注释"用户表"/"审计表"/"外表样本" 进 `name`；`orders` / `p1_case`
   空表注释回退表名；列注释"用户ID"/"用户名"/"金额" 进 `name`，无注释回退列名 ✅
3. **类型解析**：`varchar(64)` → (64,0,0)；`numeric(10,2)` → (0,10,2)；
   `integer` 位宽不计入 precision → (0,0,0)；**裸 `numeric` / 裸 `varchar` → (0,0,0)**
   （修复前在此崩溃）✅
4. **P1-1 复现路径**：真实抽取数据上 `character_maximum_length` / `numeric_precision`
   确为 float64 NaN（测试内断言 `pd.isna`）；旧逻辑在真实数据上实测抛
   `ValueError: cannot convert float NaN to integer`，修复后 `load_columns` 正常 ✅
5. **视图/外表**：`user_view`（id, label）与 `ext_users`（id, name）的列均导出，
   与 TrinoLoader `SHOW TABLES` 范围一致，共 15 列 ✅
6. **order_no（attnum 重排）**：`audit_log` 删列后 attnum 空洞（1,3），
   order_no 重排为 [1, 2] 列序 [log_id, action]；全部表 order_no 连续从 1 递增 ✅

## 验证命令与结果

| 命令 | 结果 |
|---|---|
| `uv run pytest tests/test_postgres_loader.py -q` | 34 passed |
| `GOVIO_REAL_DB=1 uv run pytest tests/test_real_postgres.py -v` | 6 passed（真实 postgres:17） |
| `uv run pytest tests/test_real_postgres.py -q`（无 GOVIO_REAL_DB） | 6 skipped |
| `uv run pytest tests/ -q`（无容器，全量） | 298 passed, 12 skipped |
| `uv run ruff check`（上述三文件） | All checks passed |
| `uv run ruff format --check`（同上三文件） | 3 files already formatted |

> 注：本机 shell 下 `uv run` 报 "Failed to canonicalize script path"（同 V4），
> 验证用 `.venv/Scripts/python.exe -m ...` 等价执行（同一虚拟环境）。

## 遗留问题

1. `pytest.mark.integration` 标记未在 `pyproject.toml` 注册（所有权外文件），
   运行集成测试时有 `PytestUnknownMarkWarning`（非阻断，测试全绿，同 V4 遗留）。
2. `GOVIO_REAL_DB=1` 时集成测试假定 postgres:17 已在 localhost:15432 就绪
   （连接参数可用 GOVIO_PG_HOST / GOVIO_PG_PORT / GOVIO_PG_USER /
   GOVIO_PG_PASSWORD 覆盖），容器生命周期由外部命令管理（见上文）。
3. `tests/test_real_postgres.py` 夹具直接调用 `loader._fetch` 取证 NaN 前提，
   属有意为之的真实路径验证；若后续将 `_fetch` 改为公开 API 需同步该测试。
