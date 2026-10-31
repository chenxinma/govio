# V6 CLI/查询执行核验（T6）— 进度记录

## 目标

1. P2-a：scheme 处理两处不对称修复
   - `observe_data/core/database.py` `_resolve_sqlalchemy_target` 补 `trino+http://`
     → `trino://` + `connect_args {"http_scheme": "http"}`（`config.connect_args`
     显式值仍优先）；`trino+http(s)` 原先直接进 `create_engine` 会触发
     NoSuchModuleError（trino 方言只注册 trino 驱动）。
   - `cli/meta.py` 的 `url.drivername` 精确匹配改大小写不敏感（`MySQL://` 等写法
     与 observe 的 `scheme.lower()` 行为统一）；trino http_scheme 判定同步改
     大小写不敏感。
2. P2-g：`tests/test_meta_cli_sources.py::test_trino_url_missing_user_with_password_rejected`
   补断言退出码 == 1 与报错文案含「必须同时指定 user」。
3. P2-c：`docs/specs/config.md`「Datasource URL Formats」表补齐
   postgresql://、postgres://、裸 mysql:// 归一与 trino+http/https 语义
   （与 observe-data.md 归一表一致）。
4. 真实库端到端（trino:483 / mysql:8 / postgres:17）：`DatabaseManager.execute_sql`
   统一路径真实查询 + `govio-cli meta meta --source mysql/postgres/trino` 真实抽取
   到 CSV 并核对产物，固化为 `tests/test_real_observe.py`（integration 默认 skip）。

## 改动文件清单

| 文件 | 改动 |
|---|---|
| `src/govio/observe_data/core/database.py` | `_resolve_sqlalchemy_target` 新增 `_TRINO_URL_HTTP_SCHEMES`（trino+http→http、trino+https→https），两者均归一为 `trino://` + `connect_args.http_scheme`；`config.connect_args` 同名键优先的合并顺序不变（`{**extra, **config}`） |
| `src/govio/cli/meta.py` | `url.drivername` 匹配改 `.lower()` 大小写不敏感（scheme 校验 + trino http_scheme 判定两处） |
| `tests/test_meta_cli_sources.py` | P2-g：补 `exc.value.code == 1` 与「必须同时指定 user」断言；新增 `test_url_scheme_match_is_case_insensitive`、`test_trino_uppercase_https_scheme_builds_https_auth` |
| `tests/test_observe_multi_db.py` | 归一参数表补 `trino+http://`、`TRINO+HTTP://` 用例；新增 `test_create_engine_trino_http_merges_connect_args`（含 config.connect_args 同名键优先断言） |
| `tests/test_real_observe.py` | 新建：真实库 integration 用例 6 个，`@pytest.mark.integration` + `GOVIO_REAL_DB != "1"` 时 skip；样例 schema `govio_observe_it` 三库自建（幂等 drop + create） |
| `docs/specs/config.md` | 「Datasource URL Formats」表补齐 mysql/postgresql/trino 归一语义 + 大小写不敏感说明 |
| `docs/specs/observe-data.md` | URL Scheme Normalization 表补 `trino+http://` 行 |
| `docs/tasks/V6-cli-verify/` | 本记录 + status.json |

## 验证命令与结果

| 命令 | 结果 |
|---|---|
| `uv run python -m pytest tests/test_meta_cli_sources.py tests/test_observe_multi_db.py -q` | 37 passed |
| `uv run python -m pytest tests/ -q`（无容器） | 303 passed, 18 skipped |
| `uv run ruff check src/ tests/` | All checks passed |
| `GOVIO_REAL_DB=1 uv run python -m pytest tests/test_real_observe.py -v` | 6 passed（真实 trino:483 + mysql:8 + postgres:17） |
| `uv run python -m pytest tests/test_real_observe.py -q`（无 GOVIO_REAL_DB） | 6 skipped |
| `govio-cli meta meta --source mysql --db mysql://root:***@localhost:13306/govio_it --schemas govio_it ...` | ✓ 1 数据源/2 表/6 字段，5 个 CSV 产物，中文注释（用户表/用户ID/金额）原样落盘 |
| `govio-cli meta meta --source postgres --db postgres://postgres:***@localhost:15432/postgres ...` | ✓ 1 数据源/2 表/6 字段，CSV 核对通过 |
| `govio-cli meta meta --source trino --db trino+http://govio@localhost:18080/memory ...` | ✓ 1 数据源/2 表/6 字段（trino+http scheme 走新归一路径） |
| `DatabaseManager.execute_sql` 对 trino+http / 裸 mysql / 裸 postgres 真实查询 | 三库均返回期望 2 行数据（统一 SQLAlchemy 路径） |

### 真实库容器（wslc）使用记录

- 启动：`wslc run -d --name govio-trino483 -p 18080:8080 trino:483`、
  `wslc run -d --name govio-mysql8 -e MYSQL_ROOT_PASSWORD=govio123 -p 13306:3306 mysql:8`、
  `wslc run -d --name govio-pg17 -e POSTGRES_PASSWORD=govio123 -p 15432:5432 postgres:17`
- 就绪探测：pg_isready / mysqladmin ping / trino `/v1/info` 轮询，约 10s 全就绪
- 清理：`wslc stop` + `wslc remove` 三个容器，`wslc list` 无残留

### 真实库验证细节

- trino:483 自带 catalogs：jmx / memory / system / tpcds / tpch；样例数据建在
  `memory.govio_observe_it`（CREATE SCHEMA/TABLE + INSERT）。
- DatabaseManager 统一路径验证用 `trino+http://`（P2-a 修复点：原实现 NoSuchModuleError）、
  裸 `mysql://`、裸 `postgres://` 三种归一写法各查 `SELECT id, name ...`。
- meta CLI 抽取产物核对：PhysicalTable.csv / Col.csv / Datasource.csv /
  HAS_COLUMN.csv / OWNS.csv，表全限定名
  `{datasource}.{schema}.{table}`，Datasource.source_type 取 --source。

## 遗留问题

1. `wslc exec ... mysql -e "<含中文的 DDL>"` 链路（Git Bash → wslc → WSL → 容器）
   会把命令行里的中文双重编码（存进库即 mojibake，HEX 实证 C3A7E2809D...）；
   改用宿主侧 pymysql（charset=utf8mb4）建样例后注释正确。集成测试的建库
   fixture 已统一走宿主 Python 客户端，不受该陷阱影响；人工 wslc exec 建含中文
   注释的样例库时需避免把中文放在命令行参数里。
2. `uv run pytest ...` 在当前 Git Bash 环境报 "Failed to canonicalize script path"
   （uv 脚本解析问题），等价命令 `uv run python -m pytest ...` 正常；本次验证
   均用后者。
3. `pytest.mark.integration` 未在 pyproject 注册（PytestUnknownMarkWarning），
   与 V4/V5 现状一致；注册属 pyproject.toml 所有者决策，本轮未动。
4. mysql / postgres 真实抽取核验用例（`test_real_mysql.py` / `test_real_postgres.py`）
   不在本任务所有权内，未改动；本任务真实库用例独立为 `test_real_observe.py`，
   样例 schema 命名 `govio_observe_it` 避免与 V4/V5 的 `govio_it` 相互覆盖。
