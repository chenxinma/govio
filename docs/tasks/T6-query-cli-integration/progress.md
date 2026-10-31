# T6 多数据库 CLI 接入与查询执行统一 — progress.md

## 目标

把 mysql / postgres / trino 接入元数据导入（`govio-cli meta meta --source`）与 observe 查询执行链路：

1. `meta meta --source` 新增 `mysql` / `postgres` / `trino` 取值（原 `tds` / `duckdb`）
2. observe 查询执行支持 postgresql 与 trino，核实 mysql 可用
3. 新增打桩测试（不依赖真实数据库）
4. `docs/specs/metadata.md`、`cli.md`、`observe-data.md` 与实现严格同步

## `--db` 参数语义（已在 cli.md 文档化）

| `--source` | `--db` 格式 | 示例 |
|---|---|---|
| `duckdb` | DuckDB 文件路径 | `./data/local.duckdb` |
| `mysql` | 连接 URL（scheme: `mysql` / `mysql+pymysql`） | `mysql://user:pass@host:3306/db` |
| `postgres` | 连接 URL（scheme: `postgres` / `postgresql` / `postgresql+psycopg`） | `postgresql://user:pass@host:5432/db` |
| `trino` | 连接 URL（scheme: `trino` / `trino+http` / `trino+https`） | `trino://user:pass@host:8080/catalog` |

- URL 省略 user / port 时取各加载器默认值（mysql: `root`/3306，postgres: `postgres`/5432，trino: `trino`/8080）
- trino URL 路径必须恰好为 catalog 一段，schema 由 `--schemas` 指定；含密码时构造 `trino.auth.BasicAuthentication`（须同时给 user）；`trino+https` 表示 HTTPS
- URL scheme 与 `--source` 不匹配、URL 非法或缺 host → `step_meta_export` 打印 `❌ ...` 返回 None，`cmd_meta` 退出码 1
- URL 解析用 SQLAlchemy 的 `make_url`（复用已装依赖，正确处理转义/IPv6/非法端口）

## 驱动 / 查询执行选型（observe）

统一走 SQLAlchemy 引擎 + `pandas.read_sql`，理由：

1. `DatabaseManager` 现状即「非 duckdb 一律 create_engine」，扩展只需 scheme 归一，无需新增后端分支
2. trino 选 **trino 包自带 `trino.sqlalchemy` 方言**而非裸 dbapi：engine 懒连接、`connect_args` 透传（`http_scheme`/证书等）、与 mysql/postgres 共用同一 `execute_sql` 路径；`pd.read_sql` 对该方言可用（engine 离线创建已验证）
3. postgres 走 **sqlalchemy + psycopg3**（`postgresql+psycopg://`，psycopg[binary] 已是依赖）；裸 `postgresql://` 默认找 psycopg2（未安装），故归一
4. mysql 现状已走 sqlalchemy，但裸 `mysql://` 默认找 mysqldb（未安装），同样归一到 `mysql+pymysql://`
5. `trino+https://` 归一为 `trino://` + `connect_args.http_scheme="https"`（trino 方言只注册 `trino` 驱动，HTTPS 由 trino dbapi 的 `http_scheme` 控制）；`config.connect_args` 同名键优先

## 改动文件清单

| 文件 | 改动 |
|---|---|
| `src/govio/cli/meta.py` | `--source` 增加 mysql/postgres/trino；新增 `_build_metadata_loader`（URL → Loader 构造参数）、`SOURCE_LABELS` / `SCHEMA_HINTS` / `SOURCE_URL_SCHEMES` / `SOURCE_URL_DEFAULTS`；`step_meta_export` / `cmd_meta` 非 TDS 分支泛化（`source_type` 取 `--source`，报错风格与 duckdb 分支一致）；argparse help 同步 |
| `src/govio/observe_data/core/database.py` | 新增 `_resolve_sqlalchemy_target`（scheme 归一：mysql→mysql+pymysql、postgres/postgresql→postgresql+psycopg、trino+https→trino+http_scheme）；`_ensure` 对归一后 URL + 合并 connect_args 调 create_engine |
| `tests/test_meta_cli_sources.py` | 新建：16 个用例（argparse 取值/拒绝、参数必填、URL→构造参数传递、scheme 不匹配/非法 URL/trino 缺 catalog、全链路 CSV 导出与 source_type），全部打桩 Loader |
| `tests/test_observe_multi_db.py` | 新建：16 个用例（scheme 归一参数化、create_engine 参数传递、execute_sql 引擎路由、duckdb 原生回归、load_dataframe 链路），打桩 create_engine/read_sql |
| `docs/specs/metadata.md` | 新增 MysqlLoader / PostgresLoader 小节，TrinoLoader 补构造参数与方法 |
| `docs/specs/cli.md` | `meta meta` 支持来源、`--db` 语义表、参数守卫提示表（修正过时的 "duckdb without --db" 行）、`step_meta_export` / `_build_metadata_loader` 说明 |
| `docs/specs/observe-data.md` | DatabaseManager URL scheme 归一表、统一查询执行路径说明 |
| `docs/tasks/T6-query-cli-integration/` | 本目录：progress.md、status.json |

`src/govio/cli/main.py` 无需改动：`meta` 子命令在 main.py 中仅以 `argparse.REMAINDER` 透传参数，`--source` 等 argparse 定义实际在 `meta.py` 的 `meta()` 中（已核实并更新）。

## 验证命令与结果

```text
uv run python -m pytest tests/test_meta_cli_sources.py tests/test_observe_multi_db.py tests/test_trino_loader.py -q
  → 53 passed
uv run ruff check src/ tests/
  → All checks passed!
uv run python -m pytest tests/test_meta_cli_guard.py tests/test_meta_export_ids.py -q
  → 33 passed（未被破坏）
uv run python -m pytest tests/ -q --ignore=tests/test_metadata_benchmark.py
  → 257 passed, 1 skipped, 1 failed（tests/test_query_output.py::test_named_records_file_and_ascii_inline，
    Decimal 序列化 822 vs '822'，为改动前已存在的失败，query.py 不在导入链/任务范围内）
```

（注：本环境 `uv run pytest` 直调报 "Failed to canonicalize script path"，改用等价的 `uv run python -m pytest`。）

## 遗留问题

1. `tests/test_query_output.py::test_named_records_file_and_ascii_inline` 预先存在失败（`govio.cli.query.output_result` 的 Decimal 序列化），与本任务无关，未处理（不在任务所有权清单内）。
2. `docs/specs/config.md` 的 "Datasource URL Formats" 表尚无 PostgreSQL 行（observe 现已支持 `postgresql://`），该文件不在任务所有权清单内，未改动。
3. trino URL 的 query 参数（`access_token` 等高级认证）在 `meta meta` 路径未解析（仅 user/password + BasicAuthentication）；observe 路径可经 `config.connect_args` 透传。
