# T5 PostgreSQL 元数据加载器 — progress.md

## 目标

新增 `PostgresLoader(MetadataLoader)`，从 PostgreSQL 抽取表/列元数据：

- 表/列来自 `information_schema.tables` / `information_schema.columns`
- 表注释 `obj_description(oid, 'pg_class')`、列注释 `col_description(oid, attnum)`（`ordinal_position` 即 `pg_attribute.attnum`）
- 类型解析（`_parse_type`）：`character varying/character/bit(n)` → size；`numeric/decimal` → precision/scale；其余 → (0,0,0)（整型 `numeric_precision` 为位宽语义，不计入 precision，与 TrinoLoader 语义一致）
- 输出 DataFrame 列结构与 TrinoLoader/DuckDBLoader 完全一致，`data_entity_type` 用 `POSTGRES_TABLE` / `POSTGRES_COLUMN`
- 支持 `datasource_name` 全限定（`qualify`）与 schema 过滤（`table_schema = ANY(%s)` 参数化）
- `order_no` 按 `ordinal_position` 排序后每表从 1 连续递增（删列导致 attnum 空洞时重排，保证 `recommender.py` 依赖的 `order_no == 1` 首列语义）
- 注释为空（NULL / ""）时 `name` 回退为表名/列名
- CLI 接入不在本任务范围（T6 负责）

## 驱动选型

选择 **psycopg[binary]>=3.2.0（psycopg3）**，理由：

1. PostgreSQL 官方维护的现代驱动，支持 Python 3.13；`[binary]` 附带预编译 libpq，Windows/CI 免本地编译。
2. 原生把 Python list 适配为数组，`WHERE table_schema = ANY(%s)` 参数化 schema 过滤无需拼 SQL（降低注入风险）。
3. `cursor.description` 提供 `Column.name`，`_fetch` 模式与 TrinoLoader 一致。
4. 备选方案：psycopg2 已进入维护模式（仅 bugfix）；pg8000 纯 Python 但性能与生态较弱，故不采用。

## 改动文件清单

| 文件 | 改动 |
|---|---|
| `src/govio/metadata/postgres_loader.py` | 新增：`PostgresLoader`、`_parse_type`、两条 SQL（表/列 + 注释） |
| `tests/test_postgres_loader.py` | 新增：27 个用例，monkeypatch 打桩 `psycopg.connect`，不依赖真实数据库 |
| `pyproject.toml` | dependencies 新增 `psycopg[binary]>=3.2.0` |
| `docs/tasks/T5-postgres-loader/` | 本目录：progress.md、status.json |

测试覆盖：表清单、列清单（列结构逐列断言与 Trino/DuckDB 一致）、注释回退（NULL/空串，表+列）、类型解析（`_parse_type` 参数化 + load_columns 集成）、`order_no` 递增与删列空洞重排、空 schema（空 DataFrame + 不发起连接）、多 schema（含未声明 schema 过滤）、`datasource_name` qualify、连接参数透传。

## 验证命令与结果

```text
uv run --frozen --with "psycopg[binary]" python -m pytest tests/test_postgres_loader.py -q
  → 27 passed

uv run --frozen --with "psycopg[binary]" python -m pytest tests/test_postgres_loader.py tests/test_trino_loader.py tests/test_duckdb_loader.py -q
  → 57 passed

uv run --frozen --with "psycopg[binary]" python -m pytest tests/ -q
  → 237 passed, 1 skipped, 1 failed
  （失败项 tests/test_query_output.py::test_named_records_file_and_ascii_inline 为既有失败，
    仅涉及 govio.cli.query 的 Decimal 序列化，与本任务无关，改动前即失败）

uv run --frozen python -m ruff check src/ tests/
  → All checks passed!

uv run --frozen python -m ruff format --check src/govio/metadata/postgres_loader.py tests/test_postgres_loader.py
  → 2 files already formatted（已执行 ruff format 规整）
```

说明：验证用 `uv run --frozen --with "psycopg[binary]"` 是为了不重写 `uv.lock`（见遗留问题 1）。

## 遗留问题

1. **uv.lock 未更新**（不在任务所有权清单内，禁止改动）：`pyproject.toml` 新增依赖后需由主会话执行 `uv lock`（或 `uv sync`）刷新 `uv.lock`；在此之前新环境跑全量测试需带 `--with "psycopg[binary]"`。
2. **docs/specs/metadata.md 未同步**：AGENTS.md 文档同步规则要求更新，但本任务明确排除 `docs/specs/`，建议 T6 CLI 接入时一并更新。
3. **CLI 接入**（`cli/meta.py`）由 T6 负责，本任务未触碰。
4. **未做真实 PostgreSQL 联调**：测试全部打桩；SQL 按 PostgreSQL 标准信息 schema 视图编写（含 pg_namespace/pg_class 关联取 OID），建议集成阶段连真实库冒烟验证（尤其是删列后 `col_description` 取值）。
5. 同工作树存在他任务文件（`src/govio/metadata/mysql_loader.py`、`tests/test_mysql_loader.py`），只读未动。
