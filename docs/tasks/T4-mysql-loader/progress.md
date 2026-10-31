# T4 MySQL 元数据加载器 — 进度

## 目标

新增 `MysqlLoader(MetadataLoader)`，用 pymysql 读取 MySQL `information_schema.tables` / `information_schema.columns` 抽取表与列元数据（含表注释、列注释，类型解析出 size/precision/scale），输出 DataFrame 列结构与 TrinoLoader / DuckDBLoader 完全一致，`data_entity_type` 用 `MYSQL_TABLE` / `MYSQL_COLUMN`，支持 `datasource_name` 全限定（qualify）与 schema 过滤。CLI 接入不在本任务范围（T6 负责）。

## 改动文件清单

| 文件 | 变更 |
|---|---|
| `src/govio/metadata/mysql_loader.py` | 新增：`MysqlLoader` + 模块级 `_parse_type`（从 COLUMN_TYPE 解析 size/precision/scale） |
| `tests/test_mysql_loader.py` | 新增：25 个测试，monkeypatch 打桩 `pymysql.connect`，不依赖真实数据库 |
| `docs/tasks/T4-mysql-loader/progress.md` | 本文件 |
| `docs/tasks/T4-mysql-loader/status.json` | 任务状态 |
| `docs/tasks/T4-mysql-loader/spec-draft.md` | docs/specs/metadata.md 的 MysqlLoader 小节草稿（待 T6/文档同步时并入） |

未触碰任务所有权之外的任何文件（`cli/meta.py`、`docs/specs/`、`metadata/__init__.py` 等均未改）。

## 关键设计

- 构造签名：`MysqlLoader(host, schemas, port=3306, user="root", password="", database="", datasource_name="")`；`database` 传给 `pymysql.connect` 作为默认库，并写入输出 `database_name` 列（空串时输出空串，与 DuckDBLoader 一致）。
- 表来源：`information_schema.tables`，`WHERE TABLE_TYPE = 'BASE TABLE' AND TABLE_SCHEMA IN (...)`（视图不进 PhysicalTable；如需含视图删掉该条件即可）。
- 列来源：`information_schema.columns` JOIN `information_schema.tables`（保证只含 BASE TABLE 的列），取 `COLUMN_TYPE` / `COLUMN_COMMENT` / `ORDINAL_POSITION`。
- `order_no` 直接取 `ORDINAL_POSITION`（每表从 1 递增）。
- 表/列注释为空（`None` / `NaN` / `""`）时 `name` 回退为 `table_name` / `column_name`。
- `dtype` = 原始 `COLUMN_TYPE`（如 `decimal(10,2)`），`data_type` 与之一致（对齐 TrinoLoader）。
- schema 过滤用 `%s` 占位符参数化传递（`IN (%s, ...)`），值不拼进 SQL；`schemas` 为空时直接返回带正确列名的空 DataFrame、不发起连接。
- `_parse_type` 与 TrinoLoader 同语义（decimal → precision/scale，varchar/char → size，其余 0），因任务所有权不允许改共享模块，故在本模块内自带一份。
- 输出列结构（与 TrinoLoader/DuckDBLoader 逐列一致）：
  - `PhysicalTable`: `full_table_name, schema, table_name, name, data_entity_type, database_name`
  - `Col`: `column, column_name, name, full_table_name, data_entity_type, dtype, size, precision, scale, order_no, data_type`

## 验证命令与结果

| 命令 | 结果 |
|---|---|
| `uv run python -m pytest tests/test_mysql_loader.py -v` | 25 passed（表清单/列清单/注释回退/None 注释回退/类型解析 decimal(10,2)·varchar(64)·bigint/order_no 递增/空 schema/多 schema/qualify/connect kwargs/连接关闭/参数化过滤/子类与属性委托） |
| `uv run ruff check src/govio/metadata/mysql_loader.py tests/test_mysql_loader.py` | All checks passed |
| `uv run ruff format --check src/govio/metadata/mysql_loader.py tests/test_mysql_loader.py` | 2 files already formatted（已按仓库惯例执行过 `ruff format`） |
| `uv run python -m pytest tests/ -q` | 210 passed, 1 skipped, 1 failed — 失败为 `tests/test_query_output.py::test_named_records_file_and_ascii_inline`，与本任务无关（该测试不含任何对 mysql_loader 的引用，本任务 diff 仅新增文件；为工作树中既有失败，疑似 pandas 版本数值类型 `'822'` vs `822.0` 断言不匹配） |
| `git diff --cached --name-only` | 空（无任何 staged 文件，未执行任何 git 写操作） |

## 遗留问题

1. `tests/test_query_output.py::test_named_records_file_and_ascii_inline` 在工作树中既有失败（`'822'` vs `822.0` 类型不匹配），与 T4 无关，未处理。
2. CLI 接入（`--source mysql` 等）由 T6 完成；`metadata/__init__.py` 未导出 `MysqlLoader`（该文件不在 T4 所有权内，且现有 TrinoLoader/DuckDBLoader 也未从该处导出，测试均直接从子模块导入）。
3. 仅抽取 BASE TABLE（不含视图）为实现取舍，如产品需要视图，调整 `mysql_loader.py` 中 `TABLE_TYPE = 'BASE TABLE'` 过滤即可。
4. `docs/specs/metadata.md` 的同步更新按任务要求未执行，草稿见同目录 `spec-draft.md`，由后续文档同步任务并入。
5. 同工作树内存在其他并行任务的改动（`pyproject.toml` 新增 `psycopg[binary]`、`postgres_loader.py` 等），均不属于 T4，未触碰。
