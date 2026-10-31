# govio.cli -- Command Line Interface

Entry point: `govio-cli` -> `govio.cli:main`

## main.py

Uses `argparse` with subparsers:

| Subcommand | Description |
|---|---|
| `onboard` | Interactive setup wizard; non-interactive datasource add/remove via flags |
| `backend` | Display current graph backend type |
| `query -c QUERY` | Knowledge graph query (Cypher or Python) |
| `meta` | Knowledge graph maintenance command group |
| `observe` | Data table observation command group |
| `sql` | Metric SQL assembly command group |
| `-V/--version` | Show package version |

### Lazy imports (startup speed)

Heavy modules are imported on demand so fast paths (`-V`, `backend`, `observe info`) start in ~0.1s:

- Subcommand modules (`onboard` / `query` / `meta` / `observe` / `sql`) are imported inside `main()`'s dispatch branch, not at module top level
- `govio/__init__.py` and `govio/metadata/__init__.py` expose their API via PEP 562 `__getattr__` lazy exports (`from govio import FalkorDBGraph` etc. still works)
- `observe.py` imports `render_chart` (matplotlib), `visualize_relations` (networkx), `load_dataframe` (duckdb) inside the matching `cmd_*` function

---

## ConfigManager

`config.py`

Manages `~/.govio/config.yaml`. Auto-migrates plaintext passwords to encrypted storage.

### Constructor

```python
ConfigManager(config_path: Path | None = None)  # Default: ~/.govio/config.yaml
```

### Methods

```python
exists() -> bool
load() -> dict[str, Any]              # Raises FileNotFoundError; auto-encrypts plaintext passwords
save(config: dict[str, Any]) -> None  # YAML with unicode support
validate(config: dict[str, Any]) -> bool  # Raises ValueError on issues
```

### Validation Rules

- `graph`: required section
- `graph.backend`: required, `"networkx"`, `"falkordb"`, or `"ladybug"`
- `graph.networkx.gml_path`: required if backend is networkx
- `graph.falkordb.host/port/graph`: required if backend is falkordb
- `graph.ladybug.db_path`: required if backend is ladybug
- `datasources.*`: optional, each entry must have `url` key

---

## onboard.py

Interactive setup wizard + non-interactive datasource management.

### Interactive wizard behavior

- No config: prompts for graph backend (networkx/falkordb/ladybug), then datasource add/delete/edit loop
- Existing config with graph backend: offers "skip backend, edit datasources only"; otherwise asks before overwriting
- Datasources-only config (created via `--add-datasource`): configures graph backend while preserving existing datasources

### Non-interactive datasource flags

| Flag | Description |
|---|---|
| `--add-datasource NAME` | Add datasource (requires `--url`) |
| `--remove-datasource NAME` | Delete datasource |
| `--url URL` | Connection URL, e.g. `mysql+pymysql://user:pass@host:3306/db` (password auto-masked + Fernet-encrypted) |
| `--password P` | Password provided separately, injected into a password-less `scheme://user@host` URL |
| `--connect-args KEY=VALUE` | Repeatable extra connection args (values coerced to bool/int/float) |
| `--overwrite` | Replace existing datasource with the same name |

Errors exit with code 1; adding a datasource never touches other config sections.

```bash
govio-cli onboard --add-datasource prod --url "mysql+pymysql://user:pass@host:3306/db" --connect-args charset=utf8mb4 --connect-args timeout=30
govio-cli onboard --add-datasource staging --url "postgresql://user@host:5432/db" --password secret
govio-cli onboard --remove-datasource staging
```

### Key Functions

```python
validate_csv_directory(csv_dir: Path) -> bool        # Checks PhysicalTable.csv exists
_coerce_scalar(value) -> Any                         # str -> bool/int/float/value
prompt_connect_args(existing=None) -> dict           # Interactive key=value input
parse_cli_connect_args(pairs) -> dict                # Parse CLI key=value pairs (raises ValueError)
_encrypt_url_password(url) -> dict                   # {url: masked, encrypted_password?}
_attach_password(url, password) -> str               # Inject password into password-less URL (raises ValueError)
add_datasource(name, url, connect_args=None, password=None, overwrite=False, config_manager=None) -> dict
remove_datasource(name, config_manager=None) -> None
prompt_graph_config() -> dict                        # Interactive graph backend section
prompt_datasource_config(existing=None) -> dict | None
onboard_datasource_cli(add_name=None, remove_name=None, url=None, password=None, connect_args=None, overwrite=False) -> None
onboard() -> None                                    # Interactive wizard entry
```

---

## query.py

Knowledge graph query interface.

```python
query(query_text: str | None) -> None    # Dispatches to networkx, falkordb, or ladybug
cmd_networkx(code, gml_path) -> None     # exec() with graph in scope, expects `result` var
cmd_falkordb(cypher, host, port, graph_name) -> None  # Validates MATCH, executes Cypher
cmd_ladybug(cypher, db_path, ...) -> None  # Validates MATCH, executes Cypher via Ladybug
output_result(data, columns=None) -> None  # 输出契约见下
_fail(message: str) -> NoReturn          # ❌ 前缀错误到 stderr + sys.exit(1)
```

**输出契约**（`output_result`）：

- 结果永远写 `.govio/output-<时间戳含微秒>.json`：UTF-8、JSON Lines、records 带字段名（Cypher 后端字段名来自 RETURN 别名，经 `query_with_columns()` 透传；文件是权威结果）
- stdout 摘要行：`Result output: <path> rows: N columns: [...]`，columns 经 `ensure_ascii` 转义，整行 ASCII 安全（防 bash/PowerShell 控制台编码差异破坏中文 JSON）
- 行数 ≤ `INLINE_LIMIT`（50）时 stdout 额外内联一行 JSON records（`ensure_ascii=True`，同样编码安全）
- 空结果只输出 `Data not found.`，不写文件
- networkx 后端（Python exec 的 `result`）无列名元信息，位置键 `"0"`/`"1"` 兑底

Logs to `~/.govio/logs/query_{YYYYMMDD}.log`.

**Security**: `cmd_networkx` uses `exec()` for arbitrary Python code execution.

### 退出码契约（query）

| 场景 | 退出码 | 输出 |
|---|---|---|
| 查询成功（有结果） | 0 | stdout 结果摘要（+内联）+ 结果文件 |
| 查询成功但结果为空 | 0 | `Data not found.`（空结果不是失败） |
| 查询执行失败（Python 语法错、Cypher 错、后端异常） | 1 | `❌ 查询执行失败: ...` → stderr |
| 连接失败（图库/数据库不可达、GML 不存在） | 1 | `❌ 连接图库失败: ...` / `❌ GML file not found: ...` → stderr |
| 参数缺失或非法（缺 `-c`、非 MATCH 查询、配置缺字段） | 1 | `❌ ...` → stderr |
| 未预期异常（兜底） | 1 | `❌ 查询执行失败: ...` → stderr（不裸奔 traceback） |

约定：成功 0 / 失败非 0（统一 1）。错误信息一律 `❌` 前缀走 stderr，stdout 不混入错误。

---

## meta.py

Knowledge graph maintenance command group: `govio-cli meta`.

### Subcommands

All subcommands are independent, CLI-only (no config file dependency). All inputs are explicit CLI arguments.

Exception: `import-schema` reads `config.datasources` to resolve the DuckDB file path from a datasource name.

| Subcommand | Description |
|---|---|
| `meta meta` | Import TDS/DuckDB/MySQL/PostgreSQL/Trino metadata (Datasource, PhysicalTable, Col, HAS_COLUMN, OWNS) |
| `meta std` | Import data standards (Standard nodes, TDS only) |
| `meta compliance` | Export existing standard-column associations (COMPLIES_WITH, TDS only; `--datasources-file` 必填) |
| `meta rel` | Import table relationships (RELATES_TO edges) |
| `meta metric` | Import metric/dimension definitions (Metric, Dimension + 5 edge types) |
| `meta graph` | Graph database management (update/rebuild/clear + assets) |
| `meta recommend` | Data standard recommendation (schema scope from `--datasource`) |
| `meta schema` | Print input JSON Schema (metric / relationship / datasource) |
| `meta import-schema` | Import metadata from a configured DuckDB datasource to graph (shortcut for meta + graph) |

Recommended order: `meta` → `std` → `compliance` → `rel` → `metric` → `graph`

### Data Sources

`meta meta` 覆盖单一信息来源（`--source`）；各来源分多次运行，通过 CSV merge 合流：

- **tds**: 从元数据库读取；`--kundb`、`--workspace-uuid`、`--datasources-file` 必填
  - 抽取范围 = 声明文件各条目 `filter.schemas` 的并集（**不接受 `--schemas`**）
  - 一次运行可产出多个 `Datasource` 节点（声明文件全部条目）及其 `OWNS` 归属
- **duckdb / mysql / postgres / trino**: 从一个实际数据源读取；`--db`、`--schemas`、`--datasource` 必填
  - 过滤由 CLI `--schemas` 执行；一次运行对应一个 `Datasource`
  - 提供 `--datasources-file` 时按声明校验/合并定义，否则按 CLI 参数自动声明（`source_type` 取 `--source`，`filter.schemas` 取 `--schemas`）

`--db` 语义随 `--source` 变化：

| `--source` | `--db` 格式 | 示例 |
|---|---|---|
| `duckdb` | DuckDB 文件路径 | `./data/local.duckdb` |
| `mysql` | 连接 URL（scheme: `mysql` / `mysql+pymysql`） | `mysql://user:pass@host:3306/db` |
| `postgres` | 连接 URL（scheme: `postgres` / `postgresql` / `postgresql+psycopg`） | `postgresql://user:pass@host:5432/db` |
| `trino` | 连接 URL（scheme: `trino` / `trino+http` / `trino+https`） | `trino://user:pass@host:8080/catalog` |

- URL 省略 user / port 时取各加载器默认值（mysql: `root`/3306，postgres: `postgres`/5432，trino: `trino`/8080）
- trino URL 路径必须恰好为 catalog 一段，schema 由 `--schemas` 指定（路径带第二段或缺 catalog 报错）
- trino URL 含密码时构造 `trino.auth.BasicAuthentication`（此时必须同时给 user）；`trino+https` 表示 HTTPS 连接
- URL scheme 与 `--source` 不匹配、URL 非法或缺 host 时，`step_meta_export` 打印 `❌ ...` 并返回 None（`cmd_meta` 退出码 1）

DuckDB 文件通常只有单个用户 schema `main`；缺省 `--schemas` 会静默导出 0 张表，因此必须显式指定。

### Empty-Result Guard

`step_meta_export` aborts before writing any CSV when no table is found (`df_tables.empty`):

```python
_describe_duckdb_schemas(db_path) -> str   # read-only "main(5 张表)、..." hint for the error message
```

- Returns `None`, `cmd_meta` exits with code 1
- Error text: `❌ 未发现任何表: schema [...] 在元数据源中不存在或为空（未写入任何 CSV）`
- For `duckdb`, the message appends the importable schemas of the file via `DuckDBLoader.list_schemas()`, so callers never need to connect to the source database themselves
- Renaming a schema during import is not supported; `full_table_name` 恒为 `<datasource_name>.<schema>.<table>`

### Schema Discovery on Missing `--schemas`

When `--schemas` is omitted in 非 TDS 模式, `cmd_meta` prints an error message **plus** source-dependent hints:

| Source | Hint |
|---|---|
| any non-tds | 错误行含来源相关 schema 语义（`duckdb`: `DuckDB 文件默认 schema 为 main`；`mysql`: `MySQL 中 schema 即库名`；`postgres`: `PostgreSQL 常用 schema 为 public`；`trino`: `Trino schema 为 catalog 下的命名空间`） |
| `duckdb` with `--db` | 额外列出可导入 schema（`DuckDBLoader.list_schemas()`，如 `📖 test.duckdb 中可导入的 schema: main(5 张表)`） |
| `mysql` / `postgres` / `trino` | 无额外提示（无法离线列举 schema，按错误行的 schema 语义重试） |
| `tds` | 不接受 `--schemas`（抽取范围由 `--datasources-file` 的 `filter.schemas` 决定） |

`--db` / `--datasource` 缺省时先于 `--schemas` 报错（`❌ {Source} 模式需要指定 --db（文件路径/连接 URL）` / `❌ {Source} 模式需要指定 --datasource`）。

### Step Functions

Each subcommand maps to an independent, idempotent step function:

```python
step_meta_export(output, source, db_path, schemas, datasource_name, datasource_file, kundb, workspace_uuid) -> tuple[df_tables, df_columns] | None
step_std_export(output, kundb, workspace_uuid) -> None
step_compliance_export(output, kundb, workspace_uuid) -> None
step_rel_export(output, relationship_file) -> None
step_metric_export(output, metric_file) -> bool
```

`step_meta_export` 的 `db_path` 即 `--db` 值：duckdb 为文件路径，mysql / postgres / trino 为连接 URL。

加载器构造由 `_build_metadata_loader` 完成（`src/govio/cli/meta.py`）：

```python
_build_metadata_loader(source, db, schemas, datasource_name) -> MetadataLoader
# duckdb -> DuckDBLoader；mysql -> MysqlLoader；postgres -> PostgresLoader；trino -> TrinoLoader
# URL scheme 与 --source 不匹配、URL 非法或缺 host 时抛 ValueError
```

### CSV Merge Helpers

Incremental merge support via:

```python
merge_node_csv(new_df, csv_path, node_type, key_col) -> pd.DataFrame
merge_edge_csv(new_df, csv_path, dedup_cols) -> pd.DataFrame
```

### `schema` Subcommand

`govio-cli meta schema {metric|relationship|datasource}` — 输出标准 JSON Schema（UTF-8），供外部 agent 生成可被 `meta metric` / `meta rel` / `meta meta` 导入的标准数据。

```bash
govio-cli meta schema metric
govio-cli meta schema relationship -o relationship_schema.json
govio-cli meta schema datasource
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `schema_name` (positional) | Yes | — | `metric` / `relationship` / `datasource` |
| `-o/--output` | No | stdout | 输出文件路径 |

输出内容与包内 schema 文件一一对应：`src/govio/metadata/metric_schema.json`、`relationship_schema.json`、`datasource_schema.json`。

### `recommend` Subcommand

`meta recommend` 用 `--datasource NAME` 替代原 `--app-map`：schema 分析范围从该 `Datasource` 的 `filter.schemas` 解析，解析顺序为 `--datasources-file` → `--csv-dir/Datasource.csv`。

### Graph Management

```python
_update_graph(output, graph_mode, assets_dir) -> bool   # "update" (incremental) or "rebuild"
_clear_graph(assets_dir) -> bool                        # Clear graph database
_generate_assets(assets_dir) -> None                    # schema.md, names, metrics_index.md
```

`meta graph` accepts `--assets-dir <path>` (default `.agents/skills/govio/assets`). The path is resolved to an absolute path and passed to all three helpers. `_generate_assets()` prints the absolute path on success.

Graph backend config is read from `~/.govio/config.yaml` (`graph` section). Supports all three backends:
- FalkorDB: upsert/import/delete
- Ladybug: upsert/import/delete
- NetworkX: incremental/rebuild GML/delete

### `import-schema` Subcommand

Shortcut that combines `meta meta` + `meta graph` for DuckDB datasources already configured via `govio-cli onboard`. Reads the datasource URL from `~/.govio/config.yaml`, validates it is `duckdb://`, extracts the file path, then runs the full pipeline: CSV export → graph update → assets generation.

```bash
govio-cli meta import-schema --datasource mydb --schemas main,analytics
govio-cli meta import-schema --datasource mydb --schemas main --output ./data --assets-dir ./assets --mode rebuild
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `--datasource` | Yes | — | Datasource name from `config.datasources` (must be DuckDB) |
| `--schemas` | Yes | — | Comma-separated schema list |
| `--output` | No | `./output` | CSV intermediate directory |
| `--assets-dir` | No | `.agents/skills/govio/assets` | Assets output directory |
| `--mode` | No | `update` | `update` / `rebuild` / `clear` |

Only supports DuckDB datasources (`duckdb://` URL). Non-DuckDB datasources cause exit with code 1.

同时生成 `Datasource.csv` / `OWNS.csv`：`datasource_name` 取 `--datasource`，`source_type=duckdb`，`filter.schemas` 取 `--schemas`。

---

## sql.py

Metric SQL assembly: `govio-cli sql build`.

### Subcommands

| Subcommand | Arguments | Description |
|---|---|---|
| `build` | `-f/--file JSON_FILE`, `-o/--output SQL_FILE` | Assemble metric SQL from JSON spec |

### Usage

```bash
# Print to stdout
govio-cli sql build -f query.json

# Output to file
govio-cli sql build -f query.json -o output.sql

# From stdin
cat query.json | govio-cli sql build
```

---

## std_recommend.py

```python
std_recommend(output_dir, kundb, workspace_uuid, datasource_name, schemas, csv_dir) -> None
```

All parameters are explicit function arguments. `schemas` 由 `meta recommend` 从 `Datasource.filter.schemas` 解析后传入，调用 `data_standard_recommend()`。Called by `meta recommend` CLI subcommand.

---

## observe.py

Data table observation subcommand dispatcher: `govio-cli observe`.

### Subcommands

| Command | Arguments | Description |
|---|---|---|
| `info` | `--datasource`, `--df`, `--name NAME [--rows N]` | Query datasource/DataFrame info |
| `load` | `--name`, `(--datasource\|--memory)`, `--sql`, `-o OUTPUT` | Load DataFrame from DB or memory |
| `release` | `--name NAME` or `--all` | Release (delete) DataFrames |
| `compare` | `--source`, `--target`, `--join-columns COLS` | Compare two DataFrames |
| `explore` | `--dataframes` (optional) | Explore relationships |
| `visualize-relations` | `--relations-file FILE` | Generate visualization |
| `chart` | `--name`, `--type bar\|line`, `--x`, `--y`, `-o OUTPUT` | Generate PNG chart |

### `info` Subcommand

Merges previous `show-datasource`, `list`, and `show` into one unified command:
- No flags: returns overview (datasource names + DataFrame list)
- `--datasource`: list configured datasource names
- `--df`: list loaded DataFrames
- `--name NAME [--rows N]`: show DataFrame structure and sample data (read-only)

### `load --memory` Mode

Loads all registered DataFrames into an in-memory DuckDB, executes SQL against them, and stores result back. Enables DataFrame-to-DataFrame transformations without database access.

```bash
# Load from DB
govio-cli observe load --name orders --datasource prod_db --sql "SELECT ..."

# Transform in memory
govio-cli observe load --name summary --memory --sql "SELECT customer_id, SUM(amount) FROM orders GROUP BY customer_id"
```

### `chart` Subcommand

Generates PNG charts from loaded DataFrames:

```bash
govio-cli observe chart --name sales --type bar --x region --y revenue -o /tmp/sales.png
govio-cli observe chart --name monthly --type line --x month --y revenue -o /tmp/trend.png
```

Supports `bar` and `line` chart types. Includes Chinese font fallback chain.

### `load -o` Output

When `-o/--output` is specified, writes DataFrame content as JSON (records orientation, UTF-8, indented) to the specified file path.

### 退出码契约（observe load）

| 场景 | 退出码 | 输出 |
|---|---|---|
| 加载成功（含 0 行结果） | 0 | stdout 结果 JSON（`success: true`） |
| SQL 错误 / 连接失败 / 内存加载失败 | 1 | stdout 保留结果 JSON（`success: false, error: ...`，契约向后兼容）+ `❌ {error}` → stderr |
| 参数缺失（如 `--datasource` 未设且非 `--memory`） | 1 | `❌ ...` → stderr |
| 未预期异常（observe 入口兜底） | 1 | `❌ 执行失败: ...` → stderr（不裸奔 traceback） |

约定与 query 一致：成功 0 / 失败非 0；`load` 失败时 stdout 的 result JSON 契约不变，仅额外向 stderr 输出 `❌` 错误并以退出码 1 结束。其余 observe 子命令（compare/explore/chart 等）本次不在退出码改造范围内。
