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
query(query_text) -> None    # Dispatches to networkx, falkordb, or ladybug
cmd_networkx(code, gml_path) -> None     # exec() with graph in scope, expects `result` var
cmd_falkordb(cypher, host, port, graph_name) -> None  # Validates MATCH, executes Cypher
cmd_ladybug(cypher, db_path, ...) -> None  # Validates MATCH, executes Cypher via Ladybug
output_result(data) -> None  # >20 rows -> JSON file, else stdout
```

Logs to `~/.govio/logs/query_{YYYYMMDD}.log`.

**Security**: `cmd_networkx` uses `exec()` for arbitrary Python code execution.

---

## meta.py

Knowledge graph maintenance command group: `govio-cli meta`.

### Subcommands

All subcommands are independent, CLI-only (no config file dependency). All inputs are explicit CLI arguments.

| Subcommand | Description |
|---|---|
| `meta meta` | Import TDS/DuckDB metadata (PhysicalTable, Col, HAS_COLUMN) |
| `meta app` | Import application list (Application + USE edges) |
| `meta std` | Import data standards (Standard nodes, TDS only) |
| `meta compliance` | Export existing standard-column associations (COMPLIES_WITH, TDS only) |
| `meta rel` | Import table relationships (RELATES_TO edges) |
| `meta metric` | Import metric/dimension definitions (Metric, Dimension + 5 edge types) |
| `meta graph` | Graph database management (update/rebuild/clear + assets) |
| `meta recommend` | Data standard recommendation |

Recommended order: `meta` → `app` → `std` → `compliance` → `rel` → `metric` → `graph`

### Data Sources

The `meta` subcommand supports three data source modes (`--source`):
- **tds**: Read from metadata database only; `--kundb`, `--workspace-uuid`, `--schemas` are required
- **duckdb**: Read from local DuckDB file only; `--db`, `--schemas` are required (skips Standard data)
- **both**: Merge TDS + DuckDB (DuckDB wins on conflict); TDS-side params same as tds mode

`--schemas` is mandatory for every source mode. Omitting it used to export zero tables silently, because `DuckDBLoader` filters with `schema_name IN (SELECT unnest(?))`. DuckDB files normally hold a single user schema named `main`.

### Empty-Result Guard

`step_meta_export` aborts before writing any CSV when no table is found (`df_tables.empty`):

```python
_describe_duckdb_schemas(db_path) -> str   # read-only "main(5 张表)、..." hint for the error message
```

- Returns `None`, `cmd_meta` exits with code 1
- Error text: `❌ 未发现任何表: schema [...] 在元数据源中不存在或为空（未写入任何 CSV）`
- For `duckdb` / `both`, the message appends the importable schemas of the file via `DuckDBLoader.list_schemas()`, so callers never need to connect to the source database themselves
- Renaming a schema during import is not supported; node names always come from `full_table_name = <schema>.<table>`

### Step Functions

Each subcommand maps to an independent, idempotent step function:

```python
step_meta_export(output, source, db_path, schemas, db_name, kundb, workspace_uuid) -> tuple[df_tables, df_columns] | None
step_app_export(output, app_list_file, app_map_file, db_name) -> None
step_std_export(output, kundb, workspace_uuid) -> None
step_compliance_export(output, kundb, workspace_uuid) -> None
step_rel_export(output, relationship_file) -> None
step_metric_export(output, metric_file) -> bool
```

### CSV Merge Helpers

Incremental merge support via:

```python
merge_node_csv(new_df, csv_path, node_type, key_col) -> pd.DataFrame
merge_edge_csv(new_df, csv_path, dedup_cols) -> pd.DataFrame
```

### Graph Management

```python
_update_graph(output, graph_mode, assets_dir) -> bool   # "update" (incremental) or "rebuild"
_clear_graph(assets_dir) -> bool                        # Clear graph database
_generate_assets(assets_dir) -> None                    # schema.md, names, metrics_index.md
```

`meta graph` accepts `--assets-dir <path>` (default `.agent/skills/govio/assets`). The path is resolved to an absolute path and passed to all three helpers. `_generate_assets()` prints the absolute path on success.

Graph backend config is read from `~/.govio/config.yaml` (`graph` section). Supports all three backends:
- FalkorDB: upsert/import/delete
- Ladybug: upsert/import/delete
- NetworkX: incremental/rebuild GML/delete

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
std_recommend(output_dir, kundb, workspace_uuid, app_map, csv_dir) -> None
```

All parameters are explicit function arguments. Loads `df_app_db_map` from JSON, calls `data_standard_recommend()`. Called by `meta recommend` CLI subcommand.

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
