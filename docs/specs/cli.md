# govio.cli -- Command Line Interface

Entry point: `govio-cli` -> `govio.cli:main`

## main.py

Uses `argparse` with subparsers:

| Subcommand | Description |
|---|---|
| `onboard` | Interactive setup wizard |
| `backend` | Display current graph backend type |
| `query -c QUERY` | Knowledge graph query (Cypher or Python) |
| `meta` | Knowledge graph maintenance command group |
| `observe` | Data table observation command group |
| `sql` | Metric SQL assembly command group |
| `-V/--version` | Show package version |

---

## ConfigManager

`config.py`

Manages `~/.govio/config.yaml`. Supports auto-migration from old flat format to new nested format, and password encryption.

### Constructor

```python
ConfigManager(config_path: Path | None = None)  # Default: ~/.govio/config.yaml
```

### Methods

```python
exists() -> bool
load() -> dict[str, Any]              # Raises FileNotFoundError; auto-migrates old format
save(config: dict[str, Any]) -> None  # YAML with unicode support
validate(config: dict[str, Any]) -> bool  # Raises ValueError on issues
```

### Auto-migration

On `load()`, automatically migrates:
1. **Old flat format** → new nested format (`metadata`, `graph`, `datasources` sections)
2. **Plaintext passwords** → encrypted storage (`encrypted_password` field, backed up to `.yaml.bak`)

### Validation Rules

- `graph.backend`: required, `"networkx"`, `"falkordb"`, or `"ladybug"`
- `graph.networkx.gml_path`: required if backend is networkx
- `graph.falkordb.host/port/graph`: required if backend is falkordb
- `graph.ladybug.db_path`: required if backend is ladybug
- `datasources.*`: optional, each entry must have `url` key

---

## MetaConfigManager

`config.py`

Manages `~/.govio/meta_config.yaml` for the `meta` command group.

### Constructor

```python
MetaConfigManager(config_path: Path | None = None)  # Default: ~/.govio/meta_config.yaml
```

### Methods

```python
exists() -> bool
load() -> dict[str, Any]
save(config: dict[str, Any]) -> None
migrate_from_config(config: dict) -> dict[str, Any]  # Extract metadata fields from main config
load_or_migrate() -> dict[str, Any]  # Load or auto-migrate from main config
```

---

## onboard.py

Interactive setup wizard. Three modes:

1. `--new-networkx`: skip CSV generation, generate GML from existing CSV
2. `--new-falkordb`: skip CSV generation, import CSV to FalkorDB
3. Full interactive: prompts for CSV config, backend choice, datasource config

### Key Functions

```python
onboard(new_falkordb=None, new_networkx=None) -> None
validate_csv_directory(csv_dir: Path) -> bool       # Checks PhysicalTable.csv exists
prompt_csv_config(config_manager) -> dict            # Interactive CSV config
generate_csv(config: dict) -> None                   # Calls make_csv()
prompt_backend_choice() -> str                       # "networkx", "falkordb", or "ladybug"
prompt_networkx_config() -> dict                     # CSV dir + GML generation
prompt_falkordb_config(csv_dir: Path) -> dict        # Host, port, graph, import
delete_falkordb_graph(host, port, graph_name) -> None
import_csv_to_falkordb(csv_dir, host, port, graph_name) -> None
prompt_connect_args(existing=None) -> dict           # Interactive key=value input
prompt_datasource_config(existing=None) -> dict | None
```

`import_csv_to_falkordb` handles all node files (PhysicalTable, Col, Application, Standard, Metric, Dimension) and relation files (HAS_COLUMN, USE, RELATES_TO, USES_TABLE, REFERS_COLUMN, DERIVED_FROM, DIMENSION_USED, SUPERSEDES).

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

| Subcommand | Description |
|---|---|
| `meta sync` | Full pipeline (interactive or CLI mode) |
| `meta sync meta` | Import TDS/DuckDB metadata (PhysicalTable, Col, HAS_COLUMN) |
| `meta sync app` | Import application list (Application + USE edges) |
| `meta sync std` | Import data standards (Standard nodes, TDS only) |
| `meta sync compliance` | Export existing standard-column associations (COMPLIES_WITH, TDS only) |
| `meta sync rel` | Import table relationships (RELATES_TO edges) |
| `meta sync metric` | Import metric/dimension definitions (Metric, Dimension + 5 edge types) |
| `meta sync graph` | Update graph database + generate assets |
| `meta recommend` | Data standard recommendation |
| `meta config` | Interactive meta config management |

### Data Sources

The `sync` pipeline supports three data source modes:
- **TDS**: Read from metadata database only
- **DuckDB**: Read from local DuckDB file only (skips Standard data)
- **Both**: Merge TDS + DuckDB (DuckDB wins on conflict)

### Step Functions

Each `sync` subcommand maps to an independent, idempotent step function:

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

### Graph Update

```python
_update_graph(output, graph_mode) -> bool   # "update" (incremental) or "rebuild"
_generate_assets() -> None                   # schema.md, names, metrics_index.md
```

Supports all three backends: FalkorDB (upsert/import), Ladybug (upsert/import), NetworkX (incremental/rebuild GML).

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
std_recommend() -> None
```

Reads meta config, loads `df_app_db_map` from JSON, calls `data_standard_recommend()`. Requires: `kundb`, `workspace_uuid`, `app_map`, `csv_dir`.

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
