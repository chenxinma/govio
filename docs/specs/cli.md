# govio.cli -- Command Line Interface

Entry point: `govio-cli` -> `govio.cli:main`

## main.py

Uses `argparse` with subparsers:

| Subcommand | Arguments | Description |
|---|---|---|
| `onboard` | `--new-falkordb CSV_DIR`, `--new-networkx CSV_DIR` | Interactive setup wizard |
| `std-recommend` | (none) | Data standard recommendation |
| `query` | `-c/--code QUERY` | Knowledge graph query |
| `observe` | REMAINDER | Data table observation |

---

## ConfigManager

`config.py`

Manages `~/.govio/config.yaml`.

### Constructor

```python
ConfigManager(config_path: Path | None = None)  # Default: ~/.govio/config.yaml
```

### Methods

```python
exists() -> bool
load() -> dict[str, Any]              # Raises FileNotFoundError
save(config: dict[str, Any]) -> None  # YAML with unicode support
validate(config: dict[str, Any]) -> bool  # Raises ValueError on issues
```

### Validation Rules

- `backend`: required, `"networkx"` or `"falkordb"`
- `networkx.gml_path`: required if backend is networkx
- `falkordb.host/port/graph`: required if backend is falkordb
- `csv_dir`: optional, path must exist
- `graph_dir`: optional, path must exist
- `datasources`: optional dict, each entry must have `url` key

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
prompt_backend_choice() -> str                       # "networkx" or "falkordb"
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
query(query_text) -> None    # Dispatches to networkx or falkordb
cmd_networkx(code, gml_path) -> None     # exec() with graph in scope, expects `result` var
cmd_falkordb(cypher, host, port, graph_name) -> None  # Validates MATCH, executes Cypher
output_result(data) -> None  # >20 rows -> JSON file, else stdout
```

Logs to `~/.govio/logs/query_{YYYYMMDD}.log`.

**Security**: `cmd_networkx` uses `exec()` for arbitrary Python code execution.

---

## std_recommend.py

```python
std_recommend() -> None
```

Reads config, loads `df_app_db_map` from JSON, calls `data_standard_recommend()`. Requires: `kundb`, `workspace_uuid`, `app_map`, `csv_dir`.

---

## observe.py

Data table observation subcommand dispatcher.

### Subcommands

| Command | Arguments | Description |
|---|---|---|
| `show-datasource` | (none) | List configured datasources |
| `load` | `--name`, `--datasource`, `--sql` | Load DataFrame from SQL |
| `list` | (none) | List loaded DataFrames |
| `release` | `--name` or `--all` | Release (delete) DataFrames |
| `compare` | `--source`, `--target`, `--join-columns` | Compare two DataFrames |
| `explore` | `--dataframes` (optional) | Explore relationships |
| `visualize-relations` | `--relations` (JSON) | Generate visualization |
