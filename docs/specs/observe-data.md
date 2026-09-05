# govio.observe_data -- Data Observation Module

Data exploration, comparison, charting, and relationship inference for loaded DataFrames.

## Config

`observe_data/config.py`

```python
@dataclass
class DataSourceConfig:
    url: str
    connect_args: dict[str, Any] = field(default_factory=dict)
    encrypted_password: str | None = None  # Encrypted password from config
```

---

## ObserveStore

`core/observe_store.py`

File-backed DataFrame storage using parquet files.

### Constants

```python
OBSERVE_DIR = Path(".govio/observe")
DATAFRAMES_DIR = OBSERVE_DIR / "dataframes"
MANIFEST_FILE = OBSERVE_DIR / "manifest.json"
```

### Data Structures

```python
@dataclass
class DataFrameInfo:
    name: str
    datasource: str
    sql: str
    file: str
    loaded_at: str
    rows: int
    columns: int
    column_info: list[dict]

@dataclass
class Manifest:
    version: str = "1.0"
    dataframes: dict[str, dict[str, Any]]
```

### Methods

```python
store(name, df, datasource, sql) -> DataFrameInfo   # Save parquet + manifest
get(name) -> pd.DataFrame | None                     # Load from parquet
list() -> list[DataFrameInfo]                         # List all stored
release(name) -> bool                                 # Delete parquet + manifest entry
exists(name) -> bool
```

---

## DatabaseManager

`core/database.py`

Multi-datasource database connection manager.

### Constructor

```python
DatabaseManager(datasources: dict[str, DataSourceConfig])
```

- DuckDB URLs: `duckdb://path` -- supports file and directory modes
- Other URLs: SQLAlchemy engines

### Methods

```python
get_engine(datasource: str) -> Engine      # Raises ValueError if not SQLAlchemy
execute_sql(datasource: str, sql: str) -> pd.DataFrame  # DuckDB or SQLAlchemy
```

---

## TableComparator

`core/comparator.py`

DataFrame comparison using datacompy.

```python
compare_schema(source, target) -> dict
# Returns: match (bool), source_columns, target_columns, common_columns, source_only, target_only

compare_data(source, target, join_columns) -> dict
# Uses datacompy Compare. Returns: report (str)

compare(source, target, join_columns) -> dict
# Combines schema + data comparison
```

---

## RelationExplorer

`core/explorer.py`

Relationship inference between DataFrames.

```python
find_column_similarity(df1, df2) -> list[dict]
# SequenceMatcher threshold > 0.7
# Returns: [{'column', 'match_column', 'similarity'}]

infer_foreign_keys(source_df, source_name, target_df, target_name) -> list[dict]
# Matches _id/id suffix columns, checks value overlap > 0.5
# Returns: [{'source_table', 'source_column', 'target_table', 'target_column', 'confidence'}]

explore(dataframes: dict[str, pd.DataFrame]) -> dict
# Runs FK inference + column similarity on all pairs
# Returns: {'foreign_keys': [...], 'column_similarities': [...]}
```

---

## RelationVisualizer

`core/visualizer.py`

Converts relation data to graph/JSON formats.

```python
to_networkx(relations) -> nx.DiGraph     # Directed graph with edge attributes
to_json(relations) -> dict               # {'nodes': [...], 'edges': [...]}
visualize(relations) -> dict             # Alias for to_json()
```

---

## Chart Renderer

`core/chart.py`

Generates PNG charts from DataFrames.

```python
render_chart(df, chart_type, x_col, y_col, output_path) -> dict
```

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `df` | `pd.DataFrame` | Source data |
| `chart_type` | `str` | `"bar"` or `"line"` |
| `x_col` | `str` | X axis column (category/time) |
| `y_col` | `str` | Y axis column (numeric, single series) |
| `output_path` | `str` | Output PNG file path |

### Returns

```json
{"success": true, "output": "/abs/path/to/chart.png"}
```

or `{"success": false, "error": "..."}` on failure.

### Features

- Bar chart and line chart support (single series)
- Chinese font fallback chain: Noto Sans CJK SC / WenQuanYi Zen Hei / SimHei / Microsoft YaHei / Arial Unicode MS
- Negative number display fix

---

## Tool Functions

`observe_data/tools/`

| Function | Input | Returns |
|---|---|---|
| `list_dataframes(store)` | `ObserveStore` | `{'dataframes': [{name, rows, columns, column_info}]}` |
| `list_datasources(db_manager)` | `DatabaseManager` | `[{name, driver, url}]` |
| `load_dataframe(store, db_manager, datasource, name, sql)` | mixed | `{'success': True/False, ...}` |
| `load_from_memory(store, name, sql)` | `ObserveStore` | Loads all registered DataFrames into in-memory DuckDB, executes SQL, stores result. Returns `{'success': True/False, ..., source_tables: [...]}` |
| `release_dataframe(store, name)` | `ObserveStore` | `{'success': True/False}` |
| `release_all_dataframes(store)` | `ObserveStore` | `{'success': True, 'released': [], 'count': int}` |
| `visualize_relations(relations)` | list or dict | `{'success': True, 'nodes': [], 'edges': []}` |

---

## CLI Subcommands

| Command | Arguments | Description |
|---|---|---|
| `info` | `--datasource`, `--df`, `--name NAME [--rows N]` | Query datasource/DataFrame info (unified command) |
| `load` | `--name`, `(--datasource\|--memory)`, `--sql`, `-o OUTPUT` | Load DataFrame from DB or memory DuckDB |
| `release` | `--name NAME` or `--all` | Release DataFrames |
| `compare` | `--source`, `--target`, `--join-columns COLS` | Compare two DataFrames |
| `explore` | `--dataframes` (optional) | Explore relationships |
| `visualize-relations` | `--relations-file FILE` | Generate visualization |
| `chart` | `--name`, `--type bar\|line`, `--x`, `--y`, `-o OUTPUT` | Generate PNG chart |
