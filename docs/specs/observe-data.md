# govio.observe_data -- Data Observation Module

Data exploration, comparison, and relationship inference for loaded DataFrames.

## Config

`observe_data/config.py`

```python
@dataclass
class DataSourceConfig:
    url: str
    connect_args: dict[str, Any] = field(default_factory=dict)

@dataclass
class Config:
    datasources: dict[str, DataSourceConfig]

load_config(path: Path) -> Config
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

## DataFrameStore

`core/dataframe_store.py`

In-memory singleton DataFrame storage (older/alternative to ObserveStore).

```python
store(name, df) -> DataFrameInfo
get(name) -> pd.DataFrame | None
list() -> list[DataFrameInfo]
release(name) -> bool
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

## Tool Functions

`observe_data/tools/`

| Function | Input | Returns |
|---|---|---|
| `list_dataframes(store)` | `ObserveStore` | `{'dataframes': [{name, rows, columns, column_info}]}` |
| `list_datasources(db_manager)` | `DatabaseManager` | `[{name, driver, url}]` |
| `load_dataframe(store, db_manager, datasource, name, sql)` | mixed | `{'success': True/False, ...}` |
| `release_dataframe(store, name)` | `ObserveStore` | `{'success': True/False}` |
| `release_all_dataframes(store)` | `ObserveStore` | `{'success': True, 'released': [], 'count': int}` |
| `visualize_relations(relations)` | list or dict | `{'success': True, 'nodes': [], 'edges': []}` |
