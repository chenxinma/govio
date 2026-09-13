# govio.metadata -- Metadata Loading and Recommendation

Pipeline for extracting metadata from databases, loading app/standard info, defining relationships and metrics, recommending data standards, and generating node IDs.

## TDSLoader

`database.py` (formerly `DatabaseLoader`)

Extracts table and column metadata from TDS (metadata database) via SQLAlchemy.

### Constructor

```python
TDSLoader(
    db: str,                          # SQLAlchemy connection URL
    workspace_uuid: str,              # Tenant workspace identifier
    schema_limits: list[str] | None,  # Optional schema name filter
    app_names: list[str] | None       # Optional app names (must match schema_limits length)
)
```

If `schema_limits` and `app_names` both provided and same length, builds `app_names_map` dict.

### Properties

| Property | Columns |
|---|---|
| `PhysicalTable` | `full_table_name`, `schema`, `table_name`, `name`, `data_entity_type`, `database_name` |
| `Col` | `column`, `column_name`, `name`, `full_table_name`, `data_entity_type`, `dtype`, `size`, `precision`, `scale`, `order_no`, `data_type` |

### Methods

```python
load_tables() -> pd.DataFrame    # SQL query joining database_table, database, datasource
load_columns() -> pd.DataFrame   # SQL query with data type conversion
```

Oracle type conversion (`_convert_data_type`): NVARCHAR2/VARCHAR2 -> VARCHAR, NUMBER -> DECIMAL/BIGINT/INTEGER, fallback to lowercase.

---

## DuckDBLoader

`duckdb_loader.py`

Loads metadata from a local DuckDB file using the duckdb Python library directly (no SQLAlchemy). Produces DataFrames with the same column schema as TDSLoader.

### Constructor

```python
DuckDBLoader(db_path: str, schemas: list[str])
```

### Properties

| Property | Description |
|---|---|
| `PhysicalTable` | Same schema as TDSLoader, `data_entity_type = "DUCKDB_TABLE"` |
| `Col` | Same schema as TDSLoader, `data_entity_type = "DUCKDB_COLUMN"` |

### Methods

```python
load_tables() -> pd.DataFrame    # Queries duckdb_tables() system table
load_columns() -> pd.DataFrame   # Queries information_schema.columns + duckdb_columns()
list_schemas() -> list[tuple[str, int]]  # Read-only (schema 名, 表数量)，排除内部 schema
```

`list_schemas()` 排除 DuckDB 内部 schema（`information_schema` / `pg_catalog` 以及 `system` / `temp` 目录），仅用于 `--schemas` 写错时由 CLI 给出可用 schema 提示。

构造参数 `schemas` 中出现不存在的 schema 时，`load_tables()` / `load_columns()` 返回空 DataFrame（不报错）；空结果由 `step_meta_export` 的守卫转为失败退出。

---

## TrinoLoader

`trino_loader.py`

Loads metadata from a Trino database connector.

---

## AppInfoLoader

`application.py`

Loads application metadata from Excel.

### Constructor

```python
AppInfoLoader(app_list_file: str | Path, app_limits: list[str] | None = None)
```

- Reads sheet named "应用清单"
- `app_limits`: optional filter

### Properties

| Property | Columns |
|---|---|
| `Application` | `app_id`, `name`, `app_name_en`, `app_type`, `business_domain`, `manager`, `network_area`, `maintenance_level`, `external_vendor` |

---

## StandardLoader

`standard.py`

Loads data standards and compliance info from governance DB.

### Constructor

```python
StandardLoader(db: str, workspace_uuid: str)
```

### Properties

| Property | Columns |
|---|---|
| `Standard` | `standard_id`, `name`, plus dynamically pivoted attribute columns |
| `StdCompliance` | `standard_id`, `standard_name`, `database_name`, `full_table_name`, `column_name`, `name`, `data_entity_type`, `dtype`, `size`, `precision`, `scale` |

### Methods

```python
load_standard_connects() -> pd.DataFrame  # Joins standard_conn, standard_basic, navigation
load_standards() -> pd.DataFrame          # CTE + pivot key-value attributes into columns
```

---

## RelationshipLoader

`relationship.py`

Validates and loads table relationships from JSON.

### Constants

```python
VALID_RELATIONSHIP_TYPES = {"one_to_one", "one_to_many", "many_to_one", "many_to_many"}
```

### Constructor

```python
RelationshipLoader(json_path: str, df_tables: pd.DataFrame, df_columns: pd.DataFrame)
```

### Methods

```python
load_json() -> dict                                              # Parse JSON, requires version + relationships
validate_relationship(rel: dict, index: int) -> bool             # Check required fields and type
validate_table_and_columns(rel: dict, index: int) -> bool        # Check tables/columns exist (case-insensitive)
load_relationships() -> pd.DataFrame                             # Full pipeline -> edge rows
```

Result columns: `source`, `target`, `relationship_type`, `description`, `source_columns`, `target_columns`

### Convenience Function

```python
load_relationships(json_path, df_tables, df_columns) -> pd.DataFrame
```

### JSON Format

```json
{
  "version": "...",
  "relationships": [
    {
      "source": {"PhysicalTable": "schema.table", "Cols": ["col1"]},
      "target": {"PhysicalTable": "schema.table2", "Cols": ["col2"]},
      "relationship_type": "one_to_many",
      "description": "..."
    }
  ]
}
```

---

## Node ID Generator

`node_id.py`

Generates deterministic 10-character string IDs for graph nodes.

### ID Format

```
<2-char type prefix><SHA256(business_key)[:8] uppercase hex>
```

| Node Type | Prefix | Business Key Column |
|---|---|---|
| PhysicalTable | `PT` | `full_table_name` |
| Col | `CO` | `column` |
| Application | `AP` | `app_id` |
| Standard | `ST` | `standard_id` |
| Metric | `ME` | `code` |
| Dimension | `DI` | `code` |

### Functions

```python
make_id(node_type: str, business_key: str) -> str
# Returns 10-char string ID, e.g., "PTA1B2C3D4"

assign_node_ids(df: pd.DataFrame, node_type: str, key_col: str) -> None
# In-place adds "node_id" column to df. Raises ValueError on missing keys or ID collisions.

write_node_csv(df: pd.DataFrame, path: Path, node_type: str) -> None
# Writes CSV with :ID(NodeType) as first column header.
```

---

## StandardRecommender

`recommender.py`

k-NN collaborative filtering for recommending data standards to non-compliant columns. Uses TF-IDF character n-gram vectorization with cosine similarity.

### Constants

```python
DEFAULT_WEIGHTS = {'table': 0.20, 'name': 0.26, 'comment': 0.22, 'type': 0.22, 'numeric': 0.10}
DEFAULT_K_NEIGHBORS = 5
DEFAULT_TOP_N = 3
MIN_SIMILARITY = 0.7
NGRAM_N = 2
```

### Constructor

```python
StandardRecommender(
    std_compliance: pd.DataFrame,
    weights: dict[str, float] | None = None,
    k_neighbors: int = DEFAULT_K_NEIGHBORS,
    top_n: int = DEFAULT_TOP_N,
    min_similarity: float = MIN_SIMILARITY
)
```

Features: table name, column name, column comment, data type (encoded), numeric features (size, precision, scale). Weights normalized to sum 1.0. Pre-computes feature matrix on init.

### Methods

```python
find_k_neighbors(column: pd.Series, exclude_columns: set[str] | None) -> list[tuple[int, float]]
recommend(column: pd.Series) -> list[dict[str, Any]]
batch_recommend(columns: pd.DataFrame, exclude_compliant: bool = True) -> pd.DataFrame
evaluate(test_columns: pd.DataFrame, test_standards: dict[str, str]) -> dict[str, float]
```

### Factory Function

```python
create_recommender(std_compliance, weights=None, k_neighbors=5, top_n=3) -> StandardRecommender
```

---

## MetricLoader

`metric.py`

Loads and validates metric definitions from JSON against `metric_schema.json`.

### Constructor

```python
MetricLoader(metric_file: str, df_tables: pd.DataFrame, df_columns: pd.DataFrame)
```

Validates:
- JSON Schema (draft-07)
- Source tables exist in metadata
- Derived_from references exist
- Dimension codes exist in shared_dimensions
- No cycles in derived_from DAG

### Properties

| Property | Type | Columns |
|---|---|---|
| `Metric` | Node DataFrame | `code`, `name`, `business_definition`, `type`, `formula`, `unit`, `data_type`, `owner`, `update_frequency`, `statistical_scope`, `time_scope`, `source_layer`, `version`, `effective_from` |
| `Dimension` | Node DataFrame | `code`, `name`, `granularity`, `values_example` |
| `uses_table_edges` | Edge DataFrame | `:START_ID(Metric)`, `:END_ID(PhysicalTable)` |
| `refers_column_edges` | Edge DataFrame | `:START_ID(Metric)`, `:END_ID(Col)`, `role` |
| `derived_from_edges` | Edge DataFrame | `:START_ID(Metric)`, `:END_ID(Metric)` |
| `dimension_used_edges` | Edge DataFrame | `:START_ID(Metric)`, `:END_ID(Dimension)`, `usage_type` |
| `supersedes_edges` | Edge DataFrame | `:START_ID(Metric)`, `:END_ID(Metric)`, `change_description` |

### Convenience Function

```python
load_metrics(metric_file, df_tables, df_columns) -> MetricLoader
```

---

## Metric Schema (`metric_schema.json`)

JSON Schema (draft-07) for metric definitions:

- **Root**: `version` (must be `"1.0"`), `metrics` (array, minItems 1), optional `shared_dimensions`
- **metric**: requires `code`, `name`, `business_definition`, `type` (`"atomic"` | `"derived"`), `unit`, `data_type`, `source_layer` (`"DWD"` | `"DWS"` | `"DM"`)
  - `type == "atomic"` -> requires `source_tables`
  - `type == "derived"` -> requires `derived_from` + `formula`
- **source_table**: requires `full_table_name`, optional `columns` array
- **source_column**: requires `column_name`, `role` (`"measure"` | `"filter"` | `"dimension_ref"`)
- **dimension**: requires `code`, `name`
- **dimension_ref**: requires `code`, `usage_type` (`"filter"` | `"group"` | `"slice"`)

---

## gen_networkx.py

Converts CSV node/edge files to NetworkX GML format.

```python
load_nodes(csv_dir: str) -> list[dict]
load_edges(csv_dir: str) -> pd.DataFrame
build_graph(csv_dir: str, output_gml: str, incremental: bool = False)
gml_generate() -> None  # CLI: --csv, -o/--output
```

When `incremental=True`, merges new CSV data into existing GML graph instead of rebuilding from scratch.

Supported node CSVs: PhysicalTable, Col, Application, Standard, Metric, Dimension
Supported edge CSVs: HAS_COLUMN, USE, COMPLIES_WITH, RELATES_TO, USES_TABLE, REFERS_COLUMN, DERIVED_FROM, DIMENSION_USED, SUPERSEDES

---

## utility.py

CLI orchestration functions.

```python
reorder_index(dfs: list[pd.DataFrame], start: int = 1) -> None
make_csv(output, db, workspace_uuid, app_list_file, df_app_db_map,
         relationship_file=None, metric_file=None) -> None
data_standard_recommend(output, db, workspace_uuid, df_app_db_map) -> None
```

`data_standard_recommend` uses custom weights: `table=0.25, name=0.35, comment=0.25, type=0.05, numeric=0.10`.
