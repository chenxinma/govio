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
    datasource_map: dict[str, str] | None = None  # schema -> datasource_name 归属映射
)
```

`datasource_map` 将每个 schema 归属到一个 `Datasource`，产出的 `full_table_name` / `column` 为带数据源前缀的全限定标识（见 [data-model.md](data-model.md#identity-format)）。TDS 导入的抽取范围来自 datasource 声明文件中 `filter.schemas` 的并集。

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
DuckDBLoader(db_path: str, schemas: list[str], datasource_name: str = "")
```

`datasource_name` 用于生成全限定标识，并在导出时生成 `Datasource` / `OWNS` 归属；`schemas` 为非 TDS 导入路径的实际过滤参数（由 CLI `--schemas` 指定）。缺省空串表示不加前缀，仅供 `list_schemas()` 等只读探测场景。

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

Loads metadata from a Trino database connector. 与 `DuckDBLoader` 相同，构造参数末尾可传 `datasource_name: str = ""` 生成全限定标识与 `Datasource` / `OWNS` 归属。

---

## DatasourceLoader

`datasource.py`

加载并校验 datasource 声明文件（`datasource.json`，schema 见 [Datasource Schema](#datasource-schema-datasource_schemajson)），生成 `Datasource` 节点数据与 schema 归属映射。

### Constructor

```python
DatasourceLoader(datasource_file: str | Path)  # 加载并完成 JSON Schema + 语义校验
```

### Methods

```python
get(datasource_name: str) -> DatasourceDef        # 按名取声明（缺失抛 KeyError）
schemas_for(datasource_name: str) -> list[str]    # 该数据源 filter.schemas
datasource_for_schema(schema: str) -> str | None  # schema -> datasource_name 反查
all_schemas() -> list[str]                        # filter.schemas 并集（TDS 抽取范围）
datasource_map() -> dict[str, str]                # schema -> datasource_name 映射
matches_table(schema: str, table_name: str) -> bool  # filter 执行（按归属数据源）
```

### Properties

| Property | Type | Columns |
|---|---|---|
| `Datasource` | Node DataFrame | `datasource_name`, `comment`, `source_type`, `filter` |
| `defs` | `list[DatasourceDef]` | 全部数据源声明 |

`DatasourceDef`（`datasource_name` / `source_type` / `name` / `filter`）提供 `schemas`、`filter_json`、`matches_table()`、`to_row()`；模块级工具函数：`qualify()`（加数据源前缀）、`resolve_datasource()`（schema 反查，未归属抛 ValueError）、`make_datasource_def()`（非 TDS 自动声明）、`filter_frames()`（按 filter 过滤表/列）、`build_owns_edges()`（OWNS 边）。

语义约束：

- `datasource_name` 全局唯一（英文，与 observe 的 `config.datasources` key 一致），重复即报错
- `comment` 为中文备注，可选
- 同一 schema 不得出现在多个条目的 `filter.schemas` 中（归属唯一）
- 非 TDS 导入且未提供声明文件时，由 CLI 参数构造等价定义：`source_type` 取导入源类型，`filter.schemas` 取 `--schemas`

---

## StandardLoader

`standard.py`

Loads data standards and compliance info from governance DB.

### Constructor

```python
StandardLoader(db: str, workspace_uuid: str, datasource_map: dict[str, str] | None = None)
```

提供 `datasource_map` 时，`StdCompliance` 的 `full_table_name` / `column` 为全限定标识（与 Col.csv 一致，供 COMPLIES_WITH 匹配）；缺省保持 `schema.table[.column]` 原始标识。

### Properties

| Property | Columns |
|---|---|
| `Standard` | `standard_id`, `name`, plus dynamically pivoted attribute columns |
| `StdCompliance` | `standard_id`, `standard_name`, `database_name`, `full_table_name`, `column`, `column_name`, `name`, `data_entity_type`, `dtype`, `size`, `precision`, `scale` |

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
validate() -> None                                               # jsonschema 校验 relationship_schema.json
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
  "version": "1.0",
  "relationships": [
    {
      "source": {"PhysicalTable": "hr_prod.ihrodb.employee", "Cols": ["dept_id"]},
      "target": {"PhysicalTable": "hr_prod.ihrodb.department", "Cols": ["dept_id"]},
      "relationship_type": "many_to_one",
      "description": "..."
    }
  ]
}
```

`PhysicalTable` / `Cols` 使用全限定标识（见 [data-model.md](data-model.md#identity-format)），结构约束由 `relationship_schema.json` 定义（见 [Relationship Schema](#relationship-schema-relationship_schemajson)）。

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
| Datasource | `DS` | `datasource_name` |
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
- **source_table**: requires `full_table_name`（全限定标识 `<datasource_name>.<schema>.<table>`）, optional `columns` array
- **source_column**: requires `column_name`, `role` (`"measure"` | `"filter"` | `"dimension_ref"`)
- **dimension**: requires `code`, `name`
- **dimension_ref**: requires `code`, `usage_type` (`"filter"` | `"group"` | `"slice"`)

文件路径：`src/govio/metadata/metric_schema.json`；可通过 `govio-cli meta schema metric` 输出，供外部 agent 生成标准指标定义。

---

## Relationship Schema (`relationship_schema.json`)

JSON Schema (draft-07) for table relationship definitions（文件路径 `src/govio/metadata/relationship_schema.json`）：

- **Root**: `version` (must be `"1.0"`), `relationships` (array, minItems 1)，`additionalProperties: false`
- **relationship**: requires `source`, `target`, `relationship_type`
  - `relationship_type`: `"one_to_one"` | `"one_to_many"` | `"many_to_one"` | `"many_to_many"`
  - `source` / `target`: requires `PhysicalTable`（全限定标识）、`Cols`（string array, minItems 1，支持复合键）
  - optional `description`: string
- 可通过 `govio-cli meta schema relationship` 输出

---

## Datasource Schema (`datasource_schema.json`)

JSON Schema (draft-07) for datasource declaration files（文件路径 `src/govio/metadata/datasource_schema.json`）：

- **Root**: `version` (must be `"1.0"`), `datasources` (array, minItems 1)，`additionalProperties: false`
- **datasource**: requires `datasource_name`, `source_type`
  - optional `comment`: 中文备注
  - `datasource_name`: 英文唯一名，与 observe 的 `config.datasources` key 一致
  - `source_type`: 开放字符串，已知取值 `duckdb` / `tds` / `mysql` / `postgres` / `oracle` / `hive` / `trino`
  - optional `filter`: requires `schemas` (array, minItems 1)；optional `include_tables` / `exclude_tables`（glob 数组）
- **语义校验**（超出 JSON Schema，由 `DatasourceLoader` 执行）：`datasource_name` 唯一、schema 归属唯一
- 可通过 `govio-cli meta schema datasource` 输出

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

Supported node CSVs: Datasource, PhysicalTable, Col, Standard, Metric, Dimension
Supported edge CSVs: HAS_COLUMN, OWNS, COMPLIES_WITH, RELATES_TO, USES_TABLE, REFERS_COLUMN, DERIVED_FROM, DIMENSION_USED, SUPERSEDES

---

## utility.py

CLI orchestration functions.

```python
reorder_index(dfs: list[pd.DataFrame], start: int = 1) -> None
make_csv(output, db, workspace_uuid, datasource_file,
         relationship_file=None, metric_file=None) -> None
data_standard_recommend(output, db, workspace_uuid, schemas: list[str],
                        datasource_name: str, csv_dir: Path | None = None) -> None
```

`make_csv` 产出 `PhysicalTable.csv`、`Col.csv`、`HAS_COLUMN.csv`、`Datasource.csv`、`OWNS.csv`，可选追加 `RELATES_TO.csv` 与指标相关 CSV；TDS 抽取范围来自 `datasource_file` 各条目 `filter.schemas` 的并集。`schemas` 由调用方从 `Datasource.filter.schemas` 解析后传入；`csv_dir` 为已导入 CSV 目录（读取 `Col.csv` / `Standard.csv`，缺省取 `output`）。

`data_standard_recommend` uses custom weights: `table=0.25, name=0.35, comment=0.25, type=0.05, numeric=0.10`.
