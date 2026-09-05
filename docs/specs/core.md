# govio.core -- Factory, Asset Generation, and SQL Builder

## GraphFactory

`graph_factory.py`

Static factory for creating graph backend instances from config.

```python
GraphFactory.create(config: dict[str, Any]) -> NetworkXGraph | FalkorDBGraph | LadybugGraph
```

### Config Requirements

- `backend`: `"networkx"`, `"falkordb"`, or `"ladybug"` (required)
- `graph.networkx.gml_path`: required if backend is networkx
- `graph.falkordb.host`, `graph.falkordb.port`, `graph.falkordb.graph`: required if backend is falkordb
- `graph.ladybug.db_path`: required if backend is ladybug
  - `graph.ladybug.buffer_pool_size`: optional (default 256MB)
  - `graph.ladybug.max_db_size`: optional (default 1GB)

### Errors

- `ValueError`: missing fields or unsupported backend
- `FileNotFoundError`: GML file missing (NetworkX)

---

## AssetsGenerator

`assets_generator.py`

Generates documentation assets from a graph backend.

### Constructor

```python
AssetsGenerator(graph: NetworkXGraph | FalkorDBGraph | LadybugGraph, output_dir: Path)
```

Creates output directory if not exists.

### Methods

```python
generate_schema() -> None        # Writes schema.md
generate_names() -> None         # Dispatches to backend-specific name generation
generate_metric_index() -> None  # Writes metrics_index.md (atomic/derived tables)
generate_all() -> None           # Calls all three
```

### Name Generation

**NetworkX**: writes `names/node_names.md` in JSON Lines format:
```json
{"id": "...", "name": "...", "node_type": "..."}
```

**FalkorDB / Ladybug**: for each Application, queries tables and columns, writes `names/{name}_{app_name_en}.md`:
```markdown
# full_table_name table_name
- column_name col_name
```

---

## SQL Builder

`sql_builder.py`

Assembles metric query SQL from structured JSON specifications. Supports atomic metrics (direct table queries) and derived metrics (CTE-based combinations).

### Core Function

```python
build_metric_sql(
    metrics: list[dict],
    dimensions: list[str] | None = None,
    filters: dict[str, str] | None = None,
    order_by: str | None = None,
    limit: int = 100,
    cte_refs: dict[str, str] | None = None,
) -> str
```

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `metrics` | `list[dict]` | Metric definitions with `code`, `name`, `type`, `source_table`, `formula`, `actual_column`, `time_column` |
| `dimensions` | `list[str]` | Group-by dimension fields |
| `filters` | `dict[str, str]` | WHERE conditions (e.g., `{"report_ym": "202605"}`) |
| `order_by` | `str` | ORDER BY clause |
| `limit` | `int` | LIMIT (default 100) |
| `cte_refs` | `dict[str, str]` | References to pre-loaded DataFrames as CTEs |

### Metric Types

| Type | SQL Strategy |
|---|---|
| 原子 (Atomic) | Direct `SELECT ... FROM source_table` with optional GROUP BY |
| 派生 (Derived) | CTE referencing atomic metric CTEs, applying formula |

### Validation

- `report_ym` is a **mandatory filter** when the source table uses it as a time column (拉链表). Raises `ValueError` if missing.
- Each atomic metric must have `source_table`.
- Each derived metric must have `formula`.

### SQL Structure

```
WITH
  atomic_<table> AS (SELECT dims, SUM(metrics) FROM table WHERE filters GROUP BY dims),
  derived_<code> AS (SELECT dims, formula FROM atomic_<table>)
SELECT ... FROM final CTE
ORDER BY ...
LIMIT N
```

### CLI Interface

```bash
govio-cli sql build -f query.json [-o output.sql]
cat query.json | govio-cli sql build
```

See [govio-query skill](../../skills/govio-query/SKILL.md) for the full JSON request format specification.
