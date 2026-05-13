# govio.core -- Factory and Asset Generation

## GraphFactory

`graph_factory.py`

Static factory for creating graph backend instances from config.

```python
GraphFactory.create(config: dict[str, Any]) -> NetworkXGraph | FalkorDBGraph
```

### Config Requirements

- `backend`: `"networkx"` or `"falkordb"` (required)
- `networkx.gml_path`: required if backend is networkx
- `falkordb.host`, `falkordb.port`, `falkordb.graph`: required if backend is falkordb

### Errors

- `ValueError`: missing fields or unsupported backend
- `FileNotFoundError`: GML file missing (NetworkX)

---

## AssetsGenerator

`assets_generator.py`

Generates documentation assets from a graph backend.

### Constructor

```python
AssetsGenerator(graph: NetworkXGraph | FalkorDBGraph, output_dir: Path)
```

Creates output directory if not exists.

### Methods

```python
generate_schema() -> None        # Writes schema.md
generate_names() -> None         # Dispatches to NetworkX or FalkorDB name generation
generate_metric_index() -> None  # Writes metrics_index.md (atomic/derived tables)
generate_all() -> None           # Calls all three
```

### Name Generation

**NetworkX**: writes `names/node_names.md` in JSON Lines format:
```json
{"id": "...", "name": "...", "node_type": "..."}
```

**FalkorDB**: for each Application, queries tables and columns, writes `names/{name}_{app_name_en}.md`:
```markdown
# full_table_name table_name
- column_name col_name
```
