# govio.graph -- Graph Backends

Provides two graph backend implementations with a unified `schema` property.

## NetworkXGraph

`networkx_graph.py`

Loads a GML file into an in-memory NetworkX directed graph.

### Constructor

```python
NetworkXGraph(graph: str | PathLike = "ontology.gml")
```

- Raises `FileNotFoundError` if file missing
- Calls `refresh_schema()` on init

### Properties

| Property | Type | Description |
|---|---|---|
| `schema` | `str` | Human-readable schema: node types, edge types, edge relationships |
| `G` | `nx.Graph` | Underlying NetworkX graph object |

### Methods

```python
refresh_schema() -> None
```

Scans all nodes/edges to build schema dict:
- `node_types`: `type -> attribute keys`
- `edge_relationships`: `(src_type)-[rel_type]->(dst_type) -> attribute keys`

Nodes must have `node_type` attribute. Edges use `edge_type` (defaults to `"connected_to"`).

---

## FalkorDBGraph

`falkordb_graph.py`

Connects to a FalkorDB (Redis-based) graph database.

### Constructor

```python
FalkorDBGraph(graph: str = "ontology", host: str = 'localhost', port: int = 6379)
```

- Connects via `FalkorDB(host=host, port=port)`
- Selects graph by name
- Calls `refresh_schema()` on init

### Properties

| Property | Type | Description |
|---|---|---|
| `schema` | `str` | Formatted schema: node labels, properties, relationship patterns, relationship properties |

### Methods

```python
refresh_schema() -> None
```

Queries FalkorDB for labels, node properties, relationship triples, and relationship properties.

```python
query(query: str, params: dict = {}) -> list[dict[str, Any]]
```

Executes a read-only Cypher query via `self._g.ro_query()`. Raises `ValueError` on invalid Cypher.

### Private Methods

| Method | Returns | Description |
|---|---|---|
| `_get_labels()` | `Generator[str]` | Yields all node labels via `CALL db.labels()` |
| `_get_property_names(node)` | `Generator[str]` | Yields distinct property keys for a label |
| `_get_relateships()` | `Generator[dict]` | Yields `{start, type, end}` for all relationship patterns |
| `_get_rel_properties()` | `Generator[dict]` | Yields `{types, keys}` for relationship properties |
| `_wrap_name(name)` | `str` | Wraps reserved names in backticks |
