# govio.graph -- Graph Backends

Provides three graph backend implementations with a unified `schema` property and `query()` method.

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
query(query: str, params: dict | None = None) -> list[list[Any]]
query_with_columns(query: str, params: dict | None = None) -> tuple[list[str], list[list[Any]]]
```

`query` executes a read-only Cypher query via `self._g.ro_query()`. Raises `ValueError` on invalid Cypher.
`query_with_columns` additionally returns column names from `result.header`（`[[name, type], ...]`，即 RETURN 别名）；`query()` 等价于取其第二项。

### Private Methods

| Method | Returns | Description |
|---|---|---|
| `_get_labels()` | `Generator[str]` | Yields all node labels via `CALL db.labels()` |
| `_get_property_names(node)` | `Generator[str]` | Yields distinct property keys for a label |
| `_get_relateships()` | `Generator[dict]` | Yields `{start, type, end}` for all relationship patterns |
| `_get_rel_properties()` | `Generator[dict]` | Yields `{types, keys}` for relationship properties |
| `_wrap_name(name)` | `str` | Wraps reserved names in backticks |

---

## LadybugGraph

`ladybug_graph.py`

Embedded local graph database backed by `.lbdb` files. Uses the `ladybug` Python library. Interface aligned with FalkorDBGraph.

### Constructor

```python
LadybugGraph(
    db_path: str | PathLike = "ontology.lbdb",
    *,
    buffer_pool_size: int = 256 * 1024 * 1024,  # 256MB
    max_db_size: int = 1 * 1024 * 1024 * 1024,   # 1GB
    read_only: bool = False,
)
```

- Creates `lb.Database` and `lb.Connection`
- Calls `refresh_schema()` on init
- `max_db_size` must be explicitly set (default 8TB mmap fails on most environments)

### Properties

| Property | Type | Description |
|---|---|---|
| `schema` | `str` | Formatted schema string with nodes, relations, and relationship patterns |
| `conn` | `lb.Connection` | Raw Ladybug connection for advanced usage |

### Methods

```python
refresh_schema() -> None
query(query: str, params: dict | None = None) -> list[list[Any]]
query_with_columns(query: str, params: dict | None = None) -> tuple[list[str], list[list[Any]]]
```

`query` executes Cypher and returns `list[list]` (data rows without headers). Raises `ValueError` on invalid Cypher.
`query_with_columns` additionally returns column names via `get_column_names()`（RETURN 别名）；`query()` 等价于取其第二项。

### Private Methods

| Method | Description |
|---|---|
| `_show_tables()` | Returns `[{name, type}]` via `CALL SHOW_TABLES()` |
| `_node_properties(label)` | Returns `(property_names, pk_name)` via `CALL TABLE_INFO()` |
| `_rel_endpoints(rel)` | Discovers `(src_label, dst_label)` for a relationship type |
| `_bt(name)` | Wraps identifier in backticks |
| `_bt_val(value)` | Escapes string literal for Cypher |

### Notes

- Ladybug uses **backticks** for identifiers (double quotes cause parse errors)
- `query()` returns `list[list]` (not `list[dict]`), consistent with FalkorDB's `ro_query` result format；`query_with_columns()` 额外返回 RETURN 别名列表（CLI 输出 records 的字段名来源）
- Ladybug Cypher 方言限制（详见 `skills/govio-query/reference-ladybug.md` 的"Cypher 方言坑"）：
  - Col 的 `column` 属性是保留字，必须反引号 `` c.`column` ``；且其值为全限定名，按列名过滤用 `column_name`
  - `WITH ... ORDER BY` 必须跟 `SKIP`/`LIMIT`
  - 不支持窗口函数（`OVER ()` 解析报错），用 `collect`/`size` 或分两段查询
  - `round(x, n)` 必须带小数位参数；无 `toFloat()`，浮点除法用 `x * 1.0 / y`
  - 变量/别名不能用保留字（如 `all`、`order`）
  - 聚合后仅分组键/聚合结果在作用域，不能再引用原变量

---

## CSV Loaders

Node/edge 文件清单与命名约定见 [data-model.md](data-model.md#csv-conventions)（节点：`Datasource`、`PhysicalTable`、`Col`、`Standard`、`Metric`、`Dimension`）。

### FalkorDB Bulk Loader

`falkordb_loader.py`

```python
import_csv_to_falkordb(csv_dir, host, port, graph_name) -> None  # Full rebuild (DELETE + INSERT)
upsert_csv_to_falkordb(csv_dir, host, port, graph_name) -> None   # Incremental MERGE
```

### Ladybug Loader

`ladybug_loader.py`

```python
import_csv_to_ladybug(csv_dir, db_path, buffer_pool_size, max_db_size) -> None  # Full rebuild
upsert_csv_to_ladybug(csv_dir, db_path, buffer_pool_size, max_db_size) -> None   # Incremental MERGE
```

属性类型：默认列为 STRING，数值型属性列 `size`/`precision`/`scale`/`order_no` 建为 INT64
（读 CSV 时 `pd.to_numeric` 转换，缺失值兑底 0），保证 `ORDER BY order_no` 等为数值语义。
存量 STRING 列的旧 `.lbdb` 需全量重建才能获得 INT64 列。
