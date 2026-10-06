# Configuration

## Config File Locations

| File | Path | Purpose |
|---|---|---|
| Main config | `~/.govio/config.yaml` | Graph backend, datasources, global settings |
| Observe store | `.govio/observe/` | DataFrame persistence (parquet + manifest) |

Managed by `ConfigManager` in `govio.cli.config`.

## Main Config Schema (Nested Format)

```yaml
# Graph backend section
graph:
  backend: "networkx"  # or "falkordb" or "ladybug"

  # NetworkX config (if backend == "networkx")
  networkx:
    gml_path: "skills/govio/assets/ontology.gml"

  # FalkorDB config (if backend == "falkordb")
  falkordb:
    host: "localhost"
    port: 6379
    graph: "ontology"

  # Ladybug config (if backend == "ladybug")
  ladybug:
    db_path: "ontology.lbdb"
    buffer_pool_size: 268435456   # 256MB, optional
    max_db_size: 1073741824       # 1GB, optional

# Datasources for observe module
datasources:
  mydb:
    url: "mysql+pymysql://user:****@host/db"
    connect_args:
      ssl: true
    encrypted_password: "gAAAAA..."  # auto-encrypted on first load
  local_duckdb:
    url: "duckdb://path/to/data"
```

### Auto-migration

`ConfigManager.load()` automatically migrates:

1. **Old flat format → nested format**: Fields like `backend`, `kundb`, `networkx`, etc. are reorganized into `metadata`, `graph`, and `datasources` sections. A backup is saved to `config.yaml.bak`.

2. **Plaintext passwords → encrypted storage**: Passwords embedded in datasource URLs are extracted, encrypted via `govio.crypto`, and stored in the `encrypted_password` field. The URL is masked.

## Validation Rules

| Field | Required | Rule |
|---|---|---|
| `graph.backend` | Yes | `"networkx"`, `"falkordb"`, or `"ladybug"` |
| `graph.networkx.gml_path` | If backend=networkx | File must exist |
| `graph.falkordb.host` | If backend=falkordb | -- |
| `graph.falkordb.port` | If backend=falkordb | -- |
| `graph.falkordb.graph` | If backend=falkordb | -- |
| `graph.ladybug.db_path` | If backend=ladybug | -- |
| `datasources.*` | No | Each entry must have `url` key |

## Graph Model Alignment

在新模型下，`datasources` 配置与图模型中的 `Datasource` 节点对齐：

- `datasources` 中已配置的名称应优先映射为 `Datasource` 节点，且 `accessible=true`, `virtual=false`
- 未在 `datasources` 中配置、但由 `meta meta` 显式指定的数据源名称，应生成 `accessible=false`, `virtual=true` 的 `Datasource` 节点
- 该映射用于治理范围表达，不要求 `filter` 字段与 `--schemas` 参数完全自动一致

## Datasource URL Formats

| Type | URL Pattern | Notes |
|---|---|---|
| MySQL | `mysql+pymysql://user:pass@host/db` | Via SQLAlchemy |
| DuckDB file | `duckdb://path/to/file.duckdb` | Direct DuckDB connection |
| DuckDB dir | `duckdb://path/to/directory` | Uses `SET file_search_path` |
| Trino | `trino://user@host/catalog/schema` | Via trino-python-client |

## Observe Store Paths

```
.govio/
  observe/
    manifest.json          # DataFrame registry
    dataframes/
      {name}.parquet       # Stored DataFrames
  output-{timestamp}.json  # Query results > 20 rows
  logs/
    query_{YYYYMMDD}.log   # Query logs
```

## Relationship JSON Format

```json
{
  "version": "1.0",
  "relationships": [
    {
      "source": {"PhysicalTable": "schema.table1", "Cols": ["col1"]},
      "target": {"PhysicalTable": "schema.table2", "Cols": ["col2"]},
      "relationship_type": "one_to_many",
      "description": "..."
    }
  ]
}
```

Valid `relationship_type` values: `one_to_one`, `one_to_many`, `many_to_one`, `many_to_many`

## Metric JSON Format

See [metadata.md](metadata.md#metric-schema-metric_schemajson) for the full JSON Schema definition.
