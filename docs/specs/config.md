# Configuration

## Config File Location

Default: `~/.govio/config.yaml`

Managed by `ConfigManager` in `govio.cli.config`.

## Full Config Schema

```yaml
# Metadata extraction source
kundb: "mysql+pymysql://user:pass@host/db"
app_list: "path/to/app_list.xlsx"
app_map: "path/to/app_map.json"
relationship: "path/to/relationships.json"  # optional
metric: "path/to/metrics.json"              # optional

# Paths
csv_dir: "./output"
output_dir: "./output"
workspace_uuid: "82ee37374b314a938bf28170ab4db7cf"

# Graph backend
backend: "networkx"  # or "falkordb"

# NetworkX config (if backend == "networkx")
networkx:
  gml_path: "skills/govio/assets/ontology.gml"

# FalkorDB config (if backend == "falkordb")
falkordb:
  host: "localhost"
  port: 6379
  graph: "ontology"

# Observe module datasources
datasources:
  mydb:
    url: "mysql+pymysql://user:pass@host/db"
    connect_args:
      ssl: true
  local_duckdb:
    url: "duckdb://path/to/data"
```

## Validation Rules

| Field | Required | Rule |
|---|---|---|
| `backend` | Yes | `"networkx"` or `"falkordb"` |
| `networkx.gml_path` | If backend=networkx | File must exist |
| `falkordb.host` | If backend=falkordb | -- |
| `falkordb.port` | If backend=falkordb | -- |
| `falkordb.graph` | If backend=falkordb | -- |
| `csv_dir` | No | Path must exist |
| `graph_dir` | No | Path must exist |
| `datasources.*` | No | Each entry must have `url` key |

## Datasource URL Formats

| Type | URL Pattern | Notes |
|---|---|---|
| MySQL | `mysql+pymysql://user:pass@host/db` | Via SQLAlchemy |
| DuckDB file | `duckdb://path/to/file.duckdb` | Direct DuckDB connection |
| DuckDB dir | `duckdb://path/to/directory` | Uses `SET file_search_path` |

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
