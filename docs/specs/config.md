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

`config.yaml` 的 `datasources` 与图模型中的 `Datasource` 节点是两个独立概念，仅靠 `datasource_name` 名字契约关联：

- `config.datasources` 是 **observe 运行时连接配置**，只管可达性
- `Datasource` 节点是 **治理侧信息记录**（`datasource_name` / `name` / `source_type` / `filter`），不保存任何连接或可达性状态
- 两者没有同步要求：连接配置的增删改不要求更新图，反之亦然

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

## Datasource Declaration File

`meta meta`（TDS 模式）与 `meta recommend` 使用的治理侧声明文件（示例路径 `data/datasource.json`），schema 见 [metadata.md](metadata.md#datasource-schema-datasource_schemajson)，可用 `govio-cli meta schema datasource` 输出。

```json
{
  "version": "1.0",
  "datasources": [
    {
      "datasource_name": "hr_prod",
      "name": "人力资源生产库",
      "source_type": "oracle",
      "filter": {
        "schemas": ["ihrodb", "IHRO_BILL"],
        "include_tables": ["*"],
        "exclude_tables": ["tmp_*", "bak_*"]
      }
    }
  ]
}
```

- `filter.schemas` 决定 TDS 导入的抽取范围；同一 schema 不得归属多个 datasource
- `include_tables` / `exclude_tables` 为 `fnmatch` glob，大小写不敏感，exclude 优先
- 首版初始数据可由 `data/app_map.json` 转换：按 `name` 分组聚合 `schema` 为 `filter.schemas`，`datasource_name` 取 `name`（后续可换英文标识），`source_type` 需人工补充

## Relationship JSON Format

```json
{
  "version": "1.0",
  "relationships": [
    {
      "source": {"PhysicalTable": "hr_prod.ihrodb.employee", "Cols": ["dept_id"]},
      "target": {"PhysicalTable": "hr_prod.ihrodb.department", "Cols": ["dept_id"]},
      "relationship_type": "one_to_many",
      "description": "..."
    }
  ]
}
```

Valid `relationship_type` values: `one_to_one`, `one_to_many`, `many_to_one`, `many_to_many`

完整结构约束见 [metadata.md](metadata.md#relationship-schema-relationship_schemajson)；`govio-cli meta schema relationship` 可输出该 schema。

## Metric JSON Format

See [metadata.md](metadata.md#metric-schema-metric_schemajson) for the full JSON Schema definition.
