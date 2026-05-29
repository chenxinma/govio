# Govio Spec Documentation

Govio (Governance + IO) is a data governance knowledge graph platform. It extracts metadata from relational databases, builds graph structures, and provides data standard recommendation via collaborative filtering.

- **Version**: 0.2.7
- **Python**: >= 3.13
- **Build**: hatchling
- **CLI**: `govio-cli` -> `govio.cli:main`

## Module Specs

| Module | Spec | Description |
|---|---|---|
| `govio.graph` | [graph.md](graph.md) | Graph backends (NetworkX, FalkorDB) |
| `govio.metadata` | [metadata.md](metadata.md) | Metadata loading, recommendation, metric definition |
| `govio.core` | [core.md](core.md) | Graph factory, asset generation |
| `govio.cli` | [cli.md](cli.md) | CLI entry points and subcommands |
| `govio.observe_data` | [observe-data.md](observe-data.md) | Data observation, comparison, exploration |
| Data Model | [data-model.md](data-model.md) | Node types, edge types, CSV conventions |
| Configuration | [config.md](config.md) | Config file format, datasource definitions |

## Dependencies

| Package | Purpose |
|---|---|
| pandas | Core data manipulation |
| sqlalchemy | Database abstraction |
| networkx | Local graph backend |
| falkordb | FalkorDB graph client |
| falkordb-bulk-loader | Bulk CSV import |
| scikit-learn | TF-IDF vectorization, cosine similarity |
| openpyxl | Excel file reading |
| jsonschema | JSON Schema validation |
| duckdb | DuckDB data source |
| datacompy | DataFrame comparison |
| pyyaml | YAML config I/O |
| tqdm | Progress bars |
