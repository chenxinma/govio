# Govio Spec Documentation

Govio (Governance + IO) is a data governance knowledge graph platform. It extracts metadata from relational databases, builds graph structures, and provides data standard recommendation via collaborative filtering.

- **Version**: 0.5.4
- **Python**: >= 3.13
- **Build**: hatchling
- **CLI**: `govio-cli` -> `govio.cli:main`

## Module Specs

| Module | Spec | Description |
|---|---|---|
| `govio.graph` | [graph.md](graph.md) | Graph backends (NetworkX, FalkorDB, Ladybug) |
| `govio.metadata` | [metadata.md](metadata.md) | Metadata loading, recommendation, metric definition, node ID generation |
| `govio.core` | [core.md](core.md) | Graph factory, asset generation, SQL builder |
| `govio.cli` | [cli.md](cli.md) | CLI entry points and subcommands |
| `govio.observe_data` | [observe-data.md](observe-data.md) | Data observation, comparison, exploration, charting |
| `govio.crypto` | (inline) | Credential encryption/decryption for config passwords |
| Data Model | [data-model.md](data-model.md) | Node types, edge types, CSV conventions |
| Configuration | [config.md](config.md) | Config file format, datasource definitions, meta config |

## Skills

Govio ships with AI assistant skills under `skills/`:

| Skill | Description |
|---|---|
| `govio` | 主控 Skill，需求识别与路由入口 |
| `govio-meta` | 知识图谱维护（同步元数据、推荐标准、管理配置） |
| `govio-query` | 元数据/指标查询（Cypher/Python 查询、SQL 组装） |
| `govio-observe` | 数据探查与比对（加载、探索、比对、图表） |
| `govio-eda` | EDA 探索性数据分析（4 阶段探查流程） |

## Dependencies

| Package | Purpose |
|---|---|
| pandas | Core data manipulation |
| sqlalchemy | Database abstraction (TDS, MySQL) |
| networkx | Local graph backend |
| falkordb | FalkorDB graph client |
| falkordb-bulk-loader | Bulk CSV import to FalkorDB |
| ladybug | Embedded local graph database (.lbdb files) |
| scikit-learn | TF-IDF vectorization, cosine similarity |
| openpyxl | Excel file reading |
| jsonschema | JSON Schema validation |
| duckdb | DuckDB data source |
| datacompy | DataFrame comparison |
| pyyaml | YAML config I/O |
| tqdm | Progress bars |
| questionary | Interactive CLI prompts |
| matplotlib | Chart rendering (bar/line) |
| mcp | Model Context Protocol support |
| trino | Trino database connector |
| pymysql | MySQL database connector |
| cryptography | Password encryption for config |
