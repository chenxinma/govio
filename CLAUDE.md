# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Govio (Governance + IO) is a data governance knowledge graph platform. It extracts metadata from relational databases, builds graph structures (via FalkorDB, Ladybug, or NetworkX), and provides data standard recommendation using collaborative filtering (k-NN).

## Commands

```bash
# Install dependencies (uses uv package manager, Tsinghua mirror)
uv sync
uv sync --group dev

# Run tests
uv run pytest tests/
uv run pytest tests/ -v
uv run pytest tests/ --cov=src/govio

# Lint/format
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# CLI
govio-cli onboard           # Interactive setup wizard
govio-cli backend            # Show current graph backend
govio-cli query -c "..."     # Knowledge graph query
govio-cli meta meta --source duckdb --db x.duckdb --schemas main --output ./output   # 导入元数据 -> CSV
govio-cli meta graph --output ./output --mode update  # CSV -> 图库 + assets
govio-cli meta recommend     # Data standard recommendation
govio-cli observe info       # Show datasources + DataFrames
govio-cli observe load ...   # Load DataFrame from DB or memory
govio-cli observe compare ... # Compare two DataFrames
govio-cli observe chart ...   # Generate PNG chart
govio-cli sql build -f query.json  # Assemble metric SQL
```

## Documentation Sync Rule

**修改 skills 或功能实现时，必须同步更新 `docs/specs/` 下对应的设计文档。**

| 变更内容 | 需同步更新的 spec 文档 |
|---|---|
| CLI 命令、子命令、参数 | `docs/specs/cli.md` |
| 配置格式、配置迁移、加密 | `docs/specs/config.md` |
| 图后端、CSV loader | `docs/specs/graph.md` |
| 元数据加载器、推荐器、指标、node_id | `docs/specs/metadata.md` |
| GraphFactory、AssetsGenerator、sql_builder | `docs/specs/core.md` |
| ObserveStore、DatabaseManager、comparator、chart | `docs/specs/observe-data.md` |
| 节点/边类型、CSV 格式、ID 格式 | `docs/specs/data-model.md` |
| 依赖变更、模块增删、版本变更 | `docs/specs/README.md` |

## Architecture

### Source layout: `src/govio/`

**`cli/`** — CLI entry points (`govio-cli`):
- `main.py` — argparse dispatch to subcommands
- `config.py` — `ConfigManager` (main config) + `MetaConfigManager` (meta config), auto-migration
- `meta.py` — `meta` command group: sync (full pipeline + step functions), recommend, config
- `observe.py` — `observe` command group: info, load, release, compare, explore, chart
- `query.py` — Knowledge graph query (dispatches to networkx/falkordb/ladybug)
- `sql.py` — `sql build` command for metric SQL assembly
- `onboard.py` — Interactive setup wizard
- `std_recommend.py` — Data standard recommendation entry

**`core/`** — Shared core logic:
- `graph_factory.py` — `GraphFactory.create()`: creates NetworkXGraph/FalkorDBGraph/LadybugGraph from config
- `assets_generator.py` — `AssetsGenerator`: generates schema.md, names index, metrics_index.md
- `sql_builder.py` — `build_metric_sql()`: assembles metric query SQL (CTE, atomic/derived)

**`graph/`** — Graph database backends:
- `networkx_graph.py` — In-memory graph via NetworkX GML files
- `falkordb_graph.py` — FalkorDB (Redis-based) graph client using Cypher
- `falkordb_loader.py` — CSV bulk import/upsert to FalkorDB
- `ladybug_graph.py` — Ladybug embedded graph database (.lbdb files), Cypher queries
- `ladybug_loader.py` — CSV bulk import/upsert to Ladybug

**`metadata/`** — Metadata loading and processing:
- `database.py` — `TDSLoader`: extracts table/column metadata from TDS via SQLAlchemy
- `duckdb_loader.py` — `DuckDBLoader`: loads metadata from local DuckDB files
- `trino_loader.py` — `TrinoLoader`: loads metadata from Trino
- `application.py` — `AppInfoLoader`: loads app metadata from Excel
- `standard.py` — `StandardLoader`: loads data standards and compliance info
- `relationship.py` — `RelationshipLoader`: validates and loads table relationships from JSON
- `recommender.py` — `StandardRecommender`: k-NN collaborative filtering for data standard recommendation
- `metric.py` — `MetricLoader`: loads metric/dimension definitions from JSON
- `node_id.py` — Deterministic 10-char string ID generation (SHA256-based)
- `gen_networkx.py` — CSV → GML conversion with incremental merge support
- `utility.py` — CLI orchestration: make_csv, data_standard_recommend

**`observe_data/`** — Data observation module:
- `config.py` — DataSourceConfig, load_config
- `core/observe_store.py` — ObserveStore (parquet-backed DataFrame persistence)
- `core/database.py` — DatabaseManager (multi-datasource connection manager)
- `core/comparator.py` — TableComparator (datacompy-based comparison)
- `core/explorer.py` — RelationExplorer (FK inference, column similarity)
- `core/chart.py` — render_chart (bar/line PNG with Chinese font support)
- `core/visualizer.py` — RelationVisualizer (networkx/JSON output)
- `tools/` — CLI-facing tool functions for each observe subcommand

### Graph backends

| Backend | Config Key | Query Language | Storage |
|---|---|---|---|
| NetworkX | `graph.networkx` | Python (exec) | `.gml` file |
| FalkorDB | `graph.falkordb` | Cypher | Redis-based |
| Ladybug | `graph.ladybug` | Cypher | `.lbdb` embedded file |

### Node ID format

Node IDs are deterministic 10-char strings: `<2-char prefix><SHA256(business_key)[:8]>`.
Prefixes: PT (PhysicalTable), CO (Col), AP (Application), ST (Standard), ME (Metric), DI (Dimension).

### Key conventions

- Python 3.13+, uses modern type hints (`X | None` syntax)
- All metadata loaders return pandas DataFrames
- Node identities use dotted format: `db.schema.table.column`
- Chinese language is used in comments, print statements, and documentation
- Always use `encoding="utf-8"` for file reads/writes (Windows cp936 compatibility)
