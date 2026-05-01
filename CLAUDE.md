# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Govio is a Python data governance knowledge graph platform. It extracts metadata from MySQL databases, builds graph structures (via FalkorDB or NetworkX), and provides data standard recommendation using collaborative filtering (k-NN). The name combines "Governance" + "IO" (data interaction/flow).

## Commands

```bash
# Install dependencies (uses uv package manager, Tsinghua mirror)
uv sync
uv sync --group dev        # includes falkordb-bulk-loader

# Run tests (pytest-style tests in tests/, but using unittest runner in README)
uv run pytest tests/

# CLI entry points (defined in pyproject.toml [project.scripts])
metadata --kundb "mysql+pymysql://user:pass@host/db" --app-list "path.xlsx" --app-map "path.json" -o ./output
metadata --kundb "..." --app-list "..." --app-map "..." -m recommend -o ./output
metadata --kundb "..." --app-list "..." --app-map "..." --relationship rel.json -o ./output
gml_generate --csv ./output -o ./output
```

## Architecture

### Source layout: `src/govio/`

**`__init__.py`** — Public API surface: exports `run`, `gml_generate`, `FalkorDBGraph`, `NetworkXGraph`.

**`graph/`** — Graph database abstractions:
- `networkx_graph.py` — In-memory graph via NetworkX GML files. Provides `schema` property (node/edge type inspection) and direct `G` access to the underlying `nx.DiGraph`.
- `falkordb_graph.py` — FalkorDB (Redis-based) graph client using Cypher queries.

**`metadata/`** — Data loading and processing pipeline:
- `database.py` — `DatabaseLoader`: extracts table/column metadata from MySQL via SQLAlchemy. Exposes `PhysicalTable` and `Col` as DataFrames. Requires `workspace_uuid` and `schema_limits`.
- `application.py` — `AppInfoLoader`: loads app metadata from Excel (openpyxl).
- `standard.py` — `StandardLoader`: loads data standards and compliance info from governance DB.
- `relationship.py` — `RelationshipLoader` / `load_relationships()`: validates table relationships from JSON (supports one_to_one, one_to_many, many_to_one, many_to_many). Returns edges DataFrame.
- `recommender.py` — `create_recommender()`: k-NN collaborative filtering for recommending data standards to non-compliant columns. Uses configurable weights (table, name, comment, type, numeric).
- `metric.py` — `MetricLoader` / `load_metrics()`: loads metric definitions from JSON (validated by `metric_schema.json`), produces Metric/Dimension node DataFrames and edge DataFrames (USES_TABLE, REFERS_COLUMN, DERIVED_FROM, DIMENSION_USED, SUPERSEDES). Validates source table references, derived_from references, and DAG property.
- `gen_networkx.py` — `build_graph()`: converts CSV node/edge files to NetworkX GML format. Reads specific CSV naming conventions (`:ID(NodeType)` columns for nodes, `:START_ID`/`:END_ID` for edges).
- `utility.py` — CLI entry point (`run()`), orchestrates the full pipeline: load metadata → generate CSVs → optionally produce GML. Also contains `data_standard_recommend()` for batch recommendation mode.

### Graph model

Node types: `PhysicalTable`, `Col`, `Application`, `Standard`, `Metric`, `Dimension`
Edge types: `HAS_COLUMN` (table→col), `USE` (app→table), `COMPLIES_WITH` (col→standard), `RELATES_TO` (table→table), `USES_TABLE` (metric→table), `REFERS_COLUMN` (metric→col), `DERIVED_FROM` (metric→metric), `DIMENSION_USED` (metric→dimension), `SUPERSEDES` (metric→metric)

`Calculation` node type and `CALCULATED_BY`/`BASED_ON` edges are reserved for future shared calculation templates.

CSV files use FalkorDB bulk-import header conventions (`:ID(Type)`, `:START_ID(Type)`, `:END_ID(Type)`). The GML generator parses these headers to reconstruct typed graphs.

### Data flow

1. `metadata` CLI → DatabaseLoader + AppInfoLoader + StandardLoader → CSV files (node + edge)
2. `metadata --relationship` appends RELATES_TO.csv
3. `metadata` with `--metric` (via onboard) appends Metric.csv, Dimension.csv, and metric edge CSVs
4. `metadata -m recommend` generates COMPLIES_WITH.csv via recommender
4. `gml_generate` → CSV files → NetworkX GML graph
5. `NetworkXGraph` loads GML for query/inspection

### Key conventions

- Python 3.13+, uses modern type hints (`X | None` syntax, not `Optional[X]`)
- All metadata loaders return pandas DataFrames
- Node identities use dotted format: `db.schema.table.column`
- `workspace_uuid` is hardcoded in utility.py but parameterized in loader classes
- Tests use pytest (despite README mentioning unittest) — run with `uv run pytest`
- Chinese language is used in comments, print statements, and documentation
