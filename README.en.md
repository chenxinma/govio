# Govio

[中文](README.md) | **English**

✅ Core word formation: Gov (Governance, the core of data governance) + io (Data IO, data interaction / data flow, the core vehicle of data governance)
✅ Deep meaning: with "data governance" as the core and "data interaction" as the delivery path, it carries metadata management, data standard unification, and data quality validation in one place, empowering compliant governance and efficient flow across the entire data lifecycle.

A data governance knowledge graph platform providing metadata query, table/column comparison, metric SQL assembly, data standard recommendation, metric lineage analysis, data exploration & comparison, and EDA analysis.

## Features

- **Metadata query**: query metadata such as names, descriptions, origins, and status of data assets
- **Table/column comparison**: compare column differences between tables (column names, data types, nullability, etc.)
- **Metric SQL assembly**: automatically assemble query SQL from metric JSON specs
- **Data standard recommendation**: recommend suitable data standards for unstandardized columns via collaborative filtering
- **Metric management**: bring business metrics into the knowledge graph, supporting metric lineage tracing, impact analysis, data provenance, and dimension discovery
- **Data exploration & comparison**: load data source data, explore table relations, compare data differences, and generate visualization charts
- **EDA analysis**: standard 4-stage exploratory data analysis flow (profiling → infer relations → verify relations → consistency check)
- **Multiple graph backends**: NetworkX (local GML), FalkorDB (Redis graph database), Ladybug (embedded `.lbdb`)
- **Multiple metadata sources**: metadata can be read from TDS (KunDB/MySQL metadata database), DuckDB, or both merged

## Installation

> 📖 For a detailed installation guide, sample data import, and Agent integration steps, see the [Installation Guide](docs/install.md) (currently in Chinese).

```bash
# Install uv (if not already installed)
# https://docs.astral.sh/uv/getting-started/installation/

# Install from PyPI
uv tool install govio

# Or build from source
uv build
./start.sh
```

`start.sh` runs `uv tool install --from "$WHL" govio --force` internally, installing `govio-cli` as a persistent command-line tool.

**Development mode:**

```bash
# Install project dependencies
uv sync

# Install dev dependencies (incl. falkordb-bulk-loader, pytest, ruff)
uv sync --group dev

# Run tests
uv run pytest tests/
```

## From Metadata to Knowledge Graph

Govio turns enterprise metadata into a knowledge graph, supporting three graph database backends and multiple metadata sources:

```
┌───────────────────────────────────┐    ┌─────────────────────────────┐
│  Metadata sources                 │    │  Data sources (for observe) │
│  TDS(KunDB/MySQL) / DuckDB / Both │    │  MySQL / DuckDB ...         │
└────────────────┬──────────────────┘    └──────────────┬──────────────┘
                 │                                      │
                 │ govio-cli meta <subcommand>          │ govio-cli onboard
                 │ (explicit args, no config file)      │ (graph backend + datasources)
                 ▼                                      ▼
┌──────────────────────────────────────────────────────────────────────┐
│  ① meta meta/std/compliance/rel/metric -> CSV (incremental merge)    │
│  ② meta graph --mode update -> update graph -> generate assets       │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│          Graph backend (selected via govio-cli onboard)              │
│      NetworkX (GML)   │   FalkorDB   │   Ladybug (.lbdb embedded)    │
└────────────────────────────────────────┬─────────────────────────────┘
                                         │
                                         ▼
                    ┌────────────────────────────────────┐
                    │  .agents/skills/govio/assets       │
                    │  schema.md / metrics_index.md      │
                    │  ontology.gml / names/             │
                    └────────────────────────────────────┘
                                         │
                                         ▼
                  govio-cli query / observe / sql  +  Skills toolset
```

## Quick Start

Govio provides the interactive CLI tool `govio-cli`. The typical workflow has three steps:

### Step 1: Install govio-cli

```bash
uv build && ./start.sh
```

### Step 2: Run the Onboard wizard

```bash
govio-cli onboard
```

The wizard guides you through:

1. **Choosing a graph backend**
   - `networkx`: local GML file
   - `falkordb`: connect to a FalkorDB graph database (host, port, graph name)
   - `ladybug`: Ladybug embedded graph database (default `~/.govio/ontology.lbdb`)

2. **Configuring data sources (optional, used by the `observe` command)**
   - Add MySQL, DuckDB, and other data sources
   - Passwords are encrypted automatically; the config file is safe to share

Configuration is saved to `~/.govio/config.yaml`.

### Step 3: Import metadata into the knowledge graph

Each `meta` subcommand runs independently with explicit arguments (no config file dependency), following a two-step workflow: **subcommand generates CSV → `meta graph` imports into the graph**:

```bash
# ① Generate/merge CSV (--datasource and --schemas are required; the default schema of a DuckDB file is main)
govio-cli meta meta --source duckdb --db ./data/chocolate.db --datasource chocolate --schemas chocolate --output ./data/meta

# ② Incrementally import into the graph + generate assets
govio-cli meta graph --output ./data/meta --mode update
```

The `--output` directory is the only shared state between subcommands: always reuse the same directory for incremental merging (idempotent). Pointing at the wrong directory loses previous increments.

For a first-time build, run the required subcommands in dependency order (`meta` must run first). For later top-ups, run only the subcommand with new input plus `meta graph`.

When done, copy `skills/govio*` into your Agent's skills directory to query in natural language.

<details>
<summary>meta meta + meta graph example run</summary>

```bash
$ govio-cli meta meta --source duckdb --db ./data/chocolate.db --datasource chocolate --schemas chocolate --output ./data/meta
Reading metadata from DuckDB...
✓ Metadata exported: 4 tables, 17 columns

$ govio-cli meta graph --output ./data/meta --mode update
Incrementally updating Ladybug graph (C:\Users\Administrator\.govio\ontology.lbdb)...
  PhysicalTable.csv: 4 rows -> PhysicalTable
  Col.csv: 17 rows -> Col
  HAS_COLUMN.csv: 17 rows -> HAS_COLUMN
✓ Ladybug data updated

Generating assets...
✓ Assets generated at: D:\Work\gov-io\govio\.agents\skills\govio\assets
  (resolved from .agents/skills/govio/assets under the current working directory; use --assets-dir to choose another directory; if this differs from the assets directory your application reads, merge them yourself)

✅ graph update complete!
```

If `--schemas` is wrong or the source has no tables, the command fails directly, writes no CSV, and lists the schemas available in that database:

```bash
$ govio-cli meta meta --source duckdb --db ./data/sales.duckdb --schemas sales --output ./data/meta
❌ No tables found: schema [sales] does not exist or is empty in the metadata source (no CSV written)
   Importable schemas in ./data/sales.duckdb: main (5 tables)
   Please check --schemas and retry
```

</details>

## CLI Command Overview

| Command | Purpose |
|------|------|
| `govio-cli onboard` | Initial setup wizard (graph backend + data sources) |
| `govio-cli backend` | Show current graph backend type |
| `govio-cli meta` | Knowledge graph maintenance (import subcommands / `graph` / `recommend`) |
| `govio-cli query` | Knowledge graph query (Cypher or Python) |
| `govio-cli observe` | Data exploration (load / compare / explore / chart) |
| `govio-cli sql` | Metric SQL assembly (`build`) |
| `govio-cli -V` | Show version |

### meta Command Group

All subcommands are fully independent and CLI-only (no config file dependency). They share state via CSV files under the `--output` directory and support incremental merging (idempotent).

| Subcommand | Purpose | Output CSV |
|--------|------|----------|
| `govio-cli meta meta` | Import TDS/DuckDB metadata | Datasource, PhysicalTable, Col, HAS_COLUMN, OWNS |
| `govio-cli meta std` | Import data standards (TDS only) | Standard |
| `govio-cli meta compliance` | Export existing standard associations (TDS only) | COMPLIES_WITH |
| `govio-cli meta rel` | Import table relations | RELATES_TO |
| `govio-cli meta metric` | Import metric/dimension definitions | Metric, Dimension + 5 edge types |
| `govio-cli meta schema` | Output standard JSON Schemas (metric/relationship/datasource) | — (printed or written to file) |
| `govio-cli meta graph` | Update/rebuild/clear graph + generate assets | graph + assets |
| `govio-cli meta recommend` | Data standard recommendation | recommendation results |
| `govio-cli meta import-schema` | One-shot import from a configured DuckDB data source (metadata + graph + assets) | graph + assets |

Recommended order for a first-time build: `meta → std → compliance → rel → metric → graph`. For top-ups, run only the subcommand with new input plus `graph`.

```bash
# DuckDB metadata (--db / --datasource / --schemas required)
govio-cli meta meta --source duckdb --db /path/to/meta.duckdb --datasource mydb --schemas main --output ./data/meta

# TDS metadata (--kundb / --workspace-uuid / --datasources-file required; extraction scope comes from filter.schemas in the declaration file)
govio-cli meta meta --source tds --kundb "mysql+pymysql://user:pass@host:port/catalog" \
  --workspace-uuid <uuid> --datasources-file ./data/datasources.json --output ./data/meta

# Table relations / metric definitions
govio-cli meta rel --file ./data/relationships.json --output ./data/meta
govio-cli meta metric --file ./data/metrics.json --output ./data/meta

# Import into the graph + generate assets
govio-cli meta graph --output ./data/meta --mode update
```

**Metadata sources (`meta meta --source`):**

| Source | Description |
|------|------|
| `tds` | Read from the metadata database only; `--kundb`, `--workspace-uuid`, `--datasources-file` required; extraction scope comes from `filter.schemas` in the declaration file |
| `duckdb` | Read from a DuckDB file only; `--db`, `--datasource`, `--schemas` required; **skips Standard data standards** |

**`meta graph --mode`:**

| Mode | Description |
|------|------|
| `update` | Incremental MERGE (default), safe to re-run |
| `rebuild` | Delete and re-insert |
| `clear` | Clear the graph only, no import |

**Naming & failure behavior:**

- Node names come from `full_table_name = <schema>.<table>`; node_id is auto-generated by hashing the business key
- `--schemas` must contain schema names that actually exist in the source database (the default schema of a DuckDB file is `main`); renaming on import is not supported in the current version
- If a schema does not exist or the source has no tables, the command fails without writing CSV, and the error lists the importable schemas of that database
- Assets are written to `.agents/skills/govio/assets` under the current working directory by default (use `--assets-dir` to choose another directory), and the absolute path is printed at the end; if this differs from the directory your application reads, merge them yourself (improvement tracked in `docs/roadmap.md`)

### Query Tool

```bash
# Backend chosen automatically from config.yaml (FalkorDB/Ladybug use Cypher, NetworkX uses Python)
govio-cli query -c "MATCH (n:PhysicalTable) RETURN n.name LIMIT 5"
```

**Query rules:**

- FalkorDB / Ladybug: use the Cypher query language; queries must start with `MATCH`
- NetworkX: use Python code operating on `g` (the NetworkX graph object); assign the result to `result`

```bash
# NetworkX mode (Python code, assign the result to result)
govio-cli query -c "result = [n for n, d in g.nodes(data=True) if d.get('node_type') == 'PhysicalTable'][:5]"
```

> Results over 20 rows are written to `.govio/output-<timestamp>.json` automatically; smaller results are printed as JSON. Query logs go to `~/.govio/logs/query_<date>.log`.

### Skills Toolset

Once configured, copy `skills/govio*` into your Agent directory to query in natural language. Configuration is read automatically from `~/.govio/config.yaml`.

**Skill list:**

| Skill | Purpose |
|-------|------|
| `govio` | Master skill — identifies the request type and routes to sub-skills |
| `govio-meta` | Knowledge graph maintenance (import metadata, graph update, recommendation) |
| `govio-query` | Metadata/metric queries (applications, tables, columns, metric questions) |
| `govio-observe` | Data exploration & comparison (load, explore, compare, chart) |
| `govio-eda` | EDA exploratory data analysis (standard 4-stage flow) |

**Directory structure:**

```
skills/govio/
├── SKILL.md              # Skill definition (used by AI Agents)
├── assets/               # Asset files (auto-generated by meta graph)
│   ├── schema.md         # Graph database schema
│   ├── metrics_index.md  # Metric index (grouped by atomic/derived)
│   ├── ontology.gml      # NetworkX GML data file
│   └── names/            # Node name index
│       └── *.md          # One file per application (Cypher backends)
└── ...
```

**Using with AI Agents:** after loading `SKILL.md`, query directly in natural language, e.g.:
- "How many tables does the CRM application have?"
- "Find all table names containing 'user'"
- "What is this month's billing revenue?"

### CSV File Format Requirements

The CSV files generated by the `meta` subcommands follow the FalkorDB bulk-import header convention (`:ID(Type)`, `:START_ID(Type)`, `:END_ID(Type)`).

**Node files:**
- `PhysicalTable.csv`: physical table nodes
- `Col.csv`: column nodes
- `Datasource.csv`: data source nodes
- `Standard.csv`: data standard nodes
- `Metric.csv`: metric nodes (optional, generated from metric definition JSON)
- `Dimension.csv`: dimension nodes (optional, generated from metric definition JSON)

**Edge files:**
- `HAS_COLUMN.csv`: table contains column
- `OWNS.csv`: data source governs physical table
- `COMPLIES_WITH.csv`: column complies with standard (exported by `meta compliance`)
- `RELATES_TO.csv`: relations between tables
- `USES_TABLE.csv`: metric's source table (optional)
- `REFERS_COLUMN.csv`: metric references column (optional)
- `DERIVED_FROM.csv`: derived metric depends on upstream metric (optional)
- `DIMENSION_USED.csv`: metric-dimension relation (optional)
- `SUPERSEDES.csv`: metric version evolution (optional)

**CSV format examples:**

```csv
# PhysicalTable.csv
:ID(PhysicalTable),name,full_table_name
table1,用户表,DB.SCHEMA.TABLE1

# Col.csv
:ID(Col),name,column_name,full_table_name
col1,用户ID,USER_ID,DB.SCHEMA.TABLE1

# HAS_COLUMN.csv
:START_ID(PhysicalTable),:END_ID(Col)
table1,col1
```

## Python API

### Using the Graph Database

**NetworkX mode:**

```python
from govio import NetworkXGraph

# Load a NetworkX graph
graph = NetworkXGraph(graph="./output/ontology.gml")

# Inspect the graph schema
print(graph.schema)

# Query the graph with Python
nodes = list(graph.G.nodes(data=True))
cols = [n for n, data in graph.G.nodes(data=True) if data.get("node_type") == "Col"]

for u, v, data in graph.G.edges(data=True):
    print(f"{u} --[{data.get('edge_type')}]--> {v}")
```

**FalkorDB mode:**

```python
from govio import FalkorDBGraph

# Connect to a FalkorDB graph database
graph = FalkorDBGraph(host="localhost", port=6379, graph="ontology")

# Inspect the graph schema
print(graph.schema)

# Query with Cypher
result = graph.query("MATCH (n:PhysicalTable) RETURN n.name LIMIT 10")
```

**Ladybug mode (embedded, Cypher-compatible with FalkorDB):**

```python
from govio import LadybugGraph

# Open a local .lbdb file (no extra service needed)
graph = LadybugGraph("~/.govio/ontology.lbdb", read_only=True)

# Inspect the graph schema
print(graph.schema)

# Query with Cypher
result = graph.query("MATCH (n:PhysicalTable) RETURN n.name LIMIT 10")
```

### Loading Metadata

```python
from govio.metadata.database import TDSLoader
from govio.metadata.standard import StandardLoader
from govio.metadata.duckdb_loader import DuckDBLoader
from govio.metadata.metric import MetricLoader

# Load table/column metadata from TDS (metadata database)
tds_loader = TDSLoader(
    db="mysql+pymysql://user:pass@host/db",
    workspace_uuid="your-uuid",
    schema_limits=["schema1", "schema2"],
)
tables = tds_loader.PhysicalTable
columns = tds_loader.Col

# Load table/column metadata from DuckDB
duck_loader = DuckDBLoader(
    db_path="xxx.duckdb",
    schemas=["schema1", "schema2"],
)
tables = duck_loader.PhysicalTable
columns = duck_loader.Col

# Load data standards (TDS only)
std_loader = StandardLoader(
    db="mysql+pymysql://user:pass@host/db",
    workspace_uuid="your-uuid",
)
standards = std_loader.Standard
std_compliance = std_loader.StdCompliance

# Load metric definitions
metric_loader = MetricLoader(
    metric_file="path/to/metrics.json",
    df_tables=tables,
    df_columns=columns,
)
metrics = metric_loader.Metric          # metric node DataFrame
dimensions = metric_loader.Dimension    # dimension node DataFrame
```

### Metric Definition JSON Format

Metric definitions use a JSON file validated against a JSON Schema:

```json
{
  "version": "1.0",
  "shared_dimensions": [
    { "code": "ym", "name": "年月", "granularity": "月" },
    { "code": "business_unit", "name": "事业部" }
  ],
  "metrics": [
    {
      "code": "bill_income_amt",
      "name": "当月账单收入",
      "business_definition": "当月实际确认的账单收入金额",
      "type": "atomic",
      "unit": "万元",
      "data_type": "decimal(18,2)",
      "source_layer": "DWS",
      "source_tables": [
        {
          "full_table_name": "db.schema.t_bill",
          "columns": [
            { "column_name": "bill_income_amt", "role": "measure" }
          ]
        }
      ],
      "dimensions": [
        { "code": "ym", "usage_type": "group" },
        { "code": "business_unit", "usage_type": "slice" }
      ]
    },
    {
      "code": "burndown_amt",
      "name": "存量消耗额",
      "business_definition": "预计当月产生的账单收入减去危机金额",
      "type": "derived",
      "formula": "forecast_income_amt - risk_amt",
      "unit": "万元",
      "data_type": "decimal(18,2)",
      "source_layer": "DM",
      "derived_from": ["forecast_income_amt", "risk_amt"],
      "dimensions": [
        { "code": "ym", "usage_type": "group" }
      ]
    }
  ]
}
```

### Metric Query Examples

```python
# FalkorDB / Ladybug: metric lineage tracing
result = graph.query(
    "MATCH p=(m:Metric {code: 'burndown_amt'})-[:DERIVED_FROM*1..5]->(up:Metric) "
    "RETURN up.code, up.name, up.type"
)

# FalkorDB / Ladybug: impact analysis (what downstream metrics a change affects)
result = graph.query(
    "MATCH (m:Metric {code: 'bill_income_amt'})<-[:DERIVED_FROM*1..5]-(dep:Metric) "
    "RETURN dep.code, dep.name, dep.formula"
)

# FalkorDB / Ladybug: data provenance (a metric's source tables)
result = graph.query(
    "MATCH (m:Metric {code: 'bill_income_amt'})-[:USES_TABLE]->(t:PhysicalTable) "
    "RETURN t.full_table_name, t.name"
)

# FalkorDB / Ladybug: dimension discovery (how a metric can be broken down)
result = graph.query(
    "MATCH (m:Metric {code: 'burndown_amt'})-[d:DIMENSION_USED]->(dim:Dimension) "
    "RETURN dim.code, dim.name, d.usage_type"
)

# NetworkX: metric lineage tracing
import networkx as nx
metric_node = [n for n, d in graph.G.nodes(data=True)
               if d.get('node_type') == 'Metric' and d.get('code') == 'burndown_amt'][0]
ancestors = nx.ancestors(graph.G, metric_node)
lineage = [(n, graph.G.nodes[n].get('code')) for n in ancestors
           if graph.G.nodes[n].get('node_type') == 'Metric']
```

### Metric SQL Assembly

```python
from govio import build_metric_sql

sql_text = build_metric_sql(
    metrics=[
        {"code": "bill_income_amt", "name": "当月账单收入", "type": "原子",
         "source_table": "db.schema.t_bill", "actual_column": "bill_income_amt"},
    ],
    dimensions=["ym"],
    filters={"report_ym": "202501"},
    order_by="metric_value DESC",
    limit=100,
)
print(sql_text)
```

Or via the CLI:

```bash
# Assemble from a JSON spec file
govio-cli sql build -f spec.json -o output.sql

# Pass via stdin (metrics is a list of metric dicts; type takes the values "原子"/"派生")
echo '{"metrics": [{"code": "bill_income_amt", "name": "当月账单收入", "type": "原子", "source_table": "db.schema.t_bill", "actual_column": "bill_income_amt"}], "dimensions": ["ym"], "filters": {"report_ym": "202501"}, "limit": 100}' | govio-cli sql build
```

### Using the Data Standard Recommender

```python
from govio.metadata.recommender import create_recommender
from govio.metadata.database import TDSLoader

# Load columns that already comply with standards
std_loader = StandardLoader(db="mysql+pymysql://user:pass@host/db", workspace_uuid="your-uuid")
std_compliance = std_loader.StdCompliance

# Create the recommender
WEIGHTS = {
    'table': 0.25,     # table name weight (only table_name extracted from full_table_name)
    'name': 0.35,      # column name weight
    'comment': 0.25,   # column comment weight
    'type': 0.05,      # data type weight
    'numeric': 0.10    # numeric feature weight
}
recommender = create_recommender(
    std_compliance=std_compliance,
    weights=WEIGHTS,
    k_neighbors=5,  # use 5 nearest neighbors
    top_n=3,        # return top 3 recommendations
)

# Batch recommendation
tds_loader = TDSLoader(db, workspace_uuid, ["schema_name"])
all_columns = tds_loader.Col  # all columns
recommendations = recommender.batch_recommend(all_columns)
```

## Graph Model

**Node types:** `PhysicalTable`, `Col`, `Datasource`, `Standard`, `Metric`, `Dimension`

**Edge types:**

| Edge type | Direction | Meaning |
|--------|------|------|
| `HAS_COLUMN` | table -> col | table contains column |
| `OWNS` | datasource -> table | data source governance ownership |
| `COMPLIES_WITH` | col -> standard | column complies with standard |
| `RELATES_TO` | table -> table | relations between tables |
| `USES_TABLE` | metric -> table | metric's source table |
| `REFERS_COLUMN` | metric -> col | metric references column |
| `DERIVED_FROM` | metric -> metric | derived metric depends on upstream metric |
| `DIMENSION_USED` | metric -> dimension | metric-dimension relation |
| `SUPERSEDES` | metric -> metric | metric version evolution |

> The `Calculation` node type and `CALCULATED_BY`/`BASED_ON` edges are reserved for shared calculation templates and are not enabled yet.

Node identity uses dot notation: `db.schema.table.column`.

## License

[MIT License]
