# Data Model -- Node Types, Edge Types, CSV Conventions

## Node Types

| Node | ID Column | Key Attributes |
|---|---|---|
| PhysicalTable | `:ID(PhysicalTable)` | `full_table_name`, `schema`, `table_name`, `name`, `data_entity_type`, `database_name` |
| Col | `:ID(Col)` | `column`, `column_name`, `name`, `full_table_name`, `data_type`, `dtype`, `size`, `precision`, `scale`, `order_no` |
| Application | `:ID(Application)` | `app_id`, `name`, `app_name_en`, `app_type`, `business_domain`, `manager`, `network_area`, `maintenance_level`, `external_vendor` |
| Standard | `:ID(Standard)` | `standard_id`, `name`, plus dynamically pivoted attribute columns |
| Metric | `:ID(Metric)` | `code`, `name`, `business_definition`, `type`, `formula`, `unit`, `data_type`, `owner`, `update_frequency`, `statistical_scope`, `time_scope`, `source_layer`, `version`, `effective_from` |
| Dimension | `:ID(Dimension)` | `code`, `name`, `granularity`, `values_example` |

## Edge Types

| Edge | From | To | Extra Attributes |
|---|---|---|---|
| HAS_COLUMN | PhysicalTable | Col | -- |
| USE | Application | PhysicalTable | -- |
| COMPLIES_WITH | Col | Standard | -- |
| RELATES_TO | PhysicalTable | PhysicalTable | `relationship_type`, `description`, `source_columns`, `target_columns` |
| USES_TABLE | Metric | PhysicalTable | -- |
| REFERS_COLUMN | Metric | Col | `role` |
| DERIVED_FROM | Metric | Metric | -- |
| DIMENSION_USED | Metric | Dimension | `usage_type` |
| SUPERSEDES | Metric | Metric | `change_description` |

## Reserved for Future

- Node: `Calculation`
- Edges: `CALCULATED_BY`, `BASED_ON`

## CSV Conventions

FalkorDB bulk-import header format:

### Node CSVs

```
:ID(NodeType),attr1,attr2,...
1,值1,值2,...
```

- Column 1: `:ID(NodeType)` -- sequential integer ID
- Remaining columns: node attributes

### Edge CSVs

```
:START_ID(NodeType),:END_ID(NodeType),attr1,...
1,2,值1,...
```

- Column 1: `:START_ID(SourceType)` -- source node ID
- Column 2: `:END_ID(TargetType)` -- target node ID
- Remaining columns: edge attributes

### Naming Convention

- Node files: `{NodeType}.csv` (e.g., `PhysicalTable.csv`, `Col.csv`)
- Edge files: `{EDGE_TYPE}.csv` (e.g., `HAS_COLUMN.csv`, `USE.csv`)

## Identity Format

Node identities use dotted format: `db.schema.table.column`

## Graph Output

`gen_networkx.py` reads CSV files and produces a GML file (`ontology.gml`) containing a `nx.DiGraph` with all nodes and edges.
