# Govio 图数据库本体模型

```mermaid
erDiagram
    PhysicalTable {
        string full_table_name
        string schema
        string table_name
        string name
        string data_entity_type
        string database_name
    }

    Col {
        string column
        string column_name
        string name
        string full_table_name
        string data_entity_type
        string dtype
        int size
        int precision
        int scale
        int order_no
        string data_type
    }

    Application {
        string app_id
        string name
        string app_name_en
        string app_type
        string business_domain
        string manager
        string network_area
        string maintenance_level
        string external_vendor
    }

    Standard {
        string standard_id
        string name
        string adaptability
        string alias
        string basis
        string core_system
        string data_category
        string data_expression
        int data_length
        string data_type
        string definition
        string name_en
        string source
        string standard_status
        string ref_code_define
        string business_rule
    }

    Metric {
        string code
        string name
        string business_definition
        string type
        string unit
        string data_type
        string owner
        string update_frequency
        string time_scope
        string source_layer
        string version
        string statistical_scope
        string formula
    }

    Dimension {
        string code
        string name
        string granularity
        string values_example
    }

    PhysicalTable ||--o{ Col : "HAS_COLUMN"
    PhysicalTable ||--o{ PhysicalTable : "RELATES_TO"
    Application ||--o{ PhysicalTable : "USE"
    Col ||--o{ Standard : "COMPLIES_WITH"
    Metric ||--o{ PhysicalTable : "USES_TABLE"
    Metric ||--o{ Col : "REFERS_COLUMN"
    Metric ||--o{ Dimension : "DIMENSION_USED"
    Metric ||--o{ Metric : "DERIVED_FROM"
    Metric ||--o{ Metric : "SUPERSEDES"
```

## 关系说明

| 关系 | 源节点 | 目标节点 | 属性 |
|------|--------|----------|------|
| HAS_COLUMN | PhysicalTable | Col | - |
| RELATES_TO | PhysicalTable | PhysicalTable | relationship_type, description, source_columns, target_columns |
| USE | Application | PhysicalTable | - |
| COMPLIES_WITH | Col | Standard | - |
| USES_TABLE | Metric | PhysicalTable | - |
| REFERS_COLUMN | Metric | Col | role |
| DIMENSION_USED | Metric | Dimension | usage_type |
| DERIVED_FROM | Metric | Metric | - |
| SUPERSEDES | Metric | Metric | - |
