# Govio 语义层模型

基于 `metric_schema.json` 定义的指标语义层结构。

## 整体架构

```mermaid
graph TB
    subgraph 语义层
        M[Metric 指标]
        D[Dimension 维度]
        ST[SourceTable 来源表]
        SC[SourceColumn 来源字段]
    end

    subgraph 数据层
        PT[PhysicalTable 物理表]
        Col[Col 字段]
    end

    M -->|USES_TABLE| PT
    M -->|REFERS_COLUMN| Col
    M -->|DIMENSION_USED| D
    M -->|DERIVED_FROM| M
    ST -->|映射| PT
    SC -->|映射| Col
```

## 指标分类与约束

```mermaid
graph LR
    M[Metric] --> T{type}
    T -->|atomic| A[原子指标]
    T -->|derived| DR[衍生指标]

    A -->|必须| ST[source_tables]
    A -->|可选| FORM1[formula]

    DR -->|必须| DF[derived_from]
    DR -->|必须| FORM2[formula]
    DR -->|禁止| ST2[source_tables]
```

## SourceTable 与 SourceColumn 关系

```mermaid
erDiagram
    Metric ||--o{ SourceTable : "source_tables"
    SourceTable ||--o{ SourceColumn : "columns"
    SourceColumn {
        string column_name
        string role "measure | filter | dimension_ref"
    }
    SourceTable {
        string full_table_name
    }
```

## 指标与维度引用

```mermaid
erDiagram
    Metric ||--o{ DimensionRef : "dimensions"
    DimensionRef {
        string code "引用 Dimension.code"
        string usage_type "filter | group | slice"
    }
```

## 完整语义层 ER 图

```mermaid
erDiagram
    Metric {
        string code PK
        string name
        string business_definition
        string type "atomic | derived"
        string formula
        string unit
        string data_type
        string owner
        string update_frequency
        string statistical_scope
        string time_scope
        string source_layer "DWD | DWS | DM"
        int version
        date effective_from
    }

    Dimension {
        string code PK
        string name
        string granularity
        string values_example
    }

    SourceTable {
        string full_table_name
    }

    SourceColumn {
        string column_name
        string role "measure | filter | dimension_ref"
    }

    DimensionRef {
        string code
        string usage_type "filter | group | slice"
    }

    Metric ||--o{ SourceTable : "source_tables"
    SourceTable ||--o{ SourceColumn : "columns"
    Metric ||--o{ DimensionRef : "dimensions"
    DimensionRef }o--|| Dimension : "引用"
    Metric ||--o{ Metric : "derived_from"
```

## 数据血缘示意

```mermaid
graph LR
    subgraph DWD[明细层 DWD]
        T1[物理表]
        C1[字段]
    end

    subgraph DWS[汇总层 DWS]
        AM[原子指标]
    end

    subgraph DM[集市层 DM]
        DM_M[衍生指标]
    end

    T1 -->|USES_TABLE| AM
    C1 -->|REFERS_COLUMN| AM
    AM -->|DERIVED_FROM| DM_M
```

## 约束规则

| 规则 | 说明 |
|------|------|
| atomic 指标必须声明 `source_tables` | 原子指标需要明确数据来源 |
| derived 指标必须声明 `derived_from` + `formula` | 衍生指标依赖其他指标并定义计算公式 |
| derived 指标不可声明 `source_tables` | 衍生指标的数据来源通过依赖指标间接获得 |
| `derived_from` 必须构成 DAG | 指标依赖不允许循环 |
| `source_layer` 枚举约束 | 仅允许 DWD / DWS / DM |
| `role` 枚举约束 | 字段角色仅允许 measure / filter / dimension_ref |
| `usage_type` 枚举约束 | 维度使用方式仅允许 filter / group / slice |
