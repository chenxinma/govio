---
name: govio
description: 数据治理知识图谱,当需要做“元数据查询、表字段比较、SQL 生成”等数据治理相关工作时运行。
allowed-tools: Read, Grep, Glob
---

# Data Governance Knowledge Graph

作为一名数据治理专家，根据知识图谱中的信息提供数据治理的支持。

## Purpose

元数据查询：用户可以询问关于数据资产的元数据信息，如数据资产的名称、描述、来源、状态等。
表字段比较：用户可以比较不同表之间的字段差异，如字段名称、数据类型、是否必填等。
SQL 生成：用户可以根据需求描述，自动生成符合要求的 SQL 语句。

## Best Practices

For advanced usage, see [reference.md](reference.md).

1. 使用 Read, Grep, Glob 查询`assets/names/`获得被记载的标准名称
2. 严格按照 `assets/schema.md` 的图数据库结构生成`cypher`脚本

## Resource Resolution

When this skill is loaded, the base directory is provided:

```
Base directory: /path/to/data-gov-knowledge-graph
```

Relative paths resolve from base directory:
- `reference.md` → `/path/to/govio/reference.md`
- `scripts/load_schema.py` → `/path/to/govio/scripts/load_schema.py`
- `scripts/load_names.py` → `/path/to/govio/scripts/load_names.py`
- `scripts/query.py` → `/path/to/govio/scripts/query.py`
- `assets/schema.md` → `/path/to/govio/assets/schema.md`
- `assets/names/*.md`  → `/path/to/govio/assets/names/*.md`

## Dependencies

Required dependencies (install if not available):

- **environment**: (for init assets)
    1. load schema `uv run scripts/load_schema.py` 
    2. load relevant names `uv run scripts/load_names.py`
- **execute**: `uv run python -c ...`
- **query cypher**: `uv run scripts/query.py --cypher "MATCH ..."`
