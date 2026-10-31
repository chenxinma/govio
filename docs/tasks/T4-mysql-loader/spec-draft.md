# spec 草稿：docs/specs/metadata.md 新增 MysqlLoader 小节

> 按任务要求未直接修改 `docs/specs/`，本草稿供后续文档同步任务并入 `docs/specs/metadata.md`
> （建议放在 TrinoLoader 小节之后）。内容与 `src/govio/metadata/mysql_loader.py` 实现一致。

## MysqlLoader

`mysql_loader.py`

Loads metadata from a MySQL instance via pymysql, reading `information_schema.tables` / `information_schema.columns`. 与 `DuckDBLoader` 相同，构造参数末尾可传 `datasource_name: str = ""` 生成全限定标识与 `Datasource` / `OWNS` 归属。仅抽取 `TABLE_TYPE = 'BASE TABLE'`（不含视图）。

### Constructor

```python
MysqlLoader(
    host: str,
    schemas: list[str],
    port: int = 3306,
    user: str = "root",
    password: str = "",
    database: str = "",
    datasource_name: str = "",
)
```

- `schemas`：要导出的 schema（MySQL 中 schema 即库，对应 `TABLE_SCHEMA`），参数化过滤；为空时返回空 DataFrame。
- `database`：连接默认库（传给 `pymysql.connect` 的 `database`），同时写入输出 `database_name` 列（空串时不选库，`database_name` 输出空串）。

### Properties

| Property | Description |
|---|---|
| `PhysicalTable` | Same schema as TDSLoader, `data_entity_type = "MYSQL_TABLE"` |
| `Col` | Same schema as TDSLoader, `data_entity_type = "MYSQL_COLUMN"` |

### Type mapping

`dtype` / `data_type` 取 `information_schema.columns.COLUMN_TYPE` 原始字符串（如 `decimal(10,2)`）；`size` / `precision` / `scale` 从该字符串解析：

| COLUMN_TYPE | size | precision | scale |
|---|---|---|---|
| `decimal(10,2)` | 0 | 10 | 2 |
| `varchar(64)` | 64 | 0 | 0 |
| `bigint` 等无参类型 | 0 | 0 | 0 |

`order_no` 取 `ORDINAL_POSITION`（每表从 1 递增）。表/列注释为空时 `name` 回退为 `table_name` / `column_name`。
