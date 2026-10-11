# SQLite 库表结构（元数据基准库）

## 表结构

### datasource —— 数据源（业务应用系统）

| 列 | 类型 | 说明 |
|---|---|---|
| `id` | TEXT | 主键 |
| `datasource_name` | TEXT | 数据源名称（如 MDM、IHRO、SSOP） |
| `comment` | TEXT | 业务说明（中文，如"外包项目管理系统"） |
| `source_type` | TEXT | 来源类型（如 trino） |
| `filter` | TEXT | 采集过滤配置（JSON，如 `{"schemas": [...], "include_tables": [...]}`） |

### physical_table —— 物理表

| 列 | 类型 | 说明 |
|---|---|---|
| `id` | TEXT | 主键 |
| `full_table_name` | TEXT | 表全名，形如 `<datasource_name>.<schema>.<table_name>` |
| `schema` | TEXT | 库名（如 ihrodb、SSOP_USER） |
| `table_name` | TEXT | 表名（如 CUST_CUSTOMER、t_invoice） |
| `name` | TEXT | 表中文名（如"客户信息表"） |
| `data_entity_type` | TEXT | 实体类型（如 ORACLE_TABLE） |
| `database_name` | TEXT | 数据库中文名 |
| `datasource_id` | TEXT | 所属数据源 → `datasource.id`（N:1） |

### col —— 字段

| 列 | 类型 | 说明 |
|---|---|---|
| `id` | TEXT | 主键 |
| `column` | TEXT | 字段全限定名 `<datasource>.<schema>.<table>.<column>` |
| `column_name` | TEXT | 字段名（如 BANK_ACCOUNT、bank_account，大小写不敏感） |
| `name` | TEXT | 字段中文名（如"银行账号"） |
| `full_table_name` | TEXT | 所属表全名（与 `physical_table.full_table_name` 同值） |
| `data_entity_type` | TEXT | 实体类型（如 ORACLE_COLUMN） |
| `dtype` | TEXT | 数据类型（如 VARCHAR2、NUMBER、varchar） |
| `size` / `precision` / `scale` | INTEGER | 长度 / 精度 / 小数位 |
| `order_no` | INTEGER | 字段顺序号 |
| `data_type` | TEXT | 类型全称（如 `varchar(200)`、`decimal(18, 2)`） |
| `table_id` | TEXT | 所属表 → `physical_table.id`（N:1） |

## 关系

```
datasource (1) ──< physical_table (1) ──< col
   ↑ datasource_id            ↑ table_id
```

## 注意

- `column` 列是全限定名，按字段名过滤请用 `column_name`；
- `full_table_name` 的第一段即 `datasource_name`，可用来按系统过滤；
- SQLite `LIKE` 的 `%`/`_` 是通配符，匹配字面下划线要写 `ESCAPE`；
- 题目若涉及本结构不存在的信息（如表行数），正确行为是回答无法获得，不要编造。
