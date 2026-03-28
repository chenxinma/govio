---
name: mcp-dataset-ops
description: Govio MCP 数据集操作合集。包含数据源查看、数据加载、DataFrame 管理和资源释放功能。当需要执行数据抽取、查看可用数据源、管理内存中的 DataFrame 或释放资源时使用。整合了 list-ds、list-dfs、load-df、release-df 四个操作。
---

# MCP Dataset Operations

数据集操作的统一入口，包含四个核心功能：

| 功能 | 工具 | 用途 |
|---|---|---|
| 列出数据源 | `list_ds()` | 查看可用数据库连接 |
| 列出 DataFrame | `list_dfs()` | 查看已加载的数据 |
| 加载数据 | `load_df()` | 从数据库抽取数据到内存 |
| 释放资源 | `release_df()` | 清理 DataFrame 占用的内存 |

## 典型工作流

```
list_ds() → 选择数据源 → load_df() → 数据处理 → release_df()
                                    ↓
                              list_dfs() 监控
```

---

## 功能 1: 列出数据源 (list_ds)

### 使用场景

- 开始数据治理任务前确认可用数据源
- 检查数据源配置是否正确加载
- 了解有哪些数据库可以连接

### 调用方式

无需参数：

```python
list_ds()
```

### 返回结果

```json
{
  "datasources": [
    {
      "name": "prod_db",
      "type": "postgresql",
      "host": "localhost",
      "port": 5432,
      "database": "production"
    },
    {
      "name": "legacy_db",
      "type": "mysql",
      "host": "192.168.1.100",
      "port": 3306,
      "database": "legacy"
    }
  ]
}
```

### 示例

**场景**: 用户说"查看可用的数据源"

**执行**:
```
[调用 mcp-dataset-ops.list_ds]
```

**返回**:
```
可用数据源:
1. prod_db (PostgreSQL) - production 数据库
2. legacy_db (MySQL) - legacy 数据库
3. analytics_db (PostgreSQL) - 分析数据库
```

---

## 功能 2: 列出 DataFrame (list_dfs)

### 使用场景

- 查看当前有哪些数据可用
- 监控内存使用情况
- 确认数据是否已成功加载
- 决定是否需要释放某些 DataFrame

### 调用方式

无需参数：

```python
list_dfs()
```

### 返回结果

```json
{
  "dataframes": [
    {
      "name": "customers",
      "rows": 1000,
      "columns": 5,
      "memory_usage": "45.2 KB"
    },
    {
      "name": "orders",
      "rows": 5000,
      "columns": 8,
      "memory_usage": "128.5 KB"
    }
  ]
}
```

### 示例

**场景**: 用户说"查看当前加载了哪些数据"

**执行**:
```
[调用 mcp-dataset-ops.list_dfs]
```

**返回**:
```
已加载的 DataFrame:
1. customers (1,000 行 × 5 列, 45.2 KB)
2. orders (5,000 行 × 8 列, 128.5 KB)

总计: 2 个 DataFrame, 占用 173.7 KB
```

---

## 功能 3: 加载数据 (load_df)

### 使用场景

- 从数据库抽取数据进行分析
- 加载要比对的源数据和目标数据
- 准备数据用于关系探测

### 参数

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| datasource | string | 是 | 数据源名称（从 list_ds 获取） |
| name | string | 是 | DataFrame 标识名（小写字母+下划线） |
| sql | string | 是 | SQL 查询语句 |

### 调用方式

```python
load_df(
    datasource="prod_db",
    name="customers",
    sql="SELECT customer_id, name, email FROM customers WHERE created_at > '2024-01-01'"
)
```

### 返回结果

成功:
```json
{
  "success": true,
  "name": "customers",
  "rows": 1000,
  "columns": 3,
  "column_info": [
    {"name": "customer_id", "dtype": "int64"},
    {"name": "name", "dtype": "object"},
    {"name": "email", "dtype": "object"}
  ]
}
```

失败:
```json
{
  "success": false,
  "error": "数据源 'prod_db' 不存在"
}
```

### 最佳实践

1. **命名规范**: DataFrame 名称使用小写+下划线，如 `customers_2024_q1`
2. **限制数据量**: 使用 WHERE 条件避免加载过多数据
3. **选择必要列**: 只 SELECT 需要的列，减少内存占用
4. **验证结果**: 检查返回的 rows > 0 确认数据加载成功

### 示例

**场景**: 用户说"从生产库加载客户数据"

**执行**:
```
[调用 mcp-dataset-ops.load_df]
参数:
- datasource: prod_db
- name: customers
- sql: SELECT customer_id, name, email, created_at FROM customers
```

**返回**:
```
数据加载成功:
- DataFrame: customers
- 行数: 10,245
- 列数: 4
- 列: customer_id, name, email, created_at
```

---

## 功能 4: 释放资源 (release_df)

### 使用场景

- 数据治理任务完成后清理资源
- 内存不足时释放不需要的数据
- 工作流结束时统一清理

### 参数

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| name | string | 是 | DataFrame 名称 |

### 调用方式

```python
release_df(name="customers")
```

### 返回结果

成功:
```json
{
  "success": true,
  "released": "customers",
  "memory_freed": "45.2 KB"
}
```

失败:
```json
{
  "success": false,
  "error": "DataFrame 'customers' 不存在"
}
```

### 示例

**场景**: 数据比对完成后释放资源

**执行**:
```
[调用 mcp-dataset-ops.release_df]
参数:
- name: legacy_customers

[调用 mcp-dataset-ops.release_df]
参数:
- name: new_customers
```

**返回**:
```
资源释放完成:
- legacy_customers: 已释放 (45.2 KB)
- new_customers: 已释放 (38.5 KB)

总计释放: 83.7 KB
```

### 批量释放

释放所有 DataFrame：

```python
dfs = list_dfs()
for df in dfs["dataframes"]:
    release_df(df["name"])
```

---

## 完整工作流示例

### 数据迁移验证流程

```python
# 1. 查看数据源
list_ds()
# → 获得可用数据源列表

# 2. 加载源数据
load_df(
    datasource="legacy_db",
    name="legacy_customers",
    sql="SELECT customer_id, name, email FROM customers"
)

# 3. 加载目标数据
load_df(
    datasource="new_db",
    name="new_customers",
    sql="SELECT customer_id, name, email FROM customers"
)

# 4. 查看加载状态
list_dfs()
# → 确认两个 DataFrame 已加载

# 5. 进行数据比对 (使用 mcp-compare-dfs)
# ...

# 6. 释放资源
release_df("legacy_customers")
release_df("new_customers")

# 7. 确认清理完成
list_dfs()
# → 应该返回空列表
```

---

## 调用格式

调用本 Skill 中的功能时，使用以下格式：

```
[调用 mcp-dataset-ops.list_ds]

[调用 mcp-dataset-ops.list_dfs]

[调用 mcp-dataset-ops.load_df]
参数:
- datasource: prod_db
- name: customers
- sql: SELECT * FROM customers

[调用 mcp-dataset-ops.release_df]
参数:
- name: customers
```

---

## 最佳实践

1. **及时释放**: 完成分析后立即释放不需要的 DataFrame
2. **工作流结尾释放**: 每个 Plan 的最后一步应该是资源清理
3. **小批量加载**: 避免一次性加载大量数据
4. **命名规范**: 使用有意义的 DataFrame 名称，便于管理和追踪
5. **监控内存**: 定期调用 list_dfs 监控内存使用

## 与其他 Skill 的协作

| 后续操作 | 关联 Skill |
|---|---|
| 数据比对 | `mcp-compare-dfs` |
| 关系探测 | `mcp-explore-relations` |
| 主控协调 | `govio-mcp` |
