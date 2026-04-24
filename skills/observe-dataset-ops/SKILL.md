---
name: observe-dataset-ops
description: Govio 数据集操作合集。包含数据源查看、数据加载、DataFrame 管理和资源释放功能。当需要执行数据抽取、查看可用数据源、管理 DataFrame 或释放资源时使用。整合了 show-datasource、list、load、release 四个 CLI 子命令。
---

# Observe Dataset Operations

数据集操作的统一入口，包含四个核心功能，对应 `uvx --from skills/govio/assets/govio-*.whl govio-cli observe` 的子命令：

| 功能 | CLI 命令 | 用途 |
|---|---|---|
| 列出数据源 | `uvx --from skills/govio/assets/govio-*.whl govio-cli observe show-datasource` | 查看可用数据库连接 |
| 列出 DataFrame | `uvx --from skills/govio/assets/govio-*.whl govio-cli observe list` | 查看已加载的数据 |
| 加载数据 | `uvx --from skills/govio/assets/govio-*.whl govio-cli observe load <name> <datasource> <sql>` | 从数据库抽取数据并持久化 |
| 释放资源 | `uvx --from skills/govio/assets/govio-*.whl govio-cli observe release <name>` | 删除 DataFrame 文件 |

## 前置条件

1. 已运行 `uvx --from skills/govio/assets/govio-*.whl govio-cli onboard` 完成初始化
2. `~/.govio/config.yaml` 中配置了数据源
3. DataFrame 持久化在 `.govio/observe/dataframes/` 目录下

## 典型工作流

```
show-datasource → 选择数据源 → load → 数据处理 → release
                                      ↓
                                list 监控
```

---

## 功能 1: 列出数据源 (show-datasource)

### 使用场景

- 开始数据治理任务前确认可用数据源
- 检查数据源配置是否正确加载
- 了解有哪些数据库可以连接

### 调用方式

```bash
uvx --from skills/govio/assets/govio-*.whl govio-cli observe show-datasource
```

### 返回结果

```json
{
  "success": true,
  "datasources": [
    {
      "name": "prod_db",
      "driver": "mysql",
      "url": "mysql://user:pass@localhost:3306/production"
    },
    {
      "name": "legacy_db",
      "driver": "postgresql",
      "url": "postgresql://user:pass@192.168.1.100:5432/legacy"
    }
  ]
}
```

### 示例

**场景**: 用户说"查看可用的数据源"

**执行**:
```bash
uvx --from skills/govio/assets/govio-*.whl govio-cli observe show-datasource
```

**返回**:
```
可用数据源:
1. prod_db (mysql) - mysql://user:pass@localhost:3306/production
2. legacy_db (postgresql) - postgresql://user:pass@192.168.1.100:5432/legacy
```

---

## 功能 2: 列出 DataFrame (list)

### 使用场景

- 查看当前有哪些数据可用
- 确认数据是否已成功加载
- 检查 DataFrame 的行数、列数和字段类型
- 决定是否需要释放某些 DataFrame

### 调用方式

```bash
uvx --from skills/govio/assets/govio-*.whl govio-cli observe list
```

### 返回结果

```json
{
  "dataframes": [
    {
      "name": "customers",
      "rows": 1000,
      "columns": 5,
      "column_info": [
        {"name": "customer_id", "dtype": "int64"},
        {"name": "name", "dtype": "object"},
        {"name": "email", "dtype": "object"}
      ]
    }
  ]
}
```

### 示例

**场景**: 用户说"查看当前加载了哪些数据"

**执行**:
```bash
uvx --from skills/govio/assets/govio-*.whl govio-cli observe list
```

**返回**:
```
已加载的 DataFrame:
1. customers (1,000 行 × 5 列)
   列: customer_id(int64), name(object), email(object), phone(object), created_at(datetime64)
2. orders (5,000 行 × 8 列)
   列: order_id(int64), customer_id(int64), product_id(int64), quantity(int64), ...
```

---

## 功能 3: 加载数据 (load)

### 使用场景

- 从数据库抽取数据进行分析
- 加载要比对的源数据和目标数据
- 准备数据用于关系探测

### 参数

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| name | positional | 是 | DataFrame 名称（小写字母+下划线） |
| datasource | positional | 是 | 数据源名称（从 show-datasource 获取） |
| sql | positional | 是 | 查询 SQL 语句 |

### 调用方式

```bash
uvx --from skills/govio/assets/govio-*.whl govio-cli observe load customers prod_db "SELECT customer_id, name, email FROM customers WHERE created_at > '2024-01-01'"
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
```bash
uvx --from skills/govio/assets/govio-*.whl govio-cli observe load customers prod_db "SELECT customer_id, name, email, created_at FROM customers"
```

**返回**:
```
数据加载成功:
- DataFrame: customers
- 行数: 10,245
- 列数: 4
- 列: customer_id(int64), name(object), email(object), created_at(datetime64)
```

---

## 功能 4: 释放资源 (release)

### 使用场景

- 数据治理任务完成后清理资源
- 不需要的数据及时删除
- 工作流结束时统一清理

### 参数

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| name | positional | 是 | DataFrame 名称 |

### 调用方式

```bash
uvx --from skills/govio/assets/govio-*.whl govio-cli observe release customers
```

### 返回结果

成功:
```json
{
  "success": true,
  "message": "DataFrame 'customers' 已释放"
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
```bash
uvx --from skills/govio/assets/govio-*.whl govio-cli observe release legacy_customers
uvx --from skills/govio/assets/govio-*.whl govio-cli observe release new_customers
```

**返回**:
```
资源释放完成:
- legacy_customers: 已释放
- new_customers: 已释放
```

### 批量释放

查看当前加载的 DataFrame 后逐个释放：

```bash
# 先查看有哪些
uvx --from skills/govio/assets/govio-*.whl govio-cli observe list

# 然后逐个释放
uvx --from skills/govio/assets/govio-*.whl govio-cli observe release legacy_customers
uvx --from skills/govio/assets/govio-*.whl govio-cli observe release new_customers

# 确认清理完成
uvx --from skills/govio/assets/govio-*.whl govio-cli observe list
# → 应该返回空列表
```

---

## 完整工作流示例

### 数据迁移验证流程

```bash
# 1. 查看数据源
uvx --from skills/govio/assets/govio-*.whl govio-cli observe show-datasource
# → 获得可用数据源列表

# 2. 加载源数据
uvx --from skills/govio/assets/govio-*.whl govio-cli observe load legacy_customers legacy_db "SELECT customer_id, name, email FROM customers"

# 3. 加载目标数据
uvx --from skills/govio/assets/govio-*.whl govio-cli observe load new_customers new_db "SELECT customer_id, name, email FROM customers"

# 4. 查看加载状态
uvx --from skills/govio/assets/govio-*.whl govio-cli observe list
# → 确认两个 DataFrame 已加载

# 5. 进行数据比对 (使用 observe-compare-dfs)
# ...

# 6. 释放资源
uvx --from skills/govio/assets/govio-*.whl govio-cli observe release legacy_customers
uvx --from skills/govio/assets/govio-*.whl govio-cli observe release new_customers

# 7. 确认清理完成
uvx --from skills/govio/assets/govio-*.whl govio-cli observe list
# → 应该返回空列表
```

---

## 最佳实践

1. **及时释放**: 完成分析后立即释放不需要的 DataFrame
2. **工作流结尾释放**: 每个 Plan 的最后一步应该是资源清理
3. **小批量加载**: 避免一次性加载大量数据
4. **命名规范**: 使用有意义的 DataFrame 名称，便于管理和追踪
5. **监控状态**: 定期调用 list 监控已加载的 DataFrame

## 与其他 Skill 的协作

| 后续操作 | 关联 Skill |
|---|---|
| 数据比对 | `observe-compare-dfs` |
| 关系探测 | `observe-explore-relations` |
| 主控协调 | `govio-observe` |
