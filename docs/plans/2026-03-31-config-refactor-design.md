# 数据源配置重构设计

## 背景

当前 `DataSourceConfig` 使用分散的字段（driver, host, port, database, username, password）描述数据库连接，需要改为支持更灵活的 URL 格式，并增加目录数据源支持（使用 DuckDB 读取 parquet、csv 等文件）。

## 目标

1. 支持新的配置格式：`url` + `connect_args`
2. 支持目录数据源（DuckDB）
3. 使用 duckdb:// 协议区分目录数据源

## 设计

### DataSourceConfig 重构

简化配置类，只保留两个字段：

```python
@dataclass
class DataSourceConfig:
    """数据源配置"""
    url: str
    connect_args: dict[str, Any] = field(default_factory=dict)
```

### 配置示例

```json
{
  "trino_db": {
    "url": "trino://user:pass@host:port/database",
    "connect_args": {
      "http_scheme": "https",
      "timezone": "Asia/Shanghai"
    }
  },
  "local_data": {
    "url": "duckdb:///data/warehouse"
  }
}
```

### DatabaseManager 修改

区分两种引擎类型：

1. **SQLAlchemy 引擎**：用于传统数据库（Trino, MySQL, PostgreSQL 等）
2. **DuckDB 连接**：用于目录数据源

DuckDB 数据源特点：
- 使用内存模式（`:memory:`）
- 延迟加载：不在初始化时加载数据
- 用户通过 DuckDB 函数查询文件：
  - `read_parquet('/path/to/file.parquet')`
  - `read_csv('/path/to/file.csv')`
  - `read_json('/path/to/file.json')`

### 错误处理

- 初始化时验证目录路径存在性
- 捕获并包装连接异常

## 变更文件

| 文件 | 变更 |
|------|------|
| `src/govio/mcp/config.py` | 简化 DataSourceConfig |
| `src/govio/mcp/core/database.py` | 添加 DuckDB 支持 |

## 新增依赖

- `duckdb` 包
