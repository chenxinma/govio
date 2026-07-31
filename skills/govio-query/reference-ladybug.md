# LadybugGraph 使用说明

## 简介

LadybugGraph 用于连接和查询 Ladybug 嵌入式图数据库（本地 `.lbdb` 文件），提供以下核心功能：
- 打开本地 Ladybug 数据库文件
- 执行 Cypher 查询（与 FalkorDB 语法基本一致）
- 自动构建图模式信息
- 返回查询结果

与 FalkorDB 相比，Ladybug 无需独立服务进程，数据持久化在单个 `.lbdb` 文件中，适合本地/单机场景。

## 初始化连接

```python
from govio import LadybugGraph

g = LadybugGraph("/path/to/ontology.lbdb")
```

配置文件中的对应字段：

```yaml
graph:
  backend: ladybug
  ladybug:
    db_path: /home/user/.govio/ontology.lbdb
    # 可选，默认值如下（字节）
    buffer_pool_size: 268435456   # 256 MiB
    max_db_size: 1073741824       # 1 GiB
```

> `max_db_size` 必须显式设置：Ladybug 默认按 8TB 预留 mmap 地址空间，在多数环境会失败，故内置默认 1GiB。元数据图谱足够；数据量极大时可在配置中调大。

## 核心属性

### `schema` 属性
获取图的模式信息：
```python
print(g.schema)
```

### `query()` 方法
执行 Cypher 查询，返回 `list[list]`（每行为一个列表）：
```python
data = g.query("MATCH (n) RETURN n LIMIT 10")
```

带参数查询（`$param` 语法，与 FalkorDB 一致）：
```python
data = g.query(
    "MATCH (app:Application {app_name_en: $code})-[:USE]->(t:PhysicalTable) RETURN t.full_table_name",
    {"code": "AEP"},
)
```

## 图结构

与 FalkorDB 后端共享同一套节点/关系模型（见 `assets/schema.md`）：

**节点类型**：`PhysicalTable`、`Col`、`Application`、`Standard`、`Metric`、`Dimension`

**关系类型**：`HAS_COLUMN`、`USE`、`COMPLIES_WITH`、`RELATES_TO`、`USES_TABLE`、`REFERS_COLUMN`、`DERIVED_FROM`、`DIMENSION_USED`、`SUPERSEDES`

> Ladybug 严格类型，所有属性以 STRING 存储（与 falkordb_loader 的 `dtype=str` 策略一致）。

## 与 FalkorDB 的差异

| 项 | FalkorDB | Ladybug |
|----|----------|---------|
| 部署 | 独立 Redis 服务 | 本地 `.lbdb` 文件，无需服务 |
| 标识符引用 | 反引号 `` ` `` | 反引号 `` ` ``（双引号 `"` 不可用） |
| 类型系统 | 动态 | 严格类型，COPY 不可忽略类型转换错误 |
| 列表下标 | 0-based | 1-based（建议用 struct 字段访问 `row.name` 规避） |
| `type(r)` | 可用 | **不可用**，关系类型用 `label(r)` |
| 查询入口 | `MATCH` | 必须以 `MATCH` 开头 |

## 执行案例

### 元数据查询

#### 查询所有应用
```bash
uv run python -c "
from govio import LadybugGraph
g = LadybugGraph('/home/user/.govio/ontology.lbdb')
cypher = '''
MATCH (app:Application)
RETURN app.name, app.app_name_en, app.business_domain
LIMIT 300
'''
print(g.query(cypher))
"
```

#### 查询某应用使用的所有物理表
```bash
uv run python -c "
from govio import LadybugGraph
g = LadybugGraph('/home/user/.govio/ontology.lbdb')
cypher = '''
MATCH (app:Application {name: 'AEP'})-[:USE]->(t:PhysicalTable)
RETURN t.name AS table_name, t.full_table_name
LIMIT 300
'''
print(g.query(cypher))
"
```

#### 查询某表的所有字段
```bash
uv run python -c "
from govio import LadybugGraph
g = LadybugGraph('/home/user/.govio/ontology.lbdb')
cypher = '''
MATCH (t:PhysicalTable {name: 'T_INVOICE'})-[:HAS_COLUMN]->(c:Col)
RETURN c.column_name, c.data_type
ORDER BY c.order_no
LIMIT 300
'''
print(g.query(cypher))
"
```

### 聚合查询

#### 统计每个应用使用的表数量
```bash
uv run python -c "
from govio import LadybugGraph
g = LadybugGraph('/home/user/.govio/ontology.lbdb')
cypher = '''
MATCH (app:Application)-[:USE]->(t:PhysicalTable)
RETURN app.name AS app_name, count(t) AS table_count
ORDER BY table_count DESC
LIMIT 300
'''
print(g.query(cypher))
"
```

#### 统计每种节点类型的数量
```bash
uv run python -c "
from govio import LadybugGraph
g = LadybugGraph('/home/user/.govio/ontology.lbdb')
cypher = '''
MATCH (n)
RETURN label(n) AS node_type, count(n) AS count
LIMIT 300
'''
print(g.query(cypher))
"
```

## 通过 CLI 查询

```bash
# 查看当前后端
govio-cli backend

# 执行 Cypher 查询（必须以 MATCH 开头）
govio-cli query -c 'MATCH (app:Application) RETURN app.name, app.app_name_en LIMIT 300'
```
