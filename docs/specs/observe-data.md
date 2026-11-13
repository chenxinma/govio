# govio.observe_data -- Data Observation Module

Data exploration, comparison, charting, and relationship inference for loaded DataFrames.

## 分层结构（core 与 tools）

`observe_data` 分两层：`core/` 是业务内核，`tools/` 是面向 CLI / Agent 的适配薄层；包根的 `config.py` 提供两层共用的数据源配置类型。

### core —— 业务内核

| 模块 | 公开符号 | 职责 |
|---|---|---|
| `observe_store.py` | `ObserveStore`、`DataFrameInfo`、`Manifest` | parquet 落盘的 DataFrame 存储（`store` / `get` / `list` / `release` / `exists`）与 manifest 读写（`load` / `save` / `add` / `remove`） |
| `database.py` | `DatabaseManager`、`_resolve_sqlalchemy_target` | 多数据源连接与只读 SQL 执行（`datasources` 只读属性 / `get_engine` / `execute_sql`）、URL scheme 归一 |
| `comparator.py` | `TableComparator` | 结构比对与数据比对（`compare_schema` / `compare_data` / `compare`），数据比对用 datacompy `Compare` |
| `explorer.py` | `RelationExplorer` | 外键推断与列名相似度（`infer_foreign_keys` / `find_column_similarity` / `explore`） |
| `visualizer.py` | `RelationVisualizer` | 关系图数据生成（`to_networkx` / `to_json` / `visualize`） |
| `chart.py` | `render_chart`（内部 `_setup_font`） | matplotlib 渲染 bar/line PNG，含中文字体回退 |

内核不感知命令行：不解析 argv、不打印、不 `sys.exit`。错误以异常传播（如 `DatabaseManager` 数据源不存在时的 `ValueError`），或交由调用方按返回值判断（`ObserveStore.get` 未命中返回 `None`、`release` 返回 `bool`）；只有 `render_chart` 自己返回 `{'success', 'output' | 'error'}` 结构。

### config.py —— 两层共用的配置类型

`DataSourceConfig`（`url` / `connect_args` / `encrypted_password`，`resolved_url` 还原真实密码）与 `Config` 是内核的输入类型。`load_config(path)` 读 JSON 配置文件，目前只被测试用来构造 `Config`；CLI 实际走 `cli/config.py` 的 `ConfigManager`，在 `cli/observe.py` 的 `get_db_manager()` 里把 `datasources` 段逐条转成 `DataSourceConfig`。

### tools —— CLI / Agent 适配薄层

| 模块 | 函数 | 依赖的内核符号 |
|---|---|---|
| `list_dataframes.py` | `list_dataframes(store)` | `ObserveStore.list` |
| `list_datasources.py` | `list_datasources(db_manager)` | `DatabaseManager`（现状：直接读其 `_datasources` 属性） |
| `load_dataframe.py` | `load_dataframe(store, db_manager, datasource, name, sql)`、`load_from_memory(store, name, sql)` | `DatabaseManager.execute_sql` + `ObserveStore.store`；`load_from_memory` 自建内存 DuckDB 并注册已存储的表 |
| `release_dataframe.py` | `release_dataframe(store, name)`、`release_all_dataframes(store)` | `ObserveStore.get` / `release` / `list` |
| `visualize_relations.py` | `visualize_relations(relations)` | `RelationVisualizer.visualize`（先归一 `column_similarity` 记录的字段名） |

薄层三条约定：

- 入参是调用方已构造好的内核对象（`ObserveStore` / `DatabaseManager`）；薄层不读配置、不建连接
- 返回值统一为 `dict`，失败用 `{'success': False, 'error': ...}` 表达，不抛异常、不决定退出码
- 只做入参整形与结果整形，不新增业务规则（阈值、比对算法、落盘格式都在 `core/`）

### 调用方向

```
cli/observe.py
├── 构造内核对象：ObserveStore、DatabaseManager（get_db_manager + DataSourceConfig）
├── 薄层路径：info / load / release / visualize-relations  →  observe_data.tools.*  →  observe_data.core.*
└── 直连内核：compare → TableComparator，explore → RelationExplorer，chart → render_chart
```

- `tools/` → `core/` 单向依赖；`core/` 不 import `tools/`，两层都不 import `govio.cli`
- 退出码、stdout/stderr 的 JSON 形状、`❌` 提示等 CLI 契约只由 `cli/observe.py` 决定（见下文「失败语义与退出码」与 `docs/specs/cli.md`）
- 新增 Agent 可调用动作优先加薄层函数以复用结构化返回；纯算法与存储能力放 `core/`

---

## Config

`observe_data/config.py`

```python
@dataclass
class DataSourceConfig:
    url: str
    connect_args: dict[str, Any] = field(default_factory=dict)
    encrypted_password: str | None = None  # Encrypted password from config
```

---

## ObserveStore

`core/observe_store.py`

File-backed DataFrame storage using parquet files.

`pandas` is imported on demand (only when reading/writing parquet); manifest listing (`list()`, `exists()`, `release()`) runs without pandas loaded.

### Constants

```python
OBSERVE_DIR = Path(".govio/observe")
DATAFRAMES_DIR = OBSERVE_DIR / "dataframes"
MANIFEST_FILE = OBSERVE_DIR / "manifest.json"
```

### Data Structures

```python
@dataclass
class DataFrameInfo:
    name: str
    datasource: str
    sql: str
    file: str
    loaded_at: str
    rows: int
    columns: int
    column_info: list[dict]

@dataclass
class Manifest:
    version: str = "1.0"
    dataframes: dict[str, dict[str, Any]]
```

### Methods

```python
store(name, df, datasource, sql) -> DataFrameInfo   # Save parquet + manifest
get(name) -> pd.DataFrame | None                     # Load from parquet
list() -> list[DataFrameInfo]                         # List all stored
release(name) -> bool                                 # Delete parquet + manifest entry
exists(name) -> bool
```

---

## DatabaseManager

`core/database.py`

Multi-datasource database connection manager.

### Constructor

```python
DatabaseManager(datasources: dict[str, DataSourceConfig])
```

- DuckDB URLs: `duckdb://path` -- supports file and directory modes
- Other URLs: SQLAlchemy engines（scheme 先经 `_resolve_sqlalchemy_target` 归一）
- Connections are lazy: the constructor only records configs, each datasource is connected on first `execute_sql` / `get_engine` call (`_ensure(name)`). Listing datasources never opens a connection.
- `duckdb` / `sqlalchemy` / `pandas` are imported on demand, not at module import.

### URL Scheme Normalization

```python
_resolve_sqlalchemy_target(url: str) -> tuple[str, dict[str, Any]]
# 返回 (归一后的 SQLAlchemy URL, 追加的 connect_args)；其余 URL 原样返回
```

| Config URL scheme | 实际连接 | 说明 |
|---|---|---|
| `duckdb://` | duckdb 原生连接 | 文件 / 目录两种模式，不走 SQLAlchemy |
| `mysql://` | `mysql+pymysql://` | 归一到 pymysql（默认 mysqldb 驱动未安装） |
| `mysql+pymysql://` | 同左 | 原样 |
| `postgresql://` / `postgres://` | `postgresql+psycopg://` | 归一到 psycopg3 |
| `postgresql+psycopg://` | 同左 | 原样 |
| `trino://` | trino.sqlalchemy 方言（trino-python-client 自带） | 原样；URL 含密码时方言自动构造 BasicAuthentication |
| `trino+http://` | `trino://` + `connect_args.http_scheme="http"` | HTTP；`config.connect_args` 同名键优先 |
| `trino+https://` | `trino://` + `connect_args.http_scheme="https"` | HTTPS；`config.connect_args` 同名键优先 |

### Methods

```python
get_engine(datasource: str) -> Engine      # Raises ValueError if not SQLAlchemy
execute_sql(datasource: str, sql: str) -> pd.DataFrame  # DuckDB 或 pandas.read_sql（统一执行路径）
```

mysql / postgresql / trino 的查询执行统一走 SQLAlchemy 引擎 + `pandas.read_sql`，`load_dataframe` 无需按后端分支；trino 的 http_scheme / 证书等高级连接参数可经 `config.connect_args` 透传（合并时 config 显式配置优先）。

---

## TableComparator

`core/comparator.py`

DataFrame comparison using datacompy.

```python
compare_schema(source, target) -> dict
# Returns: match (bool), source_columns, target_columns, common_columns, source_only, target_only

compare_data(source, target, join_columns) -> dict
# Uses datacompy Compare. Returns: report (str)

compare(source, target, join_columns) -> dict
# Combines schema + data comparison
```

---

## RelationExplorer

`core/explorer.py`

Relationship inference between DataFrames.

```python
find_column_similarity(df1, df2) -> list[dict]
# SequenceMatcher threshold > 0.7
# Returns: [{'column', 'match_column', 'similarity'}]

infer_foreign_keys(source_df, source_name, target_df, target_name) -> list[dict]
# Matches _id/id suffix columns, checks value overlap > 0.5
# Returns: [{'source_table', 'source_column', 'target_table', 'target_column', 'confidence'}]

explore(dataframes: dict[str, pd.DataFrame]) -> dict
# Runs FK inference + column similarity on all pairs
# Returns: {'foreign_keys': [...], 'column_similarities': [...]}
```

---

## RelationVisualizer

`core/visualizer.py`

Converts relation data to graph/JSON formats.

```python
to_networkx(relations) -> nx.DiGraph     # Directed graph with edge attributes
to_json(relations) -> dict               # {'nodes': [...], 'edges': [...]}
visualize(relations) -> dict             # Alias for to_json()
```

---

## Chart Renderer

`core/chart.py`

Generates PNG charts from DataFrames.

```python
render_chart(df, chart_type, x_col, y_col, output_path) -> dict
```

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `df` | `pd.DataFrame` | Source data |
| `chart_type` | `str` | `"bar"` or `"line"` |
| `x_col` | `str` | X axis column (category/time) |
| `y_col` | `str` | Y axis column (numeric, single series) |
| `output_path` | `str` | Output PNG file path |

### Returns

```json
{"success": true, "output": "/abs/path/to/chart.png"}
```

or `{"success": false, "error": "..."}` on failure.

### Features

- Bar chart and line chart support (single series)
- Chinese font fallback chain: Noto Sans CJK SC / WenQuanYi Zen Hei / SimHei / Microsoft YaHei / Arial Unicode MS
- Negative number display fix

---

## Tool Functions

`observe_data/tools/`

| Function | Input | Returns |
|---|---|---|
| `list_dataframes(store)` | `ObserveStore` | `{'dataframes': [{name, rows, columns, column_info}]}` |
| `list_datasources(db_manager)` | `DatabaseManager` | `[{name, driver, url}]` |
| `load_dataframe(store, db_manager, datasource, name, sql)` | mixed | `{'success': True/False, ...}`；失败时 `{'success': False, 'error': ...}`（ValueError 原文透传，其余包 `SQL 执行错误: ...`） |
| `load_from_memory(store, name, sql)` | `ObserveStore` | Loads all registered DataFrames into in-memory DuckDB, executes SQL, stores result. Returns `{'success': True/False, ..., source_tables: [...]}`；失败时 `{'success': False, 'error': ...}` |
| `release_dataframe(store, name)` | `ObserveStore` | `{'success': True/False}` |
| `release_all_dataframes(store)` | `ObserveStore` | `{'success': True, 'released': [], 'count': int}` |
| `visualize_relations(relations)` | list or dict | `{'success': True, 'nodes': [], 'edges': []}` |

---

## CLI Subcommands

| Command | Arguments | Description |
|---|---|---|
| `info` | `--datasource`, `--df`, `--name NAME [--rows N]` | Query datasource/DataFrame info (unified command) |
| `load` | `--name`, `(--datasource\|--memory)`, `--sql`, `-o OUTPUT` | Load DataFrame from DB or memory DuckDB |
| `release` | `--name NAME` or `--all` | Release DataFrames |
| `compare` | `--source`, `--target`, `--join-columns COLS` | Compare two DataFrames |
| `explore` | `--dataframes` (optional) | Explore relationships |
| `visualize-relations` | `--relations-file FILE` | Generate visualization |
| `chart` | `--name`, `--type bar\|line`, `--x`, `--y`, `-o OUTPUT` | Generate PNG chart |

### 失败语义与退出码（全部 observe 子命令）

`tools/` 薄层保持 `{'success': True/False, 'error': ...}` 契约不变（向后兼容）：`load_dataframe` / `load_from_memory` / `release_dataframe` / `visualize_relations` 失败时均返回 `{'success': False, 'error': ...}`；`core/chart.py` 的 `render_chart` 虽不属薄层，同样按该契约返回 dict。

CLI 层统一把失败转换为退出码 1（与 load 一致）：`success: false`、DataFrame 不存在、参数非法（如 `--rows` 负数、release 同时给 `--name` 与 `--all`）、文件不存在、比对/探查/渲染执行错误、后端异常 → stdout 仍输出结果 JSON（含 `error` 字段，既有字段不变），另向 stderr 打印 `❌ {error}` 并以退出码 1 结束；成功（含 0 行结果、空关系列表、`released: []` 等空结果）退出 0。未预期异常由 `observe()` 入口兜底为 `❌ 执行失败: ...` + 退出码 1（不裸奔 traceback）。详见 `docs/specs/cli.md`「退出码契约（observe 全部子命令）」。
