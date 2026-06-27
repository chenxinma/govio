# String ID 与单库导出 — 设计文档

- 日期: 2026-06-27
- 状态: 已确认，待写实施计划
- 版本 bump: 0.2.12 → 0.3.0（破坏性，ID 格式不兼容旧图库）

## 背景与动机

当前所有节点 ID 由 `metadata/utility.py:reorder_index` 分配全局连续整数（1..N），跨节点类型连续编号，写入 `:ID(Label)` 列。falkordb-bulk-loader 默认按 STRING 处理并在前面拼 `Label.` 命名空间，最终入库 ID 形如 `PhysicalTable.1`、`Col.5001`。

问题：
1. ID 与业务键无关联，无法从 ID 反推节点，跨次导出顺序变化时 ID 漂移。
2. `meta-export` 只能全量导出（TDS + DuckDB 合并 + 全部 app/standard/metric），无法只导入一个数据库的子图，增量更新成本高。

falkordb-bulk-loader 已支持 `--id-type=STRING`（`.venv/.../falkordb_bulk_loader/bulk_insert.py:84-89`），且默认就是 STRING，所以改用字符串业务键 hash 作为 ID 无需改 bulk-loader 调用。

## 目标

1. 节点 ID 改为 10 位 STRING：`<2 字符类型前缀><SHA256(业务键) 前 8 hex>`。
2. `meta-export` 新增 `--db-name` 单库模式，仅导出指定数据库的相关子图，不查 TDS。
3. 两条 CSV 生成路径（`cli/meta_export.py` 与 `metadata/utility.py`）同步改造。

## 非目标

- 不做旧图库平滑迁移。旧 ID（`PhysicalTable.1` 等）不再有效，需 drop 旧 graph 重新导入。
- 不改 bulk-loader 调用参数（默认 STRING 已匹配）。
- 不改各 Loader 的抽取逻辑。

## §1 ID 生成模块

新模块 `src/govio/metadata/node_id.py`：

```python
import hashlib

NODE_PREFIXES = {
    "PhysicalTable": "PT",
    "Col":           "CO",
    "Application":   "AP",
    "Standard":      "ST",
    "Metric":        "ME",
    "Dimension":     "DI",
}

def make_id(node_type: str, business_key: str) -> str:
    prefix = NODE_PREFIXES[node_type]
    h = hashlib.sha256(business_key.encode("utf-8")).hexdigest()[:8].upper()
    return f"{prefix}{h}"
```

业务键映射（来自现有列，零新增抽取）：

| 节点类型 | 业务键列 | 来源 |
|---|---|---|
| PhysicalTable | `full_table_name` (schema.table) | database.py:133 |
| Col | `column` (schema.table.column) | database.py:89 |
| Application | `app_id` | application.py:11 |
| Standard | `standard_id` | standard.py:103 |
| Metric | `code` | metric.py:159 |
| Dimension | `code` | metric.py:146 |

冲突处理：每个节点类型内部，写 CSV 前断言 `node_id` 唯一；冲突则 `sys.exit(1)` 并打印冲突的 business_key 列表。SHA256[:8] = 32 bit，万级节点冲突概率可忽略；前置检查比 bulk-loader 报错更清晰。

跨类型同业务键不冲突（前缀不同），如 `make_id("PhysicalTable","dm.t")` 与 `make_id("Col","dm.t")` 不同。

## §2 CSV 生成改造

### 节点 CSV

所有节点 DataFrame 在写 CSV 前新增 `node_id` 列（string），用 `:ID(Label)` 作列名写出，`index=False`。

helper `write_node_csv(df, path, node_type, key_col)` 避免重复模板：
- 构造 `node_id = make_id(node_type, key)` 列，列名 `:ID(Label)`
- 业务列保持原顺序
- 唯一性断言
- `to_csv(index=False)`

### 边 CSV

边的 `:START_ID`/`:END_ID` 列值从「整数 index」改为「对端节点的 string `node_id`」。改动统一是 join 时把对端 DataFrame 的 `node_id` 列带进来 rename，不再用 `reset_index().rename({"index": ...})`。

举例 `HAS_COLUMN`（meta_export.py:86-96）：

```python
df_has_column = pd.merge(
    df_tables[["full_table_name", "node_id"]].rename(
        columns={"node_id": ":START_ID(PhysicalTable)"}
    ),
    df_columns[["full_table_name", "node_id"]].rename(
        columns={"node_id": ":END_ID(Col)"}
    ),
    on="full_table_name",
    how="inner",
)[[":START_ID(PhysicalTable)", ":END_ID(Col)"]]
```

8 类边（HAS_COLUMN / USE / RELATES_TO / USES_TABLE / REFERS_COLUMN / DERIVED_FROM / DIMENSION_USED / SUPERSEDES）全部按此模式改。Metric/Dimension 边里的 `metric_offset += ` 偏移逻辑全部删除——`node_id` 已是终值，无需偏移。

### 两条路径

1. `cli/meta_export.py`（新路径，主路径）
2. `metadata/utility.py` 的 `make_csv`/`run`（老路径，README 用的入口）

两条都改，保证一致。

### reorder_index 处置

`utility.py:reorder_index` 函数保留（避免破坏外部 import），但 `meta_export.py` 和 `utility.py:make_csv` 不再调用。docstring 标记 deprecated。

### bulk-loader 调用

`onboard.import_csv_to_falkordb` 当前不传 `--id-type`，默认 STRING。新 ID 是 string，正好匹配，**无需改 bulk-loader 调用**。

## §3 单库模式（single-db export）

### 触发

`meta-export` 新增 `--db-name <app_name>`，与 `--schemas` 同时支持：

- `--schemas` 走全量模式（TDS + DuckDB 合并 + 全部 app/std/metric），按 schema 过滤。
- `--db-name` 进入单库模式：**不查 TDS**，只从 `df_app_db_map` 里按 name 选中一行，取其 schema 抽 DuckDB。
- 两者都给时取交集：`--db-name` 锁定 app→schema，`--schemas` 进一步限制该 app 下的 schema 子集。
- 两者都不给时报错（至少给一个）。

### 包含的节点/边（完整相关子图）

1. **PhysicalTable + Col**：从 DuckDB 抽该 schema。
2. **Application**：仅 `--db-name` 对应的那一个 app 节点。
3. **USE 边**：仅 app→该 schema 表。
4. **HAS_COLUMN 边**：该 schema 表的列。
5. **Standard + COMPLIES_WITH 边**：从全量 Standard 里筛出"被该 schema 列 COMPLIES_WITH 的"子集（Standard 节点本身只保留被引用的）。
6. **Metric/Dimension + 相关边**：从全量 Metric/Dimension 里筛出"USES_TABLE 命中该 schema 表的"Metric 子集；Dimension 按 DIMENSION_USED 反查被这些 Metric 用到的子集。边只保留两端都在子集内的。

### 为什么单库仍要加载全量 Standard/Metric

Standard/Metric 定义在 governance 层，没有"属于哪个 schema"的属性，只能通过 COMPLIES_WITH / USES_TABLE 反查。加载全量再过滤是最简单的实现。全量加载本身已是现状，无新成本。

### 输出目录

复用 `--output`，单库模式下写入同一目录结构，CSV 文件名不变（PhysicalTable.csv 等），只是内容是子集。dry-run 与导入逻辑不变。

### 子图过滤实现

在 `meta_export.py` 里加 `filter_subgraph_by_schema(...)` 分支，在 ID 生成之前对每个节点/边 DataFrame 做过滤。过滤逻辑集中在 `meta_export` 内部，不改各 Loader。

### 不查 TDS 的实现

单库模式下 `tds_loader` / `tds_tables` / `tds_columns` 整段跳过，`merge_metadata` 也不调用，`df_tables = duck_tables`、`df_columns = duck_columns`。

### Application 节点来源

`AppInfoLoader` 仍读全量 app_list，但 `df_apps` 过滤成只剩 `--db-name` 对应那一行（按 `df_app_db_map` 的 name → app_id 匹配）。

### 报错语义

`--db-name` 不在 `df_app_db_map.name` 里时 `sys.exit(1)` 并列出可用 name。

## §4 CLI 与命令行

`cli/main.py` 的 `meta-export` 子命令参数变化：

```python
p_meta = sub.add_parser("meta-export", help="...")
p_meta.add_argument("--db", type=str, required=True, help="DuckDB 数据库文件路径")
p_meta.add_argument("--schemas", type=str, help="要导出的 schema 列表，逗号分隔（如 dm,dwd,dws）")
p_meta.add_argument("--db-name", type=str, help="单库模式：按 app 名导出单个数据库的子图（不查 TDS）")
p_meta.add_argument("--output", type=Path, required=True, help="CSV 输出目录")
p_meta.add_argument("--dry-run", action="store_true", help="...")
```

- `--schemas` 改为非必填（单库模式下不强制）。
- 新增 `--db-name`。
- 校验：`--schemas` 和 `--db-name` 至少给一个；都不给则报错退出。
- 同时给时：`--db-name` 选 app → schema，`--schemas` 进一步收窄。

`meta_export()` 签名从 `(db_path, schemas, output, dry_run)` 改为 `(db_path, schemas, db_name, output, dry_run)`，`schemas` 改为 `list[str] | None`，`db_name` 为 `str | None`。

调用点 `cli/main.py:99` 同步更新。

## §5 测试

### `tests/test_node_id.py`（新增）

1. `make_id` 基本正确性：相同业务键 → 相同 ID；不同业务键 → 不同 ID；长度=10；前缀正确。
2. 前缀表覆盖 6 种节点类型。
3. 同类型内 ID 唯一性（构造 1000 个不同 business_key，断言无重复）。
4. 跨类型同业务键不冲突。

### `tests/test_meta_export.py`（新增或扩展）

5. 全量模式：mock TDS+DuckDB+app+std+metric，跑 `meta_export(dry_run=True)`，断言每个节点 CSV 第一列是 `:ID(Label)` 且值为 10 位 string、前缀正确；断言边的 `:START_ID`/`:END_ID` 值能在对应节点 CSV 里找到。
6. 单库模式（`--db-name`）：断言不查 TDS（mock TDSLoader 不被调用），输出只含该 schema 的表/列、单个 app、相关 standard/metric 子集。
7. 单库 + `--schemas` 交集：`--db-name` 锁 app，`--schemas` 收窄，断言输出 schema 是交集。
8. 错误路径：`--db-name` 不存在 → exit 1；`--schemas` 和 `--db-name` 都不给 → exit 1。
9. 老 `make_csv`/`run` 路径（utility.py）：跑一次 dry-run，断言同样生成 string ID。

测试用 fixture 构造最小 DataFrame，不连真实数据库；TDS/DuckDB/MetricLoader 用 `unittest.mock` patch。

## §6 Skill 文档同步与版本

### Skill 文档

经查 `skills/` 目录下没有描述 meta-export / ID 格式的现成文档，本次仅新增 `--db-name` 说明：

1. 若 `skills/govio/SKILL.md` 或 `skills/govio/govio-metadata/SKILL.md` 提到 `meta-export`，补 `--db-name` 单库模式说明。
2. 若有描述图模型/ID 格式的 reference 文件，同步 ID 格式描述（10 位 string）。

写文档时再确认具体改哪些文件。

### 版本

`pyproject.toml` 当前 0.2.12。破坏性变更（ID 格式不兼容旧图库），bump 到 `0.3.0`（minor，表示破坏性 ID 格式变更）。

### 破坏性变更声明

在 commit message 和 release notes 里明确写：旧图库需 drop 后重新 `meta-export` 导入，旧 ID（`PhysicalTable.1` 等）不再有效。

## 涉及文件清单

新增：
- `src/govio/metadata/node_id.py`
- `tests/test_node_id.py`
- `tests/test_meta_export.py`

修改：
- `src/govio/cli/meta_export.py` — ID 生成改 string、单库模式分支、`write_node_csv` helper
- `src/govio/metadata/utility.py` — `make_csv`/`run` 同步改 string ID
- `src/govio/cli/main.py` — `--db-name` 参数、`meta_export` 签名、校验
- `pyproject.toml` — 版本 0.2.12 → 0.3.0
- `skills/` 下相关 SKILL.md（写文档时确认）
