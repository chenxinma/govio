# observe CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `govio observe` CLI command that provides data table comparison, relation exploration, and table operations without a persistent daemon process. DataFrames are persisted to `.govio/observe/` directory with a manifest file for replay.

**Architecture:** The observe command is a stateless CLI that reads/writes DataFrames to parquet files in `.govio/observe/dataframes/` and tracks them in `.govio/observe/manifest.json`. Existing MCP tool functions are reused directly (without decorators) in the CLI subcommands.

**Tech Stack:** Python 3.13+, argparse (CLI), pandas/parquet (persistence), datacompy (comparison), NetworkX (visualization), duckdb/sqlalchemy (data sources)

---

## File Structure

```
.govio/observe/                    # runtime directory (git-ignored)
├── manifest.json                  # replay manifest
└── dataframes/                    # parquet files
    ├── <name>.parquet
    └── ...

src/govio/
├── mcp/
│   ├── server.py                  # DELETED
│   ├── config.py                  # KEPT (DatabaseManager, DataSourceConfig)
│   ├── core/
│   │   ├── database.py            # KEPT (DatabaseManager)
│   │   ├── comparator.py          # KEPT (TableComparator)
│   │   ├── explorer.py            # KEPT (RelationExplorer)
│   │   └── visualizer.py          # KEPT (RelationVisualizer)
│   └── tools/
│       ├── load_dataframe.py      # MODIFIED - remove MCP decorator
│       ├── release_dataframe.py   # MODIFIED - remove MCP decorator, use file-based store
│       ├── list_dataframes.py     # MODIFIED - remove MCP decorator
│       ├── compare_tables.py      # KEPT as-is (no decorator)
│       ├── explore_relations.py   # KEPT as-is (no decorator)
│       ├── visualize_relations.py # KEPT as-is (no decorator)
│       └── list_datasources.py   # KEPT as-is (no decorator)
└── cli/
    ├── main.py                    # MODIFIED - register observe command
    ├── config.py                  # MODIFIED - support datasources field
    ├── onboard.py                 # MODIFIED - add datasource config step
    ├── observe.py                 # CREATE - main observe CLI
    └── observe_store.py           # CREATE - file-based DataFrame store
```

**Files NOT touched (kept as-is):**
- `src/govio/mcp/core/dataframe_store.py` — KEPT as-is (no changes needed — see Task 5 explanation)
- `src/govio/mcp/tools/__init__.py` — MCP tool exports no longer needed
- `src/govio/__init__.py`

---

## Task 1: Modify ConfigManager — add datasources field support

**Files:**
- Modify: `src/govio/cli/config.py`

- [ ] **Step 1: Read existing config.py**

Read `src/govio/cli/config.py` to see current structure.

- [ ] **Step 2: Add datasources validation**

Add to the `validate()` method after existing field checks:

```python
if "datasources" in config:
    datasources = config["datasources"]
    if not isinstance(datasources, dict):
        raise ValueError("datasources must be a dictionary")
    for name, ds_data in datasources.items():
        if "url" not in ds_data:
            raise ValueError(f"datasource '{name}' missing 'url' field")
```

- [ ] **Step 3: Commit**

```bash
git add src/govio/cli/config.py
git commit -m "feat(config): add datasources field validation"
```

---

## Task 2: Modify onboard.py — add datasource config step

**Files:**
- Modify: `src/govio/cli/onboard.py:276-336`

- [ ] **Step 1: Read existing onboard.py**

Focus on the `onboard()` function end section (lines 276-336).

- [ ] **Step 2: Add datasource config prompt function**

Add before `onboard()`:

```python
def prompt_datasource_config() -> dict[str, Any] | None:
    """提示用户配置数据源（可选）"""
    print("\n=== 步骤 2: 数据源配置（可选）===\n")
    print("是否配置数据源供 observe 命令使用？")
    print("  - 可以添加 MySQL、DuckDB 等数据源")
    print("  - 后续可用 govio observe load 查询这些数据源")
    print()

    add_ds = input("是否添加数据源？ (yes/no) [默认: no]: ").strip().lower()
    if add_ds not in ("yes", "y"):
        return None

    datasources = {}
    while True:
        print("\n添加数据源：")
        name = input("  数据源名称: ").strip()
        if not name:
            print("  名称不能为空")
            continue

        url = input("  URL (如 mysql+pymysql://user:pass@host/db 或 duckdb:///path): ").strip()
        if not url:
            print("  URL 不能为空")
            continue

        connect_args_str = input("  连接参数 (JSON 格式，可直接回车跳过) [默认: {}]: ").strip()
        connect_args = {}
        if connect_args_str:
            import json
            try:
                connect_args = json.loads(connect_args_str)
            except json.JSONDecodeError:
                print("  JSON 格式错误，使用空参数")
                connect_args = {}

        datasources[name] = {
            "url": url,
            "connect_args": connect_args,
        }

        more = input("\n是否继续添加数据源？ (yes/no) [默认: no]: ").strip().lower()
        if more not in ("yes", "y"):
            break

    return datasources
```

- [ ] **Step 3: Integrate into onboard() function**

Find the section after saving config and before assets generation:
```python
# 在 config_manager.save(full_config) 之后，backend_file.write_text(backend + "\n") 之前插入
datasources = prompt_datasource_config()
if datasources:
    full_config["datasources"] = datasources
    # Re-save with datasources
    config_manager.save(full_config)
```

- [ ] **Step 4: Commit**

```bash
git add src/govio/cli/onboard.py
git commit -m "feat(onboard): add optional datasource configuration step"
```

---

## Task 3: Create observe_store.py — file-based DataFrame persistence

**Files:**
- Create: `src/govio/cli/observe_store.py`

- [ ] **Step 1: Write the ObserveStore class**

```python
"""observe DataFrame 文件持久化存储

DataFrame 存储在 .govio/observe/dataframes/ 目录下的 parquet 文件中。
清单保存在 .govio/observe/manifest.json。
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


OBSERVE_DIR = Path(".govio/observe")
DATAFRAMES_DIR = OBSERVE_DIR / "dataframes"
MANIFEST_FILE = OBSERVE_DIR / "manifest.json"


@dataclass
class DataFrameInfo:
    """DataFrame 信息"""
    name: str
    datasource: str
    sql: str
    file: str
    loaded_at: str
    rows: int
    columns: int
    column_info: list[dict] = field(default_factory=list)


@dataclass
class Manifest:
    """回放清单"""
    version: str = "1.0"
    dataframes: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls) -> "Manifest":
        if not MANIFEST_FILE.exists():
            return cls()
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            version=data.get("version", "1.0"),
            dataframes=data.get("dataframes", {}),
        )

    def save(self) -> None:
        OBSERVE_DIR.mkdir(parents=True, exist_ok=True)
        with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {"version": self.version, "dataframes": self.dataframes},
                f,
                ensure_ascii=False,
                indent=2,
            )

    def add(self, info: DataFrameInfo) -> None:
        self.dataframes[info.name] = {
            "datasource": info.datasource,
            "sql": info.sql,
            "file": info.file,
            "loaded_at": info.loaded_at,
            "rows": info.rows,
            "columns": info.columns,
            "column_info": info.column_info,
        }

    def remove(self, name: str) -> bool:
        if name in self.dataframes:
            del self.dataframes[name]
            return True
        return False


class ObserveStore:
    """基于文件的 DataFrame 存储"""

    def __init__(self) -> None:
        self._manifest = Manifest.load()

    def _ensure_dataframes_dir(self) -> None:
        DATAFRAMES_DIR.mkdir(parents=True, exist_ok=True)

    def store(
        self,
        name: str,
        df: pd.DataFrame,
        datasource: str,
        sql: str,
    ) -> DataFrameInfo:
        """存储 DataFrame 到 parquet 文件"""
        self._ensure_dataframes_dir()

        file_path = DATAFRAMES_DIR / f"{name}.parquet"
        df.to_parquet(file_path, index=False)

        column_info = [{"name": col, "dtype": str(df[col].dtype)} for col in df.columns]

        info = DataFrameInfo(
            name=name,
            datasource=datasource,
            sql=sql,
            file=str(file_path),
            loaded_at=datetime.now(timezone.utc).isoformat(),
            rows=len(df),
            columns=len(df.columns),
            column_info=column_info,
        )

        self._manifest.add(info)
        self._manifest.save()

        return info

    def get(self, name: str) -> pd.DataFrame | None:
        """加载 DataFrame 到内存"""
        if name not in self._manifest.dataframes:
            return None

        file_path = Path(self._manifest.dataframes[name]["file"])
        if not file_path.exists():
            # 文件不存在，清理清单
            self._manifest.remove(name)
            self._manifest.save()
            return None

        return pd.read_parquet(file_path)

    def list(self) -> list[DataFrameInfo]:
        """列出所有 DataFrame"""
        infos = []
        for name, data in self._manifest.dataframes.items():
            column_info = data.get("column_info", [])
            if not column_info:
                # 尝试从文件读取
                file_path = Path(data["file"])
                if file_path.exists():
                    df = pd.read_parquet(file_path)
                    column_info = [
                        {"name": col, "dtype": str(df[col].dtype)}
                        for col in df.columns
                    ]

            infos.append(
                DataFrameInfo(
                    name=name,
                    datasource=data["datasource"],
                    sql=data["sql"],
                    file=data["file"],
                    loaded_at=data["loaded_at"],
                    rows=data["rows"],
                    columns=data["columns"],
                    column_info=column_info,
                )
            )
        return infos

    def release(self, name: str) -> bool:
        """释放 DataFrame — 删除文件并从清单移除"""
        if name not in self._manifest.dataframes:
            return False

        file_path = Path(self._manifest.dataframes[name]["file"])
        if file_path.exists():
            file_path.unlink()

        self._manifest.remove(name)
        self._manifest.save()
        return True

    def exists(self, name: str) -> bool:
        """检查 DataFrame 是否存在"""
        return name in self._manifest.dataframes
```

- [ ] **Step 2: Commit**

```bash
git add src/govio/cli/observe_store.py
git commit -m "feat(observe): add ObserveStore for file-based DataFrame persistence"
```

---

## Task 4: Remove MCP decorators from tool files

**Key insight — no changes to dataframe_store.py or tool type hints needed:**
- The MCP tools (`release_dataframe`, `list_dataframes`) import `DataFrameStore` from `dataframe_store.py`
- `observe.py` passes `ObserveStore` instances to these tools
- Due to **duck typing**, `ObserveStore` implements `.get()`, `.list()`, `.release()` the same way
- `load_dataframe` extracts `datasource`/`sql` and calls `store.store(name, df, datasource, sql)` — compatible with `ObserveStore.store()` signature
- Type hints are not enforced at runtime — duck typing makes it work

For each file below, remove the `@mcp.tool()` decorator and `from mcp import tool` import only. The function signatures and body stay unchanged.

**Files to modify (in this order):**

- `src/govio/mcp/tools/load_dataframe.py`
- `src/govio/mcp/tools/release_dataframe.py`
- `src/govio/mcp/tools/list_dataframes.py`
- `src/govio/mcp/tools/list_datasources.py`

- [ ] **Step 1: Modify load_dataframe.py**

Remove these two lines at the top of the file:
```python
from mcp import tool  # DELETE
@mcp.tool()           # DELETE
```
Keep everything else including the `DataFrameStore` type hint in the function signature — duck typing makes it work with `ObserveStore` at runtime.

Note: `load_dataframe.py` imports `DataFrameStore` type hint from `dataframe_store.py`. Keep this import unchanged — duck typing means `ObserveStore` is interchangeable at runtime.

- [ ] **Step 2: Modify release_dataframe.py**

Remove decorator and import only. Keep `store: DataFrameStore` type hint unchanged.

- [ ] **Step 3: Modify list_dataframes.py**

Remove decorator and import only. Keep `store: DataFrameStore` type hint unchanged.

- [ ] **Step 4: Modify list_datasources.py**

Remove decorator and import. Also fix the bug: replace `config.get_url()` with `config.url` since `DataSourceConfig` only has `.url` field.

- [ ] **Step 5: Commit each file separately**

```bash
git add src/govio/mcp/tools/load_dataframe.py
git commit -m "refactor(mcp): remove MCP decorator from load_dataframe"
git add src/govio/mcp/tools/release_dataframe.py
git commit -m "refactor(mcp): remove MCP decorator from release_dataframe"
git add src/govio/mcp/tools/list_dataframes.py
git commit -m "refactor(mcp): remove MCP decorator from list_dataframes"
git add src/govio/mcp/tools/list_datasources.py
git commit -m "fix(mcp): remove MCP decorator and fix get_url bug in list_datasources"
```

---

## Task 5: Create observe.py — main observe CLI

**Files:**
- Create: `src/govio/cli/observe.py`

- [ ] **Step 1: Write the observe command**

```python
"""observe 命令 — 数据表探查 CLI

提供数据表比较、关系探索、数据表加载释放等功能。
DataFrame 持久化到 .govio/observe/ 目录。
"""

import argparse
import json
import sys
from pathlib import Path

from .config import ConfigManager
from .observe_store import ObserveStore
from ..mcp.core.database import DatabaseManager
from ..mcp.tools.explore_relations import explore_relations
from ..mcp.tools.list_dataframes import list_dataframes
from ..mcp.tools.load_dataframe import load_dataframe as mcp_load_dataframe
from ..mcp.tools.release_dataframe import release_dataframe as mcp_release_dataframe
from ..mcp.tools.visualize_relations import visualize_relations
from ..mcp.tools.list_datasources import list_datasources


def get_db_manager(config: dict) -> DatabaseManager:
    """从配置创建 DatabaseManager"""
    from ..mcp.config import DataSourceConfig
    datasources = config.get("datasources", {})
    ds_configs = {
        name: DataSourceConfig(url=ds["url"], connect_args=ds.get("connect_args", {}))
        for name, ds in datasources.items()
    }
    return DatabaseManager(ds_configs)


def cmd_show_datasource(config: dict) -> None:
    """显示数据源"""
    db_manager = get_db_manager(config)
    result = list_datasources(db_manager)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_load(config: dict, name: str, datasource: str, sql: str) -> None:
    """加载 DataFrame"""
    db_manager = get_db_manager(config)
    store = ObserveStore()
    result = mcp_load_dataframe(
        store=store,
        db_manager=db_manager,
        datasource=datasource,
        name=name,
        sql=sql,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_list(config: dict) -> None:
    """列出 DataFrame"""
    store = ObserveStore()
    result = list_dataframes(store=store)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_release(config: dict, name: str) -> None:
    """释放 DataFrame"""
    store = ObserveStore()
    result = mcp_release_dataframe(store=store, name=name)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_compare(
    config: dict, source_df: str, target_df: str, join_columns: list[str]
) -> None:
    """比对 DataFrame"""
    store = ObserveStore()

    # 需要临时 DataFrameStore 适配器来复用 compare_tables
    # 由于 compare_tables 需要 store.get()，我们用 ObserveStore 直接调用
    source = store.get(source_df)
    if source is None:
        print(json.dumps({"success": False, "error": f"DataFrame '{source_df}' 不存在"}))
        return

    target = store.get(target_df)
    if target is None:
        print(json.dumps({"success": False, "error": f"DataFrame '{target_df}' 不存在"}))
        return

    from ..mcp.core.comparator import TableComparator
    comparator = TableComparator()
    result = comparator.compare(source, target, join_columns)
    print(json.dumps({"success": True, **result}, ensure_ascii=False, indent=2))


def cmd_explore(config: dict, dataframes: list[str] | None = None) -> None:
    """探查关系"""
    store = ObserveStore()

    if dataframes is None:
        infos = store.list()
        dataframes = [info.name for info in infos]

    df_dict = {}
    for name in dataframes:
        df = store.get(name)
        if df is None:
            print(json.dumps({"success": False, "error": f"DataFrame '{name}' 不存在"}))
            return
        df_dict[name] = df

    from ..mcp.core.explorer import RelationExplorer
    explorer = RelationExplorer()
    relations = explorer.explore(df_dict)
    print(json.dumps({"success": True, "relations": relations}, ensure_ascii=False, indent=2))


def cmd_visualize(config: dict, relations_json: str) -> None:
    """可视化关系"""
    try:
        relations = json.loads(relations_json)
    except json.JSONDecodeError as e:
        print(json.dumps({"success": False, "error": f"JSON 解析失败: {e}"}))
        return

    result = visualize_relations(relations=relations)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def observe():
    """observe 命令入口"""
    parser = argparse.ArgumentParser(
        description="数据表探查命令 — 加载、比较、探索数据表",
    )
    sub = parser.add_subparsers(dest="action", required=True)

    # show-datasource
    sub.add_parser("show-datasource", help="显示已配置的数据源")

    # load <name> <datasource> <sql>
    p = sub.add_parser("load", help="加载 DataFrame")
    p.add_argument("name", help="DataFrame 名称")
    p.add_argument("datasource", help="数据源名称")
    p.add_argument("sql", help="查询 SQL")

    # list
    sub.add_parser("list", help="列出已加载的 DataFrame")

    # release <name>
    p = sub.add_parser("release", help="释放 DataFrame")
    p.add_argument("name", help="DataFrame 名称")

    # compare <source> <target> --join-columns col1,col2
    p = sub.add_parser("compare", help="比对两个 DataFrame")
    p.add_argument("source", help="源 DataFrame 名称")
    p.add_argument("target", help="目标 DataFrame 名称")
    p.add_argument("--join-columns", required=True, help="比对列，逗号分隔")

    # explore [df1 df2 ...]
    p = sub.add_parser("explore", help="探查 DataFrame 之间的关系")
    p.add_argument("dataframes", nargs="*", help="DataFrame 名称列表")

    # visualize-relations <json>
    p = sub.add_parser("visualize-relations", help="生成关系图谱")
    p.add_argument("relations", help="关系 JSON")

    args = parser.parse_args(sys.argv[1:])

    # 加载配置
    config_manager = ConfigManager()
    if not config_manager.exists():
        print("错误: 配置文件不存在，请先运行 govio onboard", file=sys.stderr)
        sys.exit(1)

    config = config_manager.load()

    if not config.get("datasources"):
        print("警告: 配置中无 datasources，请先用 onboard 配置数据源", file=sys.stderr)

    # 分发子命令
    match args.action:
        case "show-datasource":
            cmd_show_datasource(config)
        case "load":
            cmd_load(config, args.name, args.datasource, args.sql)
        case "list":
            cmd_list(config)
        case "release":
            cmd_release(config, args.name)
        case "compare":
            cols = [c.strip() for c in args.join_columns.split(",")]
            cmd_compare(config, args.source, args.target, cols)
        case "explore":
            cmd_explore(config, args.dataframes if args.dataframes else None)
        case "visualize-relations":
            cmd_visualize(config, args.relations)


if __name__ == "__main__":
    observe()
```

- [ ] **Step 2: Commit**

```bash
git add src/govio/cli/observe.py
git commit -m "feat(observe): add main observe CLI command"
```

---

## Task 6: Modify main.py — register observe command

**Files:**
- Modify: `src/govio/cli/main.py`

- [ ] **Step 1: Read existing main.py**

- [ ] **Step 2: Add observe import and register**

```python
import argparse
from .onboard import onboard
from .std_recommend import std_recommend
from .observe import observe

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="数据治理知识图谱项目，提供元数据查询、表字段比较、SQL 生成、数据标准推荐等数据治理支持功能。",
    )
    parser.add_argument('action', default='onboard', choices=['onboard','std-recommend','observe'], action='store')
    args = parser.parse_args()
    if args.action == "onboard":
        onboard()
    elif args.action == "std-recommend":
        std_recommend()
    elif args.action == "observe":
        observe()
```

- [ ] **Step 3: Commit**

```bash
git add src/govio/cli/main.py
git commit -m "feat(cli): register observe command"
```

---

## Task 7: Delete MCP server entry point

**Files:**
- Delete: `src/govio/mcp/server.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Delete server.py**

```bash
rm src/govio/mcp/server.py
```

- [ ] **Step 2: Remove mcp-server script from pyproject.toml**

In `pyproject.toml`, remove:
```toml
[project.scripts]
mcp-server = "govio.mcp.server:main"  # DELETE THIS LINE
```

Keep `govio-cli = "govio.cli:main"`.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git rm src/govio/mcp/server.py
git commit -m "chore: remove mcp-server entry point"
```

---

## Task 8: Update .gitignore

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Add observe directory**

Add to `.gitignore`:
```
# observe DataFrame storage
.govio/observe/
```

- [ ] **Step 2: Commit**

```bash
git add .gitignore
git commit -m "chore: ignore .govio/observe directory"
```

---

## Verification

After all tasks, verify:

1. `govio observe --help` works
2. `govio onboard` still works and asks about datasources at end
3. `govio observe show-datasource` works after datasource config
4. `govio observe load test mysql+pymysql://... "SELECT 1"` creates `.govio/observe/manifest.json` and `.govio/observe/dataframes/test.parquet`
5. `govio observe list` shows loaded DataFrames
6. `govio observe release test` removes the file and updates manifest
7. `git status` shows no untracked files in `.govio/observe/`
