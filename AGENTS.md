# AGENTS.md

Guidelines for AI coding agents working in this repository.

## Project Overview

Govio (Governance + IO) is a data governance knowledge graph platform. It extracts metadata from relational databases, builds graph structures, and provides data standard recommendation via collaborative filtering.

- **Language**: Python 3.13+
- **Package Manager**: uv
- **Build Backend**: hatchling
- **Version**: see `pyproject.toml`

## Documentation Sync Rule

**修改 skills 或功能实现时，必须同步更新 `docs/specs/` 下对应的设计文档。**

具体对应关系：

| 变更内容 | 需同步更新的 spec 文档 |
|---|---|
| CLI 命令、子命令、参数 | `docs/specs/cli.md` |
| 配置格式、配置迁移、加密 | `docs/specs/config.md` |
| 图后端（NetworkX/FalkorDB/Ladybug）、CSV loader | `docs/specs/graph.md` |
| 元数据加载器（TDS/DuckDB/Trino）、推荐器、指标、node_id | `docs/specs/metadata.md` |
| GraphFactory、AssetsGenerator、sql_builder | `docs/specs/core.md` |
| ObserveStore、DatabaseManager、comparator、explorer、chart | `docs/specs/observe-data.md` |
| 节点/边类型、CSV 格式、ID 格式 | `docs/specs/data-model.md` |
| 依赖变更、模块增删、版本变更 | `docs/specs/README.md` |

更新 spec 时保持与实际代码实现一致，不要留下过时的类名、方法签名或功能描述。

## Build/Lint/Test Commands

### Installation

```bash
# Install dependencies
uv sync

# Install with dev dependencies
uv sync --group dev
```

### Running Tests

```bash
# Run all tests
uv run pytest tests/

# Run a single test file
uv run pytest tests/test_relationship.py

# Run a specific test
uv run pytest tests/test_relationship.py::test_load_json_success

# Run tests with verbose output
uv run pytest tests/ -v

# Run tests with coverage
uv run pytest tests/ --cov=src/govio
```

### Linting and Formatting

```bash
# Check code with ruff
uv run ruff check src/ tests/

# Format code with ruff
uv run ruff format src/ tests/

# Fix linting issues automatically
uv run ruff check --fix src/ tests/
```

Lint 规则集固定在 `pyproject.toml` 的 `[tool.ruff.lint]`（`uvx ruff check` 与 `uv run ruff check` 结果一致）；中文内容、测试断言、错误边界宽捕获等意图性排除见该配置内注释。

### Build

```bash
# Build package
uv build

# Build wheel only
uv build --wheel
```

### Local Deploy (full cycle)

完整本地部署流程：清理旧版本 → 构建 → 安装为 uv tool → 打包 skills。

```bash
# 一键执行（等价于 start.sh）
./start.sh

# 或手动执行：

# 1. 清理 dist 下旧版本（避免残留过期包）
rm -f dist/govio-*.whl dist/govio-*.tar.gz dist/govio-skills.zip

# 2. 构建新版本
uv build

# 3. 安装 govio 为 uv tool
WHL=$(ls dist/govio-*.whl | head -1)
uv tool install --from "$WHL" govio --compile-bytecode -p 3.13 --force

# 4. 打包 skills
uv run package_skills.py
```

> **注意**: Windows cmd 下用 `del /Q dist\govio-*.whl dist\govio-*.tar.gz` 替代步骤 1。

### Type Checking

```bash
# Run pyright (if available)
uv run pyright src/
```

### CLI Usage

```bash
govio-cli onboard           # Interactive setup wizard
govio-cli backend            # Show current graph backend
govio-cli query -c "..."     # Knowledge graph query
govio-cli meta meta --source duckdb --db x.duckdb --schemas main --datasource local_duckdb --output ./output   # 导入元数据 -> CSV
govio-cli meta graph --output ./output --mode update  # CSV -> 图库 + assets
govio-cli meta recommend     # Data standard recommendation
govio-cli observe info       # Show datasources + DataFrames
govio-cli observe load ...   # Load DataFrame from DB or memory
govio-cli observe compare ... # Compare two DataFrames
govio-cli observe chart ...   # Generate PNG chart
govio-cli sql build -f query.json  # Assemble metric SQL
```

## Code Style Guidelines

### Imports

Order imports in three groups, separated by blank lines:

1. Standard library imports (alphabetical)
2. Third-party imports (alphabetical)
3. Local imports (alphabetical)

```python
import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .database import TDSLoader
from .datasource import DatasourceLoader
```

### Type Hints

Use modern Python 3.13+ type hint syntax:

```python
# Preferred
def load_tables(self, schema_limits: list[str] | None) -> pd.DataFrame:
    ...

# Avoid (old style)
from typing import List, Optional
def load_tables(self, schema_limits: Optional[List[str]]) -> pd.DataFrame:
    ...
```

Use union types with `|` for optional parameters. Use `Any` sparingly.

### Naming Conventions

- **Modules**: snake_case (`recommender.py`, `relationship.py`)
- **Classes**: PascalCase (`StandardRecommender`, `RelationshipLoader`)
- **Functions/Methods**: snake_case (`load_relationships`, `find_k_neighbors`)
- **Private methods**: prefix with underscore (`_preprocess_std_data`, `_validate_inputs`)
- **Constants**: UPPER_SNAKE_CASE (`DEFAULT_WEIGHTS`, `MIN_SIMILARITY`)
- **Properties**: snake_case with `@property` decorator (`PhysicalTable`, `Col`)

### Docstrings

Use Chinese docstrings for Chinese-language projects. Include Args and Returns sections:

```python
def validate_relationship(self, rel: dict[str, Any], index: int) -> bool:
    """验证单个关系的有效性

    Args:
        rel: 关系字典
        index: 关系索引（用于错误消息）

    Returns:
        bool: 是否有效
    """
```

### Classes

- Use `__init__` for initialization with type-annotated parameters
- Use `@property` for computed attributes that don't require parameters
- Use factory functions for complex object creation

```python
class TDSLoader:
    def __init__(self, db: str, workspace_uuid: str, schema_limits: list[str] | None = None) -> None:
        self.engine = create_engine(db)
        self.workspace_uuid = workspace_uuid

    @property
    def PhysicalTable(self) -> pd.DataFrame:
        return self.load_tables()
```

### Error Handling

- Raise descriptive exceptions with context
- Use logging module for warnings (not print statements)
- Validate inputs early in public methods

```python
def _validate_inputs(self):
    if not self.json_path.exists():
        raise FileNotFoundError(f"关系文件不存在: {self.json_path}")

    if self.df_tables.empty:
        raise ValueError("PhysicalTable DataFrame 为空")
```

### Comments

- Write comments in Chinese for this codebase
- Avoid inline comments that restate the obvious
- Use module-level docstrings to explain purpose

### File Encoding

Always specify `encoding="utf-8"` for file reads and writes (Windows locale defaults to cp936, which corrupts Chinese text and breaks cross-platform tests):

```python
# Preferred
path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
path.read_text(encoding="utf-8")

# Avoid (uses locale default encoding)
path.write_text(text)
path.read_text()
```

### File Organization

```
src/govio/
├── __init__.py              # Public exports (FalkorDBGraph, LadybugGraph, NetworkXGraph, main, build_metric_sql)
├── crypto.py                # Credential encryption/decryption
├── cli/                     # CLI entry points
│   ├── __init__.py          # main() entry
│   ├── __main__.py
│   ├── config.py            # ConfigManager
│   ├── main.py              # argparse dispatch
│   ├── meta.py              # meta command group (sync/recommend/config)
│   ├── observe.py           # observe command group
│   ├── onboard.py           # interactive setup wizard
│   ├── query.py             # knowledge graph query
│   ├── sql.py               # sql build command
│   └── std_recommend.py     # data standard recommendation
├── core/                    # Shared core logic
│   ├── __init__.py
│   ├── assets_generator.py  # schema.md, names, metrics_index.md generation
│   ├── graph_factory.py     # GraphFactory (networkx/falkordb/ladybug)
│   └── sql_builder.py       # Metric SQL assembly (CTE, atomic/derived)
├── graph/                   # Graph database backends
│   ├── __init__.py          # exports NetworkXGraph, FalkorDBGraph, LadybugGraph
│   ├── networkx_graph.py
│   ├── falkordb_graph.py
│   ├── falkordb_loader.py   # CSV bulk import/upsert to FalkorDB
│   ├── ladybug_graph.py
│   └── ladybug_loader.py    # CSV bulk import/upsert to Ladybug
├── metadata/                # Metadata loading and processing
│   ├── __init__.py
│   ├── database.py          # TDSLoader (base: MetadataLoader)
│   ├── datasource.py        # DatasourceLoader (datasource.json declaration)
│   ├── datasource_schema.json  # Datasource declaration JSON Schema
│   ├── duckdb_loader.py     # DuckDBLoader
│   ├── gen_networkx.py      # CSV → GML conversion (incremental support)
│   ├── metric.py            # MetricLoader
│   ├── metric_schema.json   # Metric definition JSON Schema
│   ├── node_id.py           # Deterministic 10-char string ID generation
│   ├── recommender.py       # StandardRecommender (k-NN)
│   ├── relationship.py      # RelationshipLoader
│   ├── relationship_schema.json  # Relationship definition JSON Schema
│   ├── standard.py          # StandardLoader
│   ├── trino_loader.py      # TrinoLoader
│   └── utility.py           # make_csv, data_standard_recommend
└── observe_data/            # Data observation module
    ├── __init__.py
    ├── config.py            # DataSourceConfig, load_config
    └── core/
        ├── __init__.py
        ├── chart.py          # render_chart (bar/line PNG)
        ├── comparator.py     # TableComparator (datacompy)
        ├── database.py       # DatabaseManager (multi-datasource)
        ├── dataframe_store.py
        ├── explorer.py       # RelationExplorer (FK inference, similarity)
        ├── observe_store.py  # ObserveStore (parquet-backed)
        └── visualizer.py     # RelationVisualizer (networkx/JSON)
    └── tools/
        ├── __init__.py
        ├── list_dataframes.py
        ├── list_datasources.py
        ├── load_dataframe.py    # load_dataframe, load_from_memory
        ├── release_dataframe.py
        └── visualize_relations.py
```

### Constants and Configuration

Define module-level constants at the top of the file:

```python
DEFAULT_WEIGHTS = {
    'table': 0.20,
    'name': 0.26,
    'comment': 0.22,
    'type': 0.22,
    'numeric': 0.10
}

DEFAULT_K_NEIGHBORS = 5
MIN_SIMILARITY = 0.7
```

### Avoid

- Adding comments that restate code
- Using `print()` for logging (use `logging` module)
- Mutable default arguments
- Bare `except` clauses
- Star imports (`from module import *`)

## Project-Specific Notes

### Entry Points

The package defines a CLI entry point in `pyproject.toml`:

- `govio-cli` -> `govio.cli:main`

Main subcommands: `onboard`, `backend`, `query`, `meta`, `observe`, `sql`

### Environment Variables

Load environment variables using `python-dotenv`:

```python
from dotenv import load_dotenv
import os

load_dotenv()
db = os.getenv("KUNDB_URL", "")
```

### Graph Backends

Govio supports three graph backends:

| Backend | Config Key | Query Language | File Format |
|---|---|---|---|
| NetworkX | `graph.networkx` | Python (exec) | `.gml` |
| FalkorDB | `graph.falkordb` | Cypher | Redis-based |
| Ladybug | `graph.ladybug` | Cypher | `.lbdb` (embedded) |

### Node ID Format

Node IDs are deterministic 10-char strings: `<2-char prefix><SHA256(business_key)[:8]>`. See `src/govio/metadata/node_id.py`.

### Testing

Tests use pytest with fixtures. Place fixtures at module level or in conftest.py:

```python
@pytest.fixture
def sample_tables():
    return pd.DataFrame({
        "full_table_name": ["db.schema.table1"],
    })
```

<!-- CODEGRAPH_START -->
## CodeGraph

In repositories indexed by CodeGraph (a `.codegraph/` directory exists at the repo root), reach for it BEFORE grep/find or reading files when you need to understand or locate code:

- **MCP tool** (when available): `codegraph_explore` answers most code questions in one call — the relevant symbols' verbatim source plus the call paths between them, including dynamic-dispatch hops grep can't follow. Name a file or symbol in the query to read its current line-numbered source. If it's listed but deferred, load it by name via tool search.
- **Shell** (always works): `codegraph explore "<symbol names or question>"` prints the same output.

If there is no `.codegraph/` directory, skip CodeGraph entirely — indexing is the user's decision.
<!-- CODEGRAPH_END -->
