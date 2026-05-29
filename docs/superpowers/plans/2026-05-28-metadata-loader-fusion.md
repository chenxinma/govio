# Metadata Loader Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify DatabaseLoader and DuckDB metadata loading under a common MetadataLoader ABC, enabling meta-export to merge TDS full + DuckDB incremental metadata into complete CSV output.

**Architecture:** Abstract base class MetadataLoader defines load_tables/load_columns interface. TDSLoader (renamed from DatabaseLoader) and DuckDBLoader implement it. meta_export.py reads TDS config from ~/.govio/config.yaml, loads both sources, merges by full_table_name/column key with DuckDB precedence, and exports all CSV types matching onboard output.

**Tech Stack:** Python 3.13+, pandas, duckdb, sqlalchemy, pyyaml, abc

---

### Task 1: Add MetadataLoader ABC to database.py

**Files:**
- Modify: `src/govio/metadata/database.py:1-5`
- Test: `tests/test_metadata_loader.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metadata_loader.py
import pytest
import pandas as pd
from govio.metadata.database import MetadataLoader


def test_metadata_loader_is_abstract():
    """MetadataLoader cannot be instantiated directly."""
    with pytest.raises(TypeError):
        MetadataLoader()


def test_metadata_loader_requires_load_tables():
    """Subclass must implement load_tables."""

    class IncompleteLoader(MetadataLoader):
        def load_columns(self):
            return pd.DataFrame()

    with pytest.raises(TypeError):
        IncompleteLoader()


def test_metadata_loader_requires_load_columns():
    """Subclass must implement load_columns."""

    class IncompleteLoader(MetadataLoader):
        def load_tables(self):
            return pd.DataFrame()

    with pytest.raises(TypeError):
        IncompleteLoader()


def test_metadata_loader_properties_delegate():
    """PhysicalTable and Col properties delegate to load_tables/load_columns."""

    class MockLoader(MetadataLoader):
        def load_tables(self):
            return pd.DataFrame({"full_table_name": ["a.b"]})

        def load_columns(self):
            return pd.DataFrame({"column": ["a.b.c"]})

    loader = MockLoader()
    assert list(loader.PhysicalTable.columns) == ["full_table_name"]
    assert list(loader.Col.columns) == ["column"]
    assert len(loader.PhysicalTable) == 1
    assert len(loader.Col) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/macx/work/python/govio && uv run pytest tests/test_metadata_loader.py -v`
Expected: FAIL with `ImportError: cannot import name 'MetadataLoader'`

- [ ] **Step 3: Write minimal implementation**

Add to top of `src/govio/metadata/database.py` (before `DatabaseLoader` class):

```python
from abc import ABC, abstractmethod

class MetadataLoader(ABC):
    @abstractmethod
    def load_tables(self) -> pd.DataFrame:
        ...

    @abstractmethod
    def load_columns(self) -> pd.DataFrame:
        ...

    @property
    def PhysicalTable(self) -> pd.DataFrame:
        return self.load_tables()

    @property
    def Col(self) -> pd.DataFrame:
        return self.load_columns()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/macx/work/python/govio && uv run pytest tests/test_metadata_loader.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/govio/metadata/database.py tests/test_metadata_loader.py
git commit -m "feat(metadata): add MetadataLoader ABC base class"
```

---

### Task 2: Rename DatabaseLoader to TDSLoader

**Files:**
- Modify: `src/govio/metadata/database.py:5-137`
- Modify: `src/govio/metadata/utility.py:6,32,204`
- Test: `tests/test_metadata_loader.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_metadata_loader.py`:

```python
def test_tds_loader_is_metadata_loader():
    """TDSLoader is a subclass of MetadataLoader."""
    from govio.metadata.database import TDSLoader
    assert issubclass(TDSLoader, MetadataLoader)


def test_database_loader_alias():
    """DatabaseLoader alias still works for backward compatibility."""
    from govio.metadata.database import DatabaseLoader, TDSLoader
    assert DatabaseLoader is TDSLoader
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/macx/work/python/govio && uv run pytest tests/test_metadata_loader.py::test_tds_loader_is_metadata_loader -v`
Expected: FAIL with `ImportError: cannot import name 'TDSLoader'`

- [ ] **Step 3: Rename class and add alias**

In `src/govio/metadata/database.py`:
- Rename `class DatabaseLoader` to `class TDSLoader`
- Add `TDSLoader` as parent class: `class TDSLoader(MetadataLoader):`
- Remove the existing `PhysicalTable` and `Col` properties (now inherited from base)
- Add alias at end of file: `DatabaseLoader = TDSLoader`

- [ ] **Step 4: Update imports in utility.py**

In `src/govio/metadata/utility.py`:
- Line 6: `from .database import DatabaseLoader` → `from .database import TDSLoader`
- Line 32: `DatabaseLoader(` → `TDSLoader(`
- Line 204: `DatabaseLoader(` → `TDSLoader(`

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /home/macx/work/python/govio && uv run pytest tests/test_metadata_loader.py -v`
Expected: PASS (6 tests)

- [ ] **Step 6: Run full test suite**

Run: `cd /home/macx/work/python/govio && uv run pytest tests/ -v`
Expected: All existing tests pass (no regressions)

- [ ] **Step 7: Commit**

```bash
git add src/govio/metadata/database.py src/govio/metadata/utility.py tests/test_metadata_loader.py
git commit -m "refactor(metadata): rename DatabaseLoader to TDSLoader with backward compat alias"
```

---

### Task 3: Create DuckDBLoader in new file

**Files:**
- Create: `src/govio/metadata/duckdb_loader.py`
- Test: `tests/test_duckdb_loader.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_duckdb_loader.py
import pytest
import pandas as pd
import duckdb
import tempfile
from pathlib import Path
from govio.metadata.duckdb_loader import DuckDBLoader
from govio.metadata.database import MetadataLoader


@pytest.fixture
def sample_duckdb(tmp_path):
    """Create a sample DuckDB with test tables and columns."""
    db_path = str(tmp_path / "test.duckdb")
    conn = duckdb.connect(db_path)
    conn.execute("CREATE SCHEMA IF NOT EXISTS test_schema")
    conn.execute("""
        CREATE TABLE test_schema.users (
            id INTEGER,
            name VARCHAR
        )
    """)
    conn.execute("COMMENT ON TABLE test_schema.users IS '用户表'")
    conn.execute("COMMENT ON COLUMN test_schema.users.id IS '用户ID'")
    conn.execute("COMMENT ON COLUMN test_schema.users.name IS '用户名'")
    conn.execute("""
        CREATE TABLE test_schema.orders (
            order_id INTEGER,
            amount DECIMAL(10,2)
        )
    """)
    conn.close()
    return db_path


def test_duckdb_loader_is_metadata_loader(sample_duckdb):
    """DuckDBLoader is a subclass of MetadataLoader."""
    loader = DuckDBLoader(sample_duckdb, ["test_schema"])
    assert isinstance(loader, MetadataLoader)


def test_duckdb_loader_load_tables(sample_duckdb):
    """DuckDBLoader loads tables with expected columns."""
    loader = DuckDBLoader(sample_duckdb, ["test_schema"])
    df = loader.load_tables()
    assert len(df) == 2
    assert "full_table_name" in df.columns
    assert "schema" in df.columns
    assert "table_name" in df.columns
    assert "name" in df.columns
    assert "data_entity_type" in df.columns
    assert all(df["data_entity_type"] == "DUCKDB_TABLE")


def test_duckdb_loader_load_columns(sample_duckdb):
    """DuckDBLoader loads columns with expected structure."""
    loader = DuckDBLoader(sample_duckdb, ["test_schema"])
    df = loader.load_columns()
    assert len(df) == 4
    assert "column" in df.columns
    assert "column_name" in df.columns
    assert "name" in df.columns
    assert "full_table_name" in df.columns
    assert "data_entity_type" in df.columns
    assert "dtype" in df.columns
    assert "data_type" in df.columns
    assert all(df["data_entity_type"] == "DUCKDB_COLUMN")


def test_duckdb_loader_properties(sample_duckdb):
    """PhysicalTable and Col properties work."""
    loader = DuckDBLoader(sample_duckdb, ["test_schema"])
    assert len(loader.PhysicalTable) == 2
    assert len(loader.Col) == 4


def test_duckdb_loader_table_comments(sample_duckdb):
    """Table comments are loaded correctly."""
    loader = DuckDBLoader(sample_duckdb, ["test_schema"])
    df = loader.load_tables()
    users_row = df[df["table_name"] == "users"].iloc[0]
    assert users_row["name"] == "用户表"


def test_duckdb_loader_column_comments(sample_duckdb):
    """Column comments are loaded correctly."""
    loader = DuckDBLoader(sample_duckdb, ["test_schema"])
    df = loader.load_columns()
    id_row = df[df["column_name"] == "id"].iloc[0]
    assert id_row["name"] == "用户ID"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/macx/work/python/govio && uv run pytest tests/test_duckdb_loader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'govio.metadata.duckdb_loader'`

- [ ] **Step 3: Implement DuckDBLoader**

Create `src/govio/metadata/duckdb_loader.py`:

```python
import duckdb
import pandas as pd

from .database import MetadataLoader


class DuckDBLoader(MetadataLoader):
    def __init__(self, db_path: str, schemas: list[str]) -> None:
        self.db_path = db_path
        self.schemas = schemas

    def load_tables(self) -> pd.DataFrame:
        conn = duckdb.connect(self.db_path, read_only=True)
        df = conn.execute(
            """
            SELECT schema_name || '.' || table_name AS full_table_name,
                   schema_name AS "schema",
                   table_name AS table_name,
                   COALESCE(comment, '') AS "name",
                   'DUCKDB_TABLE' AS data_entity_type,
                   '' AS database_name
            FROM duckdb_tables()
            WHERE schema_name IN (SELECT unnest(?))
            ORDER BY schema_name, table_name
            """,
            [self.schemas],
        ).fetchdf()
        conn.close()
        return df

    def load_columns(self) -> pd.DataFrame:
        conn = duckdb.connect(self.db_path, read_only=True)
        df = conn.execute(
            """
            SELECT c.table_schema || '.' || c.table_name || '.' || c.column_name AS "column",
                   c.column_name AS column_name,
                   COALESCE(dc.comment, '') AS "name",
                   c.table_schema || '.' || c.table_name AS full_table_name,
                   'DUCKDB_COLUMN' AS data_entity_type,
                   c.data_type AS dtype,
                   0 AS "size",
                   0 AS "precision",
                   0 AS "scale",
                   c.ordinal_position AS order_no,
                   c.data_type AS data_type
            FROM information_schema.columns c
            LEFT JOIN duckdb_columns() dc
                ON dc.schema_name = c.table_schema
                AND dc.table_name = c.table_name
                AND dc.column_name = c.column_name
            WHERE c.table_schema IN (SELECT unnest(?))
            ORDER BY c.table_schema, c.table_name, c.ordinal_position
            """,
            [self.schemas],
        ).fetchdf()
        conn.close()
        return df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/macx/work/python/govio && uv run pytest tests/test_duckdb_loader.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/govio/metadata/duckdb_loader.py tests/test_duckdb_loader.py
git commit -m "feat(metadata): add DuckDBLoader implementing MetadataLoader"
```

---

### Task 4: Refactor meta_export.py with TDS+DuckDB merge

**Files:**
- Modify: `src/govio/cli/meta_export.py`
- Test: `tests/test_meta_export.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_meta_export.py
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
from govio.cli.meta_export import merge_metadata, meta_export


def test_merge_metadata_duckdb_wins():
    """DuckDB rows take precedence over TDS rows on conflict."""
    df_tds = pd.DataFrame({
        "full_table_name": ["s1.t1", "s1.t2"],
        "name": ["TDS Table 1", "TDS Table 2"],
    })
    df_duck = pd.DataFrame({
        "full_table_name": ["s1.t1"],
        "name": ["DuckDB Table 1"],
    })
    result = merge_metadata(df_tds, df_duck, "full_table_name")
    assert len(result) == 2
    t1_row = result[result["full_table_name"] == "s1.t1"].iloc[0]
    assert t1_row["name"] == "DuckDB Table 1"


def test_merge_metadata_adds_new():
    """DuckDB-only rows are included in merge result."""
    df_tds = pd.DataFrame({
        "full_table_name": ["s1.t1"],
        "name": ["TDS Table 1"],
    })
    df_duck = pd.DataFrame({
        "full_table_name": ["s1.t2"],
        "name": ["DuckDB Table 2"],
    })
    result = merge_metadata(df_tds, df_duck, "full_table_name")
    assert len(result) == 2
    assert set(result["full_table_name"]) == {"s1.t1", "s1.t2"}


def test_merge_metadata_tds_only():
    """Works when DuckDB has no rows."""
    df_tds = pd.DataFrame({
        "full_table_name": ["s1.t1"],
        "name": ["TDS Table 1"],
    })
    df_duck = pd.DataFrame(columns=["full_table_name", "name"])
    result = merge_metadata(df_tds, df_duck, "full_table_name")
    assert len(result) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/macx/work/python/govio && uv run pytest tests/test_meta_export.py -v`
Expected: FAIL with `ImportError: cannot import name 'merge_metadata'`

- [ ] **Step 3: Implement merge_metadata and refactor meta_export**

Rewrite `src/govio/cli/meta_export.py`:

```python
from pathlib import Path

import pandas as pd

from govio.cli.config import ConfigManager
from govio.metadata.database import TDSLoader
from govio.metadata.application import AppInfoLoader
from govio.metadata.standard import StandardLoader
from govio.metadata.duckdb_loader import DuckDBLoader
from govio.metadata.utility import reorder_index
from govio.metadata.relationship import load_relationships
from govio.metadata.metric import MetricLoader


def merge_metadata(df_tds: pd.DataFrame, df_duck: pd.DataFrame, key: str) -> pd.DataFrame:
    """TDS full + DuckDB incremental. DuckDB wins on conflict."""
    combined = pd.concat([df_tds, df_duck], ignore_index=True)
    return combined.drop_duplicates(subset=[key], keep="last").reset_index(drop=True)


def meta_export(db_path: str, schemas: list[str], start_id: int, output: Path):
    output.mkdir(parents=True, exist_ok=True)

    # 1. Load config for TDS connection
    config = ConfigManager().load()
    kundb = config["kundb"]
    workspace_uuid = config["workspace_uuid"]
    app_list_file = config["app_list"]
    app_map_path = config["app_map"]

    # 2. Load TDS full metadata
    tds_loader = TDSLoader(kundb, workspace_uuid, schemas)
    app_loader = AppInfoLoader(app_list_file)
    std_loader = StandardLoader(kundb, workspace_uuid)

    df_tables_tds = tds_loader.PhysicalTable
    df_columns_tds = tds_loader.Col
    df_apps = app_loader.Application
    df_stds = std_loader.Standard

    # 3. Load DuckDB incremental
    duckdb_loader = DuckDBLoader(db_path, schemas)
    df_tables_duck = duckdb_loader.PhysicalTable
    df_columns_duck = duckdb_loader.Col

    # 4. Merge tables and columns
    df_tables = merge_metadata(df_tables_tds, df_tables_duck, "full_table_name")
    df_columns = merge_metadata(df_columns_tds, df_columns_duck, "column")

    # 5. Assign contiguous IDs
    reorder_index([df_tables, df_columns, df_apps, df_stds], start=start_id)

    # 6. Write node CSVs
    files = []

    df_tables.to_csv(output / "PhysicalTable.csv", index_label=":ID(PhysicalTable)")
    files.append("-n " + str(output / "PhysicalTable.csv"))

    df_columns.to_csv(output / "Col.csv", index_label=":ID(Col)")
    files.append("-n " + str(output / "Col.csv"))

    df_apps.to_csv(output / "Application.csv", index_label=":ID(Application)")
    files.append("-n " + str(output / "Application.csv"))

    df_stds.to_csv(output / "Standard.csv", index_label=":ID(Standard)")
    files.append("-n " + str(output / "Standard.csv"))

    # 7. Write HAS_COLUMN edge
    df_has_column = pd.merge(
        df_tables[["full_table_name"]]
        .reset_index()
        .rename(columns={"index": ":START_ID(PhysicalTable)"}),
        df_columns[["full_table_name"]]
        .reset_index()
        .rename(columns={"index": ":END_ID(Col)"}),
        on="full_table_name",
        how="inner",
    )[":START_ID(PhysicalTable)", ":END_ID(Col)"]]
    df_has_column.to_csv(output / "HAS_COLUMN.csv", index=False)
    files.append("-r " + str(output / "HAS_COLUMN.csv"))

    # 8. Write USE edge (app→table via app-map)
    df_app_db_map = pd.read_json(app_map_path)
    df_app_table = pd.merge(
        df_app_db_map,
        df_tables[["schema"]]
        .reset_index()
        .rename(columns={"index": ":END_ID(PhysicalTable)"}),
        on="schema",
        how="inner",
    )
    df_use = pd.merge(
        df_apps[["name"]]
        .reset_index()
        .rename(columns={"index": ":START_ID(Application)"}),
        df_app_table,
        on="name",
        how="inner",
    )[[":START_ID(Application)", ":END_ID(PhysicalTable)"]]
    df_use.to_csv(output / "USE.csv", index=False)
    files.append("-r " + str(output / "USE.csv"))

    # 9. Optionally load relationships
    relationship_file = config.get("relationship")
    if relationship_file:
        try:
            df_relates_to = load_relationships(relationship_file, df_tables, df_columns)
            df_relates_to.to_csv(
                output / "RELATES_TO.csv",
                index=False,
                header=[
                    ":START_ID(PhysicalTable)",
                    ":END_ID(PhysicalTable)",
                    "relationship_type",
                    "description",
                    "source_columns",
                    "target_columns",
                ],
            )
            files.append("-r " + str(output / "RELATES_TO.csv"))
            print(f"成功生成 RELATES_TO.csv，包含 {len(df_relates_to)} 个关系")
        except Exception as e:
            print(f"警告: 无法加载关系文件: {e}")

    # 10. Optionally load metrics
    metric_file = config.get("metric")
    if metric_file:
        try:
            metric_loader = MetricLoader(metric_file, df_tables, df_columns)
            df_metrics = metric_loader.Metric
            df_dimensions = metric_loader.Dimension

            metric_offset = (
                len(df_tables) + len(df_columns) + len(df_apps) + len(df_stds) + start_id
            )
            dim_offset = metric_offset + len(df_metrics)
            reorder_index([df_metrics, df_dimensions], start=metric_offset)

            df_metrics.to_csv(output / "Metric.csv", index_label=":ID(Metric)")
            files.append("-n " + str(output / "Metric.csv"))

            df_dimensions.to_csv(output / "Dimension.csv", index_label=":ID(Dimension)")
            files.append("-n " + str(output / "Dimension.csv"))

            uses_table = metric_loader.uses_table_edges.copy()
            if not uses_table.empty:
                uses_table[":START_ID(Metric)"] += metric_offset
                uses_table.to_csv(output / "USES_TABLE.csv", index=False)
                files.append("-r " + str(output / "USES_TABLE.csv"))

            refers_col = metric_loader.refers_column_edges.copy()
            if not refers_col.empty:
                refers_col[":START_ID(Metric)"] += metric_offset
                refers_col.to_csv(output / "REFERS_COLUMN.csv", index=False)
                files.append("-r " + str(output / "REFERS_COLUMN.csv"))

            derived_from = metric_loader.derived_from_edges.copy()
            if not derived_from.empty:
                derived_from[":START_ID(Metric)"] += metric_offset
                derived_from[":END_ID(Metric)"] += metric_offset
                derived_from.to_csv(output / "DERIVED_FROM.csv", index=False)
                files.append("-r " + str(output / "DERIVED_FROM.csv"))

            dim_used = metric_loader.dimension_used_edges.copy()
            if not dim_used.empty:
                dim_used[":START_ID(Metric)"] += metric_offset
                dim_used[":END_ID(Dimension)"] += dim_offset
                dim_used.to_csv(output / "DIMENSION_USED.csv", index=False)
                files.append("-r " + str(output / "DIMENSION_USED.csv"))

            supersedes = metric_loader.supersedes_edges
            if not supersedes.empty:
                supersedes.to_csv(output / "SUPERSEDES.csv", index=False)
                files.append("-r " + str(output / "SUPERSEDES.csv"))

            print(
                f"成功生成指标数据：{len(df_metrics)} 个指标, "
                f"{len(df_dimensions)} 个维度"
            )
        except Exception as e:
            print(f"警告: 无法加载指标定义文件: {e}")

    print(f"成功导出: {len(df_tables)} 张表, {len(df_columns)} 个字段, "
          f"{len(df_apps)} 个应用, {len(df_stds)} 个标准")
    print(f"ID 范围: {start_id} ~ {start_id + len(df_tables) + len(df_columns) + len(df_apps) + len(df_stds) - 1}")

    s = f"falkordb-bulk-insert {{GRAPH}} {'  '.join(files)}"
    print(f"\n{s}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/macx/work/python/govio && uv run pytest tests/test_meta_export.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Update main.py help text**

In `src/govio/cli/main.py`, line 56:
```python
p_meta = sub.add_parser("meta-export", help="从 DuckDB + TDS 合并导出全量元数据 CSV")
```

- [ ] **Step 6: Commit**

```bash
git add src/govio/cli/meta_export.py src/govio/cli/main.py tests/test_meta_export.py
git commit -m "feat(cli): refactor meta_export with TDS+DuckDB merge and full CSV output"
```

---

### Task 5: Integration verification

**Files:**
- Test: `tests/test_meta_export_integration.py` (new)

- [ ] **Step 1: Write integration test for DuckDBLoader + merge_metadata**

```python
# tests/test_meta_export_integration.py
import pandas as pd
import duckdb
from govio.metadata.duckdb_loader import DuckDBLoader
from govio.cli.meta_export import merge_metadata


def test_full_merge_pipeline(tmp_path):
    """End-to-end: DuckDB load → merge with mock TDS → correct output."""
    # Setup DuckDB
    db_path = str(tmp_path / "test.duckdb")
    conn = duckdb.connect(db_path)
    conn.execute("CREATE SCHEMA IF NOT EXISTS s1")
    conn.execute("CREATE TABLE s1.t1 (id INTEGER, name VARCHAR)")
    conn.execute("CREATE TABLE s1.t2 (val DECIMAL(10,2))")
    conn.close()

    # Load DuckDB
    loader = DuckDBLoader(db_path, ["s1"])
    df_duck_tables = loader.PhysicalTable
    df_duck_cols = loader.Col

    # Mock TDS data (s1.t1 exists in TDS too, s1.t3 is TDS-only)
    df_tds_tables = pd.DataFrame({
        "full_table_name": ["s1.t1", "s1.t3"],
        "schema": ["s1", "s1"],
        "table_name": ["t1", "t3"],
        "name": ["TDS Table 1", "TDS Table 3"],
        "data_entity_type": ["MYSQL_TABLE", "MYSQL_TABLE"],
        "database_name": ["db1", "db1"],
    })
    df_tds_cols = pd.DataFrame({
        "column": ["s1.t1.id", "s1.t3.col1"],
        "column_name": ["id", "col1"],
        "name": ["TDS ID", "TDS Col1"],
        "full_table_name": ["s1.t1", "s1.t3"],
        "data_entity_type": ["MYSQL_COLUMN", "MYSQL_COLUMN"],
        "dtype": ["int", "varchar"],
        "size": [0, 255],
        "precision": [0, 0],
        "scale": [0, 0],
        "order_no": [1, 1],
        "data_type": ["int", "varchar(255)"],
    })

    # Merge
    df_tables = merge_metadata(df_tds_tables, df_duck_tables, "full_table_name")
    df_cols = merge_metadata(df_tds_cols, df_duck_cols, "column")

    # Assertions: 3 tables (t1 from DuckDB wins, t2 DuckDB-only, t3 TDS-only)
    assert len(df_tables) == 3
    t1_row = df_tables[df_tables["table_name"] == "t1"].iloc[0]
    assert t1_row["data_entity_type"] == "DUCKDB_TABLE"  # DuckDB wins

    # Columns: s1.t1.id from DuckDB wins, s1.t1.name DuckDB-only, s1.t2.* DuckDB-only, s1.t3.col1 TDS-only
    assert "s1.t3.col1" in df_cols["column"].values  # TDS-only column preserved
    id_row = df_cols[df_cols["column"] == "s1.t1.id"].iloc[0]
    assert id_row["data_entity_type"] == "DUCKDB_COLUMN"  # DuckDB wins
```

- [ ] **Step 2: Run all tests**

Run: `cd /home/macx/work/python/govio && uv run pytest tests/test_metadata_loader.py tests/test_duckdb_loader.py tests/test_meta_export.py tests/test_meta_export_integration.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_meta_export_integration.py
git commit -m "test: add integration test for TDS+DuckDB merge pipeline"
```
