# String ID 与单库导出 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将节点 ID 从全局连续整数改为 10 位 string（类型前缀 + 业务键 SHA256[:8]），并为 `meta-export` 新增 `--db-name` 单库导出模式。

**Architecture:** 新增 `metadata/node_id.py` 提供 `make_id` / `assign_node_ids` / `write_node_csv` 三个纯函数；`cli/meta_export.py` 与 `metadata/utility.py` 两条 CSV 生成路径改用 string ID，边的 `:START_ID`/`:END_ID` 通过 join 对端 `node_id` 列得到；`meta-export` 新增 `--db-name` 进入单库模式，跳过 TDS、按 schema 过滤相关子图。

**Tech Stack:** Python 3.13, pandas, pytest, unittest.mock, hashlib。

**Spec:** `docs/superpowers/specs/2026-06-27-string-id-and-single-db-export-design.md`

---

## 文件结构

新增：
- `src/govio/metadata/node_id.py` — ID 生成与节点 CSV 写出 helper
- `tests/test_node_id.py` — node_id 模块单测
- `tests/test_meta_export_ids.py` — meta_export 全量/单库模式 ID 与子图集成测试

修改：
- `src/govio/cli/meta_export.py` — 改用 string ID、新增单库模式分支
- `src/govio/metadata/utility.py` — `make_csv` 同步改 string ID
- `src/govio/cli/main.py` — `--db-name` 参数、`meta_export` 签名、校验
- `pyproject.toml` — 版本 0.2.12 → 0.3.0
- `skills/govio/govio-metadata/SKILL.md`（若提及 meta-export）— 补 `--db-name`

---

## Task 1: node_id 模块 — make_id

**Files:**
- Create: `src/govio/metadata/node_id.py`
- Test: `tests/test_node_id.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_node_id.py`：

```python
import pytest
from govio.metadata.node_id import make_id, NODE_PREFIXES


def test_make_id_format():
    """ID = 2 字符前缀 + 8 hex，共 10 位。"""
    node_id = make_id("PhysicalTable", "dm.orders")
    assert len(node_id) == 10
    assert node_id.startswith("PT")
    assert node_id[2:].isalnum()  # 8 个 hex 字符


def test_make_id_deterministic():
    """相同业务键产生相同 ID。"""
    assert make_id("Col", "dm.orders.id") == make_id("Col", "dm.orders.id")


def test_make_id_different_keys_differ():
    """不同业务键产生不同 ID。"""
    assert make_id("Col", "dm.orders.id") != make_id("Col", "dm.orders.name")


def test_make_id_cross_type_no_collision():
    """同业务键跨类型不冲突（前缀不同）。"""
    assert make_id("PhysicalTable", "dm.t") != make_id("Col", "dm.t")


def test_make_id_all_prefixes():
    """6 种节点类型前缀正确。"""
    assert make_id("PhysicalTable", "k").startswith("PT")
    assert make_id("Col", "k").startswith("CO")
    assert make_id("Application", "k").startswith("AP")
    assert make_id("Standard", "k").startswith("ST")
    assert make_id("Metric", "k").startswith("ME")
    assert make_id("Dimension", "k").startswith("DI")


def test_make_id_unknown_type_raises():
    with pytest.raises(ValueError, match="未知节点类型"):
        make_id("Unknown", "k")


def test_make_id_empty_key_raises():
    with pytest.raises(ValueError, match="business_key"):
        make_id("PhysicalTable", "")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_node_id.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'govio.metadata.node_id'`

- [ ] **Step 3: 写最小实现**

创建 `src/govio/metadata/node_id.py`：

```python
"""节点 string ID 生成与节点 CSV 写出。

ID 格式: <2 字符类型前缀><SHA256(业务键) 前 8 hex>，共 10 位。
业务键来自各节点的天然唯一列（full_table_name / column / app_id / standard_id / code）。
"""

import hashlib
import sys
from pathlib import Path

import pandas as pd

NODE_PREFIXES = {
    "PhysicalTable": "PT",
    "Col": "CO",
    "Application": "AP",
    "Standard": "ST",
    "Metric": "ME",
    "Dimension": "DI",
}


def make_id(node_type: str, business_key: str) -> str:
    """生成 10 位 string ID。"""
    if node_type not in NODE_PREFIXES:
        raise ValueError(f"未知节点类型: {node_type}")
    if not business_key:
        raise ValueError("business_key 不能为空")
    prefix = NODE_PREFIXES[node_type]
    digest = hashlib.sha256(business_key.encode("utf-8")).hexdigest()[:8].upper()
    return f"{prefix}{digest}"
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_node_id.py -v`
Expected: 7 passed

- [ ] **Step 5: commit**

```bash
git add src/govio/metadata/node_id.py tests/test_node_id.py
git commit -m "feat(id): add make_id for 10-char string node IDs

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 2: node_id 模块 — assign_node_ids 与 write_node_csv

**Files:**
- Modify: `src/govio/metadata/node_id.py`
- Test: `tests/test_node_id.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_node_id.py` 末尾追加：

```python
import pandas as pd
from govio.metadata.node_id import assign_node_ids, write_node_csv


def test_assign_node_ids_adds_column(tmp_path):
    df = pd.DataFrame({"full_table_name": ["dm.t1", "dm.t2"], "name": ["a", "b"]})
    assign_node_ids(df, "PhysicalTable", "full_table_name")
    assert "node_id" in df.columns
    assert df["node_id"].iloc[0].startswith("PT")
    assert len(df["node_id"].iloc[0]) == 10
    assert df["node_id"].iloc[0] != df["node_id"].iloc[1]


def test_assign_node_ids_uniqueness_exit_on_dup(tmp_path, capsys):
    df = pd.DataFrame({"full_table_name": ["dm.t1", "dm.t1"]})
    with pytest.raises(SystemExit):
        assign_node_ids(df, "PhysicalTable", "full_table_name")
    captured = capsys.readouterr()
    assert "ID 冲突" in captured.err


def test_write_node_csv_header_and_id_column(tmp_path):
    df = pd.DataFrame({"full_table_name": ["dm.t1"], "name": ["a"]})
    assign_node_ids(df, "PhysicalTable", "full_table_name")
    path = tmp_path / "PhysicalTable.csv"
    write_node_csv(df, path, "PhysicalTable")
    read_back = pd.read_csv(path)
    assert ":ID(PhysicalTable)" == read_back.columns[0]
    assert read_back[":ID(PhysicalTable)"].iloc[0].startswith("PT")
    assert "full_table_name" in read_back.columns
    assert "name" in read_back.columns
    assert "node_id" not in read_back.columns


def test_write_node_csv_without_node_id_raises(tmp_path):
    df = pd.DataFrame({"full_table_name": ["dm.t1"]})
    with pytest.raises(ValueError, match="node_id"):
        write_node_csv(df, tmp_path / "x.csv", "PhysicalTable")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_node_id.py -v`
Expected: FAIL — `ImportError: cannot import name 'assign_node_ids'`

- [ ] **Step 3: 写实现**

在 `src/govio/metadata/node_id.py` 末尾追加：

```python
def assign_node_ids(df: pd.DataFrame, node_type: str, key_col: str) -> None:
    """就地给 df 加 node_id 列。断言同类型内唯一，冲突则 sys.exit(1)。"""
    keys = [str(k) for k in df[key_col]]
    ids = [make_id(node_type, k) for k in keys]
    if len(set(ids)) != len(ids):
        seen: set[str] = set()
        dups = [k for k in keys if k in seen or seen.add(k)]  # type: ignore[func-returns-value]
        print(
            f"❌ {node_type} 节点 ID 冲突，重复业务键: {dups}",
            file=sys.stderr,
        )
        sys.exit(1)
    df["node_id"] = ids


def write_node_csv(df: pd.DataFrame, path: Path, node_type: str) -> None:
    """把已带 node_id 列的 df 写成 CSV，ID 列名 :ID(Label) 置首。"""
    if "node_id" not in df.columns:
        raise ValueError(f"DataFrame 缺少 node_id 列 (node_type={node_type})")
    out = df.drop(columns=["node_id"])
    out.insert(0, f":ID({node_type})", df["node_id"])
    out.to_csv(path, index=False)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_node_id.py -v`
Expected: 11 passed

- [ ] **Step 5: commit**

```bash
git add src/govio/metadata/node_id.py tests/test_node_id.py
git commit -m "feat(id): add assign_node_ids and write_node_csv helpers

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 3: meta_export 全量模式 — 节点 CSV 改 string ID

**Files:**
- Modify: `src/govio/cli/meta_export.py:67-83`（节点 ID 分配与写出）
- Test: `tests/test_meta_export_ids.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_meta_export_ids.py`：

```python
"""meta_export string ID 集成测试。mock 全部 Loader，跑 dry-run 检查 CSV。"""
import sys
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest


def _mock_tds_tables():
    return pd.DataFrame({
        "full_table_name": ["dm.orders", "dm.customers"],
        "schema": ["dm", "dm"],
        "table_name": ["orders", "customers"],
        "name": ["Orders", "Customers"],
        "data_entity_type": ["MYSQL_TABLE", "MYSQL_TABLE"],
        "database_name": ["db", "db"],
    })


def _mock_tds_columns():
    return pd.DataFrame({
        "column": ["dm.orders.id", "dm.orders.amount", "dm.customers.id"],
        "column_name": ["id", "amount", "id"],
        "name": ["ID", "Amount", "ID"],
        "full_table_name": ["dm.orders", "dm.orders", "dm.customers"],
        "data_entity_type": ["MYSQL_COLUMN"] * 3,
        "dtype": ["int", "decimal", "int"],
        "size": [0, 10, 0],
        "precision": [0, 10, 0],
        "scale": [0, 2, 0],
        "order_no": [1, 2, 1],
        "data_type": ["int", "decimal(10,2)", "int"],
    })


def _mock_duck_tables():
    return pd.DataFrame(columns=_mock_tds_tables().columns)


def _mock_duck_columns():
    return pd.DataFrame(columns=_mock_tds_columns().columns)


def _mock_apps():
    return pd.DataFrame({
        "app_id": ["app_billing"],
        "name": ["billing"],
        "description": ["Billing app"],
    })


def _mock_stds():
    return pd.DataFrame({
        "standard_id": ["std_amount"],
        "name": ["Amount Standard"],
        "data_type": ["decimal"],
    })


def _mock_app_db_map():
    return pd.DataFrame({"name": ["billing"], "schema": ["dm"]})


@pytest.fixture
def _patched_loaders():
    """Patch all loaders + ConfigManager so meta_export runs without DB/config."""
    config = {
        "metadata": {
            "kundb": "mysql://x",
            "workspace_uuid": "ws",
            "app_list": "app.json",
            "app_map": "app_map.json",
            "relationship": None,
            "metric": None,
        },
        "graph": {},
    }
    with patch("govio.cli.meta_export.ConfigManager") as cfg_m, \
         patch("govio.cli.meta_export.TDSLoader") as tds_m, \
         patch("govio.cli.meta_export.DuckDBLoader") as duck_m, \
         patch("govio.cli.meta_export.AppInfoLoader") as app_m, \
         patch("govio.cli.meta_export.StandardLoader") as std_m:
        cfg_m.return_value.load.return_value = config
        tds_m.return_value.PhysicalTable = _mock_tds_tables()
        tds_m.return_value.Col = _mock_tds_columns()
        duck_m.return_value.PhysicalTable = _mock_duck_tables()
        duck_m.return_value.Col = _mock_duck_columns()
        app_m.return_value.Application = _mock_apps()
        std_m.return_value.Standard = _mock_stds()
        yield


def test_node_csvs_have_string_ids(_patched_loaders, tmp_path):
    from govio.cli.meta_export import meta_export
    meta_export(db_path="ignored", schemas=["dm"], db_name=None,
                output=tmp_path, dry_run=True)

    for fname, prefix, key_col in [
        ("PhysicalTable.csv", "PT", "full_table_name"),
        ("Col.csv", "CO", "column"),
        ("Application.csv", "AP", "app_id"),
        ("Standard.csv", "ST", "standard_id"),
    ]:
        df = pd.read_csv(tmp_path / fname)
        id_col = f":ID({fname.removesuffix('.csv')})"
        assert id_col == df.columns[0], f"{fname} 第一列应为 {id_col}"
        for v in df[id_col]:
            assert len(v) == 10, f"{fname} ID 长度应为 10: {v}"
            assert v.startswith(prefix), f"{fname} ID 前缀应为 {prefix}: {v}"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_meta_export_ids.py::test_node_csvs_have_string_ids -v`
Expected: FAIL — 现有代码写整数 ID，长度≠10 或前缀不匹配；且 `meta_export` 当前签名是 `(db_path, schemas, output, dry_run)`，调用会因 `db_name` kwarg 报 TypeError。

- [ ] **Step 3: 改 meta_export 签名与节点写出**

修改 `src/govio/cli/meta_export.py`。

(a) 顶部 import 增加一行：

```python
from govio.metadata.node_id import assign_node_ids, write_node_csv
```

(b) 将 `meta_export` 函数签名（第 28 行）改为：

```python
def meta_export(
    db_path: str,
    schemas: list[str] | None,
    db_name: str | None,
    output: Path,
    dry_run: bool = True,
):
```

(c) 将第 67-83 行（`reorder_index` + 4 个节点 `to_csv`）替换为：

```python
    # --- Assign string IDs ---
    df_tables = df_tables.reset_index(drop=True)
    df_columns = df_columns.reset_index(drop=True)
    df_apps = df_apps.reset_index(drop=True)
    df_stds = df_stds.reset_index(drop=True)
    assign_node_ids(df_tables, "PhysicalTable", "full_table_name")
    assign_node_ids(df_columns, "Col", "column")
    assign_node_ids(df_apps, "Application", "app_id")
    assign_node_ids(df_stds, "Standard", "standard_id")

    files = []

    # --- Node CSVs ---
    write_node_csv(df_tables, output / "PhysicalTable.csv", "PhysicalTable")
    files.append("-n " + str(output / "PhysicalTable.csv"))

    write_node_csv(df_columns, output / "Col.csv", "Col")
    files.append("-n " + str(output / "Col.csv"))

    write_node_csv(df_apps, output / "Application.csv", "Application")
    files.append("-n " + str(output / "Application.csv"))

    write_node_csv(df_stds, output / "Standard.csv", "Standard")
    files.append("-n " + str(output / "Standard.csv"))
```

- [ ] **Step 4: 暂时跳过边测试，先确认节点测试通过**

Run: `uv run pytest tests/test_meta_export_ids.py::test_node_csvs_have_string_ids -v`
Expected: PASS（边的代码此时仍用旧 reset_index，会因 df_tables 不再有 `index` 列而在写 HAS_COLUMN 时报错——见 Task 4 修复；若本步报错属预期，继续 Task 4）

> 注：若此步因边生成报错而 fail，先注释掉 `meta_export` 里 HAS_COLUMN 及之后的边生成代码临时验证节点，再在 Task 4 一并恢复。推荐直接进入 Task 4。

- [ ] **Step 5: commit**

```bash
git add src/govio/cli/meta_export.py tests/test_meta_export_ids.py
git commit -m "feat(meta-export): node CSVs use 10-char string IDs

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 4: meta_export 全量模式 — 边 CSV 改 string ID

**Files:**
- Modify: `src/govio/cli/meta_export.py:85-208`（HAS_COLUMN / USE / RELATES_TO / 5 类 metric 边）
- Test: `tests/test_meta_export_ids.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_meta_export_ids.py` 末尾追加：

```python
def test_edge_csvs_reference_valid_node_ids(_patched_loaders, tmp_path):
    from govio.cli.meta_export import meta_export
    meta_export(db_path="ignored", schemas=["dm"], db_name=None,
                output=tmp_path, dry_run=True)

    # 收集所有节点 ID
    node_ids: set[str] = set()
    for fname, label in [
        ("PhysicalTable.csv", "PhysicalTable"),
        ("Col.csv", "Col"),
        ("Application.csv", "Application"),
        ("Standard.csv", "Standard"),
    ]:
        df = pd.read_csv(tmp_path / fname)
        node_ids.update(df[f":ID({label})"].astype(str))

    # HAS_COLUMN
    has_col = pd.read_csv(tmp_path / "HAS_COLUMN.csv")
    assert ":START_ID(PhysicalTable)" in has_col.columns
    assert ":END_ID(Col)" in has_col.columns
    for v in has_col[":START_ID(PhysicalTable)"]:
        assert str(v) in node_ids, f"HAS_COLUMN START_ID {v} 不存在于节点表"
    for v in has_col[":END_ID(Col)"]:
        assert str(v) in node_ids

    # USE
    use = pd.read_csv(tmp_path / "USE.csv")
    for v in use[":START_ID(Application)"]:
        assert str(v) in node_ids
    for v in use[":END_ID(PhysicalTable)"]:
        assert str(v) in node_ids

    # HAS_COLUMN 行数 = 列数（每个列对应一张表）
    assert len(has_col) == 3
    # USE 行数 = schema 匹配的表数（dm 下 2 张表）
    assert len(use) == 2


def test_metric_edges_use_string_ids(tmp_path):
    """带 metric 的全量导出：metric/dim 节点与 5 类边都是 string ID。"""
    metric_data = {
        "metrics": [
            {
                "code": "m_total_amount",
                "name": "Total Amount",
                "business_definition": "总金额",
                "type": "atomic",
                "unit": "元",
                "data_type": "decimal",
                "source_layer": "dwd",
                "source_tables": [
                    {"full_table_name": "dm.orders", "columns": [
                        {"column_name": "amount", "role": "measure"}
                    ]}
                ],
                "dimensions": [{"code": "dim_time", "usage_type": "group_by"}],
            }
        ],
        "shared_dimensions": [
            {"code": "dim_time", "name": "Time", "granularity": "day"}
        ],
    }
    import json
    metric_file = tmp_path / "metric.json"
    metric_file.write_text(json.dumps(metric_data, ensure_ascii=False))

    config = {
        "metadata": {
            "kundb": "mysql://x", "workspace_uuid": "ws",
            "app_list": "app.json", "app_map": "app_map.json",
            "relationship": None, "metric": str(metric_file),
        },
        "graph": {},
    }
    with patch("govio.cli.meta_export.ConfigManager") as cfg_m, \
         patch("govio.cli.meta_export.TDSLoader") as tds_m, \
         patch("govio.cli.meta_export.DuckDBLoader") as duck_m, \
         patch("govio.cli.meta_export.AppInfoLoader") as app_m, \
         patch("govio.cli.meta_export.StandardLoader") as std_m:
        cfg_m.return_value.load.return_value = config
        tds_m.return_value.PhysicalTable = _mock_tds_tables()
        tds_m.return_value.Col = _mock_tds_columns()
        duck_m.return_value.PhysicalTable = _mock_duck_tables()
        duck_m.return_value.Col = _mock_duck_columns()
        app_m.return_value.Application = _mock_apps()
        std_m.return_value.Standard = _mock_stds()

        from govio.cli.meta_export import meta_export
        out = tmp_path / "out"
        meta_export(db_path="ignored", schemas=["dm"], db_name=None,
                    output=out, dry_run=True)

    # Metric / Dimension 节点
    m_df = pd.read_csv(out / "Metric.csv")
    assert ":ID(Metric)" == m_df.columns[0]
    assert m_df[":ID(Metric)"].iloc[0].startswith("ME")
    d_df = pd.read_csv(out / "Dimension.csv")
    assert d_df[":ID(Dimension)"].iloc[0].startswith("DI")

    node_ids = set()
    for fname, label in [
        ("PhysicalTable.csv", "PhysicalTable"), ("Col.csv", "Col"),
        ("Application.csv", "Application"), ("Standard.csv", "Standard"),
        ("Metric.csv", "Metric"), ("Dimension.csv", "Dimension"),
    ]:
        d = pd.read_csv(out / fname)
        node_ids.update(d[f":ID({label})"].astype(str))

    # USES_TABLE
    ut = pd.read_csv(out / "USES_TABLE.csv")
    assert len(ut) == 1
    assert str(ut[":START_ID(Metric)"].iloc[0]) in node_ids
    assert str(ut[":END_ID(PhysicalTable)"].iloc[0]) in node_ids

    # REFERS_COLUMN
    rc = pd.read_csv(out / "REFERS_COLUMN.csv")
    assert str(rc[":START_ID(Metric)"].iloc[0]) in node_ids
    assert str(rc[":END_ID(Col)"].iloc[0]) in node_ids

    # DIMENSION_USED
    du = pd.read_csv(out / "DIMENSION_USED.csv")
    assert str(du[":START_ID(Metric)"].iloc[0]) in node_ids
    assert str(du[":END_ID(Dimension)"].iloc[0]) in node_ids
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_meta_export_ids.py -v`
Expected: FAIL — 边生成仍用 `reset_index().rename({"index": ...})`，但 df 已无 `index` 列，KeyError 或产出空/错误边。

- [ ] **Step 3: 改边生成代码**

修改 `src/govio/cli/meta_export.py`。

(a) 替换 HAS_COLUMN 段（原 85-97 行）为：

```python
    # --- HAS_COLUMN edge ---
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
    df_has_column.to_csv(output / "HAS_COLUMN.csv", index=False)
    files.append("-r " + str(output / "HAS_COLUMN.csv"))
```

(b) 替换 USE 段（原 99-117 行）为：

```python
    # --- USE edge ---
    df_app_table = pd.merge(
        df_app_db_map,
        df_tables[["schema", "node_id"]].rename(
            columns={"node_id": ":END_ID(PhysicalTable)"}
        ),
        on="schema",
        how="inner",
    )
    df_use = pd.merge(
        df_apps[["name", "node_id"]].rename(
            columns={"node_id": ":START_ID(Application)"}
        ),
        df_app_table,
        on="name",
        how="inner",
    )[[":START_ID(Application)", ":END_ID(PhysicalTable)"]]
    df_use.to_csv(output / "USE.csv", index=False)
    files.append("-r " + str(output / "USE.csv"))
```

(c) 替换 metric 段（原 142-208 行）为：

```python
    # --- Optional: metrics ---
    metric_count = 0
    if metric_file:
        try:
            metric_loader = MetricLoader(metric_file, df_tables, df_columns)
            df_metrics = metric_loader.Metric.reset_index(drop=True)
            df_dimensions = metric_loader.Dimension.reset_index(drop=True)

            assign_node_ids(df_metrics, "Metric", "code")
            assign_node_ids(df_dimensions, "Dimension", "code")

            write_node_csv(df_metrics, output / "Metric.csv", "Metric")
            files.append("-n " + str(output / "Metric.csv"))

            write_node_csv(df_dimensions, output / "Dimension.csv", "Dimension")
            files.append("-n " + str(output / "Dimension.csv"))

            # positional index -> node_id 映射（MetricLoader 内部用 positional 索引）
            metric_idx_to_id = df_metrics["node_id"].tolist()
            dim_idx_to_id = df_dimensions["node_id"].tolist()
            table_idx_to_id = df_tables["node_id"].tolist()
            col_idx_to_id = df_columns["node_id"].tolist()

            # USES_TABLE 边
            uses_table = metric_loader.uses_table_edges.copy()
            if not uses_table.empty:
                uses_table[":START_ID(Metric)"] = [
                    metric_idx_to_id[i] for i in uses_table[":START_ID(Metric)"]
                ]
                uses_table[":END_ID(PhysicalTable)"] = [
                    table_idx_to_id[i] for i in uses_table[":END_ID(PhysicalTable)"]
                ]
                uses_table.to_csv(output / "USES_TABLE.csv", index=False)
                files.append("-r " + str(output / "USES_TABLE.csv"))

            # REFERS_COLUMN 边
            refers_col = metric_loader.refers_column_edges.copy()
            if not refers_col.empty:
                refers_col[":START_ID(Metric)"] = [
                    metric_idx_to_id[i] for i in refers_col[":START_ID(Metric)"]
                ]
                refers_col[":END_ID(Col)"] = [
                    col_idx_to_id[i] for i in refers_col[":END_ID(Col)"]
                ]
                refers_col.to_csv(output / "REFERS_COLUMN.csv", index=False)
                files.append("-r " + str(output / "REFERS_COLUMN.csv"))

            # DERIVED_FROM 边
            derived_from = metric_loader.derived_from_edges.copy()
            if not derived_from.empty:
                derived_from[":START_ID(Metric)"] = [
                    metric_idx_to_id[i] for i in derived_from[":START_ID(Metric)"]
                ]
                derived_from[":END_ID(Metric)"] = [
                    metric_idx_to_id[i] for i in derived_from[":END_ID(Metric)"]
                ]
                derived_from.to_csv(output / "DERIVED_FROM.csv", index=False)
                files.append("-r " + str(output / "DERIVED_FROM.csv"))

            # DIMENSION_USED 边
            dim_used = metric_loader.dimension_used_edges.copy()
            if not dim_used.empty:
                dim_used[":START_ID(Metric)"] = [
                    metric_idx_to_id[i] for i in dim_used[":START_ID(Metric)"]
                ]
                dim_used[":END_ID(Dimension)"] = [
                    dim_idx_to_id[i] for i in dim_used[":END_ID(Dimension)"]
                ]
                dim_used.to_csv(output / "DIMENSION_USED.csv", index=False)
                files.append("-r " + str(output / "DIMENSION_USED.csv"))

            # SUPERSEDES 边
            supersedes = metric_loader.supersedes_edges.copy()
            if not supersedes.empty:
                supersedes[":START_ID(Metric)"] = [
                    metric_idx_to_id[i] for i in supersedes[":START_ID(Metric)"]
                ]
                supersedes[":END_ID(Metric)"] = [
                    metric_idx_to_id[i] for i in supersedes[":END_ID(Metric)"]
                ]
                supersedes.to_csv(output / "SUPERSEDES.csv", index=False)
                files.append("-r " + str(output / "SUPERSEDES.csv"))

            print(
                f"成功生成指标数据：{len(df_metrics)} 个指标, "
                f"{len(df_dimensions)} 个维度"
            )
            metric_count = len(df_metrics)
        except Exception as e:
            print(f"警告: 无法加载指标定义文件: {e}")
            exit(1)
```

注意：RELATES_TO 段（原 119-140 行）`load_relationships` 返回的 `source`/`target` 列值来自 `relationship.py:167-168` `_get_table_id`，即 `df_tables[...].index[0]`。新方案下 df_tables 是 0..N-1 RangeIndex，故 `source`/`target` = positional index。需用 `table_idx_to_id` 映射成 node_id。

(d) RELATES_TO 段改为（替换原 119-140 行）：

```python
    # --- Optional: RELATES_TO ---
    relations_count = 0
    if relationship_file:
        try:
            df_relates_to = load_relationships(relationship_file, df_tables, df_columns)
            relations_count = len(df_relates_to)
            table_idx_to_id = df_tables["node_id"].tolist()
            if not df_relates_to.empty:
                df_relates_to["source"] = [
                    table_idx_to_id[i] for i in df_relates_to["source"]
                ]
                df_relates_to["target"] = [
                    table_idx_to_id[i] for i in df_relates_to["target"]
                ]
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
            print(f"成功生成 RELATES_TO.csv，包含 {len(df_relates_to)} 个关系 来自[{relationship_file}]")
        except Exception as e:
            print(f"警告: 无法加载关系文件: {e}")
```

`load_relationships` 返回的列名是 `source`/`target`（见 relationship.py:275-276 与 241-242），`to_csv` 的 `header=` 参数把它们重命名为 `:START_ID`/`:END_ID`，与原代码一致。

> 测试 `test_edge_csvs_reference_valid_node_ids` 不带 relationship_file，故 RELATES_TO 改动不阻塞该测试；改完后单独跑 `uv run pytest tests/test_relationship.py -v` 确认未破坏。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_meta_export_ids.py -v`
Expected: 3 passed

再跑关系测试确认未破坏：
Run: `uv run pytest tests/test_relationship.py tests/test_meta_export.py -v`
Expected: all passed

- [ ] **Step 5: commit**

```bash
git add src/govio/cli/meta_export.py tests/test_meta_export_ids.py
git commit -m "feat(meta-export): edge CSVs use string node IDs

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 5: utility.py make_csv 同步改 string ID

**Files:**
- Modify: `src/govio/metadata/utility.py:13-176`（`make_csv` 全函数）
- Test: `tests/test_meta_export_ids.py`（新增 utility 路径测试）

- [ ] **Step 1: 写失败测试**

在 `tests/test_meta_export_ids.py` 末尾追加：

```python
def test_make_csv_utility_path_uses_string_ids(tmp_path, monkeypatch):
    """老路径 utility.make_csv 也应产出 string ID 节点 CSV。"""
    from govio.metadata import utility

    monkeypatch.setattr(utility, "TDSLoader", lambda *a, **k: MagicMock(
        PhysicalTable=_mock_tds_tables(), Col=_mock_tds_columns()))
    monkeypatch.setattr(utility, "AppInfoLoader", lambda *a, **k: MagicMock(
        Application=_mock_apps()))
    monkeypatch.setattr(utility, "StandardLoader", lambda *a, **k: MagicMock(
        Standard=_mock_stds()))

    app_map = _mock_app_db_map()
    utility.make_csv(
        output=tmp_path, db="mysql://x", workspace_uuid="ws",
        app_list_file="app.json", df_app_db_map=app_map,
    )

    df = pd.read_csv(tmp_path / "PhysicalTable.csv")
    assert ":ID(PhysicalTable)" == df.columns[0]
    assert df[":ID(PhysicalTable)"].iloc[0].startswith("PT")
    assert len(df[":ID(PhysicalTable)"].iloc[0]) == 10

    has_col = pd.read_csv(tmp_path / "HAS_COLUMN.csv")
    assert ":START_ID(PhysicalTable)" in has_col.columns
    node_ids = set(df[":ID(PhysicalTable)"].astype(str))
    for v in has_col[":START_ID(PhysicalTable)"]:
        assert str(v) in node_ids
```

需要在文件顶部 import 增加 `from unittest.mock import MagicMock`。

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_meta_export_ids.py::test_make_csv_utility_path_uses_string_ids -v`
Expected: FAIL — utility.make_csv 仍用 `reorder_index` + 整数 `index_label`。

- [ ] **Step 3: 改 utility.py**

修改 `src/govio/metadata/utility.py`。

(a) 顶部 import 增加：

```python
from .node_id import assign_node_ids, write_node_csv
```

(b) `reorder_index` 函数（13-20 行）docstring 标记 deprecated：

```python
def reorder_index(dfs: list[pd.DataFrame], start: int = 1):
    """[Deprecated] 全局连续整数 ID。新代码应使用 node_id.assign_node_ids。

    保留仅为向后兼容，make_csv 已不再调用。
    """
    base_index: int = start
    for df in dfs:
        _end_index = base_index + df.shape[0]
        df["index"] = [i for i in range(base_index, _end_index)]
        df.set_index("index", drop=True, inplace=True)
        base_index = _end_index
```

(c) 替换 `make_csv` 中第 41-55 行（`reorder_index` + 4 节点 `to_csv`）为：

```python
    df_tables = df_tables.reset_index(drop=True)
    df_columns = df_columns.reset_index(drop=True)
    df_apps = df_apps.reset_index(drop=True)
    df_stds = df_stds.reset_index(drop=True)
    assign_node_ids(df_tables, "PhysicalTable", "full_table_name")
    assign_node_ids(df_columns, "Col", "column")
    assign_node_ids(df_apps, "Application", "app_id")
    assign_node_ids(df_stds, "Standard", "standard_id")

    files = []

    write_node_csv(df_tables, output / "PhysicalTable.csv", "PhysicalTable")
    files.append("-n " + str(output / "PhysicalTable.csv"))

    write_node_csv(df_columns, output / "Col.csv", "Col")
    files.append("-n " + str(output / "Col.csv"))

    write_node_csv(df_apps, output / "Application.csv", "Application")
    files.append("-n " + str(output / "Application.csv"))

    write_node_csv(df_stds, output / "Standard.csv", "Standard")
    files.append("-n " + str(output / "Standard.csv"))
```

(d) 替换 HAS_COLUMN 段（57-68 行）为：

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
    df_has_column.to_csv(output / "HAS_COLUMN.csv", index=False)
    files.append("-r " + str(output / "HAS_COLUMN.csv"))
```

(e) 替换 USE 段（70-88 行）为：

```python
    df_app_table = pd.merge(
        df_app_db_map,
        df_tables[["schema", "node_id"]].rename(
            columns={"node_id": ":END_ID(PhysicalTable)"}
        ),
        on="schema",
        how="inner",
    )
    df_use = pd.merge(
        df_apps[["name", "node_id"]].rename(
            columns={"node_id": ":START_ID(Application)"}
        ),
        df_app_table,
        on="name",
        how="inner",
    )[[":START_ID(Application)", ":END_ID(PhysicalTable)"]]
    df_use.to_csv(output / "USE.csv", index=False)
    files.append("-r " + str(output / "USE.csv"))
```

(f) 替换 RELATES_TO 段（90-108）：与 Task 4 Step 3(d) 相同——`load_relationships` 返回的 `source`/`target` 是 positional index（`relationship.py:167-168` 用 `df_tables.index[0]`），需用 `table_idx_to_id` 映射成 node_id。完整代码复用 Task 4 Step 3(d) 的 RELATES_TO 段。

(g) 替换 metric 段（110-172）为（与 Task 4 Step 3(c) 相同结构，删除 `metric_offset`/`dim_offset`/`reorder_index` 调用，改用 `assign_node_ids` + positional→node_id 映射）。完整代码与 Task 4 Step 3(c) 的 metric 段一致，仅文件路径与变量名上下文不同——直接复用。

(h) `data_standard_recommend` 函数（179-235 行）：其中第 221-222 行读回 `:ID(Col)` / `:ID(Standard)` 列——新 CSV 仍以 `:ID(Label)` 为首列，读取逻辑不变。但第 233-235 行的 rename 有预存 bug（`START_ID(Col)` 缺冒号）。修正为：

```python
        df_complies_with[[":ID(Col)", ":ID(Standard)"]].rename(
            columns={":ID(Col)": ":START_ID(Col)", ":ID(Standard)": ":END_ID(Standard)"}
        ).to_csv(output / "COMPLIES_WITH.csv", index=False)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_meta_export_ids.py -v`
Expected: 4 passed

跑全量回归：
Run: `uv run pytest tests/ -v --ignore=tests/test_mcp_database.py`
Expected: all passed（跳过需真实 DB 的 mcp_database 测试）

- [ ] **Step 5: commit**

```bash
git add src/govio/metadata/utility.py tests/test_meta_export_ids.py
git commit -m "feat(utility): make_csv uses string node IDs, fix COMPLIES_WITH header

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 6: CLI — --db-name 参数与签名

**Files:**
- Modify: `src/govio/cli/main.py:57-71, 98-99`

- [ ] **Step 1: 写失败测试**

在 `tests/test_meta_export_ids.py` 末尾追加：

```python
def test_main_requires_schemas_or_db_name(tmp_path, monkeypatch):
    """两者都不给应报错退出。"""
    import govio.cli.main as m
    monkeypatch.setattr(sys, "argv", [
        "govio-cli", "meta-export", "--db", "x.duckdb",
        "--output", str(tmp_path),
    ])
    with pytest.raises(SystemExit):
        m.main()


def test_main_db_name_unknown_exits(tmp_path, monkeypatch, _patched_loaders):
    """--db-name 不在 app_map 里应退出并列出可用 name。"""
    import govio.cli.main as m
    monkeypatch.setattr(sys, "argv", [
        "govio-cli", "meta-export", "--db", "x.duckdb",
        "--db-name", "nope", "--output", str(tmp_path),
    ])
    with pytest.raises(SystemExit):
        m.main()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_meta_export_ids.py::test_main_requires_schemas_or_db_name tests/test_meta_export_ids.py::test_main_db_name_unknown_exits -v`
Expected: FAIL — 当前 `--schemas` required，第一个测试在 argparse 层就退出但非 SystemExit 类型不匹配；且无 `--db-name`。

- [ ] **Step 3: 改 main.py**

修改 `src/govio/cli/main.py` 第 57-71 行：

```python
    # meta-export 子命令：从 DuckDB 导出元数据 CSV
    p_meta = sub.add_parser(
        "meta-export", help="从 DuckDB + TDS 合并导出元数据 CSV（支持单库模式）"
    )
    p_meta.add_argument("--db", type=str, required=True, help="DuckDB 数据库文件路径")
    p_meta.add_argument(
        "--schemas",
        type=str,
        help="要导出的 schema 列表，逗号分隔（如 dm,dwd,dws）；全量模式",
    )
    p_meta.add_argument(
        "--db-name",
        type=str,
        help="单库模式：按 app 名导出单个数据库的相关子图（不查 TDS）",
    )
    p_meta.add_argument("--output", type=Path, required=True, help="CSV 输出目录")
    p_meta.add_argument(
        "--dry-run",
        action="store_true",
        help="仅生成 CSV 并输出状态，不更新图数据和生成 assets",
    )
```

修改第 98-99 行分发逻辑：

```python
    elif args.action == "meta-export":
        schemas = args.schemas.split(",") if args.schemas else None
        meta_export(
            db_path=args.db,
            schemas=schemas,
            db_name=args.db_name,
            output=args.output,
            dry_run=args.dry_run,
        )
```

在 `meta_export` 函数内（`src/govio/cli/meta_export.py`）开头（第 29 行 `output.mkdir` 之后）加参数校验：

```python
    if not schemas and not db_name:
        print("错误: 必须指定 --schemas 或 --db-name 之一", file=sys.stderr)
        sys.exit(1)
```

并在解析 `df_app_db_map` 之后（约第 45 行后）加 `--db-name` 校验：

```python
    if db_name and db_name not in df_app_db_map["name"].values:
        print(
            f"错误: --db-name '{db_name}' 不在 app_map 中，可用: "
            f"{df_app_db_map['name'].tolist()}",
            file=sys.stderr,
        )
        sys.exit(1)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_meta_export_ids.py::test_main_requires_schemas_or_db_name tests/test_meta_export_ids.py::test_main_db_name_unknown_exits -v`
Expected: 2 passed

确认未破坏既有：
Run: `uv run pytest tests/test_meta_export.py tests/test_meta_export_integration.py -v`
Expected: all passed

- [ ] **Step 5: commit**

```bash
git add src/govio/cli/main.py src/govio/cli/meta_export.py tests/test_meta_export_ids.py
git commit -m "feat(cli): add --db-name to meta-export, make --schemas optional

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 7: 单库模式实现

**Files:**
- Modify: `src/govio/cli/meta_export.py:28-67`（加载与过滤分支）
- Test: `tests/test_meta_export_ids.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_meta_export_ids.py` 末尾追加：

```python
def test_single_db_mode_skips_tds(tmp_path):
    """--db-name 模式不查 TDS，只抽 DuckDB 该 schema。"""
    config = {
        "metadata": {
            "kundb": "mysql://x", "workspace_uuid": "ws",
            "app_list": "app.json", "app_map": "app_map.json",
            "relationship": None, "metric": None,
        },
        "graph": {},
    }
    with patch("govio.cli.meta_export.ConfigManager") as cfg_m, \
         patch("govio.cli.meta_export.TDSLoader") as tds_m, \
         patch("govio.cli.meta_export.DuckDBLoader") as duck_m, \
         patch("govio.cli.meta_export.AppInfoLoader") as app_m, \
         patch("govio.cli.meta_export.StandardLoader") as std_m:
        cfg_m.return_value.load.return_value = config
        # TDS 若被调用会返回这些——测试断言它不应被调用
        tds_m.return_value.PhysicalTable = _mock_tds_tables()
        tds_m.return_value.Col = _mock_tds_columns()
        # DuckDB 只返回 dm.orders（单库子集）
        duck_m.return_value.PhysicalTable = pd.DataFrame({
            "full_table_name": ["dm.orders"],
            "schema": ["dm"], "table_name": ["orders"],
            "name": ["Orders"], "data_entity_type": ["DUCKDB_TABLE"],
            "database_name": ["db"],
        })
        duck_m.return_value.Col = pd.DataFrame({
            "column": ["dm.orders.id", "dm.orders.amount"],
            "column_name": ["id", "amount"], "name": ["ID", "Amount"],
            "full_table_name": ["dm.orders", "dm.orders"],
            "data_entity_type": ["DUCKDB_COLUMN", "DUCKDB_COLUMN"],
            "dtype": ["int", "decimal"], "size": [0, 10],
            "precision": [0, 10], "scale": [0, 2], "order_no": [1, 2],
            "data_type": ["int", "decimal(10,2)"],
        })
        app_m.return_value.Application = _mock_apps()
        std_m.return_value.Standard = _mock_stds()

        from govio.cli.meta_export import meta_export
        out = tmp_path / "out"
        meta_export(db_path="ignored", schemas=None, db_name="billing",
                    output=out, dry_run=True)

    # TDSLoader 不应被实例化
    tds_m.assert_not_called()

    # 只导出 dm.orders 这张表
    tables = pd.read_csv(out / "PhysicalTable.csv")
    assert set(tables["full_table_name"]) == {"dm.orders"}
    cols = pd.read_csv(out / "Col.csv")
    assert set(cols["full_table_name"]) == {"dm.orders"}

    # Application 只剩 billing 一个
    apps = pd.read_csv(out / "Application.csv")
    assert len(apps) == 1
    assert apps["app_id"].iloc[0] == "app_billing"

    # USE 边只连 billing -> dm.orders
    use = pd.read_csv(out / "USE.csv")
    assert len(use) == 1


def test_single_db_with_schemas_intersection(tmp_path):
    """--db-name + --schemas 取交集：db-name 锁 dm，schemas 收窄到不存在的 schema 应空。"""
    config = {
        "metadata": {
            "kundb": "mysql://x", "workspace_uuid": "ws",
            "app_list": "app.json", "app_map": "app_map.json",
            "relationship": None, "metric": None,
        },
        "graph": {},
    }
    with patch("govio.cli.meta_export.ConfigManager") as cfg_m, \
         patch("govio.cli.meta_export.TDSLoader") as tds_m, \
         patch("govio.cli.meta_export.DuckDBLoader") as duck_m, \
         patch("govio.cli.meta_export.AppInfoLoader") as app_m, \
         patch("govio.cli.meta_export.StandardLoader") as std_m:
        cfg_m.return_value.load.return_value = config
        duck_m.return_value.PhysicalTable = _mock_tds_tables()
        duck_m.return_value.Col = _mock_tds_columns()
        app_m.return_value.Application = _mock_apps()
        std_m.return_value.Standard = _mock_stds()

        from govio.cli.meta_export import meta_export
        out = tmp_path / "out"
        # billing 对应 schema=dm，但 --schemas=dwd 与之无交集
        meta_export(db_path="ignored", schemas=["dwd"], db_name="billing",
                    output=out, dry_run=True)

    tables = pd.read_csv(out / "PhysicalTable.csv")
    assert len(tables) == 0  # 交集为空
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_meta_export_ids.py::test_single_db_mode_skips_tds tests/test_meta_export_ids.py::test_single_db_with_schemas_intersection -v`
Expected: FAIL — 当前 `meta_export` 仍调 TDSLoader，且 `db_name` 参数未用于过滤。

- [ ] **Step 3: 改 meta_export 加载段**

修改 `src/govio/cli/meta_export.py`。将第 47-65 行（TDS + DuckDB + merge + apps + stds 加载）替换为：

```python
    # --- 决定要抽取的 schema 集合 ---
    if db_name:
        # 单库模式：app_map 里该 app 对应的 schema
        app_schemas = df_app_db_map.loc[
            df_app_db_map["name"] == db_name, "schema"
        ].tolist()
        effective_schemas = (
            [s for s in app_schemas if s in schemas] if schemas else app_schemas
        )
    else:
        effective_schemas = schemas or []

    # --- Load DuckDB metadata ---
    duck_loader = DuckDBLoader(db_path, effective_schemas)
    duck_tables = duck_loader.PhysicalTable
    duck_columns = duck_loader.Col

    if db_name:
        # 单库模式：不查 TDS，DuckDB 直接作为最终元数据
        df_tables = duck_tables.reset_index(drop=True)
        df_columns = duck_columns.reset_index(drop=True)
    else:
        # 全量模式：TDS + DuckDB 合并
        tds_loader = TDSLoader(kundb, workspace_uuid, df_app_db_map["schema"].to_list())
        tds_tables = tds_loader.PhysicalTable
        tds_columns = tds_loader.Col
        df_tables = merge_metadata(tds_tables, duck_tables, "full_table_name")
        df_columns = merge_metadata(tds_columns, duck_columns, "column")

    # --- Load apps and standards ---
    app_loader = AppInfoLoader(app_list_file, df_app_db_map["name"].to_list())
    df_apps = app_loader.Application
    if db_name:
        # 单库模式：只保留该 app
        df_apps = df_apps[
            df_apps["name"] == df_app_db_map.loc[
                df_app_db_map["name"] == db_name, "name"
            ].iloc[0]
        ].reset_index(drop=True)
    std_loader = StandardLoader(kundb, workspace_uuid)
    df_stds = std_loader.Standard
```

注意：`effective_schemas` 在单库模式下传给 `DuckDBLoader`，确保只抽该 schema。全量模式下 `DuckDBLoader` 仍用 `effective_schemas`（= schemas 列表），与原行为一致。

metric 段无需改动——MetricLoader 接收已过滤的 `df_tables`/`df_columns`，自然只产出引用该 schema 表的 metric 子集（spec §3 第 6 点）。Standard 子集过滤：当前 `df_stds` 是全量，COMPLIES_WITH 边在 `data_standard_recommend` 路径才生成；`meta_export` 全量模式不生成 COMPLIES_WITH，单库模式同样不生成，故 Standard 节点保留全量不影响（spec §3 第 5 点指 COMPLIES_WITH 相关——确认 meta_export 不产出 COMPLIES_WITH，仅在 std-recommend 流程产出）。

> 若后续需在 meta_export 单库模式过滤 Standard 子集，单独追加任务；当前 spec 范围内 meta_export 不产 COMPLIES_WITH，保持全量 Standard 节点即可。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_meta_export_ids.py -v`
Expected: all passed

跑全量回归：
Run: `uv run pytest tests/ -v --ignore=tests/test_mcp_database.py`
Expected: all passed

- [ ] **Step 5: commit**

```bash
git add src/govio/cli/meta_export.py tests/test_meta_export_ids.py
git commit -m "feat(meta-export): single-db mode via --db-name, skip TDS

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 8: 版本 bump 与 skill 文档同步

**Files:**
- Modify: `pyproject.toml`
- Modify: `skills/govio/govio-metadata/SKILL.md`（若提及 meta-export / ID）
- Modify: `skills/govio/SKILL.md`（若提及图模型 ID）

- [ ] **Step 1: bump 版本**

修改 `pyproject.toml`，将 `version = "0.2.12"` 改为 `version = "0.3.0"`。

- [ ] **Step 2: 检查 skill 文档是否提及 meta-export 或 ID 格式**

Run:
```bash
uv run python -c "import pathlib; [print(p) for p in pathlib.Path('skills').rglob('*.md') if 'meta-export' in p.read_text() or ':ID(' in p.read_text() or 'reorder_index' in p.read_text()]"
```

- 若有命中：在相应文件补充：
  - 节点 ID 现为 10 位 string（`<2 字符前缀><8 hex>`），不再是全局连续整数。
  - `meta-export` 新增 `--db-name <app_name>` 单库模式（不查 TDS，导出该数据库相关子图）；`--schemas` 改为可选。
  - 破坏性：旧图库需 drop 后重新 `meta-export` 导入。
- 若无命中：跳过 skill 修改（已确认 `skills/` 下无 meta-export / `:ID(` / `reorder_index` 描述，预期无命中）。

- [ ] **Step 3: 跑全量测试确认无回归**

Run: `uv run pytest tests/ -v --ignore=tests/test_mcp_database.py`
Expected: all passed

- [ ] **Step 4: 手动 dry-run 验证（若环境允许）**

```bash
uv run govio-cli meta-export --help
# 确认 --db-name 出现在 help 输出
```

- [ ] **Step 5: commit**

```bash
git add pyproject.toml skills/
git commit -m "chore: bump version to 0.3.0 for string ID breaking change

BREAKING CHANGE: node IDs change from global sequential integers to 10-char
strings (<2-char type prefix><8 hex of SHA256(business key)>). Existing
graphs must be dropped and re-imported via meta-export.

Add --db-name single-db mode to meta-export (skips TDS, exports one db's
subgraph). --schemas is now optional.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## 完成标准

- [ ] `uv run pytest tests/ --ignore=tests/test_mcp_database.py` 全绿
- [ ] `uv run govio-cli meta-export --help` 显示 `--db-name`
- [ ] 节点 CSV 首列为 `:ID(Label)`，值为 10 位 string
- [ ] 边 CSV 的 `:START_ID`/`:END_ID` 值能在对应节点 CSV 找到
- [ ] `--db-name` 模式不调用 TDSLoader
- [ ] 版本 0.3.0，commit 含 BREAKING CHANGE 说明
