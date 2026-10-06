"""step 函数 string ID 集成测试。mock 全部 Loader，跑 dry-run 检查 CSV。"""
import json
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
    return pd.DataFrame({
        "full_table_name": ["dm.orders", "dm.customers"],
        "schema": ["dm", "dm"],
        "table_name": ["orders", "customers"],
        "name": ["Orders", "Customers"],
        "data_entity_type": ["DUCKDB_TABLE", "DUCKDB_TABLE"],
        "database_name": ["db", "db"],
    })


def _mock_duck_columns():
    return pd.DataFrame({
        "column": ["dm.orders.id", "dm.orders.amount", "dm.customers.id"],
        "column_name": ["id", "amount", "id"],
        "name": ["ID", "Amount", "ID"],
        "full_table_name": ["dm.orders", "dm.orders", "dm.customers"],
        "data_entity_type": ["DUCKDB_COLUMN"] * 3,
        "dtype": ["int", "decimal", "int"],
        "size": [0, 10, 0],
        "precision": [0, 10, 0],
        "scale": [0, 2, 0],
        "order_no": [1, 2, 1],
        "data_type": ["int", "decimal(10,2)", "int"],
    })


def _mock_stds():
    return pd.DataFrame({
        "standard_id": ["std_amount"],
        "name": ["Amount Standard"],
        "data_type": ["decimal"],
    })


def _write_datasources_file(tmp_path, name="billing", schemas=("dm",)):
    """构造 datasource 声明文件（TDS 模式与 make_csv 路径需要）。"""
    data = {
        "version": "1.0",
        "datasources": [
            {
                "datasource_name": name,
                "name": name,
                "source_type": "mysql",
                "filter": {"schemas": list(schemas)},
            }
        ],
    }
    path = tmp_path / "datasource.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return str(path)


def _run_steps(
    output,
    source="duckdb",
    db_path="ignored",
    schemas=None,
    datasource_name="billing",
    datasource_file="",
    kundb="mysql://x",
    workspace_uuid="ws",
    relationship_file=None,
    metric_file=None,
):
    """测试辅助：按顺序执行 step 函数。"""
    from govio.cli.meta import (
        step_meta_export, step_std_export,
        step_compliance_export, step_rel_export, step_metric_export,
    )

    result = step_meta_export(
        output, source=source, db_path=db_path,
        schemas=schemas, datasource_name=datasource_name,
        datasource_file=datasource_file,
        kundb=kundb, workspace_uuid=workspace_uuid,
    )
    if result is None:
        return

    if source != "duckdb":
        step_std_export(output, kundb, workspace_uuid)
        step_compliance_export(output, kundb, workspace_uuid, datasource_file)

    if relationship_file:
        step_rel_export(output, relationship_file)

    if metric_file:
        step_metric_export(output, metric_file)


# ---------------------------------------------------------------------------
# Node CSV tests
# ---------------------------------------------------------------------------

@pytest.fixture
def _patched_loaders(tmp_path):
    """Patch all loaders so step functions run without DB/config.

    Yields:
        str: datasource 声明文件路径（TDS 模式需要）
    """
    with patch("govio.cli.meta.TDSLoader") as tds_m, \
         patch("govio.cli.meta.DuckDBLoader") as duck_m, \
         patch("govio.cli.meta.StandardLoader") as std_m:
        tds_m.return_value.PhysicalTable = _mock_tds_tables()
        tds_m.return_value.Col = _mock_tds_columns()
        duck_m.return_value.PhysicalTable = _mock_duck_tables()
        duck_m.return_value.Col = _mock_duck_columns()
        std_m.return_value.Standard = _mock_stds()
        std_m.return_value.StdCompliance = pd.DataFrame(columns=["column", "standard_id"])
        yield _write_datasources_file(tmp_path)


def test_node_csvs_have_string_ids(_patched_loaders, tmp_path):
    _run_steps(
        output=tmp_path, source="tds", datasource_file=_patched_loaders,
    )

    for fname, prefix, label in [
        ("PhysicalTable.csv", "PT", "PhysicalTable"),
        ("Col.csv", "CO", "Col"),
        ("Datasource.csv", "DS", "Datasource"),
        ("Standard.csv", "ST", "Standard"),
    ]:
        df = pd.read_csv(tmp_path / fname)
        id_col = f":ID({label})"
        assert id_col == df.columns[0], f"{fname} 第一列应为 {id_col}, 实际 {df.columns[0]}"
        for v in df[id_col]:
            assert len(str(v)) == 10, f"{fname} ID 长度应为 10: {v}"
            assert str(v).startswith(prefix), f"{fname} ID 前缀应为 {prefix}: {v}"


def test_edge_csvs_reference_valid_node_ids(_patched_loaders, tmp_path):
    _run_steps(
        output=tmp_path, source="tds", datasource_file=_patched_loaders,
    )

    # 收集所有节点 ID
    node_ids: set[str] = set()
    for fname, label in [
        ("PhysicalTable.csv", "PhysicalTable"),
        ("Col.csv", "Col"),
        ("Datasource.csv", "Datasource"),
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

    # OWNS
    owns = pd.read_csv(tmp_path / "OWNS.csv")
    for v in owns[":START_ID(Datasource)"]:
        assert str(v) in node_ids
    for v in owns[":END_ID(PhysicalTable)"]:
        assert str(v) in node_ids

    assert len(has_col) == 3
    assert len(owns) == 2


# ---------------------------------------------------------------------------
# Metric tests
# ---------------------------------------------------------------------------

def test_metric_edges_use_string_ids(tmp_path):
    """带 metric 的全量导出：metric/dim 节点与 5 类边都是 string ID。"""
    metric_data = {
        "version": "1.0",
        "metrics": [
            {
                "code": "m_total_amount",
                "name": "Total Amount",
                "business_definition": "总金额",
                "type": "atomic",
                "unit": "元",
                "data_type": "decimal",
                "source_layer": "DM",
                "source_tables": [
                    {"full_table_name": "dm.orders", "columns": [
                        {"column_name": "amount", "role": "measure"}
                    ]}
                ],
                "dimensions": [{"code": "dim_time", "usage_type": "group"}],
            }
        ],
        "shared_dimensions": [
            {"code": "dim_time", "name": "Time", "granularity": "day"}
        ],
    }
    metric_file = tmp_path / "metric.json"
    metric_file.write_text(json.dumps(metric_data, ensure_ascii=False), encoding="utf-8")

    with patch("govio.cli.meta.TDSLoader") as tds_m, \
         patch("govio.cli.meta.DuckDBLoader") as duck_m, \
         patch("govio.cli.meta.StandardLoader") as std_m:
        tds_m.return_value.PhysicalTable = _mock_tds_tables()
        tds_m.return_value.Col = _mock_tds_columns()
        duck_m.return_value.PhysicalTable = _mock_duck_tables()
        duck_m.return_value.Col = _mock_duck_columns()
        std_m.return_value.Standard = _mock_stds()
        std_m.return_value.StdCompliance = pd.DataFrame(columns=["column", "standard_id"])

        out = tmp_path / "out"
        _run_steps(
            output=out, source="duckdb", db_path="ignored", schemas=["dm"],
            metric_file=str(metric_file),
        )

    # Metric / Dimension 节点
    m_df = pd.read_csv(out / "Metric.csv")
    assert ":ID(Metric)" == m_df.columns[0]
    assert m_df[":ID(Metric)"].iloc[0].startswith("ME")
    d_df = pd.read_csv(out / "Dimension.csv")
    assert d_df[":ID(Dimension)"].iloc[0].startswith("DI")

    node_ids = set()
    for fname, label in [
        ("PhysicalTable.csv", "PhysicalTable"), ("Col.csv", "Col"),
        ("Datasource.csv", "Datasource"),
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


# ---------------------------------------------------------------------------
# Utility path tests
# ---------------------------------------------------------------------------

def test_make_csv_utility_path_uses_string_ids(tmp_path, monkeypatch):
    """老路径 utility.make_csv 也应产出 string ID 节点 CSV。"""
    from govio.metadata import utility

    monkeypatch.setattr(utility, "TDSLoader", lambda *a, **k: MagicMock(
        PhysicalTable=_mock_tds_tables(), Col=_mock_tds_columns()))
    monkeypatch.setattr(utility, "StandardLoader", lambda *a, **k: MagicMock(
        Standard=_mock_stds()))

    ds_file = _write_datasources_file(tmp_path)
    utility.make_csv(
        output=tmp_path, db="mysql://x", workspace_uuid="ws",
        datasource_file=ds_file,
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

    ds_df = pd.read_csv(tmp_path / "Datasource.csv")
    assert ds_df[":ID(Datasource)"].iloc[0].startswith("DS")
    assert ds_df["datasource_name"].iloc[0] == "billing"

    owns = pd.read_csv(tmp_path / "OWNS.csv")
    assert len(owns) == 2
    ds_ids = set(ds_df[":ID(Datasource)"].astype(str))
    for v in owns[":START_ID(Datasource)"]:
        assert str(v) in ds_ids


# ---------------------------------------------------------------------------
# CLI arg validation tests
# ---------------------------------------------------------------------------

def test_meta_requires_source(tmp_path, monkeypatch, capsys):
    """meta meta 不传 --source 应报错。"""
    from govio.cli import main
    monkeypatch.setattr(sys, "argv", [
        "govio-cli", "meta", "meta",
        "--output", str(tmp_path),
    ])
    with pytest.raises(SystemExit):
        main()


def test_meta_duckdb_requires_db(tmp_path, monkeypatch, capsys):
    """meta meta --source duckdb 不传 --db 应在 step 中报错退出。"""
    from govio.cli import main
    monkeypatch.setattr(sys, "argv", [
        "govio-cli", "meta", "meta",
        "--source", "duckdb",
        "--output", str(tmp_path),
    ])
    with pytest.raises(SystemExit):
        main()
    err = capsys.readouterr().err
    assert "DuckDB" in err or "需要指定" in err


# ---------------------------------------------------------------------------
# Data source mode tests
# ---------------------------------------------------------------------------

def test_duckdb_skips_tds_and_std(tmp_path):
    """DuckDB 模式不调用 TDSLoader 和 StandardLoader，自动声明 Datasource。"""
    with patch("govio.cli.meta.TDSLoader") as tds_m, \
         patch("govio.cli.meta.DuckDBLoader") as duck_m, \
         patch("govio.cli.meta.StandardLoader") as std_m:
        tds_m.return_value.PhysicalTable = _mock_tds_tables()
        tds_m.return_value.Col = _mock_tds_columns()
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
        std_m.return_value.Standard = _mock_stds()
        std_m.return_value.StdCompliance = pd.DataFrame(columns=["column", "standard_id"])

        out = tmp_path / "out"
        _run_steps(
            output=out, source="duckdb", db_path="ignored",
            schemas=["dm"],
        )

    # TDSLoader 不应被实例化
    tds_m.assert_not_called()
    # DuckDB 模式跳过 Standard 数据标准读取
    std_m.assert_not_called()

    # 只导出 dm.orders
    tables = pd.read_csv(out / "PhysicalTable.csv")
    assert set(tables["full_table_name"]) == {"dm.orders"}

    # Datasource 自动声明（filter.schemas 取 --schemas）
    ds_df = pd.read_csv(out / "Datasource.csv")
    assert len(ds_df) == 1
    assert ds_df["datasource_name"].iloc[0] == "billing"
    assert ds_df["source_type"].iloc[0] == "duckdb"
    assert json.loads(ds_df["filter"].iloc[0]) == {"schemas": ["dm"]}

    # OWNS 边生成
    owns = pd.read_csv(out / "OWNS.csv")
    assert len(owns) == 1

    # Standard.csv 不应存在（duckdb 模式跳过）
    assert not (out / "Standard.csv").exists()
