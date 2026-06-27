"""meta_export string ID 集成测试。mock 全部 Loader，跑 dry-run 检查 CSV。"""
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
         patch("govio.cli.meta_export.pd.read_json") as read_json_m, \
         patch("govio.cli.meta_export.TDSLoader") as tds_m, \
         patch("govio.cli.meta_export.DuckDBLoader") as duck_m, \
         patch("govio.cli.meta_export.AppInfoLoader") as app_m, \
         patch("govio.cli.meta_export.StandardLoader") as std_m:
        cfg_m.return_value.load.return_value = config
        read_json_m.return_value = _mock_app_db_map()
        tds_m.return_value.PhysicalTable = _mock_tds_tables()
        tds_m.return_value.Col = _mock_tds_columns()
        duck_m.return_value.PhysicalTable = _mock_duck_tables()
        duck_m.return_value.Col = _mock_duck_columns()
        app_m.return_value.Application = _mock_apps()
        std_m.return_value.Standard = _mock_stds()
        yield


def test_node_csvs_have_string_ids(_patched_loaders, tmp_path):
    from govio.cli.meta_export import meta_export
    # Edges not yet refactored (Task 4); node CSVs are written before edge generation.
    try:
        meta_export(db_path="ignored", schemas=["dm"], db_name=None,
                    output=tmp_path, dry_run=True)
    except Exception:
        pass

    for fname, prefix, label in [
        ("PhysicalTable.csv", "PT", "PhysicalTable"),
        ("Col.csv", "CO", "Col"),
        ("Application.csv", "AP", "Application"),
        ("Standard.csv", "ST", "Standard"),
    ]:
        df = pd.read_csv(tmp_path / fname)
        id_col = f":ID({label})"
        assert id_col == df.columns[0], f"{fname} 第一列应为 {id_col}, 实际 {df.columns[0]}"
        for v in df[id_col]:
            assert len(str(v)) == 10, f"{fname} ID 长度应为 10: {v}"
            assert str(v).startswith(prefix), f"{fname} ID 前缀应为 {prefix}: {v}"
