"""meta_export string ID 集成测试。mock 全部 Loader，跑 dry-run 检查 CSV。"""
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
    meta_export(db_path="ignored", schemas=["dm"], db_name=None,
                output=tmp_path, dry_run=True)

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
         patch("govio.cli.meta_export.StandardLoader") as std_m, \
         patch("govio.cli.meta_export.pd.read_json") as read_json_m:
        cfg_m.return_value.load.return_value = config
        tds_m.return_value.PhysicalTable = _mock_tds_tables()
        tds_m.return_value.Col = _mock_tds_columns()
        duck_m.return_value.PhysicalTable = _mock_duck_tables()
        duck_m.return_value.Col = _mock_duck_columns()
        app_m.return_value.Application = _mock_apps()
        std_m.return_value.Standard = _mock_stds()
        read_json_m.return_value = _mock_app_db_map()

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


def test_make_csv_utility_path_uses_string_ids(tmp_path, monkeypatch):
    """老路径 utility.make_csv 也应产出 string ID 节点 CSV。"""
    from unittest.mock import MagicMock
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


def test_main_requires_schemas_or_db_name(tmp_path, monkeypatch, capsys):
    """两者都不给应报错退出。"""
    from govio.cli import main
    monkeypatch.setattr(sys, "argv", [
        "govio-cli", "meta-export", "--db", "x.duckdb",
        "--output", str(tmp_path),
    ])
    with pytest.raises(SystemExit):
        main()
    err = capsys.readouterr().err
    assert "必须指定 --schemas 或 --db-name" in err


def test_main_db_name_unknown_exits(tmp_path, monkeypatch, capsys):
    """--db-name 不在 app_map 里应退出并列出可用 name。"""
    from govio.cli import main

    config = {
        "metadata": {
            "kundb": "mysql://x", "workspace_uuid": "ws",
            "app_list": "app.json", "app_map": "app_map.json",
            "relationship": None, "metric": None,
        },
        "graph": {},
    }
    with patch("govio.cli.meta_export.ConfigManager") as cfg_m, \
         patch("govio.cli.meta_export.pd.read_json") as read_json_m:
        cfg_m.return_value.load.return_value = config
        read_json_m.return_value = _mock_app_db_map()
        monkeypatch.setattr(sys, "argv", [
            "govio-cli", "meta-export", "--db", "x.duckdb",
            "--db-name", "nope", "--output", str(tmp_path),
        ])
        with pytest.raises(SystemExit):
            main()
    err = capsys.readouterr().err
    assert "nope" in err
    assert "billing" in err


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
         patch("govio.cli.meta_export.StandardLoader") as std_m, \
         patch("govio.cli.meta_export.pd.read_json") as read_json_m:
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
        read_json_m.return_value = _mock_app_db_map()

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

    def _mock_duck(*args, **kwargs):
        schemas = kwargs.get("schemas") or (args[1] if len(args) > 1 else [])
        if not schemas:
            return MagicMock(
                PhysicalTable=pd.DataFrame(columns=_mock_tds_tables().columns),
                Col=pd.DataFrame(columns=_mock_tds_columns().columns),
            )
        return MagicMock(
            PhysicalTable=_mock_tds_tables(),
            Col=_mock_tds_columns(),
        )

    with patch("govio.cli.meta_export.ConfigManager") as cfg_m, \
         patch("govio.cli.meta_export.TDSLoader") as tds_m, \
         patch("govio.cli.meta_export.DuckDBLoader") as duck_m, \
         patch("govio.cli.meta_export.AppInfoLoader") as app_m, \
         patch("govio.cli.meta_export.StandardLoader") as std_m, \
         patch("govio.cli.meta_export.pd.read_json") as read_json_m:
        cfg_m.return_value.load.return_value = config
        duck_m.side_effect = _mock_duck
        app_m.return_value.Application = _mock_apps()
        std_m.return_value.Standard = _mock_stds()
        read_json_m.return_value = _mock_app_db_map()

        from govio.cli.meta_export import meta_export
        out = tmp_path / "out"
        # billing 对应 schema=dm，但 --schemas=dwd 与之无交集
        meta_export(db_path="ignored", schemas=["dwd"], db_name="billing",
                    output=out, dry_run=True)

    tables = pd.read_csv(out / "PhysicalTable.csv")
    assert len(tables) == 0  # 交集为空
