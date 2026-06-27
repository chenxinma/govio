import pandas as pd
import pytest
from govio.metadata.node_id import (
    assign_node_ids,
    make_id,
    write_node_csv,
)


def test_make_id_format():
    """ID = 2 字符前缀 + 8 hex，共 10 位。"""
    import re
    node_id = make_id("PhysicalTable", "dm.orders")
    assert len(node_id) == 10
    assert node_id.startswith("PT")
    assert re.fullmatch(r"[0-9A-F]{8}", node_id[2:])  # 8 个大写 hex


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
