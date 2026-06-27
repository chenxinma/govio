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
