"""type_parser 共享实现单测 — 方言类型解析收敛后的等价性基线

用例矩阵与期望值取自重构前三份 ``_parse_type``（mysql_loader / postgres_loader /
trino_loader）对同一批代表性输入的逐例输出快照（见
docs/tasks/phase3-slim/progress.md 第 2 项），收敛为共享实现后必须逐例一致。
"""

import pytest

from govio.metadata.type_parser import parse_type_attributes, parse_type_string

# (类型串, Trino 期望 (size, precision, scale), MySQL 期望)
# MySQL 与 Trino 的唯一差异是 unsigned / zerofill 后缀剥离
STRING_CASES = [
    ("", (0, 0, 0), (0, 0, 0)),
    ("   ", (0, 0, 0), (0, 0, 0)),
    ("  varchar(64)  ", (64, 0, 0), (64, 0, 0)),
    ("-varchar(1)", (0, 0, 0), (0, 0, 0)),
    ("TIMESTAMP", (0, 0, 0), (0, 0, 0)),
    ("bigint", (0, 0, 0), (0, 0, 0)),
    ("bigint(20) UNSIGNED ZEROFILL", (0, 0, 0), (0, 0, 0)),
    ("bit(8)", (0, 0, 0), (0, 0, 0)),
    ("char(2)", (2, 0, 0), (2, 0, 0)),
    ("decimal", (0, 0, 0), (0, 0, 0)),
    ("decimal(10)", (0, 10, 0), (0, 10, 0)),
    ("decimal(10, 2)", (0, 10, 2), (0, 10, 2)),
    ("decimal(10,2)", (0, 10, 2), (0, 10, 2)),
    ("decimal(10,2) unsigned", (0, 0, 0), (0, 10, 2)),
    ("decimal(10,2) unsigned zerofill", (0, 0, 0), (0, 10, 2)),
    ("decimal(a,b)", (0, 0, 0), (0, 0, 0)),
    ("double precision", (0, 0, 0), (0, 0, 0)),
    ("double(10,2)", (0, 0, 0), (0, 0, 0)),
    ("enum('a','b')", (0, 0, 0), (0, 0, 0)),
    ("float(5,3)", (0, 0, 0), (0, 0, 0)),
    ("int", (0, 0, 0), (0, 0, 0)),
    ("int(10) unsigned", (0, 0, 0), (0, 0, 0)),
    ("int(11) zerofill", (0, 0, 0), (0, 0, 0)),
    ("json", (0, 0, 0), (0, 0, 0)),
    ("numeric(12,4)", (0, 0, 0), (0, 0, 0)),
    ("text", (0, 0, 0), (0, 0, 0)),
    ("varchar()", (0, 0, 0), (0, 0, 0)),
    ("varchar(64", (0, 0, 0), (0, 0, 0)),
    ("varchar(64)", (64, 0, 0), (64, 0, 0)),
    ("varchar(64) unsigned", (0, 0, 0), (64, 0, 0)),
]

# (data_type, character_maximum_length, numeric_precision, numeric_scale, 期望)
ATTRIBUTE_CASES = [
    ("character varying", 10, None, None, (10, 0, 0)),
    ("character", 1, None, None, (1, 0, 0)),
    ("bit", 8, None, None, (8, 0, 0)),
    ("bit varying", 8, None, None, (8, 0, 0)),
    ("numeric", None, 8, 2, (0, 8, 2)),
    ("decimal", None, 10, 0, (0, 10, 0)),
    ("numeric", None, None, None, (0, 0, 0)),
    ("integer", None, 32, 0, (0, 0, 0)),
    ("bigint", None, 64, 0, (0, 0, 0)),
    ("text", None, None, None, (0, 0, 0)),
    ("  CHARACTER VARYING  ", 255, None, None, (255, 0, 0)),
    ("character varying", float("nan"), None, None, (0, 0, 0)),
    ("numeric", None, float("nan"), float("nan"), (0, 0, 0)),
    ("character varying", 10.0, 3.0, 1.0, (10, 0, 0)),
]


@pytest.mark.parametrize("type_str, trino_expected, mysql_expected", STRING_CASES)
def test_parse_type_string(type_str, trino_expected, mysql_expected):
    """串式类型解析：与重构前 Trino / MySQL 两份实现逐例一致"""
    assert parse_type_string(type_str) == trino_expected
    assert parse_type_string(type_str, strip_numeric_modifiers=True) == mysql_expected


@pytest.mark.parametrize(
    "data_type, char_len, num_prec, num_scale, expected", ATTRIBUTE_CASES
)
def test_parse_type_attributes(data_type, char_len, num_prec, num_scale, expected):
    """属性式类型解析：与重构前 Postgres 实现逐例一致"""
    assert parse_type_attributes(data_type, char_len, num_prec, num_scale) == expected


def test_illegal_type_string_falls_back_to_zero():
    """非法 / 不可解析的类型串一律回退 (0, 0, 0)，不抛异常"""
    for bad in [
        "",
        "   ",
        "varchar(64",
        "decimal(a,b)",
        "double precision",
        "enum('a','b')",
        "-varchar(1)",
        "varchar()",
    ]:
        assert parse_type_string(bad) == (0, 0, 0)


def test_missing_numeric_attributes_treated_as_zero():
    """属性列缺失（None / NaN）按 0 处理，不抛 ValueError"""
    assert parse_type_attributes("numeric", None, None, None) == (0, 0, 0)
    assert parse_type_attributes("numeric", None, float("nan"), 2) == (0, 0, 2)
    assert parse_type_attributes("character varying", float("nan"), None, None) == (
        0,
        0,
        0,
    )


def test_integer_precision_is_not_reported_as_decimal_precision():
    """整型的 numeric_precision 是位宽语义，不计入 precision"""
    assert parse_type_attributes("integer", None, 32, 0) == (0, 0, 0)
    assert parse_type_attributes("bigint", None, 64, 0) == (0, 0, 0)
