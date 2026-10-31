"""MysqlLoader 真实 MySQL（mysql:8）集成验证

默认跳过；设置 ``GOVIO_REAL_DB=1`` 后连接真实实例执行
（默认 localhost:13306，可用 GOVIO_MYSQL_HOST / GOVIO_MYSQL_PORT /
GOVIO_MYSQL_USER / GOVIO_MYSQL_PASSWORD 覆盖）。

样例库 ``govio_it`` 由本文件自行建删（幂等），覆盖：表/列注释、
decimal unsigned、int unsigned zerofill、varchar(0)、一张视图（BASE TABLE
过滤）、一个删过列的表（order_no 语义）。容器的启动/清理见
docs/tasks/V4-mysql-verify/progress.md。
"""

import os

import pymysql
import pytest

from govio.metadata.mysql_loader import MysqlLoader, _parse_type

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("GOVIO_REAL_DB") != "1",
        reason="GOVIO_REAL_DB=1 才执行真实库集成测试",
    ),
]

_SAMPLE_SCHEMA = "govio_it"

# 样例库 DDL：表/列注释、decimal unsigned、int unsigned zerofill、
# varchar(0)、视图、删过列的表
_SAMPLE_DDL = [
    "DROP DATABASE IF EXISTS govio_it",
    "CREATE DATABASE govio_it CHARACTER SET utf8mb4",
    "CREATE TABLE govio_it.users ("
    " id BIGINT PRIMARY KEY COMMENT '用户ID',"
    " name VARCHAR(64) NOT NULL COMMENT '用户名',"
    " email VARCHAR(128)) COMMENT='用户表'",
    "CREATE TABLE govio_it.orders ("
    " order_id BIGINT PRIMARY KEY,"
    " amount DECIMAL(10,2) UNSIGNED COMMENT '金额',"
    " qty INT(10) UNSIGNED ZEROFILL,"
    " note VARCHAR(255),"
    " marker VARCHAR(0) COMMENT '零长度 varchar')",
    "CREATE TABLE govio_it.audit_log ("
    " log_id BIGINT PRIMARY KEY COMMENT '日志ID',"
    " old_col VARCHAR(20) COMMENT '将被删除的列',"
    " action VARCHAR(32) COMMENT '动作') COMMENT='审计表'",
    "ALTER TABLE govio_it.audit_log DROP COLUMN old_col",
    "CREATE VIEW govio_it.user_view AS"
    " SELECT id, CONCAT(name, '!') AS label FROM govio_it.users",
]


def _conn_params() -> dict:
    """连接参数（可用环境变量覆盖默认值）。"""
    return {
        "host": os.getenv("GOVIO_MYSQL_HOST", "localhost"),
        "port": int(os.getenv("GOVIO_MYSQL_PORT", "13306")),
        "user": os.getenv("GOVIO_MYSQL_USER", "root"),
        "password": os.getenv("GOVIO_MYSQL_PASSWORD", "govio123"),
    }


@pytest.fixture(scope="module")
def sample_schema():
    """重建样例库 govio_it（幂等），结束后不断开服务。"""
    conn = pymysql.connect(charset="utf8mb4", **_conn_params())
    try:
        cur = conn.cursor()
        for stmt in _SAMPLE_DDL:
            cur.execute(stmt)
        conn.commit()
    finally:
        conn.close()
    return _SAMPLE_SCHEMA


@pytest.fixture(scope="module")
def loader(sample_schema):
    """连接样例库的 MysqlLoader。"""
    return MysqlLoader(
        schemas=[sample_schema],
        datasource_name="",
        **_conn_params(),
    )


def test_real_mysql_load_tables_contract(loader):
    """load_tables 列契约、表注释回退与 BASE TABLE 过滤。"""
    df = loader.load_tables()
    assert list(df.columns) == [
        "full_table_name",
        "schema",
        "table_name",
        "name",
        "data_entity_type",
        "database_name",
    ]
    assert set(df["table_name"]) == {"users", "orders", "audit_log"}
    assert "user_view" not in set(df["table_name"])
    assert all(df["data_entity_type"] == "MYSQL_TABLE")
    names = dict(zip(df["table_name"], df["name"], strict=True))
    assert names["users"] == "用户表"
    assert names["audit_log"] == "审计表"
    # 空表注释回退为表名
    assert names["orders"] == "orders"


def test_real_mysql_load_columns_contract(loader):
    """load_columns 列契约、列注释回退与 dtype 保留原始 COLUMN_TYPE。"""
    df = loader.load_columns()
    assert list(df.columns) == [
        "column",
        "column_name",
        "name",
        "full_table_name",
        "data_entity_type",
        "dtype",
        "size",
        "precision",
        "scale",
        "order_no",
        "data_type",
    ]
    assert all(df["data_entity_type"] == "MYSQL_COLUMN")
    users = df[df["full_table_name"] == "govio_it.users"].set_index("column_name")
    assert users.loc["id", "name"] == "用户ID"
    # 空列注释回退为列名
    assert users.loc["email", "name"] == "email"
    orders = df[df["full_table_name"] == "govio_it.orders"].set_index("column_name")
    # dtype / data_type 保留 unsigned 后缀原文
    assert orders.loc["amount", "dtype"] == "decimal(10,2) unsigned"
    assert orders.loc["amount", "data_type"] == "decimal(10,2) unsigned"


def test_real_mysql_type_parsing(loader):
    """真实 COLUMN_TYPE 解析：decimal unsigned 不丢 precision/scale。"""
    df = loader.load_columns().set_index("column_name")
    amount = df.loc["amount"]
    assert (amount["size"], amount["precision"], amount["scale"]) == (0, 10, 2)
    qty = df.loc["qty"]
    assert (qty["size"], qty["precision"], qty["scale"]) == (0, 0, 0)
    name = df.loc["name"]
    assert (name["size"], name["precision"], name["scale"]) == (64, 0, 0)
    # varchar(0)（DDL 可表达的唯一“无长度”形态）不报错且 size=0
    marker = df.loc["marker"]
    assert (marker["size"], marker["precision"], marker["scale"]) == (0, 0, 0)
    # mysql:8 的视图/表达式列均解析出具体长度，裸 varchar 不可达，
    # 此处钉住防御分支（单测见 tests/test_mysql_loader.py::test_parse_type）
    assert _parse_type("varchar") == (0, 0, 0)


def test_real_mysql_excludes_view_columns(loader):
    """视图列被 JOIN 的 BASE TABLE 过滤排除。"""
    df = loader.load_columns()
    assert "govio_it.user_view.label" not in set(df["column"])
    assert not any(col.startswith("govio_it.user_view.") for col in df["column"])
    # users 三列 + orders 五列 + audit_log 两列（删列后）
    assert len(df) == 10


def test_real_mysql_order_no_after_drop_column(loader):
    """删列后 order_no 仍每表从 1 连续递增（recommender 依赖首列 order_no==1）。"""
    df = loader.load_columns()
    for full_table_name, group in df.groupby("full_table_name"):
        order_no = sorted(group["order_no"].tolist())
        assert order_no == list(range(1, len(group) + 1)), full_table_name
    audit = df[df["full_table_name"] == "govio_it.audit_log"].sort_values("order_no")
    assert audit["column_name"].tolist() == ["log_id", "action"]
    assert audit["order_no"].tolist() == [1, 2]
