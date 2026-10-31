"""PostgresLoader 真实 PostgreSQL（postgres:17）集成验证

默认跳过；设置 ``GOVIO_REAL_DB=1`` 后连接真实实例执行
（默认 localhost:15432，可用 GOVIO_PG_HOST / GOVIO_PG_PORT /
GOVIO_PG_USER / GOVIO_PG_PASSWORD 覆盖）。

样例库 ``govio_it``（数据库 + 同名 schema）由本文件自行重建（幂等），覆盖：
P1-1 复现（a int + b numeric 无精度 + c varchar 无长度 与带长度/精度列混批，
属性列被 pandas 推断为 float64、NULL 变 NaN）、表/列注释、视图、外表
（postgres_fdw）、一个删过列的表（attnum 空洞重排 order_no）。
容器的启动/清理见 docs/tasks/V5-postgres-verify/progress.md。
"""

import os

import pandas as pd
import psycopg
import pytest

from govio.metadata.postgres_loader import _COLUMNS_SQL, PostgresLoader, _parse_type

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("GOVIO_REAL_DB") != "1",
        reason="GOVIO_REAL_DB=1 才执行真实库集成测试",
    ),
]

_SAMPLE_DB = "govio_it"
_SAMPLE_SCHEMA = "govio_it"

# 样例库 DDL：P1-1 复现表 p1_case（a int + b numeric 无精度 + c varchar 无长度）、
# 表/列注释、视图、外表（postgres_fdw）、删过列的表（attnum 空洞）
_SAMPLE_DDL = [
    "CREATE SCHEMA govio_it",
    "CREATE TABLE govio_it.users ("
    " id bigint PRIMARY KEY,"
    " name varchar(64) NOT NULL,"
    " email varchar(128))",
    "COMMENT ON TABLE govio_it.users IS '用户表'",
    "COMMENT ON COLUMN govio_it.users.id IS '用户ID'",
    "COMMENT ON COLUMN govio_it.users.name IS '用户名'",
    "CREATE TABLE govio_it.orders ("
    " order_id bigint PRIMARY KEY,"
    " amount numeric(10,2),"
    " qty integer)",
    "COMMENT ON COLUMN govio_it.orders.amount IS '金额'",
    "CREATE TABLE govio_it.p1_case (a integer, b numeric, c varchar)",
    "CREATE TABLE govio_it.audit_log ("
    " log_id bigint PRIMARY KEY,"
    " old_col varchar(20),"
    " action varchar(32))",
    "COMMENT ON TABLE govio_it.audit_log IS '审计表'",
    "COMMENT ON COLUMN govio_it.audit_log.log_id IS '日志ID'",
    "ALTER TABLE govio_it.audit_log DROP COLUMN old_col",
    "CREATE VIEW govio_it.user_view AS"
    " SELECT id, name || '!' AS label FROM govio_it.users",
    "CREATE EXTENSION postgres_fdw",
    "CREATE SERVER govio_fdw FOREIGN DATA WRAPPER postgres_fdw"
    " OPTIONS (host '127.0.0.1', port '5432', dbname 'govio_it')",
    "CREATE USER MAPPING FOR postgres SERVER govio_fdw"
    " OPTIONS (user 'postgres', password 'govio123')",
    "CREATE FOREIGN TABLE govio_it.ext_users (id bigint, name varchar(64))"
    " SERVER govio_fdw OPTIONS (schema_name 'govio_it', table_name 'users')",
    "COMMENT ON FOREIGN TABLE govio_it.ext_users IS '外表样本'",
]

_TABLE_COLUMNS = [
    "full_table_name",
    "schema",
    "table_name",
    "name",
    "data_entity_type",
    "database_name",
]

_COLUMN_COLUMNS = [
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


def _conn_params() -> dict:
    """连接参数（可用环境变量覆盖默认值）。"""
    return {
        "host": os.getenv("GOVIO_PG_HOST", "localhost"),
        "port": int(os.getenv("GOVIO_PG_PORT", "15432")),
        "user": os.getenv("GOVIO_PG_USER", "postgres"),
        "password": os.getenv("GOVIO_PG_PASSWORD", "govio123"),
    }


@pytest.fixture(scope="module")
def sample_schema():
    """重建样例库 govio_it（drop + create + DDL，幂等），结束后不断开服务。"""
    params = _conn_params()
    conn = psycopg.connect(dbname="postgres", **params)
    try:
        conn.autocommit = True
        conn.execute("DROP DATABASE IF EXISTS govio_it WITH (FORCE)")
        conn.execute("CREATE DATABASE govio_it")
    finally:
        conn.close()
    conn = psycopg.connect(dbname=_SAMPLE_DB, **params)
    try:
        for stmt in _SAMPLE_DDL:
            conn.execute(stmt)
        conn.commit()
    finally:
        conn.close()
    return _SAMPLE_SCHEMA


@pytest.fixture(scope="module")
def loader(sample_schema):
    """连接样例库的 PostgresLoader。"""
    return PostgresLoader(
        schemas=[sample_schema],
        datasource_name="",
        dbname=_SAMPLE_DB,
        **_conn_params(),
    )


def test_real_postgres_load_tables_contract(loader):
    """load_tables 列契约、表注释回退，视图/外表与基表同批导出。"""
    df = loader.load_tables()
    assert list(df.columns) == _TABLE_COLUMNS
    assert set(df["table_name"]) == {
        "users",
        "orders",
        "p1_case",
        "audit_log",
        "user_view",
        "ext_users",
    }
    assert all(df["data_entity_type"] == "POSTGRES_TABLE")
    assert all(df["database_name"] == "govio_it")
    names = dict(zip(df["table_name"], df["name"], strict=True))
    assert names["users"] == "用户表"
    assert names["audit_log"] == "审计表"
    # 空表注释回退为表名
    assert names["orders"] == "orders"
    assert names["p1_case"] == "p1_case"
    assert names["ext_users"] == "外表样本"


def test_real_postgres_load_columns_contract(loader):
    """load_columns 列契约、列注释回退与 dtype 保留 information_schema 原文。"""
    df = loader.load_columns()
    assert list(df.columns) == _COLUMN_COLUMNS
    assert all(df["data_entity_type"] == "POSTGRES_COLUMN")
    users = df[df["full_table_name"] == "govio_it.users"].set_index("column_name")
    assert users.loc["id", "name"] == "用户ID"
    assert users.loc["name", "name"] == "用户名"
    # 空列注释回退为列名
    assert users.loc["email", "name"] == "email"
    orders = df[df["full_table_name"] == "govio_it.orders"].set_index("column_name")
    # dtype / data_type 保留 information_schema 类型原文
    assert orders.loc["amount", "dtype"] == "numeric"
    assert orders.loc["amount", "data_type"] == "numeric"
    assert users.loc["name", "dtype"] == "character varying"


def test_real_postgres_type_parsing(loader):
    """真实类型属性解析：varchar(n) → size，numeric(p,s) → precision/scale。"""
    df = loader.load_columns().set_index("column")
    users_name = df.loc["govio_it.users.name"]
    assert (users_name["size"], users_name["precision"], users_name["scale"]) == (
        64,
        0,
        0,
    )
    amount = df.loc["govio_it.orders.amount"]
    assert (amount["size"], amount["precision"], amount["scale"]) == (0, 10, 2)
    # 整型 numeric_precision 是位宽语义，不计入 precision
    p1_a = df.loc["govio_it.p1_case.a"]
    assert (p1_a["size"], p1_a["precision"], p1_a["scale"]) == (0, 0, 0)
    # mysql:8 不可达的裸 varchar / 裸 numeric 在 Postgres 是常规 DDL
    p1_b = df.loc["govio_it.p1_case.b"]
    assert (p1_b["size"], p1_b["precision"], p1_b["scale"]) == (0, 0, 0)
    p1_c = df.loc["govio_it.p1_case.c"]
    assert (p1_c["size"], p1_c["precision"], p1_c["scale"]) == (0, 0, 0)
    assert _parse_type("character varying", None, None, None) == (0, 0, 0)


def test_real_postgres_p1_nan_repro(loader):
    """P1-1 复现路径：混合属性列 + NULL 被推断成 NaN 后仍正确解析。

    同批结果里 varchar(64)/numeric(10,2)/bigint 与裸 varchar、裸 numeric
    并存时，character_maximum_length / numeric_* 列被 pandas 推断为 float64，
    SQL NULL 变 NaN（既非 None 也非 falsy），修复前 int(NaN) 抛 ValueError。
    """
    raw = loader._fetch(_COLUMNS_SQL, ([_SAMPLE_SCHEMA],))
    # 复现前提：混合列是 float64，缺失值是 NaN 而非 Python None
    assert raw["character_maximum_length"].dtype == "float64"
    assert raw["numeric_precision"].dtype == "float64"
    p1_rows = raw[raw["table_name"] == "p1_case"].set_index("column_name")
    assert pd.isna(p1_rows.loc["c", "character_maximum_length"])
    assert pd.isna(p1_rows.loc["b", "numeric_precision"])
    assert pd.isna(p1_rows.loc["b", "numeric_scale"])
    df = loader.load_columns().set_index("column")
    p1_c = df.loc["govio_it.p1_case.c"]
    assert (p1_c["size"], p1_c["precision"], p1_c["scale"]) == (0, 0, 0)
    p1_b = df.loc["govio_it.p1_case.b"]
    assert (p1_b["size"], p1_b["precision"], p1_b["scale"]) == (0, 0, 0)


def test_real_postgres_includes_view_and_foreign_table_columns(loader):
    """视图与外表的列同样导出（与 TrinoLoader SHOW TABLES 范围一致）。"""
    df = loader.load_columns()
    by_table = {
        name: sorted(group["column_name"].tolist())
        for name, group in df.groupby("full_table_name")
    }
    assert by_table["govio_it.user_view"] == ["id", "label"]
    assert by_table["govio_it.ext_users"] == ["id", "name"]
    # users 3 + orders 3 + p1_case 3 + audit_log 2 + user_view 2 + ext_users 2
    assert len(df) == 15


def test_real_postgres_order_no_attnum_renumber(loader):
    """删列后 attnum 有空洞，order_no 仍每表从 1 连续递增。"""
    df = loader.load_columns()
    for full_table_name, group in df.groupby("full_table_name"):
        order_no = sorted(group["order_no"].tolist())
        assert order_no == list(range(1, len(group) + 1)), full_table_name
    audit = df[df["full_table_name"] == "govio_it.audit_log"].sort_values("order_no")
    # old_col 删除后 attnum 空洞（1, 3），order_no 重排为 [1, 2]
    assert audit["column_name"].tolist() == ["log_id", "action"]
    assert audit["order_no"].tolist() == [1, 2]
