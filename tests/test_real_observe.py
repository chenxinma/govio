"""DatabaseManager / meta CLI 真实库（trino:483 / mysql:8 / postgres:17）集成验证

默认跳过；设置 ``GOVIO_REAL_DB=1`` 后连接真实实例执行
（默认 localhost:18080 / 13306 / 15432，可用 GOVIO_TRINO_HOST / GOVIO_TRINO_PORT /
GOVIO_TRINO_USER、GOVIO_MYSQL_HOST / GOVIO_MYSQL_PORT / GOVIO_MYSQL_USER /
GOVIO_MYSQL_PASSWORD、GOVIO_PG_HOST / GOVIO_PG_PORT / GOVIO_PG_USER /
GOVIO_PG_PASSWORD 覆盖）。

样例 schema ``govio_observe_it`` 由本文件自行重建（幂等），覆盖：
1. ``DatabaseManager.execute_sql`` 对 trino / mysql / postgres 走统一
   SQLAlchemy 路径（含 trino+http 归一、裸 mysql:// / postgres:// 归一）；
2. ``govio-cli meta meta --source mysql/postgres/trino`` 真实抽取元数据到
   CSV 产物（PhysicalTable / Col / Datasource / HAS_COLUMN / OWNS）。
容器的启动/清理见 docs/tasks/V6-cli-verify/progress.md。
"""

import argparse
import os

import psycopg
import pymysql
import pytest
from trino.dbapi import connect as trino_connect

from govio.cli.meta import cmd_meta
from govio.observe_data.config import DataSourceConfig
from govio.observe_data.core.database import DatabaseManager

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("GOVIO_REAL_DB") != "1",
        reason="GOVIO_REAL_DB=1 才执行真实库集成测试",
    ),
]

_SAMPLE_SCHEMA = "govio_observe_it"

# 样例库 DDL：两张表 + 行数据，mysql / postgres 含中文注释（校验字符集链路）
_MYSQL_DDL = [
    "DROP DATABASE IF EXISTS govio_observe_it",
    "CREATE DATABASE govio_observe_it CHARACTER SET utf8mb4",
    "CREATE TABLE govio_observe_it.users ("
    " id BIGINT PRIMARY KEY COMMENT '用户ID',"
    " name VARCHAR(64) NOT NULL COMMENT '用户名',"
    " email VARCHAR(128)) COMMENT='用户表'",
    "CREATE TABLE govio_observe_it.orders ("
    " order_id BIGINT PRIMARY KEY,"
    " amount DECIMAL(10,2) COMMENT '金额',"
    " qty INT)",
    "INSERT INTO govio_observe_it.users VALUES (1,'alice','a@x.com'),(2,'bob','b@x.com')",
    "INSERT INTO govio_observe_it.orders VALUES (1, 9.99, 2),(2, 5.50, 1)",
]

_PG_DDL = [
    "DROP SCHEMA IF EXISTS govio_observe_it CASCADE",
    "CREATE SCHEMA govio_observe_it",
    "CREATE TABLE govio_observe_it.users ("
    " id bigint PRIMARY KEY,"
    " name varchar(64) NOT NULL,"
    " email varchar(128))",
    "COMMENT ON TABLE govio_observe_it.users IS '用户表'",
    "COMMENT ON COLUMN govio_observe_it.users.id IS '用户ID'",
    "CREATE TABLE govio_observe_it.orders ("
    " order_id bigint PRIMARY KEY,"
    " amount numeric(10,2),"
    " qty integer)",
    "INSERT INTO govio_observe_it.users VALUES (1,'alice','a@x.com'),(2,'bob','b@x.com')",
    "INSERT INTO govio_observe_it.orders VALUES (1, 9.99, 2),(2, 5.50, 1)",
]

_TRINO_DDL = [
    "DROP SCHEMA IF EXISTS memory.govio_observe_it CASCADE",
    "CREATE SCHEMA memory.govio_observe_it",
    "CREATE TABLE memory.govio_observe_it.users"
    " (id bigint, name varchar, email varchar)",
    "CREATE TABLE memory.govio_observe_it.orders"
    " (order_id bigint, amount decimal(10,2), qty integer)",
    "INSERT INTO memory.govio_observe_it.users"
    " VALUES (1,'alice','a@x.com'),(2,'bob','b@x.com')",
    "INSERT INTO memory.govio_observe_it.orders VALUES (1, 9.99, 2),(2, 5.50, 1)",
]


def _mysql_params() -> dict:
    """MySQL 连接参数（可用环境变量覆盖默认值）。"""
    return {
        "host": os.getenv("GOVIO_MYSQL_HOST", "localhost"),
        "port": int(os.getenv("GOVIO_MYSQL_PORT", "13306")),
        "user": os.getenv("GOVIO_MYSQL_USER", "root"),
        "password": os.getenv("GOVIO_MYSQL_PASSWORD", "govio123"),
    }


def _pg_params() -> dict:
    """PostgreSQL 连接参数（可用环境变量覆盖默认值）。"""
    return {
        "host": os.getenv("GOVIO_PG_HOST", "localhost"),
        "port": int(os.getenv("GOVIO_PG_PORT", "15432")),
        "user": os.getenv("GOVIO_PG_USER", "postgres"),
        "password": os.getenv("GOVIO_PG_PASSWORD", "govio123"),
    }


def _trino_params() -> dict:
    """Trino 连接参数（可用环境变量覆盖默认值）。"""
    return {
        "host": os.getenv("GOVIO_TRINO_HOST", "localhost"),
        "port": int(os.getenv("GOVIO_TRINO_PORT", "18080")),
        "user": os.getenv("GOVIO_TRINO_USER", "govio"),
    }


@pytest.fixture(scope="module")
def sample_data():
    """在三个真实库重建样例 schema govio_observe_it（幂等），不做服务级清理。"""
    mysql = _mysql_params()
    conn = pymysql.connect(charset="utf8mb4", **mysql)
    try:
        cur = conn.cursor()
        for stmt in _MYSQL_DDL:
            cur.execute(stmt)
        conn.commit()
    finally:
        conn.close()

    pg = _pg_params()
    with psycopg.connect(dbname="postgres", autocommit=True, **pg) as conn:
        for stmt in _PG_DDL:
            conn.execute(stmt)

    trino = _trino_params()
    conn = trino_connect(http_scheme="http", **trino)
    try:
        cur = conn.cursor()
        for stmt in _TRINO_DDL:
            cur.execute(stmt)
    finally:
        conn.close()
    return _SAMPLE_SCHEMA


# ---------------------------------------------------------------------------
# DatabaseManager.execute_sql 统一 SQLAlchemy 路径
# ---------------------------------------------------------------------------


def test_execute_sql_mysql_unified_path(sample_data):
    """裸 mysql:// 归一后真实查询走通统一路径。"""
    mysql = _mysql_params()
    manager = DatabaseManager(
        {
            "mysql": DataSourceConfig(
                url=f"mysql://root:{mysql['password']}@{mysql['host']}:{mysql['port']}/govio_observe_it"
            )
        }
    )
    df = manager.execute_sql(
        "mysql", "SELECT id, name FROM govio_observe_it.users ORDER BY id"
    )
    assert df.to_dict("records") == [
        {"id": 1, "name": "alice"},
        {"id": 2, "name": "bob"},
    ]


def test_execute_sql_postgres_unified_path(sample_data):
    """裸 postgres:// 归一后真实查询走通统一路径。"""
    pg = _pg_params()
    manager = DatabaseManager(
        {
            "postgres": DataSourceConfig(
                url=f"postgres://postgres:{pg['password']}@{pg['host']}:{pg['port']}/postgres"
            )
        }
    )
    df = manager.execute_sql(
        "postgres", "SELECT id, name FROM govio_observe_it.users ORDER BY id"
    )
    assert df.to_dict("records") == [
        {"id": 1, "name": "alice"},
        {"id": 2, "name": "bob"},
    ]


def test_execute_sql_trino_http_unified_path(sample_data):
    """trino+http:// 归一为 trino:// + http_scheme=http 后真实查询走通统一路径。"""
    trino = _trino_params()
    manager = DatabaseManager(
        {
            "trino": DataSourceConfig(
                url=f"trino+http://{trino['user']}@{trino['host']}:{trino['port']}/memory"
            )
        }
    )
    df = manager.execute_sql(
        "trino", "SELECT id, name FROM memory.govio_observe_it.users ORDER BY id"
    )
    assert df.to_dict("records") == [
        {"id": 1, "name": "alice"},
        {"id": 2, "name": "bob"},
    ]


# ---------------------------------------------------------------------------
# meta meta 真实抽取元数据到 CSV
# ---------------------------------------------------------------------------


def _meta_args(source: str, db: str, output) -> argparse.Namespace:
    return argparse.Namespace(
        source=source,
        db=db,
        schemas=_SAMPLE_SCHEMA,
        datasource=f"it_{source}",
        datasources_file=None,
        kundb=None,
        workspace_uuid=None,
        output=str(output),
    )


def _read_csv(path):
    import csv

    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _assert_meta_csvs(output, source: str) -> None:
    """核对 meta meta 的五个 CSV 产物：节点、边、数据源归属。"""
    ds_name = f"it_{source}"
    tables = _read_csv(output / "PhysicalTable.csv")
    cols = _read_csv(output / "Col.csv")
    datasources = _read_csv(output / "Datasource.csv")
    has_column = _read_csv(output / "HAS_COLUMN.csv")
    owns = _read_csv(output / "OWNS.csv")

    assert {r["full_table_name"] for r in tables} == {
        f"{ds_name}.{_SAMPLE_SCHEMA}.users",
        f"{ds_name}.{_SAMPLE_SCHEMA}.orders",
    }
    assert {r["column_name"] for r in cols} == {
        "id",
        "name",
        "email",
        "order_id",
        "amount",
        "qty",
    }
    assert len(datasources) == 1
    assert datasources[0]["datasource_name"] == ds_name
    assert datasources[0]["source_type"] == source
    assert len(has_column) == 6
    assert len(owns) == 2
    return tables, cols


def test_meta_cli_mysql_exports_csv(sample_data, tmp_path):
    """meta meta --source mysql 真实抽取，中文注释随 CSV 原样落盘。"""
    mysql = _mysql_params()
    db = f"mysql://root:{mysql['password']}@{mysql['host']}:{mysql['port']}/govio_observe_it"
    cmd_meta(_meta_args("mysql", db, tmp_path))
    tables, cols = _assert_meta_csvs(tmp_path, "mysql")
    names = {r["full_table_name"]: r["name"] for r in tables}
    assert names[f"it_mysql.{_SAMPLE_SCHEMA}.users"] == "用户表"
    col_names = {r["column_name"]: r["name"] for r in cols}
    assert col_names["id"] == "用户ID"
    assert col_names["amount"] == "金额"


def test_meta_cli_postgres_exports_csv(sample_data, tmp_path):
    """meta meta --source postgres 真实抽取，中文注释随 CSV 原样落盘。"""
    pg = _pg_params()
    db = f"postgres://postgres:{pg['password']}@{pg['host']}:{pg['port']}/postgres"
    cmd_meta(_meta_args("postgres", db, tmp_path))
    tables, cols = _assert_meta_csvs(tmp_path, "postgres")
    names = {r["full_table_name"]: r["name"] for r in tables}
    assert names[f"it_postgres.{_SAMPLE_SCHEMA}.users"] == "用户表"
    col_names = {r["column_name"]: r["name"] for r in cols}
    assert col_names["id"] == "用户ID"


def test_meta_cli_trino_exports_csv(sample_data, tmp_path):
    """meta meta --source trino（trino+http scheme）真实抽取到 CSV。"""
    trino = _trino_params()
    db = f"trino+http://{trino['user']}@{trino['host']}:{trino['port']}/memory"
    cmd_meta(_meta_args("trino", db, tmp_path))
    _assert_meta_csvs(tmp_path, "trino")
