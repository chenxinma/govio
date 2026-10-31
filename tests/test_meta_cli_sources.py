"""meta meta --source 多数据库分支测试。

覆盖 mysql / postgres / trino 三个来源的 CLI 参数分支与连接参数传递：
1. argparse `--source` 接受新取值，未知取值拒绝
2. 非 TDS 模式 --db / --datasource / --schemas 必填校验
3. --db 连接 URL 解析为各 Loader 构造参数（含 trino auth / http_scheme）
4. URL scheme 与 --source 不匹配、URL 非法、trino URL 路径缺 catalog 时报错
5. 全链路导出 CSV 且 Datasource.source_type 取 --source

全部打桩 Loader，不依赖真实数据库。
"""

import argparse
import sys

import pandas as pd
import pytest
from trino.auth import BasicAuthentication

from govio.cli import meta as meta_mod
from govio.cli.meta import cmd_meta, meta, step_meta_export


def _table_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "full_table_name": ["ds.appdb.orders"],
            "schema": ["appdb"],
            "table_name": ["orders"],
            "name": ["订单"],
            "data_entity_type": ["MYSQL_TABLE"],
            "database_name": ["appdb"],
        }
    )


def _column_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "column": ["ds.appdb.orders.id"],
            "column_name": ["id"],
            "name": ["主键"],
            "full_table_name": ["ds.appdb.orders"],
            "data_entity_type": ["MYSQL_COLUMN"],
            "dtype": ["bigint"],
            "size": [0],
            "precision": [0],
            "scale": [0],
            "order_no": [1],
            "data_type": ["bigint"],
        }
    )


class FakeLoader:
    """打桩元数据加载器：记录构造参数并返回固定表/列 DataFrame。"""

    def __init__(self, call_sink: dict, source: str, *args, **kwargs) -> None:
        call_sink[source] = {"args": args, "kwargs": kwargs}
        self._tables = _table_df()
        self._columns = _column_df()

    @property
    def PhysicalTable(self) -> pd.DataFrame:
        return self._tables.copy()

    @property
    def Col(self) -> pd.DataFrame:
        return self._columns.copy()


@pytest.fixture
def loader_calls(monkeypatch) -> dict:
    """把 meta 模块的三个加载器类替换为记录构造参数的桩。"""
    calls: dict = {}
    for source, attr in [
        ("mysql", "MysqlLoader"),
        ("postgres", "PostgresLoader"),
        ("trino", "TrinoLoader"),
    ]:
        monkeypatch.setattr(
            meta_mod,
            attr,
            lambda *a, _s=source, **kw: FakeLoader(calls, _s, *a, **kw),
        )
    return calls


def _args(**kw) -> argparse.Namespace:
    base = {
        "source": "mysql",
        "db": None,
        "schemas": None,
        "datasource": None,
        "datasources_file": None,
        "kundb": None,
        "workspace_uuid": None,
        "output": None,
    }
    base.update(kw)
    return argparse.Namespace(**base)


def _cmd(source: str, db: str, tmp_path, **kw) -> None:
    cmd_meta(
        _args(
            source=source,
            db=db,
            schemas="appdb",
            datasource="test_ds",
            output=str(tmp_path / "out"),
            **kw,
        )
    )


# ---------------------------------------------------------------------------
# argparse --source 分支
# ---------------------------------------------------------------------------


def test_meta_argparse_accepts_new_sources(loader_calls, tmp_path, monkeypatch):
    """meta() 的 argparse 接受 mysql / postgres / trino 取值并跑通导出。

    meta() 解析 sys.argv[1:]，故 argv 中只出现一次 meta 子命令
    （main() 转发时已把子命令参数重写进 sys.argv）。
    """
    dbs = {
        "mysql": "mysql://root:pw@h:3306/appdb",
        "postgres": "postgresql://pg:pw@h:5432/appdb",
        "trino": "trino://trino@h:8080/hive",
    }
    for source, db in dbs.items():
        out = tmp_path / source
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "govio-cli",
                "meta",
                "--source",
                source,
                "--db",
                db,
                "--schemas",
                "appdb",
                "--datasource",
                "test_ds",
                "--output",
                str(out),
            ],
        )
        meta()
        assert (out / "PhysicalTable.csv").exists(), source
        assert source in loader_calls, source


def test_meta_argparse_rejects_unknown_source(monkeypatch, tmp_path):
    """--source 取值之外的来源被 argparse 拒绝。"""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "govio-cli",
            "meta",
            "--source",
            "oracle",
            "--db",
            "x",
            "--schemas",
            "s",
            "--datasource",
            "d",
            "--output",
            str(tmp_path / "out"),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        meta()
    assert exc.value.code == 2


# ---------------------------------------------------------------------------
# 参数必填校验（与 duckdb 分支同风格）
# ---------------------------------------------------------------------------


def test_cmd_meta_mysql_requires_db(loader_calls, tmp_path, capsys):
    """mysql 模式省略 --db 时报错退出。"""
    with pytest.raises(SystemExit) as exc:
        cmd_meta(
            _args(
                source="mysql",
                schemas="appdb",
                datasource="test_ds",
                output=str(tmp_path / "out"),
            )
        )
    assert exc.value.code == 1
    assert "--db" in capsys.readouterr().err


def test_cmd_meta_trino_requires_datasource(loader_calls, tmp_path, capsys):
    """trino 模式省略 --datasource 时报错退出。"""
    with pytest.raises(SystemExit) as exc:
        cmd_meta(
            _args(
                source="trino",
                db="trino://u@h:8080/hive",
                schemas="appdb",
                output=str(tmp_path / "out"),
            )
        )
    assert exc.value.code == 1
    assert "--datasource" in capsys.readouterr().err


def test_cmd_meta_postgres_requires_schemas(loader_calls, tmp_path, capsys):
    """postgres 模式省略 --schemas 时报错退出，且提示来源相关 schema 语义。"""
    with pytest.raises(SystemExit) as exc:
        cmd_meta(
            _args(
                source="postgres",
                db="postgresql://u:pw@h:5432/appdb",
                datasource="test_ds",
                output=str(tmp_path / "out"),
            )
        )
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "--schemas" in err
    assert "public" in err


# ---------------------------------------------------------------------------
# 连接 URL -> Loader 构造参数
# ---------------------------------------------------------------------------


def test_mysql_url_passes_connection_params(loader_calls, tmp_path):
    """mysql URL 解析出 host/port/user/password/database 等参数。"""
    _cmd("mysql", "mysql://root:s3cret@db.example.com:3307/appdb", tmp_path)
    kwargs = loader_calls["mysql"]["kwargs"]
    assert kwargs["host"] == "db.example.com"
    assert kwargs["port"] == 3307
    assert kwargs["user"] == "root"
    assert kwargs["password"] == "s3cret"
    assert kwargs["database"] == "appdb"
    assert kwargs["schemas"] == ["appdb"]
    assert kwargs["datasource_name"] == "test_ds"


def test_mysql_driver_qualified_scheme_accepted(loader_calls, tmp_path):
    """mysql+pymysql:// 与 observe 配置 URL 写法一致，同样被接受。"""
    _cmd("mysql", "mysql+pymysql://root@db.example.com/appdb", tmp_path)
    kwargs = loader_calls["mysql"]["kwargs"]
    assert kwargs["host"] == "db.example.com"
    assert kwargs["port"] == 3306  # 缺省端口
    assert kwargs["user"] == "root"


def test_postgres_url_passes_connection_params(loader_calls, tmp_path):
    """postgres URL 解析出 host/port/user/password/dbname 等参数。"""
    _cmd("postgres", "postgres://pg:pw@pg.example.com:5433/pgdb", tmp_path)
    kwargs = loader_calls["postgres"]["kwargs"]
    assert kwargs["host"] == "pg.example.com"
    assert kwargs["port"] == 5433
    assert kwargs["user"] == "pg"
    assert kwargs["password"] == "pw"
    assert kwargs["dbname"] == "pgdb"
    assert kwargs["schemas"] == ["appdb"]
    assert kwargs["datasource_name"] == "test_ds"


def test_trino_url_without_password_builds_loader(loader_calls, tmp_path):
    """trino URL 解析出 catalog，无密码时 auth 为 None、http_scheme 为 http。"""
    _cmd("trino", "trino://analyst@trino.example.com:8080/hive", tmp_path)
    kwargs = loader_calls["trino"]["kwargs"]
    assert kwargs["host"] == "trino.example.com"
    assert kwargs["port"] == 8080
    assert kwargs["user"] == "analyst"
    assert kwargs["catalog"] == "hive"
    assert kwargs["http_scheme"] == "http"
    assert kwargs["auth"] is None
    assert kwargs["schemas"] == ["appdb"]
    assert kwargs["datasource_name"] == "test_ds"


def test_trino_url_with_password_builds_basic_auth(loader_calls, tmp_path):
    """trino URL 含密码时构造 BasicAuthentication 并携带 user/password。"""
    _cmd("trino", "trino+https://analyst:pw@trino.example.com:8443/hive", tmp_path)
    kwargs = loader_calls["trino"]["kwargs"]
    auth = kwargs["auth"]
    assert isinstance(auth, BasicAuthentication)
    assert auth._username == "analyst"
    assert auth._password == "pw"
    assert kwargs["http_scheme"] == "https"
    assert kwargs["port"] == 8443


def test_trino_url_missing_user_with_password_rejected(loader_calls, tmp_path):
    """trino URL 有密码无 user 时报错。"""
    with pytest.raises(SystemExit):
        _cmd("trino", "trino://:pw@trino.example.com:8080/hive", tmp_path)


# ---------------------------------------------------------------------------
# URL 报错分支
# ---------------------------------------------------------------------------


def test_step_meta_export_url_scheme_mismatch(tmp_path, capsys):
    """URL scheme 与 --source 不匹配时返回 None 并提示期望写法。"""
    result = step_meta_export(
        tmp_path / "out",
        source="postgres",
        db_path="mysql://u:pw@h:3306/db",
        schemas=["appdb"],
        datasource_name="test_ds",
    )
    assert result is None
    err = capsys.readouterr().err
    assert "不匹配" in err
    assert "postgresql" in err


def test_step_meta_export_malformed_url(tmp_path, capsys):
    """--db 不是合法 URL 时返回 None 并给出中文报错。"""
    result = step_meta_export(
        tmp_path / "out",
        source="mysql",
        db_path="not-a-url",
        schemas=["appdb"],
        datasource_name="test_ds",
    )
    assert result is None
    assert "不是合法的连接 URL" in capsys.readouterr().err


def test_step_meta_export_trino_url_requires_catalog(tmp_path, capsys):
    """trino URL 路径缺 catalog（或带 schema 段）时返回 None 并提示格式。"""
    for db in ("trino://u@h:8080", "trino://u@h:8080/hive/sf1"):
        result = step_meta_export(
            tmp_path / "out",
            source="trino",
            db_path=db,
            schemas=["appdb"],
            datasource_name="test_ds",
        )
        assert result is None, db
        assert "catalog" in capsys.readouterr().err


def test_step_meta_export_unknown_source(tmp_path, capsys):
    """未支持的 source 返回 None。"""
    result = step_meta_export(
        tmp_path / "out",
        source="oracle",
        db_path="oracle://u:pw@h:1521/orcl",
        schemas=["appdb"],
        datasource_name="test_ds",
    )
    assert result is None
    assert "不支持的信息来源" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# 全链路导出
# ---------------------------------------------------------------------------


def test_step_meta_export_mysql_writes_source_type(loader_calls, tmp_path):
    """mysql 全链路导出 CSV，Datasource.source_type 取 --source。"""
    out = tmp_path / "out"
    result = step_meta_export(
        out,
        source="mysql",
        db_path="mysql://root:pw@h:3306/appdb",
        schemas=["appdb"],
        datasource_name="test_ds",
    )
    assert result is not None

    df_ds = pd.read_csv(out / "Datasource.csv")
    assert df_ds["source_type"].iloc[0] == "mysql"
    assert df_ds["datasource_name"].iloc[0] == "test_ds"
    assert df_ds[":ID(Datasource)"].iloc[0].startswith("DS")

    df_owns = pd.read_csv(out / "OWNS.csv")
    assert len(df_owns) == 1

    df_tables, df_columns = result
    assert len(df_tables) == 1
    assert len(df_columns) == 1
