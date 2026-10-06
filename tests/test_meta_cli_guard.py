"""meta meta 参数守卫与空结果守卫测试。

覆盖四条约束：
1. DuckDB 模式 `--schemas` / `--datasource` 必填（省略会导致静默导出 0 张表）
2. TDS 模式 `--datasources-file` 必填（抽取范围取 filter.schemas）
3. 导出结果为空时直接失败，不写任何 CSV，并提示源库可用 schema
4. assets 输出路径以绝对路径打印，提示用户自行合并
"""
import argparse
from pathlib import Path
from unittest.mock import patch

import duckdb
import pytest

from govio.cli import meta as meta_mod
from govio.cli.meta import cmd_meta, step_meta_export


@pytest.fixture
def duck_db(tmp_path):
    """构造只含 main schema 的临时 DuckDB 文件。"""
    db_path = str(tmp_path / "t.duckdb")
    conn = duckdb.connect(db_path)
    conn.execute("CREATE TABLE a (x INTEGER, y VARCHAR)")
    conn.execute("CREATE TABLE b (x INTEGER)")
    conn.close()
    return db_path


def _args(**kw) -> argparse.Namespace:
    base = {
        "source": "duckdb", "db": None, "schemas": None, "datasource": None,
        "datasources_file": None, "kundb": None, "workspace_uuid": None, "output": None,
    }
    base.update(kw)
    return argparse.Namespace(**base)


# ---------------------------------------------------------------------------
# --schemas / --datasource 必填
# ---------------------------------------------------------------------------

def test_cmd_meta_requires_schemas(duck_db, tmp_path, capsys):
    """DuckDB 模式省略 --schemas 时报错退出，不再静默导出 0 张表。"""
    with pytest.raises(SystemExit) as exc:
        cmd_meta(_args(db=duck_db, datasource="test", output=str(tmp_path / "out")))
    assert exc.value.code == 1
    assert "--schemas" in capsys.readouterr().err


def test_cmd_meta_requires_datasource(duck_db, tmp_path, capsys):
    """DuckDB 模式省略 --datasource 时报错退出。"""
    with pytest.raises(SystemExit) as exc:
        cmd_meta(_args(db=duck_db, schemas="main", output=str(tmp_path / "out")))
    assert exc.value.code == 1
    assert "--datasource" in capsys.readouterr().err


def test_cmd_meta_tds_requires_datasources_file(tmp_path, capsys):
    """TDS 模式省略 --datasources-file 时报错退出。"""
    with pytest.raises(SystemExit) as exc:
        cmd_meta(_args(
            source="tds", kundb="mysql://x", workspace_uuid="ws",
            output=str(tmp_path / "out"),
        ))
    assert exc.value.code == 1
    assert "--datasources-file" in capsys.readouterr().err


def test_cmd_meta_schemas_blank_only(duck_db, tmp_path, capsys):
    """--schemas 仅含逗号/空白等同于未指定。"""
    with pytest.raises(SystemExit):
        cmd_meta(_args(
            db=duck_db, datasource="test", schemas=" , ",
            output=str(tmp_path / "out"),
        ))
    assert "--schemas" in capsys.readouterr().err


def test_cmd_meta_strips_schema_whitespace(duck_db, tmp_path):
    """schema 名两侧空白被清理后正常导出。"""
    out = tmp_path / "out"
    cmd_meta(_args(
        db=duck_db, datasource="test", schemas=" main ", output=str(out),
    ))
    assert (out / "PhysicalTable.csv").exists()
    assert (out / "Datasource.csv").exists()
    assert (out / "OWNS.csv").exists()


# ---------------------------------------------------------------------------
# 空结果守卫
# ---------------------------------------------------------------------------

def test_step_meta_export_empty_result_writes_no_csv(duck_db, tmp_path, capsys):
    """schema 不存在时返回 None，不写 CSV，并提示可用 schema。"""
    out = tmp_path / "out"
    result = step_meta_export(
        out, source="duckdb", db_path=duck_db, schemas=["sales"],
        datasource_name="test",
    )
    assert result is None

    err = capsys.readouterr().err
    assert "未发现任何表" in err
    assert "main" in err  # 错误信息中列出源库实际 schema
    assert not (out / "PhysicalTable.csv").exists()
    assert not (out / "Col.csv").exists()
    assert not (out / "Datasource.csv").exists()


def test_cmd_meta_exits_on_empty_result(duck_db, tmp_path):
    """CLI 层将空结果转为非零退出码。"""
    with pytest.raises(SystemExit) as exc:
        cmd_meta(_args(
            db=duck_db, datasource="test", schemas="sales",
            output=str(tmp_path / "out"),
        ))
    assert exc.value.code == 1


def test_step_meta_export_success(duck_db, tmp_path):
    """schema 命中时正常导出并返回 DataFrame。"""
    out = tmp_path / "out"
    result = step_meta_export(
        out, source="duckdb", db_path=duck_db, schemas=["main"],
        datasource_name="test",
    )
    assert result is not None

    df_tables, df_columns = result
    assert len(df_tables) == 2
    assert len(df_columns) == 3
    assert (out / "HAS_COLUMN.csv").exists()
    assert (out / "OWNS.csv").exists()


# ---------------------------------------------------------------------------
# assets 路径输出
# ---------------------------------------------------------------------------

def test_generate_assets_prints_absolute_path(tmp_path, capsys, monkeypatch):
    """assets 路径以绝对路径打印，接受 assets_dir 参数。"""
    monkeypatch.chdir(tmp_path)
    assets_dir = Path(".agents/skills/govio/assets")
    with patch.object(meta_mod, "ConfigManager") as cm, \
            patch.object(meta_mod, "GraphFactory"), \
            patch.object(meta_mod, "AssetsGenerator"):
        cm.return_value.load.return_value = {"graph": {"backend": "ladybug"}}
        meta_mod._generate_assets(assets_dir)

    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if "Assets 已生成到" in ln)
    printed = line.split(":", 1)[1].strip()
    assert Path(printed).is_absolute()
    assert printed.endswith(str(Path(".agents/skills/govio/assets")))
