"""重复导入幂等 + 校验报告集成测试。

固定「子命令写 CSV + meta graph --mode update 导图」的幂等承诺：同一批输入
跑两轮，两轮后图库状态一致、第二轮零变更（新增/更新均为 0）、各类型计数
不变、校验项全部通过。同时覆盖报告落盘契约：默认写入 --output 目录、
--report 可覆盖、图更新失败不产半截报告、import-schema 一步式同样产出报告。
"""

import argparse
import json
from pathlib import Path

import duckdb
import networkx as nx
import pandas as pd
import pytest

from govio.cli import meta as meta_mod
from govio.cli.meta import (
    cmd_graph,
    cmd_import_schema,
    step_meta_export,
    step_metric_export,
    step_rel_export,
)


class StubConfigManager:
    """打桩 ConfigManager：返回指定配置（不读 ~/.govio/config.yaml）"""

    def __init__(self, config: dict):
        self._config = config

    def exists(self) -> bool:
        return True

    def load(self) -> dict:
        return self._config


# ---------------------------------------------------------------------------
# 夹具与辅助
# ---------------------------------------------------------------------------


@pytest.fixture
def duck_db(tmp_path):
    """构造只含 main schema 的临时 DuckDB 文件。"""
    db_path = str(tmp_path / "t.duckdb")
    conn = duckdb.connect(db_path)
    conn.execute("CREATE TABLE a (x INTEGER, y VARCHAR)")
    conn.execute("CREATE TABLE b (x INTEGER)")
    conn.close()
    return db_path


@pytest.fixture
def rel_file(tmp_path):
    data = {
        "version": "1.0",
        "relationships": [
            {
                "source": {"PhysicalTable": "local.main.a", "Cols": ["x"]},
                "target": {"PhysicalTable": "local.main.b", "Cols": ["x"]},
                "relationship_type": "many_to_one",
                "description": "a.x 引用 b.x",
            }
        ],
    }
    path = tmp_path / "relationships.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return str(path)


@pytest.fixture
def metric_file(tmp_path):
    data = {
        "version": "1.0",
        "metrics": [
            {
                "code": "m_total_x",
                "name": "Total X",
                "business_definition": "x 总量",
                "type": "atomic",
                "unit": "元",
                "data_type": "decimal",
                "source_layer": "DM",
                "source_tables": [
                    {
                        "full_table_name": "local.main.a",
                        "columns": [{"column_name": "x", "role": "measure"}],
                    }
                ],
                "dimensions": [{"code": "dim_time", "usage_type": "group"}],
            }
        ],
        "shared_dimensions": [
            {"code": "dim_time", "name": "Time", "granularity": "day"}
        ],
    }
    path = tmp_path / "metric.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return str(path)


def _stub_config(monkeypatch, tmp_path, extra: dict | None = None):
    # assets 目录须预先存在：GML 落盘不自建父目录（既有行为，见 design.md 遗留问题）
    (tmp_path / "assets").mkdir(exist_ok=True)
    config = {
        "graph": {
            "backend": "networkx",
            "networkx": {"gml_path": str(tmp_path / "assets" / "ontology.gml")},
        }
    }
    if extra:
        config.update(extra)
    monkeypatch.setattr(meta_mod, "ConfigManager", lambda: StubConfigManager(config))


def _run_round(out: Path, assets: Path, duck_db, rel_file, metric_file, report=None):
    """一轮完整导入：meta + rel + metric 子命令 + meta graph --mode update。"""
    result = step_meta_export(
        out,
        source="duckdb",
        db_path=duck_db,
        schemas=["main"],
        datasource_name="local",
    )
    assert result is not None
    step_rel_export(out, rel_file)
    assert step_metric_export(out, metric_file)
    cmd_graph(
        argparse.Namespace(
            output=str(out), mode="update", assets_dir=str(assets), report=report
        )
    )


def _norm_attrs(data: dict) -> dict:
    """属性归一为字符串（NaN 视为空串），规避 nan != nan 的比较问题。"""
    return {k: ("" if pd.isna(v) else str(v)) for k, v in data.items()}


def _graph_snapshot(gml_path: Path) -> tuple[dict, dict]:
    g = nx.read_gml(gml_path)
    nodes = {n: _norm_attrs(d) for n, d in g.nodes(data=True)}
    edges = {(u, v): _norm_attrs(d) for u, v, d in g.edges(data=True)}
    return nodes, edges


def _read_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 幂等固化（acceptance ①）
# ---------------------------------------------------------------------------


def test_repeat_import_is_idempotent(
    duck_db, rel_file, metric_file, tmp_path, monkeypatch
):
    out, assets = tmp_path / "out", tmp_path / "assets"
    _stub_config(monkeypatch, tmp_path)

    _run_round(out, assets, duck_db, rel_file, metric_file)
    nodes1, edges1 = _graph_snapshot(assets / "ontology.gml")
    report1 = _read_report(out / "import-report.json")
    csv1 = {f.name: pd.read_csv(f, dtype=str).fillna("") for f in out.glob("*.csv")}

    _run_round(out, assets, duck_db, rel_file, metric_file)
    nodes2, edges2 = _graph_snapshot(assets / "ontology.gml")
    report2 = _read_report(out / "import-report.json")
    csv2 = {f.name: pd.read_csv(f, dtype=str).fillna("") for f in out.glob("*.csv")}

    # 两轮后图库状态一致（含属性）；图以 ID 为键，计数即无重复节点/边
    assert nodes1 == nodes2
    assert edges1 == edges2
    assert len(nodes2) > 0 and len(edges2) > 0

    # 第二轮零变更：新增/更新均为 0，全部跳过
    totals2 = report2["changes"]["totals"]
    assert totals2["added"] == 0
    assert totals2["updated"] == 0
    assert totals2["skipped"] > 0

    # 首轮全新导入：全部新增、零更新
    totals1 = report1["changes"]["totals"]
    assert totals1["added"] > 0
    assert totals1["updated"] == 0

    # 各类型计数不变
    assert report1["import_counts"] == report2["import_counts"]

    # 校验项全部通过（无重复 id / 必填缺失 / 孤儿边）
    assert all(c["status"] == "pass" for c in report2["checks"])

    # CSV 目录（共享状态）两轮内容一致
    assert set(csv1) == set(csv2)
    for name in csv1:
        pd.testing.assert_frame_equal(csv1[name], csv2[name])


# ---------------------------------------------------------------------------
# 报告落盘契约（acceptance ②）
# ---------------------------------------------------------------------------


def test_report_written_to_output_dir_by_default(
    duck_db, rel_file, metric_file, tmp_path, monkeypatch, capsys
):
    out, assets = tmp_path / "out", tmp_path / "assets"
    _stub_config(monkeypatch, tmp_path)
    _run_round(out, assets, duck_db, rel_file, metric_file)

    report_path = out / "import-report.json"
    assert report_path.exists()
    report = _read_report(report_path)
    assert report["mode"] == "update"
    assert report["backend"] == "networkx"
    assert report["import_counts"]["nodes"] == {
        "PhysicalTable": 2,
        "Col": 3,
        "Datasource": 1,
        "Metric": 1,
        "Dimension": 1,
    }
    assert report["import_counts"]["edges"]["HAS_COLUMN"] == 3
    assert report["import_counts"]["edges"]["RELATES_TO"] == 1
    assert report["changes"]["totals"]["added"] > 0
    assert report["summary"]["status"] == "pass"

    # 既有 stdout 体验不回退
    captured = capsys.readouterr().out
    assert "✓ GML 文件已更新" in captured
    assert "✅ graph 更新完成！" in captured
    assert "✓ 校验报告已写入" in captured


def test_report_override_path(
    duck_db, rel_file, metric_file, tmp_path, monkeypatch
):
    out, assets = tmp_path / "out", tmp_path / "assets"
    _stub_config(monkeypatch, tmp_path)
    custom = tmp_path / "custom-report.json"
    _run_round(out, assets, duck_db, rel_file, metric_file, report=str(custom))

    assert custom.exists()
    assert not (out / "import-report.json").exists()
    assert _read_report(custom)["summary"]["status"] == "pass"


def test_bad_input_writes_no_partial_report(tmp_path, monkeypatch, capsys):
    """图更新失败时不写报告（不留半截），退出码 1。"""
    out, assets = tmp_path / "out", tmp_path / "assets"
    out.mkdir()
    # 不可按 UTF-8 解码的 CSV：build_graph 读取即失败
    (out / "PhysicalTable.csv").write_bytes(b"\xff\xfe\x00garbage\x00")
    _stub_config(monkeypatch, tmp_path)
    custom = tmp_path / "custom-report.json"

    with pytest.raises(SystemExit) as exc:
        cmd_graph(
            argparse.Namespace(
                output=str(out), mode="update", assets_dir=str(assets), report=str(custom)
            )
        )

    assert exc.value.code == 1
    assert not (out / "import-report.json").exists()
    assert not custom.exists()
    assert "未生成校验报告" in capsys.readouterr().err


def test_import_schema_one_shot_writes_report(duck_db, tmp_path, monkeypatch):
    """一步式 import-schema 同样产出校验报告。"""
    out, assets = tmp_path / "out", tmp_path / "assets"
    _stub_config(
        monkeypatch,
        tmp_path,
        extra={"datasources": {"mydb": {"url": f"duckdb://{duck_db}"}}},
    )
    cmd_import_schema(
        argparse.Namespace(
            datasource="mydb",
            schemas="main",
            output=str(out),
            assets_dir=str(assets),
            mode="update",
            report=None,
        )
    )

    report = _read_report(out / "import-report.json")
    assert report["import_counts"]["nodes"]["PhysicalTable"] == 2
    assert report["changes"]["totals"]["added"] > 0
    assert report["summary"]["status"] == "pass"
