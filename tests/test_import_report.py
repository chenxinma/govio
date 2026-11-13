"""import_report 单元测试。

覆盖：
1. classify_rows 的新增/更新/跳过分类（含 NaN 归一）
2. import-delta.json 的累计与取走清零
3. 导入计数、三项校验（重复 node_id / 必填缺失 / 孤儿边）
4. 报告组装与落盘（单次写入，内容契约）
"""

import json

import pandas as pd

from govio.metadata.import_report import (
    DELTA_FILE,
    REPORT_FILE,
    build_import_report,
    classify_rows,
    collect_import_counts,
    record_merge_delta,
    run_validation_checks,
    take_merge_delta,
    write_import_report,
)

# ---------------------------------------------------------------------------
# classify_rows
# ---------------------------------------------------------------------------


def test_classify_rows_fresh_import_counts_all_added(tmp_path):
    df = pd.DataFrame({"key": ["k1", "k2"], "name": ["a", "b"]})
    assert classify_rows(df, None, ["key"]) == {"added": 2, "updated": 0, "skipped": 0}


def test_classify_rows_empty_input_counts_zero():
    assert classify_rows(pd.DataFrame(), pd.DataFrame({"key": ["k1"]}), ["key"]) == {
        "added": 0,
        "updated": 0,
        "skipped": 0,
    }


def test_classify_rows_splits_added_updated_skipped():
    existing = pd.DataFrame({"key": ["k1", "k2", "k3"], "name": ["a", "b", "c"]})
    new = pd.DataFrame({"key": ["k1", "k2", "k4"], "name": ["a", "B", "d"]})
    assert classify_rows(new, existing, ["key"]) == {
        "added": 1,  # k4
        "updated": 1,  # k2 值变化
        "skipped": 1,  # k1 完全相同
    }


def test_classify_rows_treats_nan_and_empty_as_equal():
    existing = pd.DataFrame({"key": ["k1"], "comment": [float("nan")]})
    new = pd.DataFrame({"key": ["k1"], "comment": [""]})
    assert classify_rows(new, existing, ["key"])["skipped"] == 1


def test_classify_rows_missing_key_column_counts_all_added():
    new = pd.DataFrame({"name": ["a", "b"]})
    existing = pd.DataFrame({"key": ["k1"], "name": ["a"]})
    assert classify_rows(new, existing, ["key"])["added"] == 2


# ---------------------------------------------------------------------------
# import-delta.json 累计
# ---------------------------------------------------------------------------


def test_record_merge_delta_accumulates_and_take_clears(tmp_path):
    record_merge_delta(
        tmp_path, "nodes", "PhysicalTable", {"added": 2, "updated": 0, "skipped": 0}
    )
    record_merge_delta(
        tmp_path, "nodes", "PhysicalTable", {"added": 0, "updated": 1, "skipped": 3}
    )
    record_merge_delta(
        tmp_path, "edges", "HAS_COLUMN", {"added": 5, "updated": 0, "skipped": 0}
    )

    changes = take_merge_delta(tmp_path)
    assert changes["nodes"]["PhysicalTable"] == {"added": 2, "updated": 1, "skipped": 3}
    assert changes["edges"]["HAS_COLUMN"] == {"added": 5, "updated": 0, "skipped": 0}
    assert changes["totals"] == {"added": 7, "updated": 1, "skipped": 3}
    assert not (tmp_path / DELTA_FILE).exists()


def test_take_merge_delta_missing_file_returns_zero(tmp_path):
    changes = take_merge_delta(tmp_path)
    assert changes["totals"] == {"added": 0, "updated": 0, "skipped": 0}
    assert changes["nodes"] == {}
    assert changes["edges"] == {}


# ---------------------------------------------------------------------------
# 导入计数与校验项
# ---------------------------------------------------------------------------


def _write_clean_dir(tmp_path):
    pd.DataFrame(
        {
            ":ID(PhysicalTable)": ["PT1", "PT2"],
            "full_table_name": ["db.s.t1", "db.s.t2"],
            "name": ["T1", "T2"],
        }
    ).to_csv(tmp_path / "PhysicalTable.csv", index=False)
    pd.DataFrame(
        {
            ":ID(Col)": ["CO1"],
            "column": ["db.s.t1.c1"],
            "name": ["C1"],
        }
    ).to_csv(tmp_path / "Col.csv", index=False)
    pd.DataFrame(
        {
            ":START_ID(PhysicalTable)": ["PT1"],
            ":END_ID(Col)": ["CO1"],
        }
    ).to_csv(tmp_path / "HAS_COLUMN.csv", index=False)


def test_collect_import_counts_counts_nodes_and_edges(tmp_path):
    _write_clean_dir(tmp_path)
    counts = collect_import_counts(tmp_path)
    assert counts["nodes"] == {"PhysicalTable": 2, "Col": 1}
    assert counts["edges"] == {"HAS_COLUMN": 1}


def test_collect_import_counts_skips_malformed_headers(tmp_path):
    _write_clean_dir(tmp_path)
    pd.DataFrame({"id": ["X1"], "name": ["bad"]}).to_csv(
        tmp_path / "Standard.csv", index=False
    )
    counts = collect_import_counts(tmp_path)
    assert "Standard" not in counts["nodes"]


def test_validation_checks_pass_on_clean_dir(tmp_path):
    _write_clean_dir(tmp_path)
    checks = {c["name"]: c for c in run_validation_checks(tmp_path)}
    assert all(c["status"] == "pass" for c in checks.values())
    assert checks["orphan_edges"]["count"] == 0


def test_duplicate_node_id_detected(tmp_path):
    _write_clean_dir(tmp_path)
    pd.DataFrame(
        {
            ":ID(PhysicalTable)": ["PT1", "PT1"],
            "full_table_name": ["db.s.t1", "db.s.t1x"],
            "name": ["T1", "T1x"],
        }
    ).to_csv(tmp_path / "PhysicalTable.csv", index=False)
    checks = {c["name"]: c for c in run_validation_checks(tmp_path)}
    dup = checks["duplicate_node_ids"]
    assert dup["status"] == "fail"
    assert dup["count"] == 1
    assert any("PT1" in e for e in dup["examples"])


def test_missing_required_detected(tmp_path):
    _write_clean_dir(tmp_path)
    pd.DataFrame(
        {
            ":ID(PhysicalTable)": ["PT1", "PT2"],
            "full_table_name": ["db.s.t1", "db.s.t2"],
            "name": ["", "T2"],
        }
    ).to_csv(tmp_path / "PhysicalTable.csv", index=False)
    pd.DataFrame(
        {
            ":START_ID(PhysicalTable)": ["PT2", ""],
            ":END_ID(Col)": ["CO1", "CO1"],
        }
    ).to_csv(tmp_path / "HAS_COLUMN.csv", index=False)
    checks = {c["name"]: c for c in run_validation_checks(tmp_path)}
    missing = checks["missing_required"]
    assert missing["status"] == "fail"
    assert missing["count"] == 2  # 节点缺 name 1 + 边缺端点 1
    assert checks["orphan_edges"]["status"] == "pass"  # 空端点不重复计孤儿


def test_orphan_edge_detected(tmp_path):
    _write_clean_dir(tmp_path)
    pd.DataFrame(
        {
            ":START_ID(PhysicalTable)": ["PT1"],
            ":END_ID(Col)": ["CO404"],
        }
    ).to_csv(tmp_path / "HAS_COLUMN.csv", index=False)
    checks = {c["name"]: c for c in run_validation_checks(tmp_path)}
    orphan = checks["orphan_edges"]
    assert orphan["status"] == "fail"
    assert orphan["count"] == 1
    assert any("CO404" in e for e in orphan["examples"])


# ---------------------------------------------------------------------------
# 报告组装与落盘
# ---------------------------------------------------------------------------


def test_build_import_report_structure_and_summary(tmp_path):
    _write_clean_dir(tmp_path)
    record_merge_delta(
        tmp_path, "nodes", "PhysicalTable", {"added": 2, "updated": 0, "skipped": 0}
    )
    report = build_import_report(
        tmp_path, mode="update", backend="networkx", changes=take_merge_delta(tmp_path)
    )
    for key in (
        "report_version",
        "generated_at",
        "csv_dir",
        "mode",
        "backend",
        "import_counts",
        "changes",
        "checks",
        "summary",
    ):
        assert key in report
    assert report["mode"] == "update"
    assert report["backend"] == "networkx"
    assert report["import_counts"]["nodes"]["PhysicalTable"] == 2
    assert report["changes"]["totals"] == {"added": 2, "updated": 0, "skipped": 0}
    assert report["summary"]["status"] == "pass"
    assert report["summary"]["checks_failed"] == 0
    assert [c["name"] for c in report["checks"]] == [
        "duplicate_node_ids",
        "missing_required",
        "orphan_edges",
    ]


def test_build_import_report_summary_fail_on_orphan(tmp_path):
    _write_clean_dir(tmp_path)
    pd.DataFrame(
        {
            ":START_ID(PhysicalTable)": ["PT1"],
            ":END_ID(Col)": ["CO404"],
        }
    ).to_csv(tmp_path / "HAS_COLUMN.csv", index=False)
    report = build_import_report(tmp_path, mode="update", backend="networkx")
    assert report["summary"]["status"] == "fail"
    assert report["summary"]["checks_failed"] == 1


def test_write_import_report_roundtrip(tmp_path):
    _write_clean_dir(tmp_path)
    report = build_import_report(tmp_path, mode="update", backend="networkx")
    path = write_import_report(report, tmp_path / REPORT_FILE)
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data == report
