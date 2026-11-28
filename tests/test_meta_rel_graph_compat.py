"""meta rel / meta graph CSV 表头兼容 + --mode update 旧库兼容回归（0.8.0 缺陷修复）。

缺陷 1：meta rel 产物 RELATES_TO.csv 可能带杂散前置列 source,target（0 有效关系
导出写脏表头、二次导出 merge 并入），meta graph 导入报 Binder 裸异常。
缺陷 2：--mode update 对旧版表结构（缺 node_id 等属性列）报
Cannot find property node_id for n 裸异常。
"""

import json

import pandas as pd
import pytest

from govio import LadybugGraph
from govio.cli.meta import step_meta_export, step_rel_export
from govio.cli.meta.merge import _load_csv_with_node_ids, merge_edge_csv
from govio.graph.ladybug_loader import (
    import_csv_to_ladybug,
    upsert_csv_to_ladybug,
)

DEDUP = [":START_ID(PhysicalTable)", ":END_ID(PhysicalTable)", "relationship_type"]
RELATES_TO_HEADER = [
    ":START_ID(PhysicalTable)",
    ":END_ID(PhysicalTable)",
    "relationship_type",
    "description",
    "source_columns",
    "target_columns",
]


@pytest.fixture
def duck_db(tmp_path):
    """构造只含 main schema 的临时 DuckDB 文件。"""
    import duckdb

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


# ---------------------------------------------------------------------------
# 缺陷 1：merge_edge_csv 写盘规范化
# ---------------------------------------------------------------------------


def test_merge_edge_csv_drops_legacy_stray_columns(tmp_path, capsys):
    """已有 CSV 带杂散列（source,target）时合并后表头规范、旧数据保留。"""
    rel_path = tmp_path / "RELATES_TO.csv"
    rel_path.write_text(
        "source,target,:START_ID(PhysicalTable),:END_ID(PhysicalTable),"
        "relationship_type,description\n"
        ",,PT0,PT9,one_to_one,旧数据\n",
        encoding="utf-8",
    )
    new_df = pd.DataFrame(
        [
            {
                ":START_ID(PhysicalTable)": "PT1",
                ":END_ID(PhysicalTable)": "PT2",
                "relationship_type": "many_to_one",
                "description": "新数据",
            }
        ]
    )

    merged = merge_edge_csv(new_df, rel_path, DEDUP)

    lines = rel_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == ",".join(
        [
            ":START_ID(PhysicalTable)",
            ":END_ID(PhysicalTable)",
            "relationship_type",
            "description",
        ]
    )
    assert len(merged) == 2
    assert merged["description"].tolist() == ["旧数据", "新数据"]
    assert "source" not in merged.columns and "target" not in merged.columns
    assert "source" in capsys.readouterr().err  # 丢弃杂散列有可读告警


def test_rel_export_with_legacy_relates_to_is_normalized_and_importable(
    duck_db, rel_file, tmp_path
):
    """缺陷 1 回归：旧版杂散列表头经 meta rel 重跑后规范，且能被 ladybug 导入。"""
    out = tmp_path / "out"
    assert step_meta_export(
        out, source="duckdb", db_path=duck_db, schemas=["main"], datasource_name="local"
    )

    # 模拟 0.8.0 产物：杂散 source,target 前置列 + 一行旧边
    tables = pd.read_csv(out / "PhysicalTable.csv")
    pt_ids = tables[":ID(PhysicalTable)"].tolist()
    (out / "RELATES_TO.csv").write_text(
        "source,target,:START_ID(PhysicalTable),:END_ID(PhysicalTable),"
        "relationship_type,description,source_columns,target_columns\n"
        f",,{pt_ids[0]},{pt_ids[1]},one_to_one,旧数据,x,x\n",
        encoding="utf-8",
    )

    step_rel_export(out, rel_file)

    header = (out / "RELATES_TO.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header == ",".join(RELATES_TO_HEADER)

    db_path = str(tmp_path / "ontology.lbdb")
    import_csv_to_ladybug(out, db_path)
    g = LadybugGraph(db_path, read_only=True)
    rows = g.query("MATCH ()-[r:RELATES_TO]->() RETURN r.relationship_type")
    assert len(rows) == 2  # 旧边保留 + 新边入库
    assert sorted(r[0] for r in rows) == ["many_to_one", "one_to_one"]


def test_rel_export_empty_result_keeps_endpoint_header(duck_db, tmp_path):
    """0 有效关系的导出不再写 source,target 表头（杂散列根因）。"""
    out = tmp_path / "out"
    assert step_meta_export(
        out, source="duckdb", db_path=duck_db, schemas=["main"], datasource_name="local"
    )
    bad_rel = tmp_path / "bad.json"
    bad_rel.write_text(
        json.dumps(
            {
                "version": "1.0",
                "relationships": [
                    {
                        "source": {"PhysicalTable": "local.main.nope", "Cols": ["x"]},
                        "target": {"PhysicalTable": "local.main.b", "Cols": ["x"]},
                        "relationship_type": "many_to_one",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    step_rel_export(out, str(bad_rel))

    header = (out / "RELATES_TO.csv").read_text(encoding="utf-8").splitlines()[0]
    cols = header.split(",")
    assert cols[:3] == DEDUP
    assert "source" not in cols and "target" not in cols


# ---------------------------------------------------------------------------
# 缺陷 1 加重面：ladybug loader 边表容错
# ---------------------------------------------------------------------------


def _write_minimal_csvs(csv_dir, edge_text):
    csv_dir.mkdir(parents=True, exist_ok=True)
    (csv_dir / "PhysicalTable.csv").write_text(
        ":ID(PhysicalTable),full_table_name,name\nPT1,db.s.a,A\nPT2,db.s.b,B\n",
        encoding="utf-8",
    )
    (csv_dir / "RELATES_TO.csv").write_text(edge_text, encoding="utf-8")


def test_ladybug_import_tolerates_stray_edge_columns(tmp_path, capsys):
    """边表带全空杂散列（source,target）也能导入：丢弃杂散列、端点列重排兜底。"""
    csv_dir = tmp_path / "csv"
    _write_minimal_csvs(
        csv_dir,
        "source,target,:START_ID(PhysicalTable),:END_ID(PhysicalTable),relationship_type\n"
        ",,PT1,PT2,many_to_one\n",
    )
    db_path = str(tmp_path / "ontology.lbdb")

    import_csv_to_ladybug(csv_dir, db_path)

    assert "source" in capsys.readouterr().err  # 杂散列丢弃有可读告警
    g = LadybugGraph(db_path, read_only=True)
    rows = g.query(
        "MATCH (:PhysicalTable)-[r:RELATES_TO]->(:PhysicalTable) RETURN r.relationship_type"
    )
    assert rows == [["many_to_one"]]


def test_ladybug_edge_header_without_endpoints_raises_readable_error(tmp_path):
    """边表缺端点列时报可读错误，不再透出 Binder 裸异常。"""
    csv_dir = tmp_path / "csv"
    _write_minimal_csvs(
        csv_dir, "source,target,relationship_type\nPT1,PT2,many_to_one\n"
    )

    with pytest.raises(ValueError, match="表头不符合规范"):
        import_csv_to_ladybug(csv_dir, str(tmp_path / "ontology.lbdb"))


# ---------------------------------------------------------------------------
# 缺陷 2：--mode update 旧版表结构兼容
# ---------------------------------------------------------------------------


def test_upsert_on_legacy_schema_skips_missing_props_with_guidance(tmp_path, capsys):
    """旧库（无 node_id 属性列）+ 带 node_id 列的 CSV：增量更新成功并给迁移指引。"""
    csv_dir = tmp_path / "csv"
    db_path = str(tmp_path / "ontology.lbdb")
    _write_minimal_csvs(csv_dir, "")
    (csv_dir / "RELATES_TO.csv").unlink()
    # 旧版表结构：节点 CSV 不含 node_id 属性列
    import_csv_to_ladybug(csv_dir, db_path)

    # 模拟 0.8.0 _load_csv_with_node_ids 产物：node_id 属性列混入节点 CSV
    (csv_dir / "PhysicalTable.csv").write_text(
        ":ID(PhysicalTable),full_table_name,name,node_id\n"
        "PT1,db.s.a,A2,PT1\n"
        "PT2,db.s.b,B2,PT2\n"
        "PT3,db.s.c,C3,PT3\n",
        encoding="utf-8",
    )

    upsert_csv_to_ladybug(
        csv_dir, db_path
    )  # 修复前报 Cannot find property node_id for n

    err = capsys.readouterr().err
    assert "node_id" in err and "--mode rebuild" in err  # 可读迁移指引
    g = LadybugGraph(db_path, read_only=True)
    rows = g.query("MATCH (n:PhysicalTable) RETURN n.name ORDER BY n.name")
    assert rows == [["A2"], ["B2"], ["C3"]]  # 其余属性正常合并


def test_load_csv_with_node_ids_keeps_canonical_header(tmp_path):
    """node_id 只在内存中统一，不向节点 CSV 回写重复列；两列并存的历史产物被治愈。"""
    pt_path = tmp_path / "PhysicalTable.csv"
    pt_path.write_text(
        ":ID(PhysicalTable),full_table_name,name\nPT1,db.s.a,A\n", encoding="utf-8"
    )

    df = _load_csv_with_node_ids(pt_path, "PhysicalTable", "full_table_name")

    assert df["node_id"].tolist() == ["PT1"]
    assert pt_path.read_text(encoding="utf-8").splitlines()[0] == (
        ":ID(PhysicalTable),full_table_name,name"
    )

    pt_path.write_text(
        ":ID(PhysicalTable),full_table_name,name,node_id\nPT1,db.s.a,A,PT1\n",
        encoding="utf-8",
    )
    _load_csv_with_node_ids(pt_path, "PhysicalTable", "full_table_name")
    assert pt_path.read_text(encoding="utf-8").splitlines()[0] == (
        ":ID(PhysicalTable),full_table_name,name"
    )
