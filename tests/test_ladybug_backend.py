"""Ladybug 后端测试：全量重建、查询、增量 upsert、工厂与配置。"""

import csv
from pathlib import Path

import pytest

from govio import LadybugGraph
from govio.cli.config import ConfigManager
from govio.core.graph_factory import GraphFactory
from govio.graph.ladybug_loader import (
    import_csv_to_ladybug,
    upsert_csv_to_ladybug,
)


def _write_csvs(csv_dir, tables=None, cols=None, apps=None, has_column=None, use=None):
    """按 FalkorDB 批量导入约定写入节点/边 CSV。"""
    csv_dir = Path(csv_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)

    def _write(name, header, rows):
        with open(csv_dir / name, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

    if tables is not None:
        _write(
            "PhysicalTable.csv",
            [":ID(PhysicalTable)", "full_table_name", "name"],
            tables,
        )
    if cols is not None:
        _write(
            "Col.csv",
            [":ID(Col)", "column_name", "name", "full_table_name", "order_no"],
            cols,
        )
    if apps is not None:
        _write(
            "Application.csv",
            [":ID(Application)", "app_id", "name", "app_name_en"],
            apps,
        )
    if has_column is not None:
        _write(
            "HAS_COLUMN.csv",
            [":START_ID(PhysicalTable)", ":END_ID(Col)"],
            has_column,
        )
    if use is not None:
        _write(
            "USE.csv",
            [":START_ID(Application)", ":END_ID(PhysicalTable)"],
            use,
        )


def test_ladybug_rebuild_and_query(tmp_path):
    csv_dir = tmp_path / "csv"
    db_path = str(tmp_path / "ontology.lbdb")

    _write_csvs(
        csv_dir,
        tables=[["PT1", "db.dbo.T1", "T1"], ["PT2", "db.dbo.T2", "T2"]],
        cols=[["CO1", "id", "ID", "db.dbo.T1", "1"]],
        apps=[["AP1", "app1", "销售系统", "AEP"]],
        has_column=[["PT1", "CO1"]],
        use=[["AP1", "PT1"]],
    )

    import_csv_to_ladybug(csv_dir, db_path)

    g = LadybugGraph(db_path)

    # schema 包含节点标签与关系签名
    assert "PhysicalTable" in g.schema
    assert "Application" in g.schema
    assert "(:Application)-[:USE]->(:PhysicalTable)" in g.schema

    # 查询所有物理表
    rows = g.query("MATCH (t:PhysicalTable) RETURN t.id, t.name ORDER BY t.name")
    assert rows == [["PT1", "T1"], ["PT2", "T2"]]

    # 带参数查询：AEP 使用的表
    rows = g.query(
        "MATCH (app:Application {app_name_en: $code})-[:USE]->(t:PhysicalTable) "
        "RETURN t.full_table_name",
        {"code": "AEP"},
    )
    assert rows == [["db.dbo.T1"]]

    # HAS_COLUMN 边
    rows = g.query(
        "MATCH (t:PhysicalTable)-[:HAS_COLUMN]->(c:Col) RETURN c.column_name"
    )
    assert rows == [["id"]]


def test_ladybug_upsert(tmp_path):
    csv_dir = tmp_path / "csv"
    db_path = str(tmp_path / "ontology.lbdb")

    # 初始重建
    _write_csvs(csv_dir, tables=[["PT1", "db.dbo.T1", "T1"]])
    import_csv_to_ladybug(csv_dir, db_path)

    # 增量：更新 PT1 名称 + 新增 PT2
    _write_csvs(
        csv_dir,
        tables=[["PT1", "db.dbo.T1", "T1_renamed"], ["PT2", "db.dbo.T2", "T2"]],
    )
    upsert_csv_to_ladybug(csv_dir, db_path)

    g = LadybugGraph(db_path)
    rows = g.query("MATCH (t:PhysicalTable) RETURN t.id, t.name ORDER BY t.name")
    # PT1 属性被更新，PT2 新增
    assert rows == [["PT1", "T1_renamed"], ["PT2", "T2"]]


def test_graph_factory_ladybug(tmp_path):
    db_path = str(tmp_path / "factory.lbdb")
    g = GraphFactory.create({"backend": "ladybug", "ladybug": {"db_path": db_path}})
    assert isinstance(g, LadybugGraph)
    assert isinstance(g.schema, str)


def test_graph_factory_ladybug_missing_db_path():
    with pytest.raises(ValueError, match="db_path"):
        GraphFactory.create({"backend": "ladybug", "ladybug": {}})


def test_config_validate_ladybug():
    # 合法配置
    ConfigManager._validate_backend(
        {"backend": "ladybug", "ladybug": {"db_path": "/tmp/x.lbdb"}}
    )
    # 缺少 db_path
    with pytest.raises(ValueError, match="db_path"):
        ConfigManager._validate_backend({"backend": "ladybug", "ladybug": {}})
