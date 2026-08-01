import csv
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import networkx as nx

from govio import FalkorDBGraph, LadybugGraph, NetworkXGraph
from govio.core.assets_generator import AssetsGenerator
from govio.graph.ladybug_loader import import_csv_to_ladybug


def create_test_gml(gml_path: Path):
    """创建测试用的 GML 文件"""
    G = nx.DiGraph()

    G.add_node("app1", name="应用1", node_type="Application", app_name_en="APP1")
    G.add_node(
        "table1", name="表1", node_type="PhysicalTable", full_table_name="SCHEMA.TABLE1"
    )
    G.add_node("col1", name="字段1", node_type="Col", column_name="COL1")
    G.add_node("col2", name="字段2", node_type="Col", column_name="COL2")

    G.add_edge("app1", "table1", edge_type="USE")
    G.add_edge("table1", "col1", edge_type="HAS_COLUMN")
    G.add_edge("table1", "col2", edge_type="HAS_COLUMN")

    nx.write_gml(G, gml_path)


def test_assets_generator_networkx_schema():
    with tempfile.TemporaryDirectory() as tmpdir:
        gml_path = Path(tmpdir) / "test.gml"
        output_dir = Path(tmpdir) / "assets"
        output_dir.mkdir()

        create_test_gml(gml_path)

        graph = NetworkXGraph(gml_path)
        generator = AssetsGenerator(graph, output_dir)

        generator.generate_schema()

        schema_path = output_dir / "schema.md"
        assert schema_path.exists()

        content = schema_path.read_text(encoding="utf-8")
        assert "NetworkX schema" in content
        assert "node_types" in content


def test_assets_generator_networkx_names():
    with tempfile.TemporaryDirectory() as tmpdir:
        gml_path = Path(tmpdir) / "test.gml"
        output_dir = Path(tmpdir) / "assets"
        output_dir.mkdir()

        create_test_gml(gml_path)

        graph = NetworkXGraph(gml_path)
        generator = AssetsGenerator(graph, output_dir)

        generator.generate_names()

        names_dir = output_dir / "names"
        assert names_dir.exists()

        node_names_path = names_dir / "node_names.md"
        assert node_names_path.exists()

        content = node_names_path.read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        assert len(lines) > 0
        node = json.loads(lines[0])
        assert "id" in node
        assert "name" in node
        assert "node_type" in node


def test_assets_generator_generate_all():
    with tempfile.TemporaryDirectory() as tmpdir:
        gml_path = Path(tmpdir) / "test.gml"
        output_dir = Path(tmpdir) / "assets"
        output_dir.mkdir()

        create_test_gml(gml_path)

        graph = NetworkXGraph(gml_path)
        generator = AssetsGenerator(graph, output_dir)

        generator.generate_all()

        assert (output_dir / "schema.md").exists()
        assert (output_dir / "names" / "node_names.md").exists()


def test_assets_generator_falkordb_names():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "assets"
        output_dir.mkdir()

        mock_graph = MagicMock(spec=FalkorDBGraph)
        mock_graph.schema = "## FalkorDB schema:\n节点：[]\n关联: []\n"

        mock_graph.query = MagicMock()
        mock_graph.query.side_effect = [
            [["APP1", "应用1"]],
            [["SCHEMA.TABLE1", "表1"]],
            [["COL1", "字段1"]],
        ]

        generator = AssetsGenerator(mock_graph, output_dir)
        generator.generate_all()

        assert (output_dir / "schema.md").exists()
        names_dir = output_dir / "names"
        assert names_dir.exists()
        app_file = names_dir / "应用1_APP1.md"
        assert app_file.exists()


def test_assets_generator_ladybug_names():
    """Ladybug 真实 Cypher 路径：generate_names 不应触发保留字解析错误。

    回归测试：Cypher 变量名 table 在 Ladybug 中是保留字（TABLE），
    曾导致 'Generated Cypher Statement is not valid' 解析失败。
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_dir = Path(tmpdir) / "csv"
        csv_dir.mkdir()
        db_path = str(Path(tmpdir) / "ontology.lbdb")

        def _write(name, header, rows):
            with open(csv_dir / name, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(header)
                w.writerows(rows)

        _write(
            "PhysicalTable.csv",
            [":ID(PhysicalTable)", "full_table_name", "name"],
            [["PT1", "db.dbo.T1", "T1"]],
        )
        _write(
            "Col.csv",
            [":ID(Col)", "column_name", "name", "full_table_name", "order_no"],
            [["CO1", "id", "ID", "db.dbo.T1", "1"]],
        )
        _write(
            "Application.csv",
            [":ID(Application)", "app_id", "name", "app_name_en"],
            [["AP1", "app1", "销售系统", "AEP"]],
        )
        _write(
            "HAS_COLUMN.csv",
            [":START_ID(PhysicalTable)", ":END_ID(Col)"],
            [["PT1", "CO1"]],
        )
        _write(
            "USE.csv",
            [":START_ID(Application)", ":END_ID(PhysicalTable)"],
            [["AP1", "PT1"]],
        )

        import_csv_to_ladybug(csv_dir, db_path)

        graph = LadybugGraph(db_path)
        output_dir = Path(tmpdir) / "assets"
        output_dir.mkdir()

        generator = AssetsGenerator(graph, output_dir)
        generator.generate_all()

        assert (output_dir / "schema.md").exists()
        names_dir = output_dir / "names"
        app_file = names_dir / "销售系统_AEP.md"
        assert app_file.exists()

        content = app_file.read_text(encoding="utf-8")
        assert "db.dbo.T1" in content
        assert "- id ID" in content
