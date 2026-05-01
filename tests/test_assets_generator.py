import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import networkx as nx

from govio import FalkorDBGraph, NetworkXGraph
from govio.core.assets_generator import AssetsGenerator


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
