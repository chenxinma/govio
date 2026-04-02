import pytest
from pathlib import Path
import tempfile
import networkx as nx
from govio.core.graph_factory import GraphFactory


def test_create_networkx_graph():
    with tempfile.TemporaryDirectory() as tmpdir:
        gml_path = Path(tmpdir) / "test.gml"
        G = nx.DiGraph()
        G.add_node(1, name="test", node_type="TestNode")
        nx.write_gml(G, gml_path)

        config = {"backend": "networkx", "networkx": {"gml_path": str(gml_path)}}

        graph = GraphFactory.create(config)
        assert graph is not None
        assert graph.G.number_of_nodes() == 1
        assert graph.G.number_of_edges() == 0


def test_create_networkx_graph_file_not_found():
    config = {"backend": "networkx", "networkx": {"gml_path": "/nonexistent/path.gml"}}

    with pytest.raises(FileNotFoundError):
        GraphFactory.create(config)


def test_create_falkordb_graph_mock():
    config = {
        "backend": "falkordb",
        "falkordb": {"host": "localhost", "port": 6379, "graph": "test_graph"},
    }

    with pytest.raises(Exception):
        GraphFactory.create(config)
