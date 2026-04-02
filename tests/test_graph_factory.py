import tempfile

import networkx as nx
import pytest
from pathlib import Path

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


@pytest.mark.skip(reason="FalkorDB not available in test environment")
def test_create_falkordb_graph_mock():
    config = {
        "backend": "falkordb",
        "falkordb": {"host": "localhost", "port": 6379, "graph": "test_graph"},
    }

    with pytest.raises(Exception):
        GraphFactory.create(config)


def test_create_unsupported_backend():
    config = {"backend": "unsupported"}

    with pytest.raises(ValueError, match="不支持的 backend"):
        GraphFactory.create(config)


def test_create_missing_backend():
    config = {}

    with pytest.raises(ValueError, match="配置缺少 'backend' 字段"):
        GraphFactory.create(config)


def test_create_networkx_missing_config():
    config = {"backend": "networkx"}

    with pytest.raises(ValueError, match="NetworkX backend 需要 'networkx' 配置"):
        GraphFactory.create(config)


def test_create_networkx_missing_gml_path():
    config = {"backend": "networkx", "networkx": {}}

    with pytest.raises(ValueError, match="NetworkX 配置缺少 'gml_path' 字段"):
        GraphFactory.create(config)


def test_create_falkordb_missing_config():
    config = {"backend": "falkordb"}

    with pytest.raises(ValueError, match="FalkorDB backend 需要 'falkordb' 配置"):
        GraphFactory.create(config)


def test_create_falkordb_missing_field():
    config = {"backend": "falkordb", "falkordb": {"host": "localhost"}}

    with pytest.raises(ValueError, match="FalkorDB 配置缺少 'port' 字段"):
        GraphFactory.create(config)
