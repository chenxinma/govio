"""
govio.graph.networkx_graph

"""
import networkx as nx

class NetworkXGraph:
    def __init__(self, graph="ontology.gml") -> None:
        self._g = nx.read_gml(graph)

        self._schema:str = ""
        self.refresh_schema()
    
    def refresh_schema(self):
        schema = {
            "node_types": {},
            "edge_relationships": {}
        }

        # 1. Inspect Nodes
        for node, data in self._g.nodes(data=True):            
            node_type = data["node_type"]

            if node_type not in schema["node_types"]:
                schema["node_types"][node_type] = set()

            # Collect all attribute keys seen for this node type
            schema["node_types"][node_type].update(data.keys())

        edge_types = set()
        # 2. Inspect Edges
        for u, v, data in self._g.edges(data=True):
            u_type = self._g.nodes[u]["node_type"]
            v_type = self._g.nodes[v]["node_type"]

            rel_type = data.get("edge_type", "connected_to")
            edge_types.add(rel_type)

            # Define relationship signature: source_type --[rel]--> target_type
            # Since G = nx.Graph() is undirected, you might want to sort types to avoid dups
            types = sorted([u_type, v_type])
            rel_key = f"({types[0]})-[{rel_type}]->({types[1]})"

            if rel_key not in schema["edge_relationships"]:
                schema["edge_relationships"][rel_key] = set()

            # Collect all attribute keys seen for this edge type
            schema["edge_relationships"][rel_key].update(data.keys())

        # Convert sets to lists for JSON serialization
    
        self._schema = (
            "## NetworkX schema:\n"
            f"node_types：{ {k: sorted(list(v)) for k, v in schema['node_types'].items()} }\n"
            f"edge_types: {list(edge_types)}\n"
            f"edge_relationships: { {k: sorted(list(v)) for k, v in schema['edge_relationships'].items()} }\n"
        )
    
    @property
    def schema(self) -> str:
        """Returns the schema of the Graph"""
        return self._schema
    

    @property
    def G(self) -> nx.Graph:
        """Returns the schema of the Graph"""
        return self._g