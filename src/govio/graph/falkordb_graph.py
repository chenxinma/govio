import textwrap
from typing import Any, Generator
from falkordb import FalkorDB

class FalkorDBGraph:
    def __init__(self, graph="ontology", host='localhost', port=6379) -> None:
        self.db = FalkorDB(host=host, port=port)
        self._g = self.db.select_graph(graph)

        self._schema:str = ""
        self.refresh_schema()
    
    def _get_labels(self) ->  Generator[str, Any, None]:
        result = self._g.query("CALL db.labels()").result_set
        for label in result:
            yield label[0]
    
    def _get_property_names(self, node: str):
        result = self._g.query(f"MATCH (n:{node}) WITH n LIMIT 20 UNWIND keys(n) AS k RETURN DISTINCT k").result_set
        for prop in result:
            yield prop[0]
    
    def _get_relateships(self):
        result = self._g.query(
            textwrap.dedent("""
                MATCH (n)-[r]->(m)
                UNWIND labels(n) as src_label
                UNWIND labels(m) as dst_label
                UNWIND type(r) as rel_type
                RETURN DISTINCT {start: src_label, type: rel_type, end: dst_label} AS output"""
            )).result_set
        for r in result:
            yield r[0]
    
    def _get_rel_properties(self):
        q = textwrap.dedent(
                """
                MATCH ()-[r]->()
                WITH keys(r) as keys, type(r) AS types
                WITH CASE WHEN keys = [] THEN [NULL] ELSE keys END AS keys, types 
                UNWIND types AS type
                UNWIND keys AS key WITH type,
                collect(DISTINCT key) AS keys 
                RETURN {types:type, keys:keys} AS output
                """)
        result = self._g.query(q).result_set
        for rp in result:
            yield rp[0]
                            
    
    def _wrap_name(self, name: str) -> str:
        """Wrap name with backticks."""
        if name in ['Column']:
            return f"`{name}`"
        return name
    

    def refresh_schema(self):
        """Refreshes the Kùzu graph schema information"""
        node_properties = []
        for node in self._get_labels():
            current_table_schema = {"properties": [], "label": self._wrap_name(node)}
            properties = self._get_property_names(node)
            for property_name in properties:
                if property_name.startswith(":ID"):
                    continue
                current_table_schema["properties"].append(
                    property_name
                )
            node_properties.append(current_table_schema)

        relationships = []
        rels = self._get_relateships()
        for r in rels:
            relationships.append(
                "(:%s)-[:%s]->(:%s)" % (self._wrap_name(r["start"]), r["type"], self._wrap_name(r["end"]))
            )

        rel_properties = [ { 'label': rp['types'], 'properties': rp['keys'] }  for rp in self._get_rel_properties() ]

        self._schema = (
            "## 图数据库结构:\n"
            f"节点：{node_properties}\n"
            f"关联: {rel_properties}\n"
            f"节点关联关系: {relationships}\n"
        )

    @property
    def schema(self) -> str:
        """Returns the schema of the Graph"""
        return self._schema
    
    def query(self, query: str, params: dict = {}) -> list[dict[str, Any]]:
        """Query FalkorDB database."""

        try:
            data = self._g.query(query, params)
            return data.result_set
        except Exception as e:
            raise ValueError(f"Generated Cypher Statement is not valid\n{e}")
