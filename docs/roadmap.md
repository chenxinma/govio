# Roadmap

## Infrastructure Improvements

### Local Embedded Graph Database Investigation
**Objective:** Replace FalkorDB with a lightweight, embedded solution to eliminate the need for a separate server process while maintaining graph storage capabilities for applications, tables, fields, and data standards.

**Current State:**
- Using FalkorDB (requires Docker/server process).
- Stores property graphs (Nodes: Application, Table, Field; Edges: relationships).
- Querying via openCypher.

**Alternatives Analyzed:**

1.  **TinkerGraph (via gremlin-python)**
    *   **Type:** In-memory property graph (Apache TinkerPop).
    *   **Pros:** Zero-config, embedded, supports powerful traversals (Gremlin) similar to Cypher, high performance.
    *   **Cons:** Gremlin syntax differs from Cypher (learning curve), persistence requires manual serialization (e.g., GraphML).
    *   **Verdict:** Strongest candidate for preserving graph traversal capabilities.

2.  **RDFLib**
    *   **Type:** RDF Triple Store.
    *   **Pros:** Pure Python, standard for semantic data, supports SPARQL (declarative).
    *   **Cons:** Triple model differs from property graphs, potential performance overhead for large datasets.
    *   **Verdict:** Good if shifting towards semantic web standards, but requires data model mapping.

3.  **NetworkX**
    *   **Type:** Pure Python Graph Library.
    *   **Pros:** Extremely lightweight, no new query language (pure Python API), easy integration.
    *   **Cons:** No declarative query language (imperative traversals), scaling limits for very large graphs.
    *   **Verdict:** Best for simplicity if complex graph queries can be replaced with Python logic.

**Proposed Migration Plan:**
1.  **Export:** Dump current data from FalkorDB (Cypher export or JSON).
2.  **Schema Mapping:** Map property graph entities to the chosen format (e.g., NetworkX objects or Gremlin vertices).
3.  **Implementation:**
    *   Update `skills/govio/scripts/query.py` to use the new library.
    *   Refactor `skills/govio/SKILL.md` examples.
4.  **Verification:** Verify SQL generation and metadata comparison workflows.

**Next Steps:**
- [ ] Select final candidate (Recommendation: **TinkerGraph** for query power or **NetworkX** for simplicity).
- [ ] Prototype a replacement for `query.py` using the selected library.
