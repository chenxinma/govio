"""govio.graph.ladybug_graph

Ladybug 嵌入式图数据库后端，接口对齐 FalkorDBGraph：
连接本地 .lbdb 文件，执行 Cypher 查询，自动构建图模式信息。

Ladybug Python API 要点（实测 ladybug 0.19.0）：
- Database 必须显式传 max_db_size，否则默认 8TB mmap 在多数环境失败。
- QueryResult.get_all() 返回 list[list]（纯数据行，不含表头）。
- 标识符用反引号包裹（双引号会解析失败）。
"""
from os import PathLike
from typing import Any

import ladybug as lb

# 默认内存上限。max_db_size 是预留虚拟地址空间（非物理内存），
# 8TB 默认值在多数环境 mmap 失败，故显式设为 1GiB；元数据图谱足够。
_DEFAULT_BUFFER_POOL = 256 * 1024 * 1024
_DEFAULT_MAX_DB = 1 * 1024 * 1024 * 1024


class LadybugGraph:
    """Ladybug 图数据库客户端。

    Args:
        db_path: .lbdb 数据库文件路径，":memory:" 为内存模式。
        buffer_pool_size: buffer pool 大小（字节）。
        max_db_size: 数据库最大尺寸（字节），必须显式设置以规避 8TB mmap 限制。
        read_only: 是否只读打开。
    """

    def __init__(
        self,
        db_path: str | PathLike = "ontology.lbdb",
        *,
        buffer_pool_size: int = _DEFAULT_BUFFER_POOL,
        max_db_size: int = _DEFAULT_MAX_DB,
        read_only: bool = False,
    ) -> None:
        db = lb.Database(
            db_path,
            buffer_pool_size=buffer_pool_size,
            max_db_size=max_db_size,
            read_only=read_only,
        )
        self._conn = lb.Connection(db)
        self._db_path = str(db_path)

        self._schema: str = ""
        self.refresh_schema()

    def _bt(self, name: str) -> str:
        """反引号包裹标识符。"""
        return f"`{name}`"

    def _show_tables(self) -> list[dict[str, Any]]:
        """返回 [{name, type}]，type 为 NODE 或 REL。"""
        res = self._conn.execute("CALL SHOW_TABLES() RETURN *")
        tables: list[dict[str, Any]] = []
        for row in res.get_all():
            # [id, name, type, database name, comment]
            tables.append({"name": row[1], "type": row[2]})
        return tables

    def _node_properties(self, label: str) -> tuple[list[str], str | None]:
        """返回 (属性名列表, 主键属性名)。"""
        res = self._conn.execute(f"CALL TABLE_INFO({self._bt_val(label)}) RETURN *")
        props: list[str] = []
        pk: str | None = None
        for row in res.get_all():
            # [property id, name, type, default expression, primary key]
            name = row[1]
            is_pk = row[4]
            props.append(name)
            if is_pk:
                pk = name
        return props, pk

    def _bt_val(self, value: str) -> str:
        """生成 Cypher 字符串字面量（单引号）。"""
        escaped = value.replace("'", "\\'")
        return f"'{escaped}'"

    def _rel_endpoints(self, rel: str) -> tuple[str | None, str | None]:
        """通过数据发现关系的 (src_label, dst_label)。表为空时返回 (None, None)。"""
        try:
            res = self._conn.execute(
                f"MATCH (a)-[r:{self._bt(rel)}]->(b) "
                f"RETURN label(a), label(b) LIMIT 1"
            )
            rows = res.get_all()
            if rows:
                return rows[0][0], rows[0][1]
        except Exception:
            pass
        return None, None

    def refresh_schema(self) -> None:
        """刷新 Ladybug 图模式信息。"""
        nodes: list[dict[str, Any]] = []
        rels: list[dict[str, Any]] = []
        relationships: list[str] = []

        for table in self._show_tables():
            name = table["name"]
            if table["type"] == "NODE":
                props, pk = self._node_properties(name)
                nodes.append(
                    {"label": name, "primary_key": pk, "properties": props}
                )
            elif table["type"] == "REL":
                props, _ = self._node_properties(name)
                src, dst = self._rel_endpoints(name)
                rels.append({"label": name, "properties": props})
                if src and dst:
                    relationships.append(
                        f"(:{src})-[:{name}]->(:{dst})"
                    )

        self._schema = (
            "## Ladybug 图数据库结构:\n"
            f"节点：{nodes}\n"
            f"关联: {rels}\n"
            f"节点关联关系: {relationships}\n"
        )

    @property
    def schema(self) -> str:
        """Returns the schema of the Graph"""
        return self._schema

    @property
    def conn(self) -> lb.Connection:
        """底层 Ladybug Connection（供需要原生 API 的场景使用）。"""
        return self._conn

    def query(self, query: str, params: dict | None = None) -> list[list[Any]]:
        """执行 Cypher 查询，返回数据行列表（每行为 list）。

        与 FalkorDBGraph.query 行为一致：返回 list[list]，错误包装为 ValueError。
        """
        try:
            result = self._conn.execute(query, params or {})
            return result.get_all()
        except Exception as e:
            raise ValueError(f"Generated Cypher Statement is not valid\n{e}")
