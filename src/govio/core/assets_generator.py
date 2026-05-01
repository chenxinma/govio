"""AssetsGenerator - 生成图谱资产文件

从图谱对象生成 schema.md 和名称索引文件。
"""

import json
from pathlib import Path

from ..graph import NetworkXGraph, FalkorDBGraph


class AssetsGenerator:
    """资产文件生成器

    根据图谱类型（NetworkXGraph 或 FalkorDBGraph）生成相应的资产文件。

    Args:
        graph: NetworkXGraph 或 FalkorDBGraph 实例
        output_dir: 输出目录路径
    """

    def __init__(
        self, graph: NetworkXGraph | FalkorDBGraph, output_dir: Path
    ) -> None:
        self.graph = graph
        self.output_dir = output_dir

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_schema(self) -> None:
        """生成 schema.md 文件"""
        schema_path = self.output_dir / "schema.md"

        with open(schema_path, mode="w", encoding="utf-8") as fp:
            fp.write(self.graph.schema)

    def generate_names(self) -> None:
        """生成名称索引文件

        NetworkXGraph: 生成 names/node_names.md (JSON Lines 格式)
        FalkorDBGraph: 生成 names/{name}_{app_name_en}.md (Markdown 格式)
        """
        names_dir = self.output_dir / "names"
        if not names_dir.exists():
            names_dir.mkdir(parents=True, exist_ok=True)

        # 检测图谱类型
        if hasattr(self.graph, "G"):
            self._generate_names_networkx(names_dir)
        else:
            self._generate_names_falkordb(names_dir)

    def _generate_names_networkx(self, names_dir: Path) -> None:
        """为 NetworkX 生成名称索引

        格式: JSON Lines，每行一个节点
        {"id": "node_id", "name": "节点名称", "node_type": "Application"}
        """
        if not type(self.graph) is NetworkXGraph:
            return None

        g = self.graph.G

        nodes = [
            {
                "id": node_id,
                "name": g.nodes[node_id].get("name") or g.nodes[node_id].get("code", ""),
                "node_type": g.nodes[node_id]["node_type"],
            }
            for node_id in g.nodes()
            if (
                g.nodes[node_id].get("name") or g.nodes[node_id].get("code")
            )
            and (
                g.nodes[node_id].get("name") != "0"
                if g.nodes[node_id].get("name")
                else True
            )
            and isinstance(
                g.nodes[node_id].get("name") or g.nodes[node_id].get("code", ""),
                str,
            )
        ]

        if nodes:
            file_path = names_dir / "node_names.md"
            with open(file_path, "w", encoding="utf-8") as f:
                for node in nodes:
                    f.write(json.dumps(node, ensure_ascii=False) + "\n")

    def _generate_names_falkordb(self, names_dir: Path) -> None:
        """为 FalkorDB 生成名称索引

        按应用分组，每个应用一个文件
        格式: {name}_{app_name_en}.md
        """
        if not type(self.graph) is FalkorDBGraph:
            return None
        # 查询所有应用
        apps_query = """
        MATCH (app:Application)
        RETURN app.app_name_en AS app_name_en, app.name AS name
        ORDER BY app.app_name_en
        """
        apps = self.graph.query(apps_query)

        # 按应用逐次处理
        for app_row in apps:
            app_name_en, name = app_row

            # 查询该应用使用的所有物理表
            tables_query = """
            MATCH (app:Application {app_name_en: $app_name_en})-[:USE]->(table:PhysicalTable)
            RETURN table.full_table_name, table.name AS table_name
            ORDER BY table.full_table_name
            """
            tables = self.graph.query(tables_query, {"app_name_en": app_name_en})

            md_content = []

            # 按物理表逐次处理
            for table_row in tables:
                full_table_name, table_name = table_row

                if not table_name or table_name == "None":
                    table_name = ""

                md_content.append(f"# {full_table_name} {table_name}")

                # 查询该物理表的所有字段
                cols_query = """
                MATCH (table:PhysicalTable {full_table_name: $full_table_name})-[:HAS_COLUMN]->(col:Col)
                RETURN col.column_name, col.name AS col_name
                ORDER BY col.order_no
                """
                cols = self.graph.query(
                    cols_query, {"full_table_name": full_table_name}
                )

                for col_row in cols:
                    column_name, col_name = col_row
                    if not col_name or col_name == "None":
                        col_name = ""
                    md_content.append(f"- {column_name} {col_name}")

                md_content.append("")  # 空行分隔

            # 写入文件
            file_path = names_dir / f"{name}_{app_name_en}.md"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(md_content))

    def generate_metric_index(self) -> None:
        """生成指标索引文件 metrics_index.md

        按原子/派生分组列出所有指标，供 LLM 快速概览。
        """
        lines = ["# 指标索引\n"]
        metrics = []

        if type(self.graph) is NetworkXGraph and hasattr(self.graph, "G"):
            g = self.graph.G
            metrics = [
                (nid, g.nodes[nid])
                for nid in g.nodes()
                if g.nodes[nid].get("node_type") == "Metric"
            ]
        elif type(self.graph) is FalkorDBGraph:
            # FalkorDB: 查询所有 Metric 节点
            try:
                result = self.graph.query(
                    "MATCH (m:Metric) RETURN m.code, m.name, m.type, "
                    "m.formula, m.business_definition, m.source_layer, m.unit "
                    "ORDER BY m.type, m.code"
                )
                metrics = [
                    (
                        row[0], # pyright: ignore
                        {
                            "code": row[0], # pyright: ignore
                            "name": row[1], # pyright: ignore
                            "type": row[2], # pyright: ignore
                            "formula": row[3], # pyright: ignore
                            "business_definition": row[4], # pyright: ignore
                            "source_layer": row[5], # pyright: ignore
                            "unit": row[6], # pyright: ignore
                        },
                    )
                    for row in result
                ]
            except Exception:
                pass

        if not metrics:
            return

        # 原子指标
        atomic = [(nid, attrs) for nid, attrs in metrics if attrs.get("type") == "atomic"]
        derived = [(nid, attrs) for nid, attrs in metrics if attrs.get("type") == "derived"]

        if atomic:
            lines.append("## 原子指标\n")
            lines.append("| 编码 | 名称 | 业务定义 | 数仓层级 | 单位 |")
            lines.append("|------|------|----------|----------|------|")
            for _, attrs in atomic:
                lines.append(
                    f"| {attrs.get('code', '')} | {attrs.get('name', '')} | "
                    f"{attrs.get('business_definition', '')} | "
                    f"{attrs.get('source_layer', '')} | {attrs.get('unit', '')} |"
                )
            lines.append("")

        if derived:
            lines.append("## 派生指标\n")
            lines.append("| 编码 | 名称 | 公式 | 业务定义 | 单位 |")
            lines.append("|------|------|------|----------|------|")
            for _, attrs in derived:
                lines.append(
                    f"| {attrs.get('code', '')} | {attrs.get('name', '')} | "
                    f"`{attrs.get('formula', '')}` | "
                    f"{attrs.get('business_definition', '')} | "
                    f"{attrs.get('unit', '')} |"
                )
            lines.append("")

        file_path = self.output_dir / "metrics_index.md"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def generate_all(self) -> None:
        """生成所有资产文件"""
        self.generate_schema()
        self.generate_names()
        self.generate_metric_index()
