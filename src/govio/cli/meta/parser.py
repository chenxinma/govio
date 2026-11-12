"""meta 命令组的 argparse 装配（`govio-cli meta`）

子命令与参数与原单文件实现完全一致：`meta / std / compliance / rel / metric /
schema / graph / recommend / import-schema`。处理器在 `commands.py`。
"""

import argparse
import sys

from .commands import (
    cmd_compliance,
    cmd_graph,
    cmd_import_schema,
    cmd_meta,
    cmd_metric,
    cmd_recommend,
    cmd_rel,
    cmd_schema,
    cmd_std,
)


def meta():
    """meta 命令入口"""
    parser = argparse.ArgumentParser(
        prog="govio-cli meta",
        description="知识图库维护 — 元数据导入、推荐、图更新",
    )
    sub = parser.add_subparsers(dest="action", required=True)

    # --- 导入子命令 ---

    # TDS 参数复用
    tds_args = argparse.ArgumentParser(add_help=False)
    tds_args.add_argument("--kundb", type=str, required=True, help="TDS 元数据库 URL")
    tds_args.add_argument(
        "--workspace-uuid",
        type=str,
        default="82ee37374b314a938bf28170ab4db7cf",
        help="工作区 UUID",
    )

    # meta meta — 元数据导入
    p_meta = sub.add_parser(
        "meta",
        help="导入 TDS/DuckDB/MySQL/PostgreSQL/Trino 元数据（Datasource, PhysicalTable, Col, HAS_COLUMN, OWNS）",
    )
    p_meta.add_argument(
        "--source",
        choices=["tds", "duckdb", "mysql", "postgres", "trino"],
        required=True,
        help="信息来源",
    )
    p_meta.add_argument(
        "--db",
        type=str,
        help="数据源位置：duckdb 为文件路径；mysql/postgres/trino 为连接 URL（如 mysql://user:pass@host:3306/db）",
    )
    p_meta.add_argument(
        "--schemas",
        type=str,
        help="要导出的 schema 列表，逗号分隔（非 TDS 模式必填；DuckDB 文件默认 schema 为 main）",
    )
    p_meta.add_argument(
        "--datasource",
        type=str,
        help="数据源名（非 TDS 模式必填，生成 Datasource 节点与 OWNS 归属）",
    )
    p_meta.add_argument(
        "--datasources-file",
        type=str,
        help="数据源声明 JSON（TDS 模式必填，抽取范围取 filter.schemas）",
    )
    p_meta.add_argument("--kundb", type=str, help="TDS 元数据库 URL（TDS 模式必须）")
    p_meta.add_argument(
        "--workspace-uuid", type=str, help="工作区 UUID（TDS 模式必须）"
    )
    p_meta.add_argument("--output", type=str, help="CSV 输出目录（默认 ./output）")
    p_meta.set_defaults(func=cmd_meta)

    # meta std — 数据标准导入
    p_std = sub.add_parser(
        "std", help="导入数据标准（Standard 节点）", parents=[tds_args]
    )
    p_std.add_argument("--output", type=str, help="CSV 输出目录（默认 ./output）")
    p_std.set_defaults(func=cmd_std)

    # meta compliance — 已有标准关联
    p_comp = sub.add_parser(
        "compliance",
        help="从 TDS 导出已有标准-字段关联（COMPLIES_WITH 边）",
        parents=[tds_args],
    )
    p_comp.add_argument(
        "--datasources-file",
        type=str,
        required=True,
        help="数据源声明 JSON（schema 归属，用于生成全限定列标识）",
    )
    p_comp.add_argument("--output", type=str, help="CSV 输出目录（默认 ./output）")
    p_comp.set_defaults(func=cmd_compliance)

    # meta rel — 表关系导入
    p_rel = sub.add_parser("rel", help="导入表关系（RELATES_TO 边）")
    p_rel.add_argument("--file", type=str, required=True, help="表关系 JSON 文件路径")
    p_rel.add_argument("--output", type=str, help="CSV 输出目录（默认 ./output）")
    p_rel.set_defaults(func=cmd_rel)

    # meta metric — 指标维度导入
    p_metric = sub.add_parser(
        "metric", help="导入指标维度定义（Metric, Dimension + 边）"
    )
    p_metric.add_argument(
        "--file", type=str, required=True, help="指标定义 JSON 文件路径"
    )
    p_metric.add_argument("--output", type=str, help="CSV 输出目录（默认 ./output）")
    p_metric.set_defaults(func=cmd_metric)

    # --- JSON Schema 输出 ---

    p_schema = sub.add_parser(
        "schema", help="输出标准 JSON Schema（供外部 agent 生成标准数据）"
    )
    p_schema.add_argument(
        "schema_name",
        choices=["metric", "relationship", "datasource"],
        help="schema 名称",
    )
    p_schema.add_argument(
        "-o", "--output", type=str, help="输出文件路径（缺省打印到 stdout）"
    )
    p_schema.set_defaults(func=cmd_schema)

    # --- 图更新 ---

    p_graph = sub.add_parser("graph", help="更新图数据库 + 生成 assets")
    p_graph.add_argument("--output", type=str, help="CSV 输出目录（默认 ./output）")
    p_graph.add_argument(
        "--assets-dir",
        type=str,
        help="assets 输出目录（默认 .agents/skills/govio/assets）",
    )
    p_graph.add_argument(
        "--mode",
        choices=["update", "rebuild", "clear"],
        default="update",
        help="更新模式: update=增量, rebuild=重建, clear=清空",
    )
    p_graph.add_argument(
        "--report",
        type=str,
        help="校验报告输出路径（默认 <output>/import-report.json）",
    )
    p_graph.set_defaults(func=cmd_graph)

    # --- 数据标准推荐 ---

    p_recommend = sub.add_parser("recommend", help="数据标准推荐")
    p_recommend.add_argument(
        "--kundb", type=str, required=True, help="TDS 元数据库 URL"
    )
    p_recommend.add_argument(
        "--workspace-uuid",
        type=str,
        default="82ee37374b314a938bf28170ab4db7cf",
        help="工作区 UUID",
    )
    p_recommend.add_argument(
        "--datasource",
        type=str,
        required=True,
        help="数据源名（分析范围取其 filter.schemas）",
    )
    p_recommend.add_argument(
        "--datasources-file",
        type=str,
        help="数据源声明 JSON（缺省时从 --csv-dir 的 Datasource.csv 解析）",
    )
    p_recommend.add_argument(
        "--csv-dir", type=str, help="已导入的 CSV 目录（默认同 --output-dir）"
    )
    p_recommend.add_argument(
        "--output-dir", type=str, help="推荐结果输出目录（默认 ./output）"
    )
    p_recommend.set_defaults(func=cmd_recommend)

    # meta import-schema — 从已配置数据源导入元数据（仅 DuckDB）
    p_import = sub.add_parser(
        "import-schema",
        help="从已配置的 DuckDB 数据源导入元数据到图库",
    )
    p_import.add_argument(
        "--datasource", type=str, required=True, help="数据源名称（仅支持 DuckDB）"
    )
    p_import.add_argument(
        "--schemas",
        type=str,
        required=True,
        help="要导入的 schema 列表，逗号分隔（DuckDB 默认 schema 为 main）",
    )
    p_import.add_argument("--output", type=str, help="CSV 输出目录（默认 ./output）")
    p_import.add_argument(
        "--assets-dir",
        type=str,
        help="assets 输出目录（默认 .agents/skills/govio/assets）",
    )
    p_import.add_argument(
        "--mode",
        choices=["update", "rebuild", "clear"],
        default="update",
        help="更新模式: update=增量, rebuild=重建, clear=清空",
    )
    p_import.add_argument(
        "--report",
        type=str,
        help="校验报告输出路径（默认 <output>/import-report.json）",
    )
    p_import.set_defaults(func=cmd_import_schema)

    args = parser.parse_args(sys.argv[1:])
    args.func(args)


if __name__ == "__main__":
    meta()


if __name__ == "__main__":
    meta()
