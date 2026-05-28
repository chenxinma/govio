import argparse
from pathlib import Path
import sys

from govio.cli.config import ConfigManager

from .meta_export import meta_export
from .onboard import onboard
from .std_recommend import std_recommend
from .observe import observe
from .query import query


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="数据治理知识图谱项目，提供元数据查询、表字段比较、SQL 生成、数据标准推荐等数据治理支持功能。",
    )
    sub = parser.add_subparsers(dest="action")

    p_onboard = sub.add_parser("onboard", help="初始化配置向导")
    p_onboard.add_argument(
        "--new-falkordb",
        type=Path,
        metavar="CSV_DIR",
        help="跳过 CSV 生成，直接将指定目录的 CSV 导入 FalkorDB",
    )
    p_onboard.add_argument(
        "--new-networkx",
        type=Path,
        metavar="CSV_DIR",
        help="跳过 CSV 生成，直接从指定目录的 CSV 生成 GML 文件",
    )
    p_std = sub.add_parser("std-recommend", help="数据标准推荐")
    p_std.add_argument("--output-dir", type=Path, help="推荐数据标准的输出目录")

    # query 子命令
    p_query = sub.add_parser("query", help="知识图谱查询")
    code_type = "NetworkX 用 Python 代码，FalkorDB 用 Cypher"
    config_manager = ConfigManager()
    if config_manager.exists():
        config = config_manager.load()
        backend = config.get("backend")
        if backend == "falkordb":
            code_type = "Cypher"
        elif backend == "networkx":
            code_type = "Python 代码"

    p_query.add_argument(
        "-c",
        "--code",
        help=f"查询语句（{code_type}）",
    )

    # meta-export 子命令：从 DuckDB 导出元数据 CSV
    p_meta = sub.add_parser("meta-export", help="从 DuckDB + TDS 合并导出全量元数据 CSV")
    p_meta.add_argument("--db", type=str, required=True, help="DuckDB 数据库文件路径")
    p_meta.add_argument(
        "--schemas",
        type=str,
        required=True,
        help="要导出的 schema 列表，逗号分隔（如 dm,dwd,dws）",
    )
    p_meta.add_argument("--output", type=Path, required=True, help="CSV 输出目录")
    p_meta.add_argument(
        "--dry-run",
        action="store_true",
        help="仅生成 CSV 并输出状态，不更新图数据和生成 assets",
    )

    # observe 子命令：保留未知参数传递给 observe()
    p_observe = sub.add_parser("observe", help="数据表探查")
    p_observe.add_argument(
        "observe_args", nargs=argparse.REMAINDER, help="observe 子命令参数"
    )

    args = parser.parse_args()

    if args.action == "onboard":
        onboard(new_falkordb=args.new_falkordb, new_networkx=args.new_networkx)
    elif args.action == "std-recommend":
        std_recommend(args.output_dir)
    elif args.action == "query":
        query(args.code)
    elif args.action == "meta-export":
        meta_export(args.db, args.schemas.split(","), args.output, dry_run=args.dry_run)
    elif args.action == "observe":
        # 将 observe 子命令参数设为 sys.argv 供 observe() 解析
        sys.argv = ["govio-cli"] + args.observe_args
        observe()
    else:
        parser.print_help()
        sys.exit(1)
