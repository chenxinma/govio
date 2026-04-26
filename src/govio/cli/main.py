import argparse
from pathlib import Path
import sys

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
    sub.add_parser("std-recommend", help="数据标准推荐")

    # query 子命令
    p_query = sub.add_parser("query", help="知识图谱查询")
    p_query.add_argument(
        "--assets",
        type=Path,
        default=Path("skills/govio/assets"),
        help="Assets 目录路径 (默认: skills/govio/assets)",
    )
    p_query.add_argument(
        "query_text",
        nargs="?",
        help="查询语句（NetworkX 用 Python 代码，FalkorDB 用 Cypher）",
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
        std_recommend()
    elif args.action == "query":
        # 将 query 参数设为 sys.argv 供 query() 解析
        q_argv = ["govio-cli", "query", "--assets", str(args.assets)]
        if args.query_text:
            q_argv.append(args.query_text)
        sys.argv = q_argv
        query()
    elif args.action == "observe":
        # 将 observe 子命令参数设为 sys.argv 供 observe() 解析
        sys.argv = ["govio-cli"] + args.observe_args
        observe()
    else:
        parser.print_help()
        sys.exit(1)
