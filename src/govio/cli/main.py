import argparse
import sys

from .onboard import onboard
from .std_recommend import std_recommend
from .observe import observe


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="数据治理知识图谱项目，提供元数据查询、表字段比较、SQL 生成、数据标准推荐等数据治理支持功能。",
    )
    sub = parser.add_subparsers(dest="action")

    sub.add_parser("onboard", help="初始化配置向导")
    sub.add_parser("std-recommend", help="数据标准推荐")

    # observe 子命令：保留未知参数传递给 observe()
    p_observe = sub.add_parser("observe", help="数据表探查")
    p_observe.add_argument(
        "observe_args", nargs=argparse.REMAINDER, help="observe 子命令参数"
    )

    args = parser.parse_args()

    if args.action == "onboard":
        onboard()
    elif args.action == "std-recommend":
        std_recommend()
    elif args.action == "observe":
        # 将 observe 子命令参数设为 sys.argv 供 observe() 解析
        sys.argv = ["govio-cli"] + args.observe_args
        observe()
    else:
        parser.print_help()
        sys.exit(1)
