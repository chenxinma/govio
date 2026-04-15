import argparse
from .onboard import onboard
from .std_recommend import std_recommend
from .observe import observe


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="数据治理知识图谱项目，提供元数据查询、表字段比较、SQL 生成、数据标准推荐等数据治理支持功能。",
    )
    parser.add_argument(
        "action",
        default="onboard",
        choices=["onboard", "std-recommend", "observe"],
        action="store",
    )
    args = parser.parse_args()
    if args.action == "onboard":
        onboard()
    elif args.action == "std-recommend":
        std_recommend()
    elif args.action == "observe":
        observe()
