import argparse
from importlib.metadata import version, PackageNotFoundError
import sys

from govio.cli.config import ConfigManager

from .onboard import onboard
from .observe import observe
from .query import query
from .meta import meta


def _get_version() -> str:
    try:
        return version("govio")
    except PackageNotFoundError:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="数据治理知识图谱项目，提供元数据查询、表字段比较、SQL 生成、数据标准推荐等数据治理支持功能。",
    )
    parser.add_argument("-V", "--version", action="version", version=f"govio {_get_version()}")
    sub = parser.add_subparsers(dest="action")

    # onboard 子命令
    sub.add_parser("onboard", help="初始化配置向导")

    sub.add_parser("backend", help="显示当前后端类型")

    # query 子命令
    p_query = sub.add_parser("query", help="知识图谱查询")
    code_type = "NetworkX 用 Python 代码，FalkorDB 用 Cypher"
    config_manager = ConfigManager()
    if config_manager.exists():
        config = config_manager.load()
        backend = (config.get("graph") or {}).get("backend")
        if backend == "falkordb":
            code_type = "Cypher"
        elif backend == "networkx":
            code_type = "Python 代码"

    p_query.add_argument(
        "-c",
        "--code",
        help=f"查询语句（{code_type}）",
    )

    # meta 子命令组
    p_meta = sub.add_parser("meta", help="知识图库维护", add_help=False)
    p_meta.add_argument(
        "meta_args", nargs=argparse.REMAINDER, help="meta 子命令参数"
    )

    # observe 子命令组
    p_observe = sub.add_parser("observe", help="数据表探查", add_help=False)
    p_observe.add_argument(
        "observe_args", nargs=argparse.REMAINDER, help="observe 子命令参数"
    )

    args, remaining = parser.parse_known_args()

    if args.action == "onboard":
        onboard()
    elif args.action == "backend":
        config_manager = ConfigManager()
        if not config_manager.exists():
            print("错误: 未找到配置文件，请先运行 govio-cli onboard", file=sys.stderr)
            sys.exit(1)
        config = config_manager.load()
        backend = (config.get("graph") or {}).get("backend")
        if not backend:
            print("错误: 配置文件中未设置后端类型", file=sys.stderr)
            sys.exit(1)
        print(backend)
    elif args.action == "query":
        query(args.code)
    elif args.action == "meta":
        sys.argv = ["govio-cli"] + args.meta_args + remaining
        meta()
    elif args.action == "observe":
        # 将 observe 子命令参数设为 sys.argv 供 observe() 解析
        sys.argv = ["govio-cli"] + args.observe_args + remaining
        observe()
    else:
        parser.print_help()
        sys.exit(1)
