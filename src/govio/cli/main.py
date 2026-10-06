import argparse
import sys
from importlib.metadata import PackageNotFoundError, version

from govio.cli.config import ConfigManager


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

    # onboard 子命令（交互向导 + 非交互式数据源管理）
    p_onboard = sub.add_parser("onboard", help="初始化配置向导（支持非交互式添加/删除数据源）")
    p_onboard.add_argument(
        "--add-datasource",
        metavar="NAME",
        help="非交互式添加数据源（需配合 --url）",
    )
    p_onboard.add_argument(
        "--remove-datasource",
        metavar="NAME",
        help="非交互式删除数据源",
    )
    p_onboard.add_argument(
        "--url",
        help="数据源连接 URL，如 mysql+pymysql://user:pass@host:3306/db（密码自动加密存储）",
    )
    p_onboard.add_argument(
        "--password",
        help="数据源密码（URL 不含密码时单独提供，避免密码出现在 URL 中）",
    )
    p_onboard.add_argument(
        "--connect-args",
        action="append",
        metavar="KEY=VALUE",
        help="连接参数，可重复使用，如 --connect-args charset=utf8mb4 --connect-args timeout=30",
    )
    p_onboard.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖同名数据源",
    )

    sub.add_parser("backend", help="显示当前后端类型")

    # query 子命令
    p_query = sub.add_parser("query", help="知识图谱查询")
    code_type = "NetworkX 用 Python 代码，FalkorDB/Ladybug 用 Cypher"
    config_manager = ConfigManager()
    if config_manager.exists():
        config = config_manager.load()
        backend = (config.get("graph") or {}).get("backend")
        if backend in ("falkordb", "ladybug"):
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

    # sql 子命令组
    p_sql = sub.add_parser("sql", help="指标 SQL 组装", add_help=False)
    p_sql.add_argument(
        "sql_args", nargs=argparse.REMAINDER, help="sql 子命令参数"
    )

    args, remaining = parser.parse_known_args()

    if args.action == "onboard":
        from .onboard import onboard, onboard_datasource_cli

        if args.add_datasource or args.remove_datasource:
            onboard_datasource_cli(
                add_name=args.add_datasource,
                remove_name=args.remove_datasource,
                url=args.url,
                password=args.password,
                connect_args=args.connect_args,
                overwrite=args.overwrite,
            )
        elif args.url or args.password or args.connect_args or args.overwrite:
            print(
                "错误: --url/--password/--connect-args/--overwrite 需配合 --add-datasource 使用",
                file=sys.stderr,
            )
            sys.exit(1)
        else:
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
        from .query import query

        query(args.code)
    elif args.action == "meta":
        from .meta import meta

        sys.argv = ["govio-cli", *args.meta_args, *remaining]
        meta()
    elif args.action == "observe":
        from .observe import observe

        # 将 observe 子命令参数设为 sys.argv 供 observe() 解析
        sys.argv = ["govio-cli", *args.observe_args, *remaining]
        observe()
    elif args.action == "sql":
        from .sql import sql

        sys.argv = ["govio-cli", *args.sql_args, *remaining]
        sql()
    else:
        parser.print_help()
        sys.exit(1)
