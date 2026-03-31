import argparse
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import sys

import pandas as pd

from govio import NetworkXGraph, FalkorDBGraph


ASSETS_DIR = Path(__file__).parent.parent / "assets"
LOCAL_DIR = Path(__file__).parent
log_dir = os.path.join(LOCAL_DIR / "../", "logs")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file = os.path.join(log_dir, f"query_{datetime.now().strftime('%Y%m%d')}.log")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)

logger = logging.getLogger(__name__)


def output_result(data):
    """输出查询结果"""
    if not data:
        print("Data not found.")
        logger.info("data not found.")
        return

    if isinstance(data, list):
        _size = len(data)
    else:
        _size = 1
        data = [data]

    if _size > 0:
        if _size > 10:
            fname = (
                ASSETS_DIR / f"output-{datetime.now().strftime('%Y%m%d%H%M%s')}.json"
            )
            df = pd.DataFrame(data)
            df.to_json(
                fname, index=False, orient="records", lines=True, force_ascii=False
            )
            print("Result output:", fname, "rows:", _size)
            logger.info("result file: %s", str(fname))
        else:
            print(
                json.dumps(data, ensure_ascii=False, default=lambda obj: obj.__dict__)
            )
        logger.info("result rows: %s", str(_size))


def cmd_networkx(args):
    """NetworkX 子命令处理"""
    gml_path = Path(args.gml_path)

    if not gml_path.exists():
        print(f"GML file not found: {gml_path}")
        sys.exit(1)

    gf = NetworkXGraph(gml_path)
    g = gf.G

    _code = args.code
    logger.info("NetworkX Code: " + _code)

    local_vars = {"g": g}
    exec(_code, locals=local_vars)
    data = local_vars.get("result")
    output_result(data)


def cmd_falkordb(args):
    """FalkorDB 子命令处理"""
    host = args.host or os.getenv("FALKORDB_HOST", "localhost")
    port = args.port or int(os.getenv("FALKORDB_PORT", "6379"))

    g = FalkorDBGraph(graph=args.graph_name, host=host, port=port)

    _cypher = args.cypher
    if not _cypher.upper().startswith("MATCH"):
        print("Please write a MATCH query.")
        sys.exit(1)

    logger.info("Cypher: " + _cypher)

    data = g.query(_cypher)
    output_result(data)


def main():
    parser = argparse.ArgumentParser(description="统一图数据库查询入口")
    subparsers = parser.add_subparsers(dest="backend", help="选择图后端")

    # NetworkX 子命令
    nx_parser = subparsers.add_parser("networkx", help="使用 NetworkX 本地图查询")
    nx_parser.add_argument(
        "--gml-path",
        type=str,
        default=str(ASSETS_DIR / "ontology.gml"),
        help="GML 文件路径",
    )
    nx_parser.add_argument(
        "--code", type=str, required=True, help="NetworkX Python 查询代码"
    )
    nx_parser.set_defaults(func=cmd_networkx)

    # FalkorDB 子命令
    falkor_parser = subparsers.add_parser("falkordb", help="使用 FalkorDB 图数据库查询")
    falkor_parser.add_argument(
        "--graph-name", type=str, default="ontology", help="图名称"
    )
    falkor_parser.add_argument(
        "--host",
        type=str,
        help="数据库主机 (默认: localhost 或 FALKORDB_HOST 环境变量)",
    )
    falkor_parser.add_argument(
        "--port", type=int, help="数据库端口 (默认: 6379 或 FALKORDB_PORT 环境变量)"
    )
    falkor_parser.add_argument(
        "--cypher", type=str, required=True, help="Cypher 查询语句"
    )
    falkor_parser.set_defaults(func=cmd_falkordb)

    args = parser.parse_args()

    if not args.backend:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
