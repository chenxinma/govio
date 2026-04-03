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


def load_config() -> dict:
    """从 assets/backend.txt 加载 backend 配置"""
    backend_file = ASSETS_DIR / "backend.txt"
    if not backend_file.exists():
        print(f"Backend 配置文件不存在: {backend_file}")
        sys.exit(1)
    backend = backend_file.read_text().strip()
    return {"backend": backend}


def load_backend() -> str:
    """从 assets/backend.txt 读取 backend 类型"""
    backend_file = ASSETS_DIR / "backend.txt"
    if not backend_file.exists():
        print(f"Backend 配置文件不存在: {backend_file}")
        sys.exit(1)
    return backend_file.read_text().strip()


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


def cmd_networkx(args, config: dict):
    """NetworkX 子命令处理"""
    gml_path = config.get("networkx", {}).get("gml_path")
    if not gml_path:
        gml_path = args.gml_path or str(ASSETS_DIR / "ontology.gml")

    gml_path = Path(gml_path)
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


def cmd_falkordb(args, config: dict):
    """FalkorDB 子命令处理"""
    falkordb_config = config.get("falkordb", {})
    host = falkordb_config.get("host", "localhost")
    port = falkordb_config.get("port", 6379)
    graph_name = falkordb_config.get("graph", "ontology")

    g = FalkorDBGraph(graph=graph_name, host=host, port=port)

    _cypher = args.cypher
    if not _cypher.upper().startswith("MATCH"):
        print("Please write a MATCH query.")
        sys.exit(1)

    logger.info("Cypher: " + _cypher)

    data = g.query(_cypher)
    output_result(data)


def main():
    if len(sys.argv) > 1:
        query_text = sys.argv[1]
    elif not sys.stdin.isatty():
        query_text = sys.stdin.read().strip()
    else:
        print("请提供查询语句（NetworkX 用 Python 代码，FalkorDB 用 Cypher）")
        print('用法: python query.py "MATCH (n) RETURN n LIMIT 10"')
        sys.exit(1)

    backend = load_backend()

    if backend == "networkx":
        cmd_networkx(Namespace(code=query_text), {"backend": backend})
    elif backend == "falkordb":
        cmd_falkordb(Namespace(cypher=query_text), {"backend": backend})
    else:
        print(f"不支持的 backend: {backend}")
        sys.exit(1)


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


if __name__ == "__main__":
    main()
