"""Govio 知识图谱统一查询入口

支持通过 CLI 或作为包内模块调用。
用法: govio query --assets <path> "查询语句"
"""
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import sys

from govio import NetworkXGraph, FalkorDBGraph
import pandas as pd


log_dir = Path.home() / ".govio" / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

log_file = log_dir / f"query_{datetime.now().strftime('%Y%m%d')}.log"
logging.basicConfig(
    filename=str(log_file),
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)

logger = logging.getLogger(__name__)


def load_backend(assets_dir: Path) -> str:
    """从 assets/backend.txt 读取 backend 类型"""
    backend_file = assets_dir / "backend.txt"
    if not backend_file.exists():
        print(f"Backend 配置文件不存在: {backend_file}")
        sys.exit(1)
    return backend_file.read_text().strip()


def output_result(data, assets_dir: Path):
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
                assets_dir / f"output-{datetime.now().strftime('%Y%m%d%H%M%s')}.json"
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


def cmd_networkx(code: str, assets_dir: Path, gml_path: str | None = None):
    """NetworkX 查询处理"""
    if not gml_path:
        gml_path = str(assets_dir / "ontology.gml")

    gml_path = Path(gml_path)
    if not gml_path.exists():
        print(f"GML file not found: {gml_path}")
        sys.exit(1)

    gf = NetworkXGraph(gml_path)
    g = gf.G

    logger.info("NetworkX Code: " + code)

    local_vars = {"g": g}
    exec(code, locals=local_vars)
    data = local_vars.get("result")
    output_result(data, assets_dir)


def cmd_falkordb(cypher: str, assets_dir: Path, host: str = "localhost", port: int = 6379, graph_name: str = "ontology"):
    """FalkorDB 查询处理"""
    g = FalkorDBGraph(graph=graph_name, host=host, port=port)

    if not cypher.upper().startswith("MATCH"):
        print("Please write a MATCH query.")
        sys.exit(1)

    logger.info("Cypher: " + cypher)

    data = g.query(cypher)
    output_result(data, assets_dir)


def query():
    """Query CLI 主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Govio 知识图谱查询")
    parser.add_argument(
        "--assets",
        type=Path,
        default=Path("skills/govio/assets"),
        help="Assets 目录路径 (默认: skills/govio/assets)",
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="查询语句（NetworkX 用 Python 代码，FalkorDB 用 Cypher）",
    )

    args = parser.parse_args()

    if args.query:
        query_text = args.query
    elif not sys.stdin.isatty():
        query_text = sys.stdin.read().strip()
    else:
        parser.print_help()
        sys.exit(1)

    assets_dir = args.assets
    backend = load_backend(assets_dir)

    if backend == "networkx":
        cmd_networkx(query_text, assets_dir)
    elif backend == "falkordb":
        cmd_falkordb(query_text, assets_dir)
    else:
        print(f"不支持的 backend: {backend}")
        sys.exit(1)


if __name__ == "__main__":
    query()
