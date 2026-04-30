"""Govio 知识图谱统一查询入口

支持通过 CLI 或作为包内模块调用。
用法: govio query --query "查询语句"
"""
from datetime import datetime
import json
import logging
from pathlib import Path
import sys

from govio import NetworkXGraph, FalkorDBGraph
from .config import ConfigManager
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
        if _size > 20:
            output_dir = Path(".") / ".govio"
            output_dir.mkdir(parents=True, exist_ok=True)
            fname = (
                output_dir / f"output-{datetime.now().strftime('%Y%m%d%H%M%s')}.json"
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


def cmd_networkx(code: str, gml_path: str):
    """NetworkX 查询处理"""
    gml_file = Path(gml_path)
    if not gml_file.exists():
        print(f"GML file not found: {gml_file}")
        sys.exit(1)

    gf = NetworkXGraph(gml_file) # pyright: ignore[reportArgumentType]
    g = gf.G

    logger.info("NetworkX Code: " + code)

    local_vars = {"g": g}
    exec(code, locals=local_vars)
    data = local_vars.get("result")
    output_result(data)


def cmd_falkordb(cypher: str, host: str = "localhost", port: int = 6379, graph_name: str = "ontology"):
    """FalkorDB 查询处理"""
    g = FalkorDBGraph(graph=graph_name, host=host, port=port)

    if not cypher.upper().startswith("MATCH"):
        print("Please write a MATCH query.")
        sys.exit(1)

    logger.info("Cypher: " + cypher)

    data = g.query(cypher)
    output_result(data)


def query(query_text):
    """Query CLI 主函数"""
    
    config_manager = ConfigManager()
    if not config_manager.exists():
        print("配置文件不存在，请先运行 govio-cli onboard")
        sys.exit(1)

    config = config_manager.load()
    backend = config.get("backend")
    if not backend:
        print("配置文件缺少 'backend' 字段，请重新运行 govio-cli onboard")
        sys.exit(1)

    if backend == "networkx":
        gml_path = config.get("networkx", {}).get("gml_path")
        if not gml_path:
            print("配置文件缺少 networkx.gml_path 字段")
            sys.exit(1)
        cmd_networkx(query_text, gml_path)
    elif backend == "falkordb":
        falkordb_config = config.get("falkordb", {})
        cmd_falkordb(
            query_text,
            host=falkordb_config.get("host", "localhost"),
            port=falkordb_config.get("port", 6379),
            graph_name=falkordb_config.get("graph", "ontology"),
        )
    else:
        print(f"不支持的 backend: {backend}")
        sys.exit(1)


