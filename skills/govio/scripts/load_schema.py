"""
环境初始化, 获得图数据库本体定义结构
"""
import argparse
from pathlib import Path
from govio import NetworkXGraph

ASSETS_DIR = Path(__file__).parent.parent / "assets"

def load_schema(g: NetworkXGraph):
    if not ASSETS_DIR.exists():
        ASSETS_DIR.mkdir()

    with open(ASSETS_DIR / "schema.md", mode="w") as fp:
        fp.write(g.schema)

if __name__ == "__main__":
    # 解析命令行参数
    g = NetworkXGraph(ASSETS_DIR / "ontology.gml")
    load_schema(g)
