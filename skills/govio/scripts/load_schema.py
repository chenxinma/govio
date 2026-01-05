"""
环境初始化, 获得图数据库本体定义结构
"""
import argparse
from pathlib import Path
from govio import FalkorDBGraph

ASSETS_DIR = Path(__file__).parent.parent / "assets"

def load_schema(g: FalkorDBGraph):
    if not ASSETS_DIR.exists():
        ASSETS_DIR.mkdir()

    with open(ASSETS_DIR / "schema.md", mode="w") as fp:
        fp.write(g.schema)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=str, help='图数据库名称', default="ontology")

    # 解析命令行参数
    args = parser.parse_args()
    g = FalkorDBGraph(graph = args.graph)
    load_schema(g)
