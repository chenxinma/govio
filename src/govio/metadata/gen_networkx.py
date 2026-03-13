"""
govio.metadata.gen_networkx
读取指定目录的csv文本（包含Node和Edge）构建图并存储为GML文件
读取内容包含：

## Nodes

- PhysicalTable.csv
- Col.csv
- Application.csv
- Standard.csv

`csv文件的第一列，列名:ID({node_name})，以node_name标明图的节点类型，后续列都作为节点的属性`
`csv文件必定包含name列作为节点名称显示`

## Edges

- HAS_COLUMN.csv     | PhysicalTable与Col的关系，物理表所包含的列
- USE.csv            | Application与PhysicalTable的关系，应用用到的物理表
- COMPLIES_WITH.csv  | Col与Standard的关系，列贯标的数据标准

`csv文件的第一、二列，列名:START_ID({node_from}),:END_ID({node_to})，定义了Edge的 from node id 和 to node id`
"""

import argparse
import os
import sys
from typing import Any
import networkx as nx
import pandas as pd
import re
from pathlib import Path
from tqdm import tqdm


def load_nodes(csv_dir: str) -> list[dict[str, Any]]:
    node_files = ["PhysicalTable.csv", "Col.csv", "Application.csv", "Standard.csv"]
    nodes_list = []
    for filename in node_files:
        filepath = Path(csv_dir) / filename
        if not filepath.exists():
            continue
        df = pd.read_csv(filepath)
        id_col = df.columns[0]
        match = re.match(r":ID\((\w+)\)", id_col)
        if not match:
            continue
        node_type = match.group(1)
        df = df.rename(columns={id_col: "id"})
        df["node_type"] = node_type
        # df = df.set_index("id")
        nodes_list.extend(df.to_dict(orient="records"))

    return nodes_list


def load_edges(csv_dir: str) -> pd.DataFrame:
    edge_files = ["HAS_COLUMN.csv", "USE.csv", "COMPLIES_WITH.csv", "RELATES_TO.csv"]
    edges_list = []
    for filename in edge_files:
        filepath = Path(csv_dir) / filename
        if not filepath.exists():
            continue
        df = pd.read_csv(filepath)
        src_col = df.columns[0]
        dst_col = df.columns[1]
        src_match = re.match(r":START_ID\((\w+)\)", src_col)
        dst_match = re.match(r":END_ID\((\w+)\)", dst_col)
        if not src_match or not dst_match:
            continue
        edge_type = Path(filename).stem
        df = df.rename(columns={src_col: "source", dst_col: "target"})
        df["edge_type"] = edge_type
        edges_list.append(df)
    if not edges_list:
        return pd.DataFrame(columns=["source", "target", "edge_type"])
    return pd.concat(edges_list)


def build_graph(csv_dir: str, output_gml: str):
    nodes_list = load_nodes(csv_dir)
    edges_df = load_edges(csv_dir)
    G = nx.DiGraph()
    for node in tqdm(nodes_list, desc="nodes"):
        id = node["id"]
        del node["id"]
        G.add_node(id, **node)
    for _, row in tqdm(edges_df.iterrows(), total=edges_df.shape[0], desc="edges"):
        G.add_edge(row["source"], row["target"], edge_type=row["edge_type"])
    nx.write_gml(G, output_gml)


def gml_generate():
    """
    读取基础元数据的生成CSV，生成GML
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="读取基础元数据的生成CSV，生成GML",
    )
    parser.add_argument("--csv", type=str, help="元数据csv目录")
    parser.add_argument("-o", "--output", type=str, default=".", help="输出目录")
    # 解析命令行参数
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print("元数据csv目录未找到")
        sys.exit()

    if not os.path.isdir(args.csv):
        print("指定的元数据csv路径应该是一个目录")
        sys.exit()

    if not os.path.exists(args.output):
        print("输出目录未找到")
        sys.exit()

    build_graph(args.csv, os.path.join(args.output, "ontology.gml"))


if __name__ == "__main__":
    gml_generate()
