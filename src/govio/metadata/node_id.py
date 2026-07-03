"""节点 string ID 生成。

ID 格式: <2 字符类型前缀><SHA256(业务键) 前 8 hex>，共 10 位。
业务键来自各节点的天然唯一列（full_table_name / column / app_id / standard_id / code）。
"""

import hashlib
from collections import Counter
from pathlib import Path

import pandas as pd

NODE_PREFIXES = {
    "PhysicalTable": "PT",
    "Col": "CO",
    "Application": "AP",
    "Standard": "ST",
    "Metric": "ME",
    "Dimension": "DI",
}


def make_id(node_type: str, business_key: str) -> str:
    """生成 10 位 string ID。"""
    if node_type not in NODE_PREFIXES:
        raise ValueError(f"未知节点类型: {node_type}")
    if not isinstance(business_key, str) or not business_key:
        raise ValueError("business_key 必须为非空字符串")
    prefix = NODE_PREFIXES[node_type]
    digest = hashlib.sha256(business_key.encode("utf-8")).hexdigest()[:8].upper()
    return f"{prefix}{digest}"


def assign_node_ids(df: pd.DataFrame, node_type: str, key_col: str) -> None:
    """就地给 df 加 node_id 列。业务键缺失或同类型内 ID 冲突时抛 ValueError。

    冲突可能来自重复业务键，也可能来自不同业务键的 SHA256[:8] 截断碰撞
    （32 位空间，万级节点下概率可忽略但非零）。错误信息区分两种情况。
    """
    keys = df[key_col].astype(str).tolist()
    for k in keys:
        if k in ("nan", "None", "<NA>", ""):
            raise ValueError(
                f"{node_type} 节点的 {key_col} 列存在缺失值（NaN/None/空），无法生成 ID"
            )
    ids = [make_id(node_type, k) for k in keys]
    if len(set(ids)) != len(ids):
        dup_keys = [k for k, c in Counter(keys).items() if c > 1]
        if dup_keys:
            raise ValueError(
                f"{node_type} 节点业务键重复: {dup_keys}"
            )
        # 重复 ID 但无重复键 → SHA256 截断碰撞
        id_counts = Counter(ids)
        colliding = [nid for nid, c in id_counts.items() if c > 1]
        raise ValueError(
            f"{node_type} 节点发生 SHA256[:8] 哈希碰撞（不同业务键产生相同 ID）: "
            f"碰撞 ID {colliding}，请扩大 hash 位数或检查业务键分布"
        )
    df["node_id"] = ids


def write_node_csv(df: pd.DataFrame, path: Path, node_type: str) -> None:
    """把已带 node_id 列的 df 写成 CSV，ID 列名 :ID(Label) 置首。"""
    if "node_id" not in df.columns:
        raise ValueError(f"DataFrame 缺少 node_id 列 (node_type={node_type})")
    out = df.drop(columns=["node_id"])
    out.insert(0, f":ID({node_type})", df["node_id"])
    out.to_csv(path, index=False)
