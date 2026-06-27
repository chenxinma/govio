"""节点 string ID 生成。

ID 格式: <2 字符类型前缀><SHA256(业务键) 前 8 hex>，共 10 位。
业务键来自各节点的天然唯一列（full_table_name / column / app_id / standard_id / code）。
"""

import hashlib
import sys
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
    if not business_key:
        raise ValueError("business_key 不能为空")
    prefix = NODE_PREFIXES[node_type]
    digest = hashlib.sha256(business_key.encode("utf-8")).hexdigest()[:8].upper()
    return f"{prefix}{digest}"


def assign_node_ids(df: pd.DataFrame, node_type: str, key_col: str) -> None:
    """就地给 df 加 node_id 列。断言同类型内唯一，冲突则 sys.exit(1)。"""
    keys = [str(k) for k in df[key_col]]
    ids = [make_id(node_type, k) for k in keys]
    if len(set(ids)) != len(ids):
        seen: set[str] = set()
        dups = [k for k in keys if k in seen or seen.add(k)]  # type: ignore[func-returns-value]
        print(
            f"❌ {node_type} 节点 ID 冲突，重复业务键: {dups}",
            file=sys.stderr,
        )
        sys.exit(1)
    df["node_id"] = ids


def write_node_csv(df: pd.DataFrame, path: Path, node_type: str) -> None:
    """把已带 node_id 列的 df 写成 CSV，ID 列名 :ID(Label) 置首。"""
    if "node_id" not in df.columns:
        raise ValueError(f"DataFrame 缺少 node_id 列 (node_type={node_type})")
    out = df.drop(columns=["node_id"])
    out.insert(0, f":ID({node_type})", df["node_id"])
    out.to_csv(path, index=False)
