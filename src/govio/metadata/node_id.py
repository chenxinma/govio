"""节点 string ID 生成与节点 CSV 写出。

ID 格式: <2 字符类型前缀><SHA256(业务键) 前 8 hex>，共 10 位。
业务键来自各节点的天然唯一列（full_table_name / column / app_id / standard_id / code）。
"""

import hashlib

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
