"""CSV 增量合并 —— meta 各导出步骤的幂等核心

节点 CSV 按业务键去重（新值覆盖旧值），边 CSV 按复合键去重；两类合并都把
新增/更新/跳过计数累计进 `import-delta.json`，供 `meta graph` 生成校验报告。
"""

import sys
from pathlib import Path

import pandas as pd

from govio.metadata.import_report import classify_rows, record_merge_delta
from govio.metadata.node_id import assign_node_ids, write_node_csv


def merge_node_csv(
    new_df: pd.DataFrame,
    csv_path: Path,
    node_type: str,
    key_col: str,
) -> pd.DataFrame:
    """将新节点数据与已有 CSV 合并，按业务键去重（新的覆盖旧的）。

    读取已有 CSV 时会去掉 :ID(NodeType) 列头前缀以统一列名，
    合并后再通过 write_node_csv 写回标准格式。

    Returns:
        合并后的 DataFrame（已重置索引，含 node_id 列）
    """
    existing = None
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        # 已有 CSV 的第一列是 :ID(NodeType)，重命名为 node_id 以统一
        id_col = f":ID({node_type})"
        if id_col in existing.columns:
            if "node_id" in existing.columns:
                # 两列并存时丢弃 :ID 列（值相同）
                existing = existing.drop(columns=[id_col])
            else:
                existing = existing.rename(columns={id_col: "node_id"})
        combined = pd.concat([existing, new_df], ignore_index=True)
        dedup_col = key_col if key_col in combined.columns else None
        if dedup_col:
            merged = combined.drop_duplicates(subset=[dedup_col], keep="last")
        else:
            print(
                f"⚠ merge_node_csv: 未找到去重列 '{key_col}'，跳过去重", file=sys.stderr
            )
            merged = combined
    else:
        merged = new_df

    delta = classify_rows(new_df, existing, [key_col])
    merged = merged.reset_index(drop=True)
    write_node_csv(merged, csv_path, node_type)
    record_merge_delta(csv_path.parent, "nodes", node_type, delta)
    return merged


def merge_edge_csv(
    new_df: pd.DataFrame,
    csv_path: Path,
    dedup_cols: list[str],
) -> pd.DataFrame:
    """将新边数据与已有 CSV 合并，按复合键去重（新的覆盖旧的）。

    Returns:
        合并后的 DataFrame
    """
    existing = None
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
        cols = [c for c in dedup_cols if c in combined.columns]
        if cols:
            merged = combined.drop_duplicates(subset=cols, keep="last")
        else:
            print(
                f"⚠ merge_edge_csv: 未找到去重列 {dedup_cols}，跳过去重",
                file=sys.stderr,
            )
            merged = combined
    else:
        merged = new_df

    delta = classify_rows(new_df, existing, dedup_cols)
    merged = merged.reset_index(drop=True)
    merged.to_csv(csv_path, index=False)
    record_merge_delta(csv_path.parent, "edges", csv_path.stem, delta)
    return merged


def _load_csv_with_node_ids(
    csv_path: Path,
    node_type: str,
    key_col: str,
) -> pd.DataFrame:
    """加载节点 CSV，若缺少 node_id 列则重新生成并写回。"""
    df = pd.read_csv(csv_path)
    if "node_id" not in df.columns:
        assign_node_ids(df, node_type, key_col)
        df.to_csv(csv_path, index=False)
    return df


# ---------------------------------------------------------------------------
# Graph / assets helpers
