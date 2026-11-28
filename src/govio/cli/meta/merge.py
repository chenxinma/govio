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


def _is_all_empty(series: pd.Series) -> bool:
    """列值全为空（NaN 或空白字符串）判断。"""
    filled = series.fillna("").astype(str).str.strip()
    return bool(filled.eq("").all())


def _normalize_edge_columns(
    df: pd.DataFrame,
    dedup_cols: list[str],
    known_cols: list[str],
) -> pd.DataFrame:
    """规范化边 CSV 列：去重列（端点 + 类型）置首、元数据列随后，
    剔除全空列（如历史产物杂散列 source,target）。"""
    ordered = [c for c in dedup_cols if c in df.columns]
    meta_cols = [c for c in known_cols if c in df.columns and c not in ordered]
    meta_cols = [c for c in meta_cols if not _is_all_empty(df[c])]
    return df[ordered + meta_cols]


def merge_edge_csv(
    new_df: pd.DataFrame,
    csv_path: Path,
    dedup_cols: list[str],
) -> pd.DataFrame:
    """将新边数据与已有 CSV 合并，按复合键去重（新的覆盖旧的）。

    写盘前规范化列：dedup_cols 置首、元数据列随后；与旧 CSV 合并时
    不识别的列（如历史产物 source,target）丢弃并告警。

    Returns:
        合并后的 DataFrame
    """
    known_cols = list(dict.fromkeys([*dedup_cols, *new_df.columns]))
    existing = None
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        unknown = [c for c in existing.columns if c not in known_cols]
        if unknown:
            print(
                f"⚠ merge_edge_csv: {csv_path.name} 已有列 {unknown} 不在规范内，合并时丢弃",
                file=sys.stderr,
            )
            existing = existing.drop(columns=unknown)
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
    merged = _normalize_edge_columns(merged, dedup_cols, known_cols)
    merged = merged.reset_index(drop=True)
    merged.to_csv(csv_path, index=False)
    record_merge_delta(csv_path.parent, "edges", csv_path.stem, delta)
    return merged


def _load_csv_with_node_ids(
    csv_path: Path,
    node_type: str,
    key_col: str,
) -> pd.DataFrame:
    """加载节点 CSV，返回带 node_id 列的 DataFrame。

    文件已有 :ID(NodeType) 列时其值即 node_id，仅在内存中统一列名，
    不回写重复的 node_id 列（回写会产出 :ID 与 node_id 并存的非标准表头，
    令 meta graph 给图库节点 SET 多余属性，旧库上报 Cannot find property）。
    两列并存的历史产物按标准格式去重写回；完全缺 ID 的旧文件重新生成。
    """
    df = pd.read_csv(csv_path)
    id_col = f":ID({node_type})"
    if id_col in df.columns:
        if "node_id" in df.columns:
            # 历史产物两列并存（值相同）：以 node_id 为准去重，规范化写回
            df = df.drop(columns=[id_col])
            write_node_csv(df, csv_path, node_type)
        else:
            df = df.rename(columns={id_col: "node_id"})
    else:
        assign_node_ids(df, node_type, key_col)
        write_node_csv(df, csv_path, node_type)
    return df


# ---------------------------------------------------------------------------
# Graph / assets helpers
