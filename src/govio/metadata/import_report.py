"""导入校验报告与增量统计。

`meta graph` / `meta import-schema` 在图库更新成功后，基于 `--output` CSV 目录
（导入任务的共享状态目录）生成 `import-report.json`：各节点/边类型导入计数、
本次新增/更新/跳过数、校验项结果（重复 node_id / 必填缺失 / 孤儿边）。

增量统计口径与图后端无关：各导入子命令 merge CSV 时按业务键把本次输入与已有
CSV 对比分类（新增 = 键不存在；更新 = 键存在但值变化；跳过 = 键存在且值相同），
累计写入 `import-delta.json`；graph 同步成功后 fold 进报告并清零。
"""

import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

REPORT_FILE = "import-report.json"
DELTA_FILE = "import-delta.json"
REPORT_VERSION = 1

# 每项校验最多列出的样例条数
MAX_EXAMPLES = 5

# 与 graph.networkx_loader.load_nodes / load_edges 的固定文件名一致
NODE_CSVS = {
    "PhysicalTable.csv": "PhysicalTable",
    "Col.csv": "Col",
    "Datasource.csv": "Datasource",
    "Standard.csv": "Standard",
    "Metric.csv": "Metric",
    "Dimension.csv": "Dimension",
}

EDGE_CSVS = {
    "HAS_COLUMN.csv": "HAS_COLUMN",
    "OWNS.csv": "OWNS",
    "COMPLIES_WITH.csv": "COMPLIES_WITH",
    "RELATES_TO.csv": "RELATES_TO",
    "USES_TABLE.csv": "USES_TABLE",
    "REFERS_COLUMN.csv": "REFERS_COLUMN",
    "DERIVED_FROM.csv": "DERIVED_FROM",
    "DIMENSION_USED.csv": "DIMENSION_USED",
    "SUPERSEDES.csv": "SUPERSEDES",
}

CHANGE_KEYS = ("added", "updated", "skipped")


def _empty_changes() -> dict[str, dict]:
    return {
        "nodes": {},
        "edges": {},
        "totals": dict.fromkeys(CHANGE_KEYS, 0),
    }


def _read_csv(path: Path) -> pd.DataFrame | None:
    """读取固定文件名 CSV；表头不符合约定时返回 None（与 load_nodes 同口径跳过）。"""
    if not path.exists():
        return None
    return pd.read_csv(path, dtype=str).fillna("")


def _node_id_col(df: pd.DataFrame) -> str | None:
    col = str(df.columns[0]) if len(df.columns) else ""
    return col if re.match(r":ID\(", col) else None


def _edge_id_cols(df: pd.DataFrame) -> tuple[str, str] | None:
    if len(df.columns) < 2:
        return None
    src, dst = str(df.columns[0]), str(df.columns[1])
    if re.match(r":START_ID\(", src) and re.match(r":END_ID\(", dst):
        return src, dst
    return None


# ---------------------------------------------------------------------------
# 增量统计（merge 时分类，graph 同步后 fold）
# ---------------------------------------------------------------------------


def classify_rows(
    new_df: pd.DataFrame,
    existing_df: pd.DataFrame | None,
    key_cols: list[str],
) -> dict[str, int]:
    """按 key_cols 把本次输入的行与已有数据对比分类。

    分类只依据本次输入的行：键不在已有数据中为新增；键存在且全部属性值
    相同为跳过；键存在但任一属性值不同为更新。属性值按字符串比较
    （NaN 视为空串）。

    Args:
        new_df: 本次输入（含 key 列）
        existing_df: 已有数据（None 或空表示全新导入）
        key_cols: 业务键列（须全部存在于 new_df，否则整批按新增计）

    Returns:
        {"added": n, "updated": n, "skipped": n}
    """
    counts = dict.fromkeys(CHANGE_KEYS, 0)
    if new_df is None or new_df.empty:
        return counts
    keys = [c for c in key_cols if c in new_df.columns]
    if not keys or existing_df is None or existing_df.empty:
        counts["added"] = len(new_df)
        return counts

    cols = list(dict.fromkeys([*keys, *new_df.columns, *existing_df.columns]))
    # 先转 object 再填空串：float64 列直接 fillna("") 会触发 pandas 弃用告警
    new = new_df.reindex(columns=cols).astype("object").fillna("").astype(str)
    existing = existing_df.reindex(columns=cols).astype("object").fillna("").astype(str)
    key_idx = [cols.index(c) for c in keys]
    attr_idx = [i for i in range(len(cols)) if i not in key_idx]

    seen: dict[tuple[str, ...], tuple[str, ...]] = {}
    for row in existing.itertuples(index=False, name=None):
        seen[tuple(row[i] for i in key_idx)] = tuple(row[i] for i in attr_idx)
    for row in new.itertuples(index=False, name=None):
        key = tuple(row[i] for i in key_idx)
        if key not in seen:
            counts["added"] += 1
        elif seen[key] == tuple(row[i] for i in attr_idx):
            counts["skipped"] += 1
        else:
            counts["updated"] += 1
    return counts


def record_merge_delta(
    csv_dir: Path, category: str, name: str, delta: dict[str, int]
) -> None:
    """把一次 merge 的分类计数累计进 <csv_dir>/import-delta.json。

    Args:
        csv_dir: 导入任务的 CSV 目录
        category: "nodes" 或 "edges"
        name: 节点类型或边类型名
        delta: classify_rows 的结果
    """
    path = csv_dir / DELTA_FILE
    data: dict[str, Any] = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = {}
    entry = data.setdefault(category, {}).setdefault(
        name, dict.fromkeys(CHANGE_KEYS, 0)
    )
    for k in CHANGE_KEYS:
        entry[k] = entry.get(k, 0) + int(delta.get(k, 0))
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def take_merge_delta(csv_dir: Path) -> dict[str, dict]:
    """读取并清零增量累计，返回报告用 changes 结构（含 totals）。"""
    path = csv_dir / DELTA_FILE
    data: dict[str, Any] = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = {}
        path.unlink()
    changes = _empty_changes()
    for category in ("nodes", "edges"):
        for name, delta in data.get(category, {}).items():
            changes[category][name] = {k: int(delta.get(k, 0)) for k in CHANGE_KEYS}
    totals = changes["totals"]
    for category in ("nodes", "edges"):
        for delta in changes[category].values():
            for k in CHANGE_KEYS:
                totals[k] += delta[k]
    return changes


# ---------------------------------------------------------------------------
# 报告内容：导入计数 + 校验项
# ---------------------------------------------------------------------------


def collect_import_counts(csv_dir: Path) -> dict[str, dict[str, int]]:
    """各节点/边类型的导入行数（只统计表头符合约定的固定文件名 CSV）。"""
    counts: dict[str, dict[str, int]] = {"nodes": {}, "edges": {}}
    for filename, label in NODE_CSVS.items():
        df = _read_csv(csv_dir / filename)
        if df is None or _node_id_col(df) is None:
            continue
        counts["nodes"][label] = len(df)
    for filename, edge_type in EDGE_CSVS.items():
        df = _read_csv(csv_dir / filename)
        if df is None or _edge_id_cols(df) is None:
            continue
        counts["edges"][edge_type] = len(df)
    return counts


def run_validation_checks(csv_dir: Path) -> list[dict[str, Any]]:
    """对 CSV 目录执行三项校验：重复 node_id / 必填缺失 / 孤儿边。

    - duplicate_node_ids: 同一节点 CSV 内 `:ID(...)` 值重复
    - missing_required: 节点缺 node_id 或 name（有 name 列时）、边缺端点 ID
    - orphan_edges: 边端点 ID 不在任何节点 CSV 中（空端点计入 missing_required）

    Returns:
        每项 {"name", "status": "pass"/"fail", "count", "examples"}（样例最多 MAX_EXAMPLES 条）
    """
    node_ids: set[str] = set()
    duplicate_examples: list[str] = []
    missing_examples: list[str] = []
    orphan_examples: list[str] = []
    duplicate_count = 0
    missing_count = 0
    orphan_count = 0

    for filename in NODE_CSVS:
        df = _read_csv(csv_dir / filename)
        if df is None:
            continue
        id_col = _node_id_col(df)
        if id_col is None:
            continue
        ids = df[id_col].tolist()
        for nid, cnt in Counter(ids).items():
            if nid and cnt > 1:
                duplicate_count += cnt - 1
                if len(duplicate_examples) < MAX_EXAMPLES:
                    duplicate_examples.append(f"{filename}: {nid} 出现 {cnt} 次")
        node_ids.update(i for i in ids if i)
        has_name = "name" in df.columns
        for row_no, row in enumerate(df.itertuples(index=False, name=None), start=2):
            row_map = dict(zip(df.columns, row, strict=True))
            problems = []
            if not str(row_map[id_col]).strip():
                problems.append(f"{id_col} 为空")
            if has_name and not str(row_map["name"]).strip():
                problems.append("name 为空")
            if problems:
                missing_count += 1
                if len(missing_examples) < MAX_EXAMPLES:
                    missing_examples.append(
                        f"{filename} 第{row_no}行: {'，'.join(problems)}"
                    )

    for filename in EDGE_CSVS:
        df = _read_csv(csv_dir / filename)
        if df is None:
            continue
        id_cols = _edge_id_cols(df)
        if id_cols is None:
            continue
        src_col, dst_col = id_cols
        for row_no, (src_raw, dst_raw) in enumerate(
            zip(df[src_col], df[dst_col], strict=True), start=2
        ):
            src, dst = str(src_raw).strip(), str(dst_raw).strip()
            if not src or not dst:
                missing_count += 1
                if len(missing_examples) < MAX_EXAMPLES:
                    missing_examples.append(f"{filename} 第{row_no}行: 边端点 ID 为空")
                continue
            if src not in node_ids or dst not in node_ids:
                orphan_count += 1
                if len(orphan_examples) < MAX_EXAMPLES:
                    orphan_examples.append(f"{filename}: {src} -> {dst} 缺少端点节点")

    def result(name: str, count: int, examples: list[str]) -> dict[str, Any]:
        return {
            "name": name,
            "status": "fail" if count else "pass",
            "count": count,
            "examples": examples,
        }

    return [
        result("duplicate_node_ids", duplicate_count, duplicate_examples),
        result("missing_required", missing_count, missing_examples),
        result("orphan_edges", orphan_count, orphan_examples),
    ]


def build_import_report(
    csv_dir: Path,
    mode: str,
    backend: str,
    changes: dict[str, dict] | None = None,
) -> dict[str, Any]:
    """组装校验报告字典（不落盘）。

    Args:
        csv_dir: 导入任务的 CSV 目录
        mode: graph 更新模式（update / rebuild）
        backend: 图后端名（networkx / falkordb / ladybug / none）
        changes: take_merge_delta 的结果；缺省按零值处理

    Returns:
        报告字典，结构见 docs/specs/cli.md「校验报告」
    """
    if changes is None:
        changes = _empty_changes()
    checks = run_validation_checks(csv_dir)
    failed = sum(1 for c in checks if c["status"] != "pass")
    return {
        "report_version": REPORT_VERSION,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "csv_dir": str(csv_dir),
        "mode": mode,
        "backend": backend,
        "import_counts": collect_import_counts(csv_dir),
        "changes": changes,
        "checks": checks,
        "summary": {
            "status": "fail" if failed else "pass",
            "checks_total": len(checks),
            "checks_failed": failed,
            "changes_totals": dict(changes["totals"]),
        },
    }


def write_import_report(report: dict[str, Any], path: Path) -> Path:
    """整体序列化后单次写入，失败时不留半截报告。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
