#!/usr/bin/env python3
"""元数据查询 benchmark：执行 + 评估 两阶段。

执行（execute）：每题 × 2 条件，用 `pi -p --model deepseek-flash` 以
“材料文档 + 任务提示词”方式回答元数据问题，文本回答落盘保存：

    graph  条件：skills/govio-meta/SKILL.md + skills/govio/assets/schema.md + 题目
    sqlite 条件：sqlite-schema.md（库表结构文档）+ 题目

评估（score）：逐回答启动 pi 提取可执行的 Cypher/SQL 语句并真实执行
（graph：`govio-cli query -c`；sqlite：基准库只读查询），按评分标准
（prompts.json 的 checks）对执行结果打分，并按 answer 输出量加权
（越简单的内容达成目的分越高）；oracle_sql 在基准库上的执行结果作为
报告中的参考答案。

辅助（build）：从 govio meta 导出的 CSV 构建基准 SQLite（oracle_sql 的执行库）。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent.parent
DEFAULT_CSV_DIR = Path("D:/work/python/govio-runtime/output")
DEFAULT_DB = ROOT / "data" / "metadata.sqlite"
DEFAULT_PROMPTS = ROOT / "prompts.json"
ARTIFACTS_DIR = ROOT / "artifacts"
DEFAULT_ANSWERS_DIR = ARTIFACTS_DIR / "answers"
DEFAULT_MODEL = "deepseek-flash"
DEFAULT_AGENT_CMD = "pi -p"

# 执行阶段的材料文档（每条件）
GRAPH_MATERIALS = [
    REPO_ROOT / "skills" / "govio-query" / "reference-ladybug.md",
    REPO_ROOT / "skills" / "govio" / "assets" / "schema.md",
]
SQLITE_MATERIALS = [ROOT / "sqlite-schema.md"]
ENV_NOTES = {
    "graph": "仅生成Cypher查询 须以 MATCH 开头，不用实际执行。",
    "sqlite": "仅生成SQL只读查询，不用实际执行。",
}
CONDITIONS = ("graph", "sqlite")

SCHEMA_SQL = """
CREATE TABLE datasource (
    id            TEXT PRIMARY KEY,
    datasource_name TEXT UNIQUE NOT NULL,
    comment       TEXT,
    source_type   TEXT,
    filter        TEXT
);
CREATE TABLE physical_table (
    id              TEXT PRIMARY KEY,
    full_table_name TEXT UNIQUE NOT NULL,
    schema          TEXT,
    table_name      TEXT,
    name            TEXT,
    data_entity_type TEXT,
    database_name   TEXT,
    datasource_id   TEXT REFERENCES datasource(id)
);
CREATE TABLE col (
    id               TEXT PRIMARY KEY,
    column           TEXT,
    column_name      TEXT,
    name             TEXT,
    full_table_name  TEXT,
    data_entity_type TEXT,
    dtype            TEXT,
    size             INTEGER,
    precision        INTEGER,
    scale            INTEGER,
    order_no         INTEGER,
    data_type        TEXT,
    table_id         TEXT REFERENCES physical_table(id)
);
CREATE INDEX ix_table_ds ON physical_table(datasource_id);
CREATE INDEX ix_table_name ON physical_table(table_name);
CREATE INDEX ix_col_table ON col(table_id);
CREATE INDEX ix_col_name ON col(column_name);
"""


# ---------------------------------------------------------------------------
# 数据准备
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _int_or_none(value: str | None) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def build_sqlite(csv_dir: Path, db: Path) -> dict[str, int]:
    """从 CSV 构建 ER 模式 SQLite：OWNS/HAS_COLUMN 边转为外键列。

    Args:
        csv_dir: CSV 目录（Datasource/PhysicalTable/Col/OWNS/HAS_COLUMN）
        db: 输出的 sqlite 路径

    Returns:
        各表行数统计
    """
    missing = [n for n in ("Datasource.csv", "PhysicalTable.csv", "Col.csv", "OWNS.csv", "HAS_COLUMN.csv")
               if not (csv_dir / n).exists()]
    if missing:
        raise FileNotFoundError(f"CSV 缺失: {missing}（--csv-dir 指向 govio meta 导出目录）")

    owns = {r[":END_ID(PhysicalTable)"]: r[":START_ID(Datasource)"] for r in _read_csv(csv_dir / "OWNS.csv")}
    has_col = {r[":END_ID(Col)"]: r[":START_ID(PhysicalTable)"] for r in _read_csv(csv_dir / "HAS_COLUMN.csv")}
    datasources = _read_csv(csv_dir / "Datasource.csv")
    tables = _read_csv(csv_dir / "PhysicalTable.csv")
    cols = _read_csv(csv_dir / "Col.csv")

    db.parent.mkdir(parents=True, exist_ok=True)
    if db.exists():
        db.unlink()
    conn = sqlite3.connect(db)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.executemany(
            "INSERT INTO datasource (id, datasource_name, comment, source_type, filter) VALUES (?,?,?,?,?)",
            [(r[":ID(Datasource)"], r["datasource_name"], r.get("comment", ""), r.get("source_type", ""), r.get("filter", ""))
             for r in datasources],
        )
        conn.executemany(
            "INSERT INTO physical_table (id, full_table_name, schema, table_name, name, data_entity_type, database_name, datasource_id)"
            " VALUES (?,?,?,?,?,?,?,?)",
            [(r[":ID(PhysicalTable)"], r["full_table_name"], r.get("schema", ""), r.get("table_name", ""),
              r.get("name", ""), r.get("data_entity_type", ""), r.get("database_name", ""),
              owns.get(r[":ID(PhysicalTable)"]))
             for r in tables],
        )
        conn.executemany(
            "INSERT INTO col (id, column, column_name, name, full_table_name, data_entity_type, dtype,"
            " size, precision, scale, order_no, data_type, table_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [(r[":ID(Col)"], r.get("column", ""), r.get("column_name", ""), r.get("name", ""),
              r.get("full_table_name", ""), r.get("data_entity_type", ""), r.get("dtype", ""),
              _int_or_none(r.get("size")), _int_or_none(r.get("precision")), _int_or_none(r.get("scale")),
              _int_or_none(r.get("order_no")), r.get("data_type", ""), has_col.get(r[":ID(Col)"]))
             for r in cols],
        )
        conn.commit()
    finally:
        conn.close()

    orphan_tables = sum(1 for r in tables if not owns.get(r[":ID(PhysicalTable)"]))
    orphan_cols = sum(1 for r in cols if not has_col.get(r[":ID(Col)"]))
    if orphan_tables or orphan_cols:
        print(f"警告: {orphan_tables} 张表缺 OWNS 边，{orphan_cols} 个字段缺 HAS_COLUMN 边", file=sys.stderr)
    return {"datasource": len(datasources), "physical_table": len(tables), "col": len(cols)}


def load_prompts(path: Path, filter_ids: list[str] | None = None, category: str | None = None) -> list[dict]:
    prompts = json.loads(path.read_text(encoding="utf-8"))["prompts"]
    if filter_ids:
        wanted = set(filter_ids)
        prompts = [p for p in prompts if p["id"] in wanted]
    if category:
        prompts = [p for p in prompts if p["category"] == category]
    return prompts


def run_oracle(db: Path, sql: str) -> str:
    """在基准库上执行 oracle SQL，渲染为报告中的参考答案。"""
    if not db.exists():
        return ""
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute(sql).fetchall()
    finally:
        conn.close()
    lines = [" | ".join("" if v is None else str(v) for v in row) for row in rows[:20]]
    if len(rows) > 20:
        lines.append(f"... 共 {len(rows)} 行")
    return "\n".join(lines) if lines else "(空)"


# ---------------------------------------------------------------------------
# 执行阶段
# ---------------------------------------------------------------------------

def compose_prompt(condition: str, question: str) -> str:
    """材料文档 + 环境说明 + 任务提示词。"""
    paths = GRAPH_MATERIALS if condition == "graph" else SQLITE_MATERIALS
    parts = [p.read_text(encoding="utf-8").strip() for p in paths]
    parts.append(ENV_NOTES[condition])
    parts.append("任务提示词：\n" + question)
    return "\n\n".join(parts)


def _resolve_exe(cmd: list[str]) -> list[str]:
    """Windows 下 subprocess 不做 PATHEXT 查找，先解析出完整可执行路径。"""
    resolved = shutil.which(cmd[0])
    cmd[0] = resolved or cmd[0]
    return cmd


def extract_final_text(raw: str) -> str:
    """pi JSONL trace 取最后一条 assistant 文本；普通输出原样返回。"""
    texts: list[str] = []
    all_json = True
    for raw_line in raw.strip().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            all_json = False
            continue
        if event is dict:
            msg = event.get("message", {})  # pyright: ignore[reportArgumentType]
            if event.get("type") == "message_end" and msg.get("role") == "assistant":  # pyright: ignore[reportCallIssue, reportCallIssue, reportOptionalMemberAccess]
                for c in msg.get("content", []):  # pyright: ignore[reportOptionalMemberAccess]
                    if isinstance(c, dict) and c.get("type") == "text" and c.get("text"):
                        texts.append(c["text"])
    if all_json and texts:
        return texts[-1]
    return raw.strip()


def run_agent_text(prompt: str, agent_cmd: str, timeout: int) -> tuple[str, str]:
    """执行一次问答，返回 (文本回答, 原始输出)。

    题目文本经 stdin 传入：Windows 下 pi 是 .cmd 包装，走 argv 会被 cmd.exe 当作
    重定向符解析（提示词里的 <、>、-> 会被吞掉）。
    """
    cmd = _resolve_exe(shlex.split(agent_cmd))
    try:
        proc = subprocess.run(
            cmd, input=prompt, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout, check=False,
        )
    except FileNotFoundError as e:
        raise SystemExit(f"agent 命令无法执行: {agent_cmd}（{e}）") from e
    except subprocess.TimeoutExpired:
        return f"[超时: {timeout}s，未作答]", ""
    raw = proc.stdout or ""
    if (proc.stderr or "").strip():
        raw += f"\n--- stderr ---\n{proc.stderr.strip()}"
    return extract_final_text(raw) or "[空回答]", raw


def full_agent_cmd(base: str, model: str) -> str:
    """pi 命令行：禁工具/扩展/上下文，纯文本问答（执行与提取阶段共用）。"""
    return f"{base} --model {model} -nt -ne -nc --no-session"


def cmd_execute(args: argparse.Namespace) -> None:
    prompts = load_prompts(Path(args.prompts), args.filter, args.category)
    conditions = CONDITIONS if args.condition == "both" else (args.condition,)
    agent_cmd = full_agent_cmd(args.agent_cmd, args.model)
    answers_dir = Path(args.answers)
    for cond in conditions:
        (answers_dir / cond).mkdir(parents=True, exist_ok=True)
    jobs = [(cond, p) for cond in conditions for p in prompts]
    with tqdm(jobs, desc="[execute]", unit="题") as bar:
        for cond, p in bar:
            bar.set_postfix_str(f"{cond}/{p['id']}")
            out = answers_dir / cond / f"{p['id']}.md"
            if out.exists() and not args.force:
                tqdm.write(f"[execute] {cond}/{p['id']} 已存在，跳过（--force 重新执行）")
                continue
            # tqdm.write(f"[execute] {cond}/{p['id']}: {p['question'][:40]}...")
            text, _raw = run_agent_text(compose_prompt(cond, p["question"]), agent_cmd, args.timeout)
            out.write_text(text, encoding="utf-8")
    print(f"执行完成，回答写入 {answers_dir}/<{'|'.join(conditions)}>/<id>.md")


# ---------------------------------------------------------------------------
# 评估阶段
# ---------------------------------------------------------------------------

EXTRACT_PROMPT = (
    "从下面的回答中提取可执行的查询语句（图库条件是 Cypher，SQLite 条件是 SQL）。\n"
    "要求：把回答中的每条语句原样放进一个 ``` 代码块，一条一块；\n"
    "除代码块外不要输出任何内容；没有可提取的语句时只输出 EMPTY。\n\n回答：\n{answer}"
)

_CODE_BLOCK_RE = re.compile(r"```[A-Za-z]*\s*\n(.*?)```", re.DOTALL)


def _code_blocks(text: str) -> list[str]:
    """提取 ``` 围栏代码块中的语句。"""
    return [b.strip() for b in _CODE_BLOCK_RE.findall(text) if b.strip()]


def _resolve_query_exe() -> list[str]:
    """解析全局 govio-cli：`uv run` 把项目 venv 排在 PATH 前，venv 里的旧版依赖读不了图库。"""
    venv = Path(sys.prefix).resolve()
    path = os.pathsep.join(p for p in os.environ.get("PATH", "").split(os.pathsep)
                           if not Path(p).resolve().is_relative_to(venv))
    exe = shutil.which("govio-cli", path=path) or shutil.which("govio-cli") or "govio-cli"
    return [exe, "query"]


def extract_statements(answer: str, agent_cmd: str, timeout: int) -> list[str]:
    """启动 pi 从回答中提取查询语句；失败时退回解析回答自身的代码块/裸语句。

    Args:
        answer: 回答文本（仅输出 Cypher/SQL）
        agent_cmd: pi 命令行
        timeout: 提取超时秒数

    Returns:
        查询语句列表（可能为空）
    """
    if not answer.strip():
        return []
    text, _ = run_agent_text(EXTRACT_PROMPT.format(answer=answer), agent_cmd, timeout)
    stmts = _code_blocks(text) or _code_blocks(answer)
    if not stmts and re.match(r"(?is)\s*(MATCH|SELECT|WITH)\b", answer):
        stmts = [answer.strip()]
    return stmts


def _normalize_rows(rows) -> list[list]:
    """执行结果归一为行列表（cypher 的 list/dict 行与 sqlite 的 tuple 行）。"""
    rows = rows if isinstance(rows, list) else [rows]
    out = []
    for r in rows:
        if isinstance(r, dict):
            out.append(list(r.values()))
        elif isinstance(r, (list, tuple)):
            out.append(list(r))
        else:
            out.append([r])
    return out


def _render_rows(rows: list[list], limit: int = 200) -> str:
    """结果行渲染为检查项可匹配的文本（单元格用 | 连接，末尾附行数）。"""
    lines = [" | ".join("" if v is None else str(v) for v in r) for r in rows[:limit]]
    if len(rows) > limit:
        lines.append(f"...（截断，仅显示前 {limit} 行）")
    lines.append(f"共 {len(rows)} 行")
    return "\n".join(lines)


def exec_cypher(stmt: str, timeout: int) -> tuple[str, int, str | None]:
    """经 `govio-cli query -c` 执行 Cypher，返回 (结果文本, 行数, 错误)。"""
    # govio-cli 要求语句以 MATCH 开头，先去掉起始注释
    stmt = re.sub(r"\A\s*(?://[^\n]*\n|/\*.*?\*/\s*)*", "", stmt, flags=re.DOTALL)
    cmd = [*_resolve_query_exe(), "-c", stmt]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=timeout, check=False)
    except FileNotFoundError as e:
        return "", 0, f"govio-cli 不可用: {e}"
    except subprocess.TimeoutExpired:
        return "", 0, f"[超时: {timeout}s]"
    out = (proc.stdout or "").strip()
    if proc.returncode != 0:
        err = (proc.stderr or out or f"exit {proc.returncode}").strip()
        return "", 0, err[-500:]  # 保留末尾：traceback 的实际报错在最后
    if out == "Data not found.":
        return _render_rows([]), 0, None
    m = re.match(r"Result output:\s*(.+?)\s+rows:\s*\d+", out)
    if m:  # govio-cli 永远把结果写成 JSONL 文件（records 带字段名），摘要行带路径
        try:
            data = Path(m.group(1)).read_text(encoding="utf-8").splitlines()
            rows = [json.loads(line) for line in data if line.strip()]
        except (OSError, json.JSONDecodeError) as e:
            return "", 0, f"结果文件读取失败: {e}"
        rows = _normalize_rows(rows)
        return _render_rows(rows), len(rows), None
    try:
        rows = json.loads(out)
    except json.JSONDecodeError:
        return "", 0, out[:500]
    rows = _normalize_rows(rows)
    return _render_rows(rows), len(rows), None


def exec_sql(stmt: str, db: Path) -> tuple[str, int, str | None]:
    """在基准库（sqlite3 只读）执行 SQL，返回 (结果文本, 行数, 错误)。"""
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    except sqlite3.Error as e:
        return "", 0, str(e)
    try:
        rows = conn.execute(stmt).fetchall()
    except sqlite3.Error as e:
        return "", 0, str(e)
    finally:
        conn.close()
    rows = _normalize_rows(rows)
    return _render_rows(rows), len(rows), None


def run_statement(cond: str, stmt: str, db: Path, timeout: int) -> dict:
    """按条件执行单条语句：graph 走 govio-cli query，sqlite 走基准库只读查询。"""
    if cond == "graph":
        text, rows, err = exec_cypher(stmt, timeout)
    else:
        text, rows, err = exec_sql(stmt, db)
    return {"statement": stmt, "ok": err is None, "rows": rows, "output": text, "error": err}


def eval_checks(checks: list[dict], answer: str) -> list[dict]:
    """逐条检查项判定，返回带 passed/weight 的结果列表（评分标准的执行体）。"""
    results = []
    folded = answer.casefold()
    for check in checks:
        ctype = check.get("type", "contains")
        weight = check.get("weight", 1)
        if ctype == "contains":
            passed = all(v.casefold() in folded for v in check["values"])
        elif ctype == "contains_any":
            passed = any(v.casefold() in folded for v in check["values"])
        elif ctype == "not_contains":
            passed = not any(v.casefold() in folded for v in check["values"])
        elif ctype == "regex":
            passed = re.search(check["pattern"], answer, re.IGNORECASE) is not None
        else:
            raise ValueError(f"未知检查项类型: {ctype}")
        results.append({**check, "weight": weight, "passed": bool(passed)})
    return results


def score_answer(checks: list[dict], answer: str) -> tuple[float, list[dict]]:
    results = eval_checks(checks, answer)
    total = sum(c["weight"] for c in results)
    got = sum(c["weight"] for c in results if c["passed"])
    return (round(got / total, 3) if total else 0.0), results


BREVITY_BUDGET = 300   # 简单回答（单条聚焦查询）的字符预算
OUTPUT_WEIGHT = 0.2    # answer 输出量在评分中的权重上限


def weigh_output(check_score: float, answer: str) -> tuple[float, float, int]:
    """按 answer 输出量折减检查得分：越简单的内容达成目的分越高。

    Args:
        check_score: 检查项得分（0~1）
        answer: 回答文本

    Returns:
        (最终得分, 简洁度 0~1, answer 字符数)
    """
    size = len(answer.strip())
    brevity = min(1.0, BREVITY_BUDGET / size) if size else 1.0
    final = check_score * (1 - OUTPUT_WEIGHT * (1 - brevity))
    return round(final, 3), round(brevity, 3), size


def score_answers(prompts: list[dict], answers_dir: Path, db: Path,
                  agent_cmd: str = DEFAULT_AGENT_CMD, timeout: int = 120) -> dict:
    """逐回答提取并执行查询语句，按评分标准对执行结果打分。"""
    results = []
    for p in prompts:
        row = {
            "id": p["id"], "category": p["category"], "difficulty": p.get("difficulty", 1),
            "question": p["question"], "expected_summary": p.get("expected_summary", ""),
            "oracle": run_oracle(db, p["oracle_sql"]) if p.get("oracle_sql") else "",
        }
        for cond in CONDITIONS:
            answer_path = answers_dir / cond / f"{p['id']}.md"
            answer = answer_path.read_text(encoding="utf-8") if answer_path.exists() else ""
            stmts = extract_statements(answer, agent_cmd, timeout)
            runs = [run_statement(cond, s, db, timeout) for s in stmts]
            result_text = "\n".join(r["output"] for r in runs if r["ok"])
            check_score, checks = score_answer(p.get("checks", []), result_text)
            score, brevity, out_size = weigh_output(check_score, answer)
            row[cond] = {"score": score, "check_score": check_score,
                         "brevity": brevity, "output_size": out_size,
                         "executed": bool(runs) and all(r["ok"] for r in runs),
                         "statements": stmts, "runs": runs,
                         "checks": checks, "excerpt": result_text[:1500]}
        results.append(row)

    summary = summarize(results)
    return {"generated_at": datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
            "model": "", "summary": summary, "results": results}


def summarize(results: list[dict]) -> dict:
    """按条件汇总：总分 / 执行成功数 / 分类均分。"""
    summary: dict[str, dict] = {}
    for cond in CONDITIONS:
        by_cat: dict[str, list[float]] = {}
        for r in results:
            by_cat.setdefault(r["category"], []).append(r[cond]["score"])
        summary[cond] = {
            "overall": round(sum(r[cond]["score"] for r in results) / len(results), 3) if results else 0.0,
            "executed": sum(1 for r in results if r[cond]["executed"]),
            "by_category": {k: round(sum(v) / len(v), 3) for k, v in by_cat.items()},
        }
    return summary


def merge_scores(old: dict | None, new: dict, order: list[dict]) -> dict:
    """把本次评出的题目并入历史结果：单 case 评估后更新全局报告。

    Args:
        old: 历史 scores.json 内容（无则 None）
        new: 本次 score_answers 结果
        order: 全量题目（决定报告逐题顺序）

    Returns:
        合并后的结果（未重评的题保留历史行，汇总按合并后全量重算）
    """
    rows = {r["id"]: r for r in (old or {}).get("results", [])}
    rows.update({r["id"]: r for r in new["results"]})
    pos = {p["id"]: i for i, p in enumerate(order)}
    results = sorted(rows.values(), key=lambda r: pos.get(r["id"], len(pos)))
    return {**new, "summary": summarize(results), "results": results}


def write_report(scores: dict, db: Path, out: Path, model: str = "") -> None:
    lines = ["# 元数据查询 Benchmark 评估报告", ""]
    if model:
        lines.append(f"执行/提取模型：`{model}`（执行阶段答题；评估阶段提取查询语句）  ")
    if db.exists():
        conn = sqlite3.connect(db)
        n_ds, n_t, n_c = (conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                          for t in ("datasource", "physical_table", "col"))
        conn.close()
        lines.append(f"数据集：`{db}`（{n_ds} 个数据源 / {n_t} 张表 / {n_c} 个字段）")
    lines += ["", f"生成时间：{scores['generated_at']}", "", "## 总览", "",
              "| 条件 | 总分 | 执行成功 |", "|---|---|---|",
              f"| 图库（govio-meta + schema.md） | {scores['summary']['graph']['overall']} | "
              f"{scores['summary']['graph']['executed']}/{len(scores['results'])} |",
              f"| SQLite（sqlite-schema.md） | {scores['summary']['sqlite']['overall']} | "
              f"{scores['summary']['sqlite']['executed']}/{len(scores['results'])} |", "",
              "## 分类对比", "", "| 分类 | 图库 | SQLite |", "|---|---|---|"]
    cats = sorted({r["category"] for r in scores["results"]})
    for cat in cats:
        lines.append(f"| {cat} | {scores['summary']['graph']['by_category'].get(cat, 0)} "
                     f"| {scores['summary']['sqlite']['by_category'].get(cat, 0)} |")
    lines += ["", "## 输出量分析", "",
              f"最终得分 = 检查项得分 × (1 − {OUTPUT_WEIGHT} × (1 − 简洁度))，"
              f"简洁度 = min(1, {BREVITY_BUDGET} / answer 字符数)；越简单的内容达成目的分越高，"
              "最多折减 20%（未达成目的不因简洁加分）。", "",
              "| 条件 | 平均输出字符 | 平均简洁度 | 检查均分 | 加权后总分 |", "|---|---|---|---|---|"]
    for cond, label in (("graph", "图库"), ("sqlite", "SQLite")):
        sides = [r[cond] for r in scores["results"]]
        n = len(sides) or 1
        lines.append(f"| {label} | {round(sum(x['output_size'] for x in sides) / n, 1)} "
                     f"| {round(sum(x['brevity'] for x in sides) / n, 3)} "
                     f"| {round(sum(x['check_score'] for x in sides) / n, 3)} "
                     f"| {scores['summary'][cond]['overall']} |")
    lines += ["", "## 汇总表", "",
              "| id | 分类 | 图库 | SQLite |", "|---|---|---|---|"]
    for r in scores["results"]:
        lines.append(f"| {r['id']} | {r['category']} | {r['graph']['score']} | {r['sqlite']['score']} |")
    lines += ["", "## 逐题明细", ""]
    for r in scores["results"]:
        lines += [f"### {r['id']}（{r['category']}，难度 {r['difficulty']}）", "",
                  f"**问题**：{r['question']}", ""]
        if r["expected_summary"]:
            lines += [f"**标准答案要点**：{r['expected_summary']}", ""]
        if r["oracle"]:
            lines += ["**参考答案（oracle SQL 结果）**：", "", "```", r["oracle"][:1500], "```", ""]
        for cond, label in (("graph", "图库"), ("sqlite", "SQLite")):
            side = r[cond]
            lang = "cypher" if cond == "graph" else "sql"
            lines += [f"**{label}回答**（得分 {side['score']}：检查 {side['check_score']} × 输出量权重，"
                      f"简洁度 {side['brevity']}，输出 {side['output_size']} 字符）：", ""]
            if not side["statements"]:
                lines += ["_未提取到查询语句_", ""]
                continue
            for run in side["runs"]:
                lines += [f"```{lang}", run["statement"], "```", ""]
                if run["ok"]:
                    lines += ["执行结果：", "", "```", run["output"][:1500], "```", ""]
                else:
                    lines += [f"_执行失败：{run['error']}_", ""]
            for c in side["checks"]:
                mark = "x" if c["passed"] else " "
                lines.append(f"- [{mark}] {c.get('desc', c.get('type'))}（权重 {c['weight']}）")
            lines += [""]
    out.write_text("\n".join(lines), encoding="utf-8")


def cmd_score(args: argparse.Namespace) -> None:
    prompts = load_prompts(Path(args.prompts), args.filter, args.category)
    agent_cmd = full_agent_cmd(args.agent_cmd, args.model)
    scores = score_answers(prompts, Path(args.answers), Path(args.db), agent_cmd=agent_cmd, timeout=args.timeout)
    scores["model"] = args.model
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    scores_json = ARTIFACTS_DIR / "scores.json"
    old = None
    if scores_json.exists():
        try:
            old = json.loads(scores_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"警告: {scores_json} 无法解析，按全量新评处理", file=sys.stderr)
    scores = merge_scores(old, scores, load_prompts(Path(args.prompts)))
    scores_json.write_text(json.dumps(scores, ensure_ascii=False, indent=1), encoding="utf-8")
    write_report(scores, Path(args.db), ARTIFACTS_DIR / "report.md", model=args.model)
    g, s = scores["summary"]["graph"], scores["summary"]["sqlite"]
    print(f"图库 {g['overall']}（执行成功 {g['executed']}/{len(scores['results'])}） | "
          f"SQLite {s['overall']}（执行成功 {s['executed']}/{len(scores['results'])}）")
    print(f"报告：{ARTIFACTS_DIR / 'report.md'}\n明细：{scores_json}")


def cmd_build(args: argparse.Namespace) -> None:
    counts = build_sqlite(Path(args.csv_dir), Path(args.db))
    print(f"已构建 {args.db}：datasource={counts['datasource']} physical_table={counts['physical_table']} col={counts['col']}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="元数据查询 benchmark（执行 + 评估）")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="CSV → 基准 SQLite（oracle_sql 执行库）")
    p_build.add_argument("--csv-dir", default=str(DEFAULT_CSV_DIR))
    p_build.add_argument("--db", default=str(DEFAULT_DB))
    p_build.set_defaults(func=cmd_build)

    p_exec = sub.add_parser("execute", help="执行阶段：pi -p 按材料+任务提示词回答并保存文本")
    p_exec.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    p_exec.add_argument("--condition", choices=["both", *CONDITIONS], default="both")
    p_exec.add_argument("--model", default=DEFAULT_MODEL, help=f"生成模型（默认 {DEFAULT_MODEL}）")
    p_exec.add_argument("--agent-cmd", default=DEFAULT_AGENT_CMD)
    p_exec.add_argument("--timeout", type=int, default=600)
    p_exec.add_argument("--answers", default=str(DEFAULT_ANSWERS_DIR))
    p_exec.add_argument("--filter", help="逗号分隔的题目 id")
    p_exec.add_argument("--category", help="只跑某个分类")
    p_exec.add_argument("--force", action="store_true", help="重新执行已存在的回答")
    p_exec.set_defaults(func=cmd_execute)

    p_score = sub.add_parser("score", help="评估阶段：提取回答中的查询语句并执行，按评分标准对执行结果打分")
    p_score.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    p_score.add_argument("--answers", default=str(DEFAULT_ANSWERS_DIR))
    p_score.add_argument("--db", default=str(DEFAULT_DB))
    p_score.add_argument("--model", default=DEFAULT_MODEL, help=f"提取查询语句的模型（默认 {DEFAULT_MODEL}）")
    p_score.add_argument("--agent-cmd", default=DEFAULT_AGENT_CMD)
    p_score.add_argument("--timeout", type=int, default=120, help="pi 提取与查询执行的超时秒数")
    p_score.add_argument("--filter", help="逗号分隔的题目 id（只重评这些题，其余保留历史结果）")
    p_score.add_argument("--category", help="只评某个分类（其余保留历史结果）")
    p_score.set_defaults(func=cmd_score)

    args = parser.parse_args(argv)
    if getattr(args, "filter", None):
        args.filter = [x.strip() for x in args.filter.split(",") if x.strip()]
    args.func(args)


if __name__ == "__main__":
    main()
