"""元数据查询 benchmark 评估程序的冒烟测试（build_sqlite / 提取执行 / eval_checks / score_answers）。"""

import csv
import importlib.util
import json
from pathlib import Path

import pytest

EVAL_PATH = Path(__file__).resolve().parent.parent / "benchmarks" / "metadata-query" / "eval.py"
spec = importlib.util.spec_from_file_location("metadata_query_eval", EVAL_PATH)
eval_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_mod)

CSV_HEADERS = {
    "Datasource.csv": [":ID(Datasource)", "datasource_name", "comment", "source_type", "filter"],
    "PhysicalTable.csv": [":ID(PhysicalTable)", "full_table_name", "schema", "table_name", "name",
                          "data_entity_type", "database_name"],
    "Col.csv": [":ID(Col)", "column", "column_name", "name", "full_table_name", "data_entity_type",
                "dtype", "size", "precision", "scale", "order_no", "data_type"],
    "OWNS.csv": [":START_ID(Datasource)", ":END_ID(PhysicalTable)"],
    "HAS_COLUMN.csv": [":START_ID(PhysicalTable)", ":END_ID(Col)"],
}


def _write_csv(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADERS[path.name])
        writer.writerows(rows)


@pytest.fixture
def csv_dir(tmp_path: Path) -> Path:
    _write_csv(tmp_path / "Datasource.csv", [["DS1", "MDM", "主数据系统", "trino", "{}"]])
    _write_csv(tmp_path / "PhysicalTable.csv", [
        ["PT1", "MDM.MDM.CUST_CUSTOMER", "MDM", "CUST_CUSTOMER", "客户信息表", "ORACLE_TABLE", "db"],
        ["PT2", "MDM.MDM.ORPHAN_TABLE", "MDM", "ORPHAN_TABLE", "孤儿表", "ORACLE_TABLE", "db"],
    ])
    _write_csv(tmp_path / "Col.csv", [
        ["CO1", "MDM.MDM.CUST_CUSTOMER.BANK_ACCOUNT", "BANK_ACCOUNT", "银行账号",
         "MDM.MDM.CUST_CUSTOMER", "ORACLE_COLUMN", "VARCHAR2", "200", "0", "0", "1", "varchar(200)"],
        ["CO2", "MDM.MDM.CUST_CUSTOMER.BAD_SIZE", "BAD_SIZE", "异常长度",
         "MDM.MDM.CUST_CUSTOMER", "ORACLE_COLUMN", "NUMBER", "abc", "0", "0", "2", "number"],
    ])
    _write_csv(tmp_path / "OWNS.csv", [["DS1", "PT1"]])
    _write_csv(tmp_path / "HAS_COLUMN.csv", [["PT1", "CO1"], ["PT1", "CO2"]])
    return tmp_path


def test_build_sqlite(csv_dir: Path, tmp_path: Path):
    db = tmp_path / "meta.sqlite"
    counts = eval_mod.build_sqlite(csv_dir, db)
    assert counts == {"datasource": 1, "physical_table": 2, "col": 2}

    import sqlite3
    conn = sqlite3.connect(db)
    rows = conn.execute(
        "SELECT d.datasource_name, t.full_table_name, c.column_name, c.size"
        " FROM col c JOIN physical_table t ON t.id = c.table_id"
        " JOIN datasource d ON d.id = t.datasource_id ORDER BY c.order_no"
    ).fetchall()
    assert rows == [("MDM", "MDM.MDM.CUST_CUSTOMER", "BANK_ACCOUNT", 200),
                    ("MDM", "MDM.MDM.CUST_CUSTOMER", "BAD_SIZE", None)]
    orphan = conn.execute("SELECT datasource_id FROM physical_table WHERE full_table_name = 'MDM.MDM.ORPHAN_TABLE'").fetchone()
    assert orphan == (None,)
    conn.close()


def test_eval_checks():
    answer = "SSOP 有 791 张表，IHRO 有 416 张。"
    results = eval_mod.eval_checks([
        {"type": "contains", "values": ["ssop", "IHRO"]},
        {"type": "contains_any", "values": ["不存在的词", "791"]},
        {"type": "not_contains", "values": ["FSGHR"]},
        {"type": "regex", "pattern": "\\b791\\b"},
        {"type": "regex", "pattern": "\\b91\\b"},  # 子串不应误命中
    ], answer)
    assert [r["passed"] for r in results] == [True, True, True, True, False]


def test_score_answers_and_report(csv_dir: Path, tmp_path: Path, monkeypatch):
    db = tmp_path / "meta.sqlite"
    eval_mod.build_sqlite(csv_dir, db)
    prompts = [{
        "id": "q1", "category": "lookup", "difficulty": 1,
        "question": "CUST_CUSTOMER 有几个字段？",
        "oracle_sql": "SELECT COUNT(*) FROM col WHERE table_id = 'PT1'",
        "expected_summary": "2 个字段",
        "checks": [
            {"type": "regex", "pattern": "\\b2\\b", "desc": "字段数 2"},
            {"type": "not_contains", "values": ["INV_NO"], "desc": "不得编造"},
        ],
    }]
    answers = tmp_path / "answers"
    (answers / "graph").mkdir(parents=True)
    (answers / "sqlite").mkdir(parents=True)
    (answers / "graph" / "q1.md").write_text(
        "```cypher\nMATCH (c:Col) RETURN count(c)\n```", encoding="utf-8")
    (answers / "sqlite" / "q1.md").write_text(
        "```sql\nSELECT COUNT(*) FROM col WHERE table_id = 'PT1'\n```", encoding="utf-8")

    monkeypatch.setattr(eval_mod, "extract_statements", lambda a, cmd, t: eval_mod._code_blocks(a))
    monkeypatch.setattr(eval_mod, "run_statement", lambda cond, stmt, db, t: {
        "statement": stmt, "ok": True, "rows": 1,
        "output": ("2\n共 1 行" if cond == "graph" else "3\n共 1 行"), "error": None})

    scores = eval_mod.score_answers(prompts, answers, db, agent_cmd="pi", timeout=5)
    assert scores["summary"]["graph"]["overall"] == 1.0
    assert scores["summary"]["graph"]["executed"] == 1
    assert scores["summary"]["sqlite"]["overall"] == 0.5  # regex \b2\b 不中，not_contains 通过
    assert scores["results"][0]["oracle"] == "2"
    assert scores["results"][0]["graph"]["check_score"] == 1.0  # 输出短，无折减
    assert scores["results"][0]["graph"]["brevity"] == 1.0
    assert scores["results"][0]["graph"]["statements"] == ["MATCH (c:Col) RETURN count(c)"]

    report = tmp_path / "report.md"
    eval_mod.write_report(scores, db, report, model="deepseek-flash")
    text = report.read_text(encoding="utf-8")
    assert "图库" in text and "SQLite" in text and "字段数 2" in text
    assert "执行结果" in text and "MATCH (c:Col) RETURN count(c)" in text
    assert "输出量分析" in text and "简洁度" in text


def test_extract_statements_fallback(monkeypatch):
    """pi 提取失败时退回回答自身的代码块/裸语句解析。"""
    monkeypatch.setattr(eval_mod, "run_agent_text", lambda prompt, cmd, t: ("", ""))
    answer = "说明：\n```sql\nSELECT 1\n```\n```sql\nSELECT 2\n```"
    assert eval_mod.extract_statements(answer, "pi", 5) == ["SELECT 1", "SELECT 2"]
    assert eval_mod.extract_statements("SELECT 3", "pi", 5) == ["SELECT 3"]
    assert eval_mod.extract_statements("", "pi", 5) == []


def test_extract_statements_via_agent(monkeypatch):
    monkeypatch.setattr(eval_mod, "run_agent_text",
                        lambda prompt, cmd, t: ("```cypher\nMATCH (n) RETURN n\n```", ""))
    assert eval_mod.extract_statements("任意回答", "pi", 5) == ["MATCH (n) RETURN n"]


def _fake_row(id_: str, cat: str, g: float, s: float) -> dict:
    return {"id": id_, "category": cat,
            "graph": {"score": g, "executed": True}, "sqlite": {"score": s, "executed": True}}


def test_merge_scores_single_case_updates_global():
    """单 case 重评只覆盖该行，其余保留历史，全局汇总重算。"""
    old = {"results": [_fake_row("a", "lookup", 1.0, 1.0), _fake_row("b", "filter", 0.0, 0.5)]}
    new = {"generated_at": "t", "model": "m", "summary": {},
           "results": [_fake_row("b", "filter", 1.0, 1.0)]}
    merged = eval_mod.merge_scores(old, new, [{"id": "a"}, {"id": "b"}, {"id": "c"}])
    assert [r["id"] for r in merged["results"]] == ["a", "b"]  # 未评的 c 不进报告
    assert merged["results"][0]["graph"]["score"] == 1.0  # a 保留历史结果
    assert merged["results"][1]["graph"]["score"] == 1.0  # b 被新结果覆盖
    assert merged["summary"]["graph"]["overall"] == 1.0
    assert merged["summary"]["graph"]["executed"] == 2
    assert merged["generated_at"] == "t"


def test_merge_scores_without_history():
    merged = eval_mod.merge_scores(None, {"generated_at": "t", "model": "", "summary": {},
                                         "results": [_fake_row("a", "lookup", 0.5, 0.5)]},
                                  [{"id": "a"}])
    assert len(merged["results"]) == 1
    assert merged["summary"]["graph"]["overall"] == 0.5


def test_weigh_output():
    """输出量权重：越简单达成目的分越高，未达成目的不因简洁加分。"""
    score, brevity, size = eval_mod.weigh_output(1.0, "MATCH (n) RETURN n")
    assert (score, brevity) == (1.0, 1.0) and size == 18
    score, brevity, size = eval_mod.weigh_output(1.0, "x" * 600)  # 简洁度 0.5 → 折减 10%
    assert (score, brevity, size) == (0.9, 0.5, 600)
    assert eval_mod.weigh_output(0.0, "x" * 600)[0] == 0.0
    assert eval_mod.weigh_output(1.0, "   ")[1] == 1.0  # 空回答不折减（得分为 0）


def test_exec_cypher_strips_leading_comments(monkeypatch):
    """govio-cli 要求以 MATCH 开头：执行前去掉起始注释。"""
    captured = {}

    def fake_run(cmd, **kw):
        captured["stmt"] = cmd[-1]
        return eval_mod.subprocess.CompletedProcess(cmd, 0, stdout="[[19]]", stderr="")

    monkeypatch.setattr(eval_mod.subprocess, "run", fake_run)
    text, rows, err = eval_mod.exec_cypher("// 说明\n/* 块注释 */\nMATCH (d) RETURN count(d)", 5)
    assert err is None and rows == 1 and "19" in text
    assert captured["stmt"].startswith("MATCH")


def test_exec_sql_read_only(csv_dir: Path, tmp_path: Path):
    db = tmp_path / "meta.sqlite"
    eval_mod.build_sqlite(csv_dir, db)
    text, rows, err = eval_mod.exec_sql("SELECT column_name FROM col ORDER BY column_name", db)
    assert err is None and rows == 2
    assert "BANK_ACCOUNT" in text and "BAD_SIZE" in text and "共 2 行" in text
    _text, _rows, err = eval_mod.exec_sql("DROP TABLE col", db)
    assert err is not None  # 只读连接拒绝写入
    _text, _rows, err = eval_mod.exec_sql("SELECT nope FROM col", db)
    assert err is not None


def test_compose_prompt_contains_materials_and_question():
    for cond, expect in (("graph", "ladybug"), ("sqlite", "schema")):
        text = eval_mod.compose_prompt(cond, "测试问题 XYZ？")
        assert "测试问题 XYZ？" in text
        assert expect in text.casefold() or "sqlite" in text.casefold()


def test_prompts_json_valid():
    data = json.loads((EVAL_PATH.parent / "prompts.json").read_text(encoding="utf-8"))
    ids = [p["id"] for p in data["prompts"]]
    assert len(ids) == len(set(ids)) >= 20
    for p in data["prompts"]:
        assert p["category"] in data["categories"]
        assert p["checks"], f"{p['id']} 缺少检查项"
        for c in p["checks"]:
            eval_mod.eval_checks([c], "")  # 检查项结构合法
