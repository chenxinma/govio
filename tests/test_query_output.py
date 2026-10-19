"""govio-cli query 输出契约：文件必产（records 带字段名）+ ASCII 安全 stdout。"""

import json
import re
from decimal import Decimal
from pathlib import Path

from govio.cli.query import output_result


def _read_run(capsys):
    """返回 (stdout 行列表, 摘要行正则匹配, 文件 records)。"""
    lines = capsys.readouterr().out.strip().splitlines()
    m = re.match(
        r"Result output:\s*(.+?)\s+rows:\s*(\d+)\s+columns:\s+(\[.*\])", lines[0]
    )
    assert m, lines[0] if lines else "<空输出>"
    path = Path(m.group(1))
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return lines, m, rows


def test_named_records_file_and_ascii_inline(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    output_result(
        [["SSOP", Decimal(822)], ["IHRO", Decimal(542)]], columns=["system", "amount"]
    )
    lines, m, rows = _read_run(capsys)
    assert m.group(2) == "2"
    assert json.loads(m.group(3)) == ["system", "amount"]
    # 文件：UTF-8 JSONL records 带真字段名（Decimal 等标量走 str 不崩）
    assert rows == [
        {"system": "SSOP", "amount": "822"},
        {"system": "IHRO", "amount": "542"},
    ]
    # stdout：摘要行 + 内联，全 ASCII（免疫 bash/PowerShell 编码差异）
    assert len(lines) == 2
    assert all(line.isascii() for line in lines)
    assert json.loads(lines[1]) == rows


def test_positional_keys_without_columns(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    output_result([["SSOP", 822]])
    lines, _, rows = _read_run(capsys)
    assert rows == [{"0": "SSOP", "1": 822}]
    assert json.loads(lines[1]) == rows


def test_large_result_file_only(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    output_result([[i, f"col_{i}"] for i in range(60)], columns=["i", "name"])
    lines, m, rows = _read_run(capsys)
    assert len(lines) == 1  # 超阈值只出摘要行，结果在文件里
    assert m.group(2) == "60"
    assert rows[0] == {"i": 0, "name": "col_0"}
    assert len(rows) == 60


def test_empty_result_no_file(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    output_result(None)
    assert capsys.readouterr().out.strip() == "Data not found."
    assert not (tmp_path / ".govio").exists()
