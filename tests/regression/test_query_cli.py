"""govio-cli query 回归用例

覆盖 NetworkX（Python exec）与 Ladybug（Cypher）两个后端的查询输出契约、
Unicode 安全、错误路径与配置校验。FalkorDB 需要 Redis 服务，不在本套件内（见 README）。
"""

import json
import re
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# 输出契约解析
# ---------------------------------------------------------------------------


def _parse_summary(out: str) -> tuple[list[str], re.Match[str], list[dict]]:
    """解析 stdout 摘要行并读取 JSONL 结果文件

    Returns:
        tuple: (stdout 行列表, 摘要行正则匹配, 结果文件 records)
    """
    lines = out.strip().splitlines()
    m = re.match(
        r"Result output:\s*(.+?)\s+rows:\s*(\d+)\s+columns:\s+(\[.*\])", lines[0]
    )
    assert m, lines[0] if lines else "<空输出>"
    rows = [
        json.loads(line)
        for line in Path(m.group(1)).read_text(encoding="utf-8").splitlines()
    ]
    return lines, m, rows


# ---------------------------------------------------------------------------
# NetworkX 后端（Python exec，result 无列名 → 位置键兜底）
# ---------------------------------------------------------------------------


def test_q01_networkx_named_file_inline_and_ascii(
    nx_gml: Path, write_config, run_cli
) -> None:
    """Q01: NetworkX 查询产出带字段名 JSONL 文件 + ASCII 内联"""
    write_config(graph={"backend": "networkx", "networkx": {"gml_path": str(nx_gml)}})
    code, out, _err = run_cli(
        "query",
        "-c",
        "result = [[n, d['name']] for n, d in g.nodes(data=True)"
        " if d['node_type'] == 'Col']",
    )
    assert code == 0
    lines, m, rows = _parse_summary(out)
    assert m.group(2) == "6"
    assert json.loads(m.group(3)) == ["0", "1"]  # exec 无列名元信息，位置键兜底
    assert rows[0] == {"0": "co00000001", "1": "customer_id"}
    assert len(rows) == 6
    assert json.loads(lines[1]) == rows  # 小结果内联
    assert all(line.isascii() for line in lines)


def test_q02_networkx_scalar_result_wrapped(
    nx_gml: Path, write_config, run_cli
) -> None:
    """Q02: 标量 result 包装为单行记录"""
    write_config(graph={"backend": "networkx", "networkx": {"gml_path": str(nx_gml)}})
    code, out, _err = run_cli("query", "-c", "result = 42")
    assert code == 0
    _lines, m, rows = _parse_summary(out)
    assert m.group(2) == "1"
    assert rows == [{"0": 42}]


def test_q03_networkx_large_result_file_only(
    nx_gml: Path, write_config, run_cli
) -> None:
    """Q03: 超过内联阈值只出摘要行，文件保留全部行"""
    write_config(graph={"backend": "networkx", "networkx": {"gml_path": str(nx_gml)}})
    code, out, _err = run_cli(
        "query", "-c", "result = [[i, 'col_' + str(i)] for i in range(60)]"
    )
    assert code == 0
    lines, m, rows = _parse_summary(out)
    assert len(lines) == 1
    assert m.group(2) == "60"
    assert rows[0] == {"0": 0, "1": "col_0"}
    assert len(rows) == 60


def test_q04_networkx_empty_result_no_file(
    nx_gml: Path, write_config, run_cli, isolated_env: Path
) -> None:
    """Q04: 空结果只输出 Data not found.，不写结果文件"""
    write_config(graph={"backend": "networkx", "networkx": {"gml_path": str(nx_gml)}})
    code, out, _err = run_cli("query", "-c", "result = []")
    assert code == 0
    assert out.strip() == "Data not found."
    assert not list((isolated_env / ".govio").glob("output-*.json"))  # 不写结果文件


def test_q05_networkx_missing_gml_file(write_config, run_cli) -> None:
    """Q05: GML 文件不存在 → 报错退出 1"""
    write_config(graph={"backend": "networkx", "networkx": {"gml_path": "missing.gml"}})
    code, out, _err = run_cli("query", "-c", "result = 1")
    assert code == 1
    assert "GML file not found" in out


# ---------------------------------------------------------------------------
# Ladybug 后端（Cypher）
# ---------------------------------------------------------------------------


def test_q06_ladybug_aliases_chinese_and_ascii_stdout(
    ladybug_db: Path, write_config, run_cli
) -> None:
    """Q06: RETURN 别名为列名；中文进文件、stdout 全 ASCII；小写 match 合法"""
    write_config(graph={"backend": "ladybug", "ladybug": {"db_path": str(ladybug_db)}})
    code, out, _err = run_cli(
        "query",
        "-c",
        "match (t:PhysicalTable) RETURN t.name AS table_name,"
        " t.comment AS comment ORDER BY t.name",
    )
    assert code == 0
    lines, m, rows = _parse_summary(out)
    assert m.group(2) == "2"
    assert json.loads(m.group(3)) == ["table_name", "comment"]
    assert rows == [
        {"table_name": "customers", "comment": "客户信息表"},
        {"table_name": "orders", "comment": "订单明细表"},
    ]
    assert all(line.isascii() for line in lines)
    file_text = Path(m.group(1)).read_text(encoding="utf-8")
    assert "客户信息表" in file_text  # 文件 UTF-8 保留中文原样


def test_q07_ladybug_cross_node_join(
    ladybug_db: Path, write_config, run_cli
) -> None:
    """Q07: 跨节点关系查询（Table-HAS_COLUMN->Col）"""
    write_config(graph={"backend": "ladybug", "ladybug": {"db_path": str(ladybug_db)}})
    code, out, _err = run_cli(
        "query",
        "-c",
        "MATCH (t:PhysicalTable)-[:HAS_COLUMN]->(c:Col)"
        " WHERE t.name = 'orders'"
        " RETURN c.name AS col, c.order_no AS ord ORDER BY c.order_no",
    )
    assert code == 0
    _lines, m, rows = _parse_summary(out)
    assert json.loads(m.group(3)) == ["col", "ord"]
    assert rows == [
        {"col": "order_id", "ord": 1},
        {"col": "amount", "ord": 2},
        {"col": "status", "ord": 3},
    ]


def test_q08_ladybug_empty_result(ladybug_db: Path, write_config, run_cli) -> None:
    """Q08: 查询无匹配 → Data not found."""
    write_config(graph={"backend": "ladybug", "ladybug": {"db_path": str(ladybug_db)}})
    code, out, _err = run_cli(
        "query",
        "-c",
        "MATCH (t:PhysicalTable) WHERE t.name = 'not_exist' RETURN t.name AS name",
    )
    assert code == 0
    assert out.strip() == "Data not found."


def test_q09_ladybug_rejects_non_match(ladybug_db: Path, write_config, run_cli) -> None:
    """Q09: 非 MATCH 语句被拒绝"""
    write_config(graph={"backend": "ladybug", "ladybug": {"db_path": str(ladybug_db)}})
    code, out, _err = run_cli("query", "-c", "CREATE (n:PhysicalTable)")
    assert code == 1
    assert out.strip() == "Please write a MATCH query."


def test_q10_ladybug_invalid_cypher_raises(
    ladybug_db: Path, write_config, run_cli
) -> None:
    """Q10: 非法 Cypher 包装为 ValueError（当前契约）"""
    write_config(graph={"backend": "ladybug", "ladybug": {"db_path": str(ladybug_db)}})
    with pytest.raises(ValueError, match="Generated Cypher Statement is not valid"):
        run_cli("query", "-c", "MATCH (n RETURN n")


def test_q11_ladybug_missing_db_path(write_config, run_cli) -> None:
    """Q11: 配置缺 ladybug.db_path → 报错退出 1"""
    write_config(graph={"backend": "ladybug", "ladybug": {}})
    code, out, _err = run_cli("query", "-c", "MATCH (n) RETURN n")
    assert code == 1
    assert "配置文件缺少 graph.ladybug.db_path 字段" in out


# ---------------------------------------------------------------------------
# 配置校验
# ---------------------------------------------------------------------------


def test_q12_missing_config(run_cli) -> None:
    """Q12: 配置文件不存在 → 提示先 onboard，退出 1"""
    code, out, _err = run_cli("query", "-c", "result = 1")
    assert code == 1
    assert "配置文件不存在，请先运行 govio-cli onboard" in out


def test_q13_missing_backend_field(write_config, run_cli) -> None:
    """Q13: graph.backend 缺失 → 报错退出 1"""
    write_config(graph={})
    code, out, _err = run_cli("query", "-c", "result = 1")
    assert code == 1
    assert "配置文件缺少 'graph.backend' 字段" in out


def test_q14_unsupported_backend(write_config, run_cli) -> None:
    """Q14: 未知 backend → 报错退出 1"""
    write_config(graph={"backend": "neo4j"})
    code, out, _err = run_cli("query", "-c", "result = 1")
    assert code == 1
    assert "不支持的 backend: neo4j" in out
