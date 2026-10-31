"""govio-cli observe load 回归用例

覆盖 --datasource（DuckDB）与 --memory 两种数据来源的加载契约、
-o JSON 输出、Unicode/NULL 边界、同名覆盖、错误路径与参数校验。
"""

import json
from pathlib import Path

from govio.observe_data.core.observe_store import ObserveStore

ORDER_COLUMNS = ["order_id", "customer_id", "amount", "status", "created_at"]


def _load(run_cli, name: str, sql: str) -> dict:
    """走 CLI 加载并返回结果 JSON"""
    code, out, _err = run_cli(
        "observe", "load", "--name", name, "--datasource", "local", "--sql", sql
    )
    assert code == 0
    return json.loads(out)


# ---------------------------------------------------------------------------
# --datasource 数据源加载
# ---------------------------------------------------------------------------


def test_l01_datasource_load_contract(
    duckdb_datasources, write_config, run_cli, isolated_env: Path
) -> None:
    """L01: 数据源加载成功契约（rows/columns/column_info）+ parquet 落盘 + 清单"""
    write_config(datasources=duckdb_datasources)
    result = _load(run_cli, "orders", "SELECT * FROM orders ORDER BY order_id")
    assert result["success"] is True
    assert result["name"] == "orders"
    assert result["rows"] == 5
    assert result["columns"] == 5
    assert [c["name"] for c in result["column_info"]] == ORDER_COLUMNS
    assert all(c["dtype"] for c in result["column_info"])

    # 落盘与清单可回放
    assert (isolated_env / ".govio" / "observe" / "manifest.json").exists()
    store = ObserveStore()
    assert store.exists("orders")
    df = store.get("orders")
    assert df is not None
    assert df["order_id"].tolist() == [1001, 1002, 1003, 1004, 1005]
    assert df["amount"].tolist()[0] == 120.5


def test_l02_output_file_unicode_and_nulls(
    duckdb_datasources, write_config, run_cli
) -> None:
    """L02: -o 输出 JSON records；中文 UTF-8 原样、NULL→null、空串/引号保留"""
    write_config(datasources=duckdb_datasources)
    code, out, _err = run_cli(
        "observe",
        "load",
        "--name",
        "customers",
        "--datasource",
        "local",
        "--sql",
        "SELECT * FROM customers ORDER BY customer_id",
        "-o",
        "customers.json",
    )
    assert code == 0
    result = json.loads(out)
    assert result["output_file"] == "customers.json"

    file_text = Path("customers.json").read_text(encoding="utf-8")
    assert "张三" in file_text  # UTF-8 原文，非 \u 转义
    records = json.loads(file_text)
    assert len(records) == 5
    assert records[0]["name"] == "张三"
    assert records[1]["region"] is None  # NULL → null
    assert records[2]["credit"] == -20.25  # 负数
    assert records[3]["name"] == "O'Brien 李"  # 引号转义
    assert records[3]["credit"] is None
    assert records[4]["name"] == ""  # 空字符串保留


def test_l03_same_name_overwrites(
    duckdb_datasources, write_config, run_cli
) -> None:
    """L03: 同名 DataFrame 覆盖加载，只保留最新结果"""
    write_config(datasources=duckdb_datasources)
    first = _load(run_cli, "orders", "SELECT * FROM orders")
    assert first["rows"] == 5

    second = _load(run_cli, "orders", "SELECT * FROM orders WHERE order_id = 1001")
    assert second["rows"] == 1

    df = ObserveStore().get("orders")
    assert df is not None
    assert df["order_id"].tolist() == [1001]


def test_l04_empty_result_still_stored(
    duckdb_datasources, write_config, run_cli
) -> None:
    """L04: 0 行结果也成功入库（rows=0，列结构保留）"""
    write_config(datasources=duckdb_datasources)
    result = _load(run_cli, "empty_orders", "SELECT * FROM orders WHERE 1=0")
    assert result["success"] is True
    assert result["rows"] == 0
    assert [c["name"] for c in result["column_info"]] == ORDER_COLUMNS
    df = ObserveStore().get("empty_orders")
    assert df is not None
    assert len(df) == 0


# ---------------------------------------------------------------------------
# --memory 内存加载
# ---------------------------------------------------------------------------


def test_l05_memory_join_over_loaded_dataframes(
    duckdb_datasources, write_config, run_cli
) -> None:
    """L05: --memory 把已加载 DataFrame 注册为同名表并执行 SQL"""
    write_config(datasources=duckdb_datasources)
    _load(run_cli, "customers", "SELECT * FROM customers")
    _load(run_cli, "orders", "SELECT * FROM orders")

    code, out, _err = run_cli(
        "observe",
        "load",
        "--name",
        "joined",
        "--memory",
        "--sql",
        "SELECT c.name AS customer_name, o.order_id FROM orders o"
        " JOIN customers c ON o.customer_id = c.customer_id ORDER BY o.order_id",
    )
    assert code == 0
    result = json.loads(out)
    assert result["success"] is True
    assert result["rows"] == 4  # order 1005 无对应客户，INNER JOIN 丢弃
    assert result["columns"] == 2
    assert set(result["source_tables"]) == {"customers", "orders"}
    df = ObserveStore().get("joined")
    assert df is not None
    assert df["customer_name"].tolist() == ["张三", "张三", "李四", "王五"]


def test_l06_memory_without_loaded_data(write_config, run_cli) -> None:
    """L06: --memory 但 store 为空 → 友好报错"""
    write_config(datasources={})
    code, out, _err = run_cli(
        "observe", "load", "--name", "x", "--memory", "--sql", "SELECT 1"
    )
    # 加载失败退出 1（docs/specs/cli.md 退出码契约；退出 0 与契约表及 test_exit_codes 冲突）
    assert code == 1
    result = json.loads(out)
    assert result["success"] is False
    assert "没有已加载的 DataFrame" in result["error"]


# ---------------------------------------------------------------------------
# 错误路径与参数校验
# ---------------------------------------------------------------------------


def test_l07_bad_sql_returns_error(
    duckdb_datasources, write_config, run_cli
) -> None:
    """L07: SQL 执行失败 → success=false，不落盘"""
    write_config(datasources=duckdb_datasources)
    code, out, _err = run_cli(
        "observe",
        "load",
        "--name",
        "bad",
        "--datasource",
        "local",
        "--sql",
        "SELECT * FROM not_exist",
    )
    # 加载失败退出 1（docs/specs/cli.md 退出码契约；退出 0 与契约表及 test_exit_codes 冲突）
    assert code == 1
    result = json.loads(out)
    assert result["success"] is False
    assert result["error"].startswith("SQL 执行错误")
    assert not ObserveStore().exists("bad")


def test_l08_unknown_datasource_returns_error(
    duckdb_datasources, write_config, run_cli
) -> None:
    """L08: 数据源未配置 → success=false"""
    write_config(datasources=duckdb_datasources)
    code, out, _err = run_cli(
        "observe",
        "load",
        "--name",
        "x",
        "--datasource",
        "nope",
        "--sql",
        "SELECT 1",
    )
    # 加载失败退出 1（docs/specs/cli.md 退出码契约；退出 0 与契约表及 test_exit_codes 冲突）
    assert code == 1
    result = json.loads(out)
    assert result["success"] is False
    assert "数据源不存在" in result["error"]


def test_l09_datasource_and_memory_mutually_exclusive(
    duckdb_datasources, write_config, run_cli
) -> None:
    """L09: --datasource 与 --memory 互斥 → 参数错误退出 2"""
    write_config(datasources=duckdb_datasources)
    code, _out, err = run_cli(
        "observe",
        "load",
        "--name",
        "x",
        "--datasource",
        "local",
        "--memory",
        "--sql",
        "SELECT 1",
    )
    assert code == 2
    assert "usage:" in err.lower()


def test_l10_missing_required_sql(write_config, run_cli) -> None:
    """L10: --sql 必填 → 参数错误退出 2"""
    write_config(datasources={})
    code, _out, err = run_cli("observe", "load", "--name", "x", "--memory")
    assert code == 2
    assert "usage:" in err.lower()


def test_l11_missing_config(run_cli) -> None:
    """L11: 配置文件不存在 → 报错退出 1"""
    code, _out, err = run_cli(
        "observe", "load", "--name", "x", "--memory", "--sql", "SELECT 1"
    )
    assert code == 1
    assert "配置文件不存在" in err
