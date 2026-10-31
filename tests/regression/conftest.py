"""query / observe load 回归测试夹具

隔离 HOME（ConfigManager 配置）与 cwd（.govio 输出、ObserveStore），
并从 tests/regression/data 构建完整测试数据：
- sample.duckdb  ← seed.sql（observe load 的数据源）
- ontology.gml   ← graph_csv/（query NetworkX 后端）
- ontology.lbdb  ← graph_csv/（query Ladybug 后端）
"""

import contextlib
import io
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import duckdb
import pytest
import yaml

DATA_DIR = Path(__file__).parent / "data"
GRAPH_CSV_DIR = DATA_DIR / "graph_csv"


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """隔离 HOME 与 cwd，避免污染真实配置与 .govio 目录"""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def run_cli(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> Callable[..., tuple[int, str, str]]:
    """以 govio-cli 真实入口执行命令

    Returns:
        callable: 调用后返回 (exit_code, stdout, stderr)
    """

    def _run(*argv: str) -> tuple[int, str, str]:
        capsys.readouterr()  # 清空夹具构建期的残留输出
        monkeypatch.setattr(sys, "argv", ["govio-cli", *argv])
        code = 0
        try:
            from govio.cli.main import main

            main()
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 1
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    return _run


@pytest.fixture
def write_config(isolated_env: Path) -> Callable[..., Path]:
    """写入 ~/.govio/config.yaml 的工厂夹具"""

    def _write(
        graph: dict[str, Any] | None = None,
        datasources: dict[str, Any] | None = None,
    ) -> Path:
        config_path = isolated_env / ".govio" / "config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config = {"graph": graph or {}, "datasources": datasources or {}}
        config_path.write_text(
            yaml.safe_dump(config, allow_unicode=True), encoding="utf-8"
        )
        return config_path

    return _write


@pytest.fixture
def sample_duckdb(isolated_env: Path) -> Path:
    """由 seed.sql 构建样例 DuckDB 数据源"""
    db_path = isolated_env / "sample.duckdb"
    script = (DATA_DIR / "seed.sql").read_text(encoding="utf-8")
    conn = duckdb.connect(str(db_path))
    conn.execute(script)
    conn.close()
    return db_path


@pytest.fixture
def duckdb_datasources(sample_duckdb: Path) -> dict[str, Any]:
    """observe load 用的数据源配置（URL 用正斜杠，跨平台安全）"""
    return {"local": {"url": f"duckdb://{sample_duckdb.as_posix()}", "connect_args": {}}}


@pytest.fixture
def nx_gml(isolated_env: Path) -> Path:
    """由 graph_csv 构建 NetworkX GML 图（复用 gen_networkx CSV→GML 管道）"""
    from govio.metadata.gen_networkx import build_graph

    gml_path = isolated_env / "ontology.gml"
    # 构建过程的 tqdm/打印不进 capsys
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        build_graph(str(GRAPH_CSV_DIR), str(gml_path))
    return gml_path


@pytest.fixture
def ladybug_db(isolated_env: Path) -> Path:
    """由 graph_csv 构建 Ladybug 图（复用 ladybug_loader 全量重建管道）"""
    from govio.graph.ladybug_loader import import_csv_to_ladybug

    db_path = isolated_env / "ontology.lbdb"
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        import_csv_to_ladybug(GRAPH_CSV_DIR, db_path)
    return db_path
