"""启动路径惰性导入自检 — 快速路径不应加载重依赖"""

import subprocess
import sys

HEAVY = ("pandas", "sklearn", "matplotlib", "duckdb", "sqlalchemy", "networkx", "falkordb")


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )


def test_cli_import_skips_heavy_deps():
    code = (
        "import sys, govio.cli; "
        f"print(' '.join(m for m in sys.modules if m.split('.')[0] in {HEAVY}))"
    )
    result = _run(code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


def test_lazy_public_api_resolves():
    code = (
        "from govio import FalkorDBGraph, LadybugGraph, NetworkXGraph, "
        "build_metric_sql, main; "
        "from govio.metadata import StandardRecommender, DEFAULT_WEIGHTS, "
        "load_relationships; print('ok')"
    )
    result = _run(code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
