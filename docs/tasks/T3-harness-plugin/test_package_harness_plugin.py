"""package_harness_plugin.py 的打包与产物自检测试

位于 T3 任务目录：tests/ 为共享目录，不在本任务所有权清单内，
故暂放此处；T7 可将本文件迁入 tests/ 统一管理。
"""

import importlib.util
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "package_harness_plugin.py"
PLUGIN_LIB_ENTRY = REPO_ROOT / "packaging" / "deepseek" / "lib" / "index.js"


def load_module():
    """按文件路径加载 scripts/package_harness_plugin.py

    Returns:
        module: 打包脚本模块
    """
    spec = importlib.util.spec_from_file_location("package_harness_plugin", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def packaging():
    """打包脚本模块（按文件路径加载）"""
    return load_module()


def test_validate_version_accepts_semver(packaging):
    """版本号校验接受常规语义化版本"""
    assert packaging.validate_version("0.6.0") == "0.6.0"
    assert packaging.validate_version("1.2.3-rc.1") == "1.2.3-rc.1"


def test_validate_version_rejects_path_injection(packaging):
    """版本号校验拒绝路径分隔符与空串（避免路径注入）"""
    for bad in ("", "bad/version", "..\\evil", "0.6.0/.."):
        with pytest.raises(ValueError):
            packaging.validate_version(bad)


def test_collect_files_excludes_pycache_pyc_and_ds_store(packaging, tmp_path):
    """统一排除规则剔除 __pycache__、.pyc、.DS_Store、node_modules"""
    (tmp_path / "keep").mkdir()
    (tmp_path / "keep" / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "x.pyc").write_bytes(b"\x00")
    (tmp_path / "b.pyc").write_bytes(b"\x00")
    (tmp_path / ".DS_Store").write_bytes(b"\x00")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "pkg.js").write_text("js", encoding="utf-8")

    files = packaging.collect_files(tmp_path)

    assert [p.relative_to(tmp_path).as_posix() for p in files] == ["keep/a.txt"]


def test_collect_files_exclude_prefix_skips_assets(packaging, tmp_path):
    """exclude_prefixes 剔除 govio/assets 派生数据，保留 skill 内部 assets"""
    (tmp_path / "govio" / "assets").mkdir(parents=True)
    (tmp_path / "govio" / "assets" / "schema.md").write_text("s", encoding="utf-8")
    (tmp_path / "govio" / "SKILL.md").write_text("k", encoding="utf-8")
    (tmp_path / "govio-observe" / "assets").mkdir(parents=True)
    (tmp_path / "govio-observe" / "assets" / "keep.json").write_text(
        "{}", encoding="utf-8"
    )

    files = packaging.collect_files(tmp_path, exclude_prefixes=("govio/assets",))

    rels = [p.relative_to(tmp_path).as_posix() for p in files]
    assert rels == ["govio-observe/assets/keep.json", "govio/SKILL.md"]


def test_rewrite_package_version_injects_version(packaging, tmp_path):
    """package.json 版本注入只改 version 字段（UTF-8 读写）"""
    pkg = tmp_path / "package.json"
    pkg.write_text(
        '{"name": "x", "version": "0.0.0", "dsh": {"bundle": {"patch": "./cordis.patch.yml"}}}',
        encoding="utf-8",
    )

    packaging.rewrite_package_version(pkg, "1.2.3")

    data = json.loads(pkg.read_text(encoding="utf-8"))
    assert data["version"] == "1.2.3"
    assert data["name"] == "x"
    assert data["dsh"] == {"bundle": {"patch": "./cordis.patch.yml"}}


def test_check_plugin_artifact_reports_missing_entries(packaging, tmp_path):
    """产物结构自检能发现缺失的必需条目"""
    problems = packaging.check_plugin_artifact(tmp_path, "0.6.0")

    assert any("lib/index.js" in p for p in problems)
    assert any("package.json" in p for p in problems)


def test_check_skills_artifact_requires_six_skill_dirs(packaging, tmp_path):
    """skills 产物自检要求 6 个 skill 目录齐全且不含 govio/assets"""
    (tmp_path / "govio").mkdir()
    (tmp_path / "govio" / "assets").mkdir()

    problems = packaging.check_skills_artifact(tmp_path)

    assert any("govio-meta/SKILL.md" in p for p in problems)
    assert any("govio/assets" in p for p in problems)


@pytest.mark.skipif(
    not PLUGIN_LIB_ENTRY.is_file(),
    reason="需要先编译 lib/（在 packaging/deepseek 下 npm install 后 npx tsc）",
)
def test_build_end_to_end_is_repeatable(tmp_path):
    """端到端打包：版本注入、skills 排除 assets、重复执行产物一致"""
    with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
        version = tomllib.load(fh)["project"]["version"]

    def run() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--output",
                str(tmp_path),
                "--skills-bundle",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

    first = run()
    assert first.returncode == 0, first.stderr
    plugin_dir = tmp_path / f"govio-dsh-{version}"
    assert (plugin_dir / "package.json").is_file()
    assert (plugin_dir / "lib" / "index.js").is_file()
    pkg = json.loads((plugin_dir / "package.json").read_text(encoding="utf-8"))
    assert pkg["version"] == version
    assert pkg["dsh"]["bundle"]["patch"] == "./cordis.patch.yml"

    skills_dir = tmp_path / "govio-dsh-skills"
    for skill in (
        "govio",
        "govio-eda",
        "govio-meta",
        "govio-observe",
        "govio-query",
        "govio-reconcile",
    ):
        assert (skills_dir / skill / "SKILL.md").is_file()
    assert not (skills_dir / "govio" / "assets").exists()

    first_files = sorted(
        p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()
    )

    second = run()
    assert second.returncode == 0, second.stderr
    second_files = sorted(
        p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()
    )
    assert first_files == second_files
