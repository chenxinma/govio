"""scripts/package_pi_extension.py 打包脚本的自动化测试

用 tmp_path/monkeypatch 搭建最小假仓库并重定向脚本路径常量，不依赖网络与真实仓库内容。
覆盖：版本注入、打包排除规则、重复执行幂等、产物结构自检失败路径。
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "package_pi_extension.py"

# 与 design.md §1.1 一致：6 个 skill 直接位于 skills 根下（测试侧硬编码，独立于脚本常量）
EXPECTED_SKILLS = ("govio", "govio-eda", "govio-meta", "govio-observe", "govio-query", "govio-reconcile")

_spec = importlib.util.spec_from_file_location("package_pi_extension", SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
package_pi_extension = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(package_pi_extension)


def _write_skill(skills_root: Path, name: str) -> None:
    """写入一个含 frontmatter name 的最小 skill

    Args:
        skills_root: skills 根目录
        name: skill 目录名（frontmatter name 与之相同）
    """
    skill_dir = skills_root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(f"---\nname: {name}\ndescription: 测试 skill\n---\n\n# {name}\n", encoding="utf-8")


def _build_fake_repo(repo: Path) -> None:
    """搭建最小假仓库：pyproject + skills 源（含应排除的垃圾文件）+ packaging/pi 包源

    Args:
        repo: 假仓库根目录
    """
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "pyproject.toml").write_text('[project]\nname = "govio"\nversion = "9.9.9"\n', encoding="utf-8")

    skills_root = repo / "skills"
    for name in EXPECTED_SKILLS:
        _write_skill(skills_root, name)
    # 应排除：govio/assets 派生数据、__pycache__、.pyc、.DS_Store
    (skills_root / "govio" / "assets").mkdir()
    (skills_root / "govio" / "assets" / "generated.json").write_text("{}", encoding="utf-8")
    (skills_root / "govio-eda" / "__pycache__").mkdir()
    (skills_root / "govio-eda" / "__pycache__" / "cache.pyc").write_bytes(b"\x00")
    (skills_root / "govio-meta" / "junk.pyc").write_bytes(b"\x00")
    (skills_root / "govio-query" / ".DS_Store").write_bytes(b"\x00")
    # 应保留：普通文件与非 govio skill 自带的 assets 资源
    (skills_root / "govio" / "eval.md").write_text("# eval\n", encoding="utf-8")
    (skills_root / "govio-observe" / "assets").mkdir()
    (skills_root / "govio-observe" / "assets" / "keep.json").write_text("{}", encoding="utf-8")

    packaging_dir = repo / "packaging" / "pi"
    (packaging_dir / "extensions").mkdir(parents=True)
    (packaging_dir / "package.json").write_text(
        json.dumps(
            {
                "name": "govio-tools",
                "version": "0.0.0",
                "pi": {"extensions": ["extensions/govio-tools.ts"], "skills": ["skills"]},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (packaging_dir / "README.md").write_text("# govio-tools\n", encoding="utf-8")
    (packaging_dir / "extensions" / "govio-tools.ts").write_text("// stub\n", encoding="utf-8")


@pytest.fixture
def fake_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """构建假仓库并把脚本的路径常量重定向到临时目录

    Args:
        tmp_path: pytest 临时目录
        monkeypatch: pytest monkeypatch

    Returns:
        Path: 假仓库根目录
    """
    repo = tmp_path / "repo"
    _build_fake_repo(repo)
    monkeypatch.setattr(package_pi_extension, "ROOT", repo)
    monkeypatch.setattr(package_pi_extension, "PACKAGING_DIR", repo / "packaging" / "pi")
    monkeypatch.setattr(package_pi_extension, "SKILLS_DIR", repo / "skills")
    monkeypatch.setattr(package_pi_extension, "SKILLS_TARGET", repo / "packaging" / "pi" / "skills")
    return repo


def _run_main(monkeypatch: pytest.MonkeyPatch, *args: str) -> int:
    """以命令行参数调用打包脚本 main()

    Args:
        monkeypatch: pytest monkeypatch（用于替换 sys.argv）
        *args: 追加的脚本参数

    Returns:
        int: 脚本退出码
    """
    monkeypatch.setattr(sys, "argv", ["package_pi_extension.py", *args])
    return package_pi_extension.main()


def _snapshot_tree(root: Path) -> dict[str, str]:
    """目录树快照：相对路径 → 内容 SHA256（用于幂等比对）

    Args:
        root: 目录根

    Returns:
        dict[str, str]: 相对路径到内容哈希的映射
    """
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_version_injected_from_pyproject(fake_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """默认版本号取 pyproject.toml 的 project.version 并注入产物 package.json"""
    assert _run_main(monkeypatch) == 0

    out_dir = fake_repo / "dist" / "govio-pi-9.9.9"
    data = json.loads((out_dir / "package.json").read_text(encoding="utf-8"))
    assert data["version"] == "9.9.9"


def test_version_cli_override(fake_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--version 覆盖 pyproject 版本号，产物目录名与注入值一致"""
    assert _run_main(monkeypatch, "--version", "1.2.3") == 0

    out_dir = fake_repo / "dist" / "govio-pi-1.2.3"
    data = json.loads((out_dir / "package.json").read_text(encoding="utf-8"))
    assert data["version"] == "1.2.3"


def test_illegal_version_rejected(fake_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """非法版本号（含空格与特殊字符）被打包脚本拒绝，不产出目录"""
    assert _run_main(monkeypatch, "--version", "bad version!") == 1

    assert not (fake_repo / "dist").exists()


def test_missing_pyproject_version_rejected(fake_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """pyproject.toml 缺少 project.version 且未指定 --version 时退出码为 1"""
    (fake_repo / "pyproject.toml").write_text('[project]\nname = "govio"\n', encoding="utf-8")

    assert _run_main(monkeypatch) == 1


def test_exclusion_rules(fake_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """排除规则：govio/assets、__pycache__、.pyc、.DS_Store 不进入 skills 副本与产物"""
    assert _run_main(monkeypatch) == 0

    out_dir = fake_repo / "dist" / "govio-pi-9.9.9"
    skills_targets = (fake_repo / "packaging" / "pi" / "skills", out_dir / "skills")
    for skills_root in skills_targets:
        assert not (skills_root / "govio" / "assets").exists()
        assert not (skills_root / "govio-eda" / "__pycache__").exists()
        assert not list(skills_root.rglob("*.pyc"))
        assert not list(skills_root.rglob(".DS_Store"))
        # 保留项：普通文件与非 govio skill 自带的 assets 资源
        assert (skills_root / "govio" / "eval.md").is_file()
        assert (skills_root / "govio-observe" / "assets" / "keep.json").is_file()


def test_repeat_run_idempotent(fake_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """重复执行：产物与 skills 副本逐文件一致，陈旧文件被清理重建"""
    assert _run_main(monkeypatch) == 0
    out_dir = fake_repo / "dist" / "govio-pi-9.9.9"
    skills_copy = fake_repo / "packaging" / "pi" / "skills"
    first_artifact = _snapshot_tree(out_dir)
    first_skills_copy = _snapshot_tree(skills_copy)

    # 注入陈旧文件，验证重复执行会先清理再重建
    (out_dir / "stale.txt").write_text("stale", encoding="utf-8")
    (skills_copy / "stale.txt").write_text("stale", encoding="utf-8")

    assert _run_main(monkeypatch) == 0
    assert _snapshot_tree(out_dir) == first_artifact
    assert _snapshot_tree(skills_copy) == first_skills_copy


def test_verify_artifact_failure_paths(fake_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """产物自检：篡改产物后各类结构问题均被报告"""
    assert _run_main(monkeypatch) == 0
    out_dir = fake_repo / "dist" / "govio-pi-9.9.9"
    assert package_pi_extension._verify_artifact(out_dir, "9.9.9") == []

    # 模拟各类产物损坏
    (out_dir / "skills" / "govio" / "assets").mkdir()
    (out_dir / "skills" / "govio-eda" / "__pycache__").mkdir()
    (out_dir / "skills" / "govio-meta" / "junk.pyc").write_bytes(b"\x00")
    (out_dir / "skills" / "govio-query" / "SKILL.md").unlink()
    (out_dir / "extensions" / "govio-tools.ts").unlink()
    (out_dir / "package.json").write_text(
        json.dumps({"version": "0.0.1", "pi": {"extensions": [], "skills": []}}) + "\n",
        encoding="utf-8",
    )

    problems = package_pi_extension._verify_artifact(out_dir, "9.9.9")
    joined = "\n".join(problems)
    assert "package.json version 未注入" in joined
    assert "pi manifest extensions 不符" in joined
    assert "pi manifest 未声明 skills" in joined
    assert "缺少 extensions/govio-tools.ts" in joined
    assert "缺少 skills/govio-query/SKILL.md" in joined
    assert "产物含排除目录" in joined
    assert "产物含缓存文件" in joined


def test_self_check_failure_returns_nonzero(fake_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """端到端自检失败：SKILL.md frontmatter name 与目录名不一致时 main() 返回 1"""
    skill_md = fake_repo / "skills" / "govio-query" / "SKILL.md"
    skill_md.write_text("---\nname: wrong-name\n---\n\n# govio-query\n", encoding="utf-8")

    assert _run_main(monkeypatch) == 1
