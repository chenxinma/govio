"""构建 pi.agent 扩展分发包（形态 A）：dist/govio-pi-<version>/

用法: uv run scripts/package_pi_extension.py [--version X.Y.Z] [--output dist]

对应 docs/tasks/T1-distribution-design/design.md §2.1/§4.2/§4.4：
1. 同步 skills/ 到 packaging/pi/skills/（排除 skills/govio/assets 派生数据与缓存文件）；
2. 清理并重建 dist/govio-pi-<version>/（复制 packaging/pi/**）；
3. 注入版本号到产物 package.json（默认取 pyproject.toml 的 project.version）；
4. 产物结构自检（6 个 skill、无 assets/缓存、pi manifest、扩展入口、版本注入）。

可重复执行：每次运行先清理旧产物与 skills 副本再重建。
排除规则与版本注入复用 scripts/_packaging_common.py，本文件只保留 pi 形态特有逻辑。
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
from pathlib import Path

# 按文件路径加载脚本时 scripts/ 不在 sys.path 上，显式加入以导入共享模块
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _packaging_common import (
    collect_files,
    copy_files,
    is_excluded_dir,
    is_excluded_file,
    read_pyproject_version,
    rewrite_package_version,
    validate_version,
)

ROOT = Path(__file__).resolve().parents[1]
PACKAGING_DIR = ROOT / "packaging" / "pi"
SKILLS_DIR = ROOT / "skills"
SKILLS_TARGET = PACKAGING_DIR / "skills"

# design.md §1.1：6 个 skill 直接位于 skills 根下，目录名与 frontmatter name 一致
EXPECTED_SKILLS = ("govio", "govio-eda", "govio-meta", "govio-observe", "govio-query", "govio-reconcile")
VERSION_PATTERN = re.compile(r"[0-9A-Za-z][0-9A-Za-z.\-+]*")

logger = logging.getLogger(__name__)


def _skill_frontmatter_name(skill_md: Path) -> str | None:
    """读取 SKILL.md frontmatter 中的 name 字段

    Args:
        skill_md: SKILL.md 路径

    Returns:
        str | None: name 字段值（未找到返回 None）
    """
    match = re.search(r"^name:\s*(\S+)\s*$", skill_md.read_text(encoding="utf-8"), re.MULTILINE)
    return match.group(1) if match else None


def _verify_artifact(out_dir: Path, version: str) -> list[str]:
    """产物结构自检

    Args:
        out_dir: 产物目录
        version: 期望版本号

    Returns:
        list[str]: 问题列表（空表示通过）
    """
    problems: list[str] = []

    package_json = out_dir / "package.json"
    if not package_json.is_file():
        problems.append("缺少 package.json")
    else:
        data = json.loads(package_json.read_text(encoding="utf-8"))
        if data.get("version") != version:
            problems.append(f"package.json version 未注入: {data.get('version')!r} != {version!r}")
        manifest = data.get("pi") or {}
        if manifest.get("extensions") != ["extensions/govio-tools.ts"]:
            problems.append(f"pi manifest extensions 不符: {manifest.get('extensions')!r}")
        if "skills" not in (manifest.get("skills") or []):
            problems.append(f"pi manifest 未声明 skills: {manifest.get('skills')!r}")

    if not (out_dir / "extensions" / "govio-tools.ts").is_file():
        problems.append("缺少 extensions/govio-tools.ts")
    if not (out_dir / "README.md").is_file():
        problems.append("缺少 README.md")

    skills_dir = out_dir / "skills"
    actual_skills = sorted(p.name for p in skills_dir.iterdir() if p.is_dir()) if skills_dir.is_dir() else []
    if actual_skills != sorted(EXPECTED_SKILLS):
        problems.append(f"skills 目录不符: {actual_skills} != {sorted(EXPECTED_SKILLS)}")
    for name in EXPECTED_SKILLS:
        skill_md = skills_dir / name / "SKILL.md"
        if not skill_md.is_file():
            problems.append(f"缺少 skills/{name}/SKILL.md")
        elif _skill_frontmatter_name(skill_md) != name:
            problems.append(f"skills/{name}/SKILL.md frontmatter name 与目录名不一致")

    for path in out_dir.rglob("*"):
        if path.is_dir() and is_excluded_dir(path):
            problems.append(f"产物含排除目录: {path.relative_to(out_dir).as_posix()}")
        elif path.is_file() and is_excluded_file(path):
            problems.append(f"产物含缓存文件: {path.relative_to(out_dir).as_posix()}")

    return problems


def main() -> int:
    """执行打包与自检

    Returns:
        int: 退出码（0 成功，非 0 失败）
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="构建 pi.agent 扩展分发包 dist/govio-pi-<version>/")
    parser.add_argument("--version", help="包版本号（默认取 pyproject.toml 的 project.version）")
    parser.add_argument("--output", default="dist", help="产物输出目录（相对路径以仓库根为基准，默认 dist）")
    args = parser.parse_args()

    try:
        version = args.version or read_pyproject_version(ROOT / "pyproject.toml")
    except (OSError, ValueError):
        logger.exception("✗ 读取版本号失败")
        return 1
    try:
        validate_version(version, VERSION_PATTERN)
    except ValueError:
        # 友好中文报错，不打印堆栈（同 package_harness_plugin.py 的 CLI 边界惯例）
        logger.error("✗ 非法版本号: %s", version)  # noqa: TRY400
        return 1

    for source_dir, label in ((PACKAGING_DIR, "扩展包源"), (SKILLS_DIR, "skills 源")):
        if not source_dir.is_dir():
            logger.error("✗ %s目录不存在: %s", label, source_dir)
            return 1

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    try:
        # 1. 同步 skills 副本到 packaging/pi/skills/（供 pi install ./packaging/pi 原地加载）
        if SKILLS_TARGET.exists():
            shutil.rmtree(SKILLS_TARGET)
        SKILLS_TARGET.mkdir(parents=True)
        skill_file_count = copy_files(collect_files(SKILLS_DIR), SKILLS_DIR, SKILLS_TARGET)

        # 2. 清理并重建产物目录
        out_dir = output_dir / f"govio-pi-{version}"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)
        file_count = copy_files(collect_files(PACKAGING_DIR), PACKAGING_DIR, out_dir)

        # 3. 版本注入
        rewrite_package_version(out_dir / "package.json", version)
    except (OSError, ValueError):
        logger.exception("✗ 打包失败")
        return 1

    # 4. 产物结构自检
    problems = _verify_artifact(out_dir, version)
    if problems:
        for problem in problems:
            logger.error("✗ 自检失败: %s", problem)
        return 1

    logger.info("✓ %s（%d 个文件，其中 skills 副本 %d 个文件）", out_dir.resolve(), file_count, skill_file_count)
    logger.info("✓ 产物结构自检通过（%d 个 skill、无 assets/缓存文件、版本 %s 已注入）", len(EXPECTED_SKILLS), version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
