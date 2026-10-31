"""打包 DeepSeek Harness（DSH）插件与配套 skills 到 dist/ 目录

用法: uv run python scripts/package_harness_plugin.py [--version X.Y.Z] [--output dist] [--skills-bundle]

产物:
    dist/govio-dsh-<version>/   DSH bundle 插件（lib/ 已编译，可 dsh plugin add link: 安装）
    dist/govio-dsh-skills/      仅在 --skills-bundle 时生成（6 个 skill 目录副本，排除 assets）

统一排除规则（design.md §4.4）: skills/govio/assets/**、**/__pycache__/**、**/*.pyc、
.DS_Store；插件源目录额外排除 node_modules/**。每次执行先清理旧产物再重建，可重复执行。
"""

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
PLUGIN_SRC = ROOT / "packaging" / "deepseek"
SKILLS_SRC = ROOT / "skills"
PYPROJECT = ROOT / "pyproject.toml"

PLUGIN_PACKAGE_NAME = "@dsh-external/govio-tools"
PLUGIN_DIST_PREFIX = "govio-dsh-"
SKILLS_DIST_NAME = "govio-dsh-skills"
SKILL_DIRS = (
    "govio",
    "govio-eda",
    "govio-meta",
    "govio-observe",
    "govio-query",
    "govio-reconcile",
)

VERSION_PATTERN = re.compile(r"^\d+(\.\d+)*([-.+][0-9A-Za-z.]+)?$")
EXCLUDED_DIR_NAMES = {"__pycache__", "node_modules"}
EXCLUDED_FILE_NAMES = {".DS_Store"}
EXCLUDED_SUFFIXES = {".pyc"}


def read_default_version(pyproject_path: Path) -> str:
    """读取 pyproject.toml 的 project.version 作为默认版本号

    Args:
        pyproject_path: pyproject.toml 路径

    Returns:
        str: 版本号
    """
    with pyproject_path.open("rb") as fh:
        data = tomllib.load(fh)
    project = data.get("project")
    if not isinstance(project, dict) or not isinstance(project.get("version"), str):
        raise ValueError(f"pyproject.toml 缺少 project.version: {pyproject_path}")
    return project["version"]


def validate_version(version: str) -> str:
    """校验版本号（限定为安全的文件名字符，避免路径注入）

    Args:
        version: 版本号，如 0.6.0

    Returns:
        str: 校验通过的版本号
    """
    if not VERSION_PATTERN.fullmatch(version):
        raise ValueError(f"非法版本号: {version!r}（示例: 0.6.0）")
    return version


def is_excluded(path: Path, root: Path) -> bool:
    """判断文件是否命中统一排除规则

    Args:
        path: 待判断文件
        root: 复制源根目录（用于计算相对路径）

    Returns:
        bool: 是否排除
    """
    rel = path.relative_to(root)
    if any(part in EXCLUDED_DIR_NAMES for part in rel.parts):
        return True
    if path.name in EXCLUDED_FILE_NAMES:
        return True
    return path.suffix in EXCLUDED_SUFFIXES


def collect_files(src: Path, exclude_prefixes: Sequence[str] = ()) -> list[Path]:
    """收集待复制文件（跳过排除项，按相对路径 POSIX 序排序）

    Args:
        src: 复制源根目录
        exclude_prefixes: 额外排除的相对路径前缀，如 ("govio/assets",)

    Returns:
        list[Path]: 待复制文件清单
    """
    files: list[Path] = []
    for path in src.rglob("*"):
        if not path.is_file() or is_excluded(path, src):
            continue
        rel_posix = path.relative_to(src).as_posix()
        if any(
            rel_posix == prefix or rel_posix.startswith(prefix + "/")
            for prefix in exclude_prefixes
        ):
            continue
        files.append(path)
    return sorted(files, key=lambda p: p.relative_to(src).as_posix())


def copy_files(files: Sequence[Path], src: Path, dst: Path) -> int:
    """按相对路径把文件复制到目标目录

    Args:
        files: 待复制文件清单
        src: 复制源根目录
        dst: 目标根目录

    Returns:
        int: 复制文件数
    """
    for path in files:
        target = dst / path.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    return len(files)


def compile_plugin(plugin_src: Path) -> None:
    """缺失 lib/index.js 时调 npx tsc 编译插件 TypeScript 源码

    Args:
        plugin_src: 插件源目录（含 tsconfig.json）
    """
    entry = plugin_src / "lib" / "index.js"
    if entry.is_file():
        logger.info("复用已有编译产物: %s", entry)
        return
    # Windows 下 npx 实为 npx.cmd，须经 shutil.which 解析后才能被 subprocess 执行
    npx = shutil.which("npx")
    if npx is None:
        raise RuntimeError(
            "未找到 npx/Node.js；请先安装 Node.js ≥ 22 与 TypeScript"
            "（在 packaging/deepseek 下执行 npm install）"
        )
    logger.info("编译插件 TypeScript: npx tsc -p %s", plugin_src)
    try:
        proc = subprocess.run(
            [npx, "tsc", "-p", "."],
            cwd=plugin_src,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError as exc:
        raise RuntimeError(
            f"无法执行 npx tsc（{exc}）；请先安装 Node.js ≥ 22 与 TypeScript"
            "（在 packaging/deepseek 下执行 npm install）"
        ) from exc
    if proc.returncode != 0:
        detail = (proc.stdout + proc.stderr).strip()[-2000:]
        raise RuntimeError(
            f"TypeScript 编译失败（退出码 {proc.returncode}）:\n{detail}\n"
            "提示: 先在 packaging/deepseek 下执行 npm install 安装 typescript 与 @deepseek-ai/* 类型依赖"
        )
    if not entry.is_file():
        detail = (proc.stdout + proc.stderr).strip()[-2000:]
        raise RuntimeError(
            f"编译后仍缺少产物: {entry}\n编译输出:\n{detail}\n"
            "提示: 先在 packaging/deepseek 下执行 npm install 安装 typescript 与 @deepseek-ai/* 类型依赖"
        )


def rewrite_package_version(package_json: Path, version: str) -> None:
    """重写 package.json 的 version 字段（UTF-8 读写）

    Args:
        package_json: 目标 package.json 路径
        version: 注入的版本号
    """
    data = json.loads(package_json.read_text(encoding="utf-8"))
    data["version"] = version
    package_json.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def build_plugin(plugin_src: Path, output_dir: Path, version: str) -> tuple[Path, int]:
    """生成插件产物目录并注入版本号（先清理旧产物，可重复执行）

    Args:
        plugin_src: 插件源目录
        output_dir: 输出目录
        version: 版本号

    Returns:
        tuple[Path, int]: （产物目录，文件数）
    """
    compile_plugin(plugin_src)
    target = output_dir / f"{PLUGIN_DIST_PREFIX}{version}"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    count = copy_files(collect_files(plugin_src), plugin_src, target)
    rewrite_package_version(target / "package.json", version)
    return target, count


def build_skills_bundle(skills_src: Path, output_dir: Path) -> tuple[Path, int]:
    """生成配套 skills 目录副本（6 个 skill 目录，排除 assets）

    Args:
        skills_src: 仓库 skills/ 目录
        output_dir: 输出目录

    Returns:
        tuple[Path, int]: （产物目录，文件数）
    """
    target = output_dir / SKILLS_DIST_NAME
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    count = copy_files(
        collect_files(skills_src, exclude_prefixes=("govio/assets",)),
        skills_src,
        target,
    )
    return target, count


def check_plugin_artifact(target: Path, version: str) -> list[str]:
    """插件产物结构自检

    Args:
        target: 插件产物目录
        version: 期望版本号

    Returns:
        list[str]: 问题清单（空表示通过）
    """
    problems: list[str] = []
    required = (
        "package.json",
        "cordis.patch.yml",
        "README.md",
        "src/index.ts",
        "tsconfig.json",
        "lib/index.js",
    )
    for rel in required:
        path = target / rel
        if not path.is_file() or path.stat().st_size == 0:
            problems.append(f"缺少或为空: {rel}")

    pkg_path = target / "package.json"
    if pkg_path.is_file():
        try:
            pkg = json.loads(pkg_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            problems.append(f"package.json 解析失败: {exc}")
        else:
            if pkg.get("name") != PLUGIN_PACKAGE_NAME:
                problems.append(
                    f"package.json name 应为 {PLUGIN_PACKAGE_NAME}，实际 {pkg.get('name')!r}"
                )
            if pkg.get("version") != version:
                problems.append(
                    f"package.json version 应为 {version}，实际 {pkg.get('version')!r}"
                )
            patch = pkg.get("dsh", {}).get("bundle", {}).get("patch")
            if patch != "./cordis.patch.yml":
                problems.append(
                    f"dsh.bundle.patch 应为 ./cordis.patch.yml，实际 {patch!r}"
                )

    patch_path = target / "cordis.patch.yml"
    if patch_path.is_file():
        text = patch_path.read_text(encoding="utf-8")
        if "insert" not in text or "govio-tools" not in text:
            problems.append("cordis.patch.yml 缺少 govio-tools 挂载行（insert）")

    for path in target.rglob("*"):
        if is_excluded(path, target):
            problems.append(f"产物含排除项: {path.relative_to(target).as_posix()}")
    return problems


def check_skills_artifact(target: Path) -> list[str]:
    """skills 产物结构自检

    Args:
        target: skills 产物目录

    Returns:
        list[str]: 问题清单（空表示通过）
    """
    problems: list[str] = []
    for skill in SKILL_DIRS:
        skill_md = target / skill / "SKILL.md"
        if not skill_md.is_file():
            problems.append(f"缺少 skill: {skill}/SKILL.md")
    if (target / "govio" / "assets").exists():
        problems.append(
            "skills 产物不应包含 govio/assets（派生数据，由 govio-cli meta graph 生成）"
        )
    for path in target.rglob("*"):
        if is_excluded(path, target):
            problems.append(f"产物含排除项: {path.relative_to(target).as_posix()}")
    return problems


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析命令行参数

    Args:
        argv: 参数列表（默认取 sys.argv[1:]）

    Returns:
        argparse.Namespace: 解析结果
    """
    parser = argparse.ArgumentParser(
        description="打包 DeepSeek Harness 插件与配套 skills 到 dist/ 目录",
    )
    parser.add_argument(
        "--version", help="版本号（默认取 pyproject.toml 的 project.version）"
    )
    parser.add_argument(
        "--output", default="dist", help="输出目录（默认 dist，相对路径以仓库根为基准）"
    )
    parser.add_argument(
        "--skills-bundle",
        action="store_true",
        help=f"额外生成 {SKILLS_DIST_NAME}/（6 个 skill 目录副本，排除 assets）",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """打包入口

    Args:
        argv: 参数列表（默认取 sys.argv[1:]）

    Returns:
        int: 退出码（0 成功，1 失败）
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    try:
        raw_version = (
            args.version
            if args.version is not None
            else read_default_version(PYPROJECT)
        )
        version = validate_version(raw_version)
        if not (PLUGIN_SRC / "package.json").is_file():
            raise FileNotFoundError(f"插件源目录缺少 package.json: {PLUGIN_SRC}")

        output_dir = Path(args.output)
        if not output_dir.is_absolute():
            output_dir = ROOT / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        plugin_target, plugin_count = build_plugin(PLUGIN_SRC, output_dir, version)
        problems = check_plugin_artifact(plugin_target, version)

        skills_target: Path | None = None
        skills_count = 0
        if args.skills_bundle:
            skills_target, skills_count = build_skills_bundle(SKILLS_SRC, output_dir)
            problems.extend(
                f"skills 产物: {p}" for p in check_skills_artifact(skills_target)
            )
    except (OSError, ValueError, RuntimeError) as exc:
        # CLI 边界转友好中文报错，不打印堆栈（同 pyproject 对 BLE001/TRY003 的意图性排除）
        logger.error("打包失败: %s", exc)  # noqa: TRY400
        return 1

    if problems:
        for problem in problems:
            logger.error("自检失败: %s", problem)
        return 1

    logger.info("✓ %s（%d 个文件）", plugin_target, plugin_count)
    if skills_target is not None:
        logger.info("✓ %s（%d 个文件）", skills_target, skills_count)
    logger.info("产物结构自检通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
