"""三支打包脚本共享的排除规则与版本注入

被以下打包脚本复用，规则口径来自 docs/tasks/T1-distribution-design/design.md §4.4：

- `package_skills.py`（仓库根）：`skills/` -> `dist/govio-skills.zip`
- `scripts/package_pi_extension.py`：`packaging/pi/` + skills 副本 -> `dist/govio-pi-<version>/`
- `scripts/package_harness_plugin.py`：`packaging/deepseek/` -> `dist/govio-dsh-<version>/`

统一排除：`<...>/govio/assets` 派生数据（assets 契约：永远落在
`<skills-root>/govio/assets`）、`__pycache__`、`node_modules`、`*.pyc`、
`.DS_Store`、`package-lock.json`。各脚本额外的相对路径前缀排除通过
`exclude_prefixes` 传入。
"""

from __future__ import annotations

import json
import re
import shutil
import tomllib
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path

# 目录段排除：缓存目录与 Node 依赖目录
EXCLUDED_DIR_NAMES = frozenset({"__pycache__", "node_modules"})
# 文件名排除：系统与锁文件
EXCLUDED_FILE_NAMES = frozenset({".DS_Store", "package-lock.json"})
# 扩展名排除：字节码缓存
EXCLUDED_SUFFIXES = frozenset({".pyc"})

# 派生数据目录约定：<skills-root>/govio/assets（design.md §1.2）
ASSETS_PARENT_NAME = "govio"
ASSETS_DIR_NAME = "assets"
ASSETS_EXCLUDE_PREFIX = "govio/assets"


def has_derived_assets(parts: Sequence[str]) -> bool:
    """路径段中是否出现 `<...>/govio/assets`（派生数据目录本身或其子孙）

    Args:
        parts: 相对源根拆出的路径段

    Returns:
        bool: 是否命中派生数据目录
    """
    return any(
        parent == ASSETS_PARENT_NAME and name == ASSETS_DIR_NAME
        for parent, name in pairwise(parts)
    )


def is_excluded_dir(path: Path) -> bool:
    """目录是否属于排除范围（缓存/依赖目录、govio/assets 派生数据）

    Args:
        path: 待判断的目录

    Returns:
        bool: 是否排除
    """
    if path.name in EXCLUDED_DIR_NAMES:
        return True
    return path.name == ASSETS_DIR_NAME and path.parent.name == ASSETS_PARENT_NAME


def is_excluded_file(path: Path) -> bool:
    """文件是否属于按名称/扩展名的排除范围

    Args:
        path: 待判断的文件

    Returns:
        bool: 是否排除
    """
    return path.name in EXCLUDED_FILE_NAMES or path.suffix in EXCLUDED_SUFFIXES


def is_excluded(path: Path, root: Path, exclude_prefixes: Sequence[str] = ()) -> bool:
    """判断路径（文件或目录）是否命中统一排除规则

    Args:
        path: 待判断路径（须位于 root 之下）
        root: 复制源根目录（用于计算相对路径）
        exclude_prefixes: 额外排除的相对路径前缀，如 ("govio/assets",)

    Returns:
        bool: 是否排除
    """
    rel = path.relative_to(root)
    if any(part in EXCLUDED_DIR_NAMES for part in rel.parts):
        return True
    if has_derived_assets(rel.parts) or is_excluded_file(path):
        return True
    rel_posix = rel.as_posix()
    return any(
        rel_posix == prefix or rel_posix.startswith(f"{prefix}/")
        for prefix in exclude_prefixes
    )


def collect_files(src: Path, exclude_prefixes: Sequence[str] = ()) -> list[Path]:
    """收集待打包文件（跳过排除项，按相对路径 POSIX 序排序）

    Args:
        src: 复制源根目录
        exclude_prefixes: 额外排除的相对路径前缀，如 ("govio/assets",)

    Returns:
        list[Path]: 待复制文件清单
    """
    files = [
        path
        for path in src.rglob("*")
        if path.is_file() and not is_excluded(path, src, exclude_prefixes)
    ]
    return sorted(files, key=lambda p: p.relative_to(src).as_posix())


def copy_files(files: Sequence[Path], src: Path, dst: Path) -> int:
    """按相对路径把文件复制到目标目录（保留 mtime，目标目录须已存在）

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


def read_pyproject_version(pyproject_path: Path) -> str:
    """读取 pyproject.toml 的 project.version 作为默认版本号

    Args:
        pyproject_path: pyproject.toml 路径

    Returns:
        str: 版本号

    Raises:
        ValueError: pyproject.toml 缺少 project.version
    """
    with pyproject_path.open("rb") as fh:
        data = tomllib.load(fh)
    project = data.get("project")
    if not isinstance(project, dict) or not isinstance(project.get("version"), str):
        raise ValueError(f"pyproject.toml 缺少 project.version: {pyproject_path}")
    return project["version"]


def validate_version(version: str, pattern: re.Pattern[str]) -> str:
    """校验版本号（限定为安全的文件名字符，避免路径注入）

    Args:
        version: 版本号，如 0.7.0
        pattern: 该分发形态的版本号格式（fullmatch 语义）

    Returns:
        str: 校验通过的版本号

    Raises:
        ValueError: 版本号不合法
    """
    if not pattern.fullmatch(version):
        raise ValueError(f"非法版本号: {version!r}（示例: 0.7.0）")
    return version


def rewrite_package_version(package_json: Path, version: str) -> None:
    """重写 package.json 的 version 字段（UTF-8 读写，其余字段保持原样）

    Args:
        package_json: 目标 package.json 路径
        version: 注入的版本号
    """
    data = json.loads(package_json.read_text(encoding="utf-8"))
    data["version"] = version
    package_json.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
