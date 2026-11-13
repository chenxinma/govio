"""打包 skills/ 目录为 dist/govio-skills.zip，排除派生数据与缓存文件

用法: uv run package_skills.py

排除规则来自 scripts/_packaging_common.py（与另两支打包脚本共用）：
skills/govio/assets 派生数据、__pycache__、*.pyc、.DS_Store 等不进分发包。
"""

import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

# 按文件路径加载脚本时 scripts/ 不在 sys.path 上，显式加入以导入共享模块
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from _packaging_common import collect_files

ROOT = Path(__file__).resolve().parent
SKILLS_DIR = ROOT / "skills"
OUTPUT = ROOT / "dist" / "govio-skills.zip"


def main() -> None:
    """写出 dist/govio-skills.zip（每次先清理旧包，可重复执行）"""
    # 清理旧版本
    if OUTPUT.exists():
        OUTPUT.unlink()

    files = sorted(collect_files(SKILLS_DIR))
    OUTPUT.parent.mkdir(exist_ok=True)
    with ZipFile(OUTPUT, "w", ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, path.relative_to(ROOT).as_posix())
    print(f"✓ {OUTPUT}（{len(files)} 个文件）")


if __name__ == "__main__":
    main()
