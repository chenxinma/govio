"""打包 skills/ 目录为 dist/govio-skills.zip，排除 skills/govio/assets 下的内容

用法: uv run package_skills.py
"""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parent
SKILLS_DIR = ROOT / "skills"
EXCLUDE_DIR = SKILLS_DIR / "govio" / "assets"
OUTPUT = ROOT / "dist" / "govio-skills.zip"


def main() -> None:
    # 清理旧版本
    if OUTPUT.exists():
        OUTPUT.unlink()

    files = sorted(
        p for p in SKILLS_DIR.rglob("*")
        if p.is_file() and EXCLUDE_DIR not in p.parents
    )
    OUTPUT.parent.mkdir(exist_ok=True)
    with ZipFile(OUTPUT, "w", ZIP_DEFLATED) as zf:
        for p in files:
            zf.write(p, p.relative_to(ROOT).as_posix())
    print(f"✓ {OUTPUT}（{len(files)} 个文件）")


if __name__ == "__main__":
    main()
