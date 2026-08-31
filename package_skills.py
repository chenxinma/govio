"""打包 skills/ 目录为 dist/govio-skill.zip，排除 skills/govio/assets 下的内容

用法: uv run python package_skills.py
"""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parent
SKILLS_DIR = ROOT / "skills"
EXCLUDE_DIR = SKILLS_DIR / "govio" / "assets"
OUTPUT = ROOT / "dist" / "govio-skill.zip"


def main() -> None:
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
