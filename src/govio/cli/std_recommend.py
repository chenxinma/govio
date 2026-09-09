import sys
from pathlib import Path

import pandas as pd

from ..metadata.utility import data_standard_recommend


def std_recommend(
    output_dir: Path,
    kundb: str,
    workspace_uuid: str,
    app_map: str,
    csv_dir: Path,
):
    """数据标准推荐主函数

    Args:
        output_dir: 推荐结果输出目录
        kundb: TDS 元数据库 URL
        workspace_uuid: 工作区 UUID
        app_map: 应用数据库映射 JSON 文件路径
        csv_dir: 已导入的 CSV 目录
    """
    if not kundb:
        print("❌ 需要指定 --kundb", file=sys.stderr)
        sys.exit(1)

    if not workspace_uuid:
        print("❌ 需要指定 --workspace-uuid", file=sys.stderr)
        sys.exit(1)

    app_map_path = Path(app_map)
    if not app_map_path.exists():
        print(f"❌ 应用映射文件不存在: {app_map_path}", file=sys.stderr)
        sys.exit(1)

    if not csv_dir.exists():
        print(f"❌ CSV 目录不存在: {csv_dir}", file=sys.stderr)
        sys.exit(1)

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    df_app_db_map = pd.read_json(app_map)

    print("\n=== 数据标准推荐 ===\n")
    print(f"数据库: {kundb}")
    print(f"工作区: {workspace_uuid}")
    print(f"应用映射: {app_map}")
    print(f"CSV 目录: {csv_dir}")
    print(f"输出目录: {output_dir}")

    try:
        data_standard_recommend(
            output=output_dir,
            db=kundb,
            workspace_uuid=workspace_uuid,
            df_app_db_map=df_app_db_map,
        )
        print("\n✓ 推荐完成！")
        if (output_dir / "COMPLIES_WITH.csv").exists():
            print(f"✓ 关系文件已生成: {output_dir / 'COMPLIES_WITH.csv'}")
    except Exception as e:
        print(f"\n❌ 推荐失败: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    std_recommend(Path(sys.argv[1]))
