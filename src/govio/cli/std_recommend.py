import sys
from pathlib import Path

import pandas as pd

from .config import ConfigManager
from ..metadata.utility import data_standard_recommend


def std_recommend():
    """数据标准推荐主函数"""
    config_manager = ConfigManager()

    if not config_manager.exists():
        print("❌ 配置文件不存在，请先运行 govio onboard 进行配置")
        sys.exit(1)

    config = config_manager.load()

    kundb = config.get("kundb", "")
    workspace_uuid = config.get("workspace_uuid", "")
    app_map = config.get("app_map", "")
    csv_dir = config.get("csv_dir", "./")
    output = config.get("output_dir", csv_dir)

    if not all([kundb, workspace_uuid, app_map, csv_dir]):
        print("❌ 配置缺少必要字段，请检查 kundb, workspace_uuid, app_map, csv_dir")
        sys.exit(1)

    csv_dir_path = Path(csv_dir)
    output_path = Path(output)
    app_map_path = Path(app_map)

    if not csv_dir_path.exists():
        print(f"❌ CSV 目录不存在: {csv_dir_path}")
        sys.exit(1)

    if not app_map_path.exists():
        print(f"❌ 应用映射文件不存在: {app_map_path}")
        sys.exit(1)

    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    df_app_db_map = pd.read_json(app_map)

    print("\n=== 数据标准推荐 ===\n")
    print(f"数据库: {kundb}")
    print(f"工作区: {workspace_uuid}")
    print(f"应用映射: {app_map}")
    print(f"CSV 目录: {csv_dir}")
    print(f"输出目录: {output}")

    try:
        data_standard_recommend(
            output=output_path,
            db=kundb,
            workspace_uuid=workspace_uuid,
            df_app_db_map=df_app_db_map,
        )
        print("\n✓ 推荐完成！")
        if (output_path / "COMPLIES_WITH.csv").exists():
            print(f"✓ 关系文件已生成: {output_path / 'COMPLIES_WITH.csv'}")
    except Exception as e:
        print(f"\n❌ 推荐失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    std_recommend()
