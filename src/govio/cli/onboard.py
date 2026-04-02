import sys
from pathlib import Path
from typing import Any

from .config import ConfigManager
from ..core.graph_factory import GraphFactory
from ..core.assets_generator import AssetsGenerator
from ..metadata.gen_networkx import build_graph


SKILLS_ASSETS_DIR = Path(".skills/govio/assets")


def validate_csv_directory(csv_dir: Path) -> bool:
    """验证 CSV 目录是否包含必需的文件

    Args:
        csv_dir: CSV 目录路径

    Returns:
        bool: 是否有效
    """
    if not csv_dir.exists() or not csv_dir.is_dir():
        return False

    required_files = ["PhysicalTable.csv"]

    for filename in required_files:
        if not (csv_dir / filename).exists():
            return False

    return True


def prompt_csv_config(config_manager: ConfigManager) -> dict[str, Any]:
    """提示用户输入 CSV 生成配置"""
    print("\n=== 步骤 1: CSV 元数据生成 ===\n")

    existing_config = {}
    if config_manager.exists():
        try:
            existing_config = config_manager.load()
        except Exception:
            pass

    default_kundb = existing_config.get("kundb", "")
    kundb = input(f"请输入元数据库 URL [默认: {default_kundb}]: ").strip()
    kundb = kundb or default_kundb

    default_app_list = existing_config.get("app_list", "")
    app_list = input(
        f"请输入应用清单 Excel 文件路径 [默认: {default_app_list}]: "
    ).strip()
    app_list = app_list or default_app_list

    default_app_map = existing_config.get("app_map", "")
    app_map = input(
        f"请输入应用数据库映射 JSON 文件路径 [默认: {default_app_map}]: "
    ).strip()
    app_map = app_map or default_app_map

    default_relationship = existing_config.get("relationship", "")
    relationship = input(
        f"请输入表关系 JSON 文件路径（可选，直接回车跳过） [默认: {default_relationship}]: "
    ).strip()
    relationship = relationship or default_relationship

    default_csv_dir = existing_config.get("csv_dir", "")
    csv_dir = input(f"请输入 CSV 输出目录 [默认: {default_csv_dir}]: ").strip()
    csv_dir = csv_dir or default_csv_dir

    return {
        "kundb": kundb,
        "app_list": app_list,
        "app_map": app_map,
        "relationship": relationship if relationship else None,
        "csv_dir": csv_dir,
    }


def generate_csv(config: dict[str, Any]) -> None:
    """根据配置生成 CSV 文件"""
    from ..metadata.utility import make_csv
    from ..metadata.database import DatabaseLoader
    from ..metadata.application import AppInfoLoader
    from ..metadata.standard import StandardLoader
    import pandas as pd

    kundb = config["kundb"]
    app_list = config["app_list"]
    app_map = config["app_map"]
    relationship = config.get("relationship")
    csv_dir = Path(config["csv_dir"])
    workspace_uuid = "82ee37374b314a938bf28170ab4db7cf"

    if not csv_dir.exists():
        csv_dir.mkdir(parents=True, exist_ok=True)

    df_app_db_map = pd.read_json(app_map, orient="records")

    make_csv(
        output=csv_dir,
        db=kundb,
        workspace_uuid=workspace_uuid,
        app_list_file=app_list,
        df_app_db_map=df_app_db_map,
        relationship_file=relationship,
    )


def prompt_backend_choice() -> str:
    """提示用户选择 backend"""
    print("\n=== Govio Onboard 向导 ===\n")
    print("请选择图数据库后端：")
    print("  1. networkx - 本地 GML 文件")
    print("  2. falkordb - FalkorDB 图数据库")

    while True:
        choice = input("\n请输入选项 (1/2) [默认: 1]: ").strip() or "1"

        if choice == "1":
            return "networkx"
        elif choice == "2":
            return "falkordb"
        else:
            print("❌ 无效选项，请输入 1 或 2")


def prompt_networkx_config() -> dict[str, Any]:
    """提示用户输入 NetworkX 配置"""
    print("\n--- NetworkX 配置 ---\n")

    generate_gml = (
        input("是否需要从 CSV 文件生成新的 GML 文件？ (yes/no) [默认: yes]: ")
        .strip()
        .lower()
    )
    generate_gml = generate_gml in ("yes", "y", "")

    if generate_gml:
        while True:
            csv_dir = input("请输入 CSV 目录路径: ").strip()
            csv_path = Path(csv_dir)

            if validate_csv_directory(csv_path):
                break
            else:
                print(f"❌ CSV 目录无效或缺少必需文件，请检查路径: {csv_dir}")

        gml_path = SKILLS_ASSETS_DIR / "ontology.gml"

        print("\n正在从 CSV 文件生成 GML 文件...")
        build_graph(str(csv_path), str(gml_path))
        print(f"✓ GML 文件已生成: {gml_path}")
    else:
        while True:
            gml_path_input = input("请输入 GML 文件路径: ").strip()
            gml_path = Path(gml_path_input)

            if gml_path.exists():
                break
            else:
                print(f"❌ GML 文件不存在: {gml_path}")

    return {"backend": "networkx", "networkx": {"gml_path": str(gml_path)}}


def prompt_falkordb_config() -> dict[str, Any]:
    """提示用户输入 FalkorDB 配置"""
    print("\n--- FalkorDB 配置 ---\n")

    host = input("请输入 FalkorDB 主机地址 [默认: localhost]: ").strip() or "localhost"
    port_str = input("请输入 FalkorDB 端口 [默认: 6379]: ").strip() or "6379"
    try:
        port = int(port_str)
    except ValueError:
        print(f"❌ 端口必须是数字: {port_str}")
        port = 6379
    graph_name = input("请输入图数据库名称 [默认: ontology]: ").strip() or "ontology"

    return {
        "backend": "falkordb",
        "falkordb": {"host": host, "port": port, "graph": graph_name},
    }


def onboard():
    """Onboard 向导主函数"""
    config_manager = ConfigManager()

    if config_manager.exists():
        print("\n⚠️  配置文件已存在")
        overwrite = input("是否覆盖现有配置？ (yes/no): ").strip().lower()
        if overwrite not in ["yes", "y"]:
            print("已取消配置")
            return

    csv_config = prompt_csv_config(config_manager)
    print("\n正在生成 CSV 文件...")
    try:
        generate_csv(csv_config)
        print(f"✓ CSV 文件已生成到: {csv_config['csv_dir']}")
    except Exception as e:
        print(f"❌ CSV 生成失败: {e}")
        sys.exit(1)

    full_config = {
        **csv_config,
        "csv_dir": str(Path(csv_config["csv_dir"]).resolve()),
    }

    backend = prompt_backend_choice()

    if backend == "networkx":
        config = prompt_networkx_config()
        full_config.update(config)
    else:
        config = prompt_falkordb_config()
        full_config.update(config)

    print("\n正在保存配置...")
    config_manager.save(full_config)
    print(f"✓ 配置已保存到: {config_manager.config_path}")

    print("\n正在生成 assets...")

    try:
        graph_obj = GraphFactory.create(config)
        generator = AssetsGenerator(graph_obj, SKILLS_ASSETS_DIR)
        generator.generate_all()

        print(f"✓ Assets 已生成到: {SKILLS_ASSETS_DIR}")
        print("\n✅ Onboard 完成！")
        print(f"\n配置文件: {config_manager.config_path}")
        print(f"Assets 目录: {SKILLS_ASSETS_DIR}")

    except Exception as e:
        print(f"\n❌ 生成 assets 失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    onboard()
