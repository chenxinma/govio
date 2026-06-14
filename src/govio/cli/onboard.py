import os
import sys
from pathlib import Path
from typing import Any

import questionary

from .config import ConfigManager
from ..core.graph_factory import GraphFactory
from ..core.assets_generator import AssetsGenerator
from ..metadata.gen_networkx import build_graph


SKILLS_ASSETS_DIR = Path("skills/govio/assets")


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
    """提示用户输入 元数据CSV 生成配置"""
    print("\n=== 步骤 1: 元数据CSV  元数据生成 ===\n")

    existing_config = {}
    if config_manager.exists():
        try:
            existing_config = config_manager.load()
        except Exception:
            pass

    # Support both nested (new) and flat (legacy) config format
    metadata_section = existing_config.get("metadata", existing_config)

    default_kundb = metadata_section.get("kundb", "")
    kundb = questionary.text(
        "请输入元数据库 URL:",
        default=default_kundb,
    ).ask() or default_kundb

    default_app_list = metadata_section.get("app_list", "")
    app_list = questionary.text(
        "请输入应用清单 Excel 文件路径:",
        default=default_app_list,
    ).ask() or default_app_list

    default_app_map = metadata_section.get("app_map", "")
    app_map = questionary.text(
        "请输入应用数据库映射 JSON 文件路径:",
        default=default_app_map,
    ).ask() or default_app_map

    default_relationship = metadata_section.get("relationship", "")
    relationship = questionary.text(
        "请输入表关系 JSON 文件路径（可选，留空跳过）:",
        default=default_relationship,
    ).ask() or default_relationship

    default_metric = metadata_section.get("metric", "")
    metric = questionary.text(
        "请输入指标定义 JSON 文件路径（可选，留空跳过）:",
        default=default_metric,
    ).ask() or default_metric

    default_csv_dir = metadata_section.get("csv_dir", "")
    csv_dir = questionary.text(
        "请输入 CSV 输出目录:",
        default=default_csv_dir,
    ).ask() or default_csv_dir

    default_workspace_uuid = metadata_section.get(
        "workspace_uuid", "82ee37374b314a938bf28170ab4db7cf"
    )
    workspace_uuid = questionary.text(
        "请输入工作区 UUID:",
        default=default_workspace_uuid,
    ).ask() or default_workspace_uuid

    return {
        "kundb": kundb,
        "app_list": app_list,
        "app_map": app_map,
        "relationship": relationship if relationship else None,
        "metric": metric if metric else None,
        "csv_dir": csv_dir,
        "workspace_uuid": workspace_uuid,
    }


def generate_csv(config: dict[str, Any]) -> None:
    """根据配置生成 CSV 文件"""
    from ..metadata.utility import make_csv
    import pandas as pd

    kundb = config["kundb"]
    app_list = config["app_list"]
    app_map = config["app_map"]
    relationship = config.get("relationship")
    metric = config.get("metric")
    csv_dir = Path(config["csv_dir"])
    workspace_uuid = config.get("workspace_uuid", "82ee37374b314a938bf28170ab4db7cf")

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
        metric_file=metric,
    )


def prompt_backend_choice() -> str:
    """提示用户选择 backend"""
    print("\n=== Govio Onboard 向导 ===\n")

    return questionary.select(
        "请选择图数据库后端：",
        choices=[
            questionary.Choice("networkx - 本地 GML 文件", value="networkx"),
            questionary.Choice("falkordb - FalkorDB 图数据库", value="falkordb"),
        ],
        default="networkx",
    ).ask()


def prompt_networkx_config() -> dict[str, Any]:
    """提示用户输入 NetworkX 配置"""
    print("\n--- NetworkX 配置 ---\n")

    generate_gml = questionary.confirm(
        "是否需要从 CSV 文件生成新的 GML 文件？",
        default=True,
    ).ask()

    if generate_gml:
        csv_dir = questionary.text(
            "请输入 CSV 目录路径:",
            validate=lambda v: True if validate_csv_directory(Path(v)) else "CSV 目录无效或缺少必需文件",
        ).ask()
        csv_path = Path(csv_dir)

        gml_path = SKILLS_ASSETS_DIR / "ontology.gml"

        print("\n正在从 CSV 文件生成 GML 文件...")
        build_graph(str(csv_path), str(gml_path))
        print(f"✓ GML 文件已生成: {gml_path}")
    else:
        gml_path_input = questionary.text(
            "请输入 GML 文件路径:",
            validate=lambda v: True if Path(v).exists() else "GML 文件不存在",
        ).ask()
        gml_path = Path(gml_path_input)

    return {"backend": "networkx", "networkx": {"gml_path": str(gml_path)}}


def delete_falkordb_graph(host: str, port: int, graph_name: str) -> None:
    """删除 FalkorDB 中的图（如果存在）"""
    import falkordb

    try:
        client = falkordb.FalkorDB(host=host, port=port)
        client.execute_command("DEL", graph_name)
        print(f"✓ 已删除现有图: {graph_name}")
    except Exception:
        pass


def import_csv_to_falkordb(
    csv_dir: Path, host: str, port: int, graph_name: str
) -> None:
    """使用 falkordb-bulk-insert 将 CSV 导入 FalkorDB

    通过 Python API 调用，避免 Windows 上 CLI 入口点损坏的问题。

    Args:
        csv_dir: CSV 文件目录
        host: FalkorDB 主机地址
        port: FalkorDB 端口
        graph_name: 图数据库名称
    """
    import subprocess

    csv_path = Path(csv_dir)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 目录不存在: {csv_path}")

    print("\n正在检查并清理已有数据...")
    delete_falkordb_graph(host, port, graph_name)

    node_files = [
        ("PhysicalTable", csv_path / "PhysicalTable.csv"),
        ("Col", csv_path / "Col.csv"),
        ("Application", csv_path / "Application.csv"),
        ("Standard", csv_path / "Standard.csv"),
        ("Metric", csv_path / "Metric.csv"),
        ("Dimension", csv_path / "Dimension.csv"),
    ]

    relation_files = [
        ("HAS_COLUMN", csv_path / "HAS_COLUMN.csv"),
        ("USE", csv_path / "USE.csv"),
    ]

    extra_rel_file = csv_path / "RELATES_TO.csv"
    if extra_rel_file.exists():
        relation_files.append(("RELATES_TO", extra_rel_file))

    # 指标相关的边文件
    metric_rel_files = [
        ("USES_TABLE", csv_path / "USES_TABLE.csv"),
        ("REFERS_COLUMN", csv_path / "REFERS_COLUMN.csv"),
        ("DERIVED_FROM", csv_path / "DERIVED_FROM.csv"),
        ("DIMENSION_USED", csv_path / "DIMENSION_USED.csv"),
        ("SUPERSEDES", csv_path / "SUPERSEDES.csv"),
    ]
    for rel_type, filepath in metric_rel_files:
        if filepath.exists():
            relation_files.append((rel_type, filepath))

    cmd = [sys.executable, "-m", "falkordb_bulk_loader.bulk_insert", graph_name]

    for label, filepath in node_files:
        if filepath.exists():
            cmd.extend(["--nodes-with-label", label, str(filepath)])

    for rel_type, filepath in relation_files:
        if filepath.exists():
            cmd.extend(["--relations-with-type", rel_type, str(filepath)])

    server_url = f"redis://{host}:{port}"
    cmd.extend(["--server-url", server_url])

    print(f"\n正在执行: {' '.join(cmd)}")

    env = {**os.environ, "PYTHONUTF8": "1"}
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    if result.returncode != 0:
        raise RuntimeError(f"falkordb-bulk-insert 失败: {result.stderr}")

    print(result.stdout)
    print("✓ CSV 数据已导入 FalkorDB")


def prompt_falkordb_config(csv_dir: Path) -> dict[str, Any]:
    """提示用户输入 FalkorDB 配置"""
    print("\n--- FalkorDB 配置 ---\n")

    host = questionary.text(
        "请输入 FalkorDB 主机地址:",
        default="localhost",
    ).ask() or "localhost"

    port_str = questionary.text(
        "请输入 FalkorDB 端口:",
        default="6379",
        validate=lambda v: True if v.isdigit() else "端口必须是数字",
    ).ask() or "6379"
    port = int(port_str)

    graph_name = questionary.text(
        "请输入图数据库名称:",
        default="ontology",
    ).ask() or "ontology"

    print("\n正在导入 CSV 数据到 FalkorDB...")
    try:
        import_csv_to_falkordb(csv_dir, host, port, graph_name)
    except Exception as e:
        print(f"❌ 导入 CSV 到 FalkorDB 失败: {e}")
        raise

    return {
        "backend": "falkordb",
        "falkordb": {"host": host, "port": port, "graph": graph_name},
    }


def prompt_connect_args(existing: dict[str, Any] | None = None) -> dict[str, Any]:
    """交互式输入连接参数（key=value 格式）

    Args:
        existing: 已有的连接参数

    Returns:
        dict: 连接参数字典
    """
    connect_args: dict[str, Any] = {}

    if existing:
        print(f"  当前连接参数: {existing}")
        keep = questionary.confirm(
            "  是否保留现有参数？",
            default=True,
        ).ask()
        if keep:
            return existing

    print("  输入连接参数 (key=value 格式，留空结束):")
    print("  示例: ssl=true, timeout=30")

    while True:
        line = questionary.text("  >").ask()
        if not line:
            break
        if "=" not in line:
            print("  格式错误，请使用 key=value 格式")
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            print("  格式错误，key 不能为空")
            continue
        value = value.strip()
        if value.lower() in ("true", "false"):
            value = value.lower() == "true"
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
        connect_args[key] = value

    return connect_args


def prompt_datasource_config(
    existing_datasources: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """提示用户配置数据源（可选）

    Args:
        existing_datasources: 已有的数据源配置

    Returns:
        dict: 数据源配置字典，None 表示无数据源
    """
    print("\n=== 数据源配置（可选）===\n")
    print("配置数据源供 observe 命令使用")
    print("可添加 MySQL、DuckDB 等数据源\n")

    datasources: dict[str, Any] = (
        dict(existing_datasources) if existing_datasources else {}
    )

    while True:
        if datasources:
            print("已配置的数据源:")
            for name, ds in datasources.items():
                print(f"  - {name}: {ds['url']}")
            print()

        action = questionary.select(
            "操作选项：",
            choices=[
                questionary.Choice("添加数据源", value="add"),
                questionary.Choice("删除数据源", value="del"),
                questionary.Choice("完成配置", value="done"),
            ],
            default="done",
        ).ask()

        if action == "add":
            name = questionary.text("  数据源名称:").ask()
            if not name:
                print("  名称不能为空")
                continue
            url = questionary.text(
                "  URL (如 mysql+pymysql://user:pass@host/db):"
            ).ask()
            if not url:
                print("  URL 不能为空")
                continue
            if name in datasources:
                overwrite = questionary.confirm(
                    f"  数据源 '{name}' 已存在，是否覆盖？",
                    default=False,
                ).ask()
                if not overwrite:
                    print("  已取消添加")
                    continue
            existing_args = datasources.get(name, {}).get("connect_args") or None
            connect_args = prompt_connect_args(existing_args)
            datasources[name] = {"url": url, "connect_args": connect_args}
            print(f"  已添加数据源: {name}")

        elif action == "del":
            if not datasources:
                print("  没有可删除的数据源")
                continue
            names = list(datasources.keys())
            removed = questionary.select(
                "  选择要删除的数据源：",
                choices=names,
            ).ask()
            if removed:
                del datasources[removed]
                print(f"  已删除: {removed}")

        elif action == "done":
            break

    return datasources if datasources else None


def onboard(new_falkordb: Path | None = None, new_networkx: Path | None = None):
    """Onboard 向导主函数

    Args:
        new_falkordb: 若提供，跳过 CSV 生成，直接将该目录的 CSV 导入 FalkorDB
        new_networkx: 若提供，跳过 CSV 生成，直接从该目录的 CSV 生成 GML 文件
    """
    config_manager = ConfigManager()

    if new_falkordb and new_networkx:
        print("❌ --new-falkordb 和 --new-networkx 不能同时使用")
        sys.exit(1)

    if new_networkx:
        csv_dir = Path(new_networkx).resolve()
        if not validate_csv_directory(csv_dir):
            print(f"❌ CSV 目录无效或缺少必需文件: {csv_dir}")
            sys.exit(1)

        existing_config = {}
        if config_manager.exists():
            try:
                existing_config = config_manager.load()
            except Exception:
                pass

        print("\n=== 跳过 CSV 生成，直接生成 GML 文件 ===")
        print(f"CSV 目录: {csv_dir}\n")

        gml_path = SKILLS_ASSETS_DIR / "ontology.gml"
        print("正在从 CSV 文件生成 GML 文件...")
        build_graph(str(csv_dir), str(gml_path))
        print(f"✓ GML 文件已生成: {gml_path}")

        graph_config = {"backend": "networkx", "networkx": {"gml_path": str(gml_path)}}
        full_config = {
            **existing_config,
            "metadata": {**existing_config.get("metadata", {}), "csv_dir": str(csv_dir)},
            "graph": graph_config,
        }

        config_manager.save(full_config)
        print(f"✓ 配置已保存到: {config_manager.config_path}")

        print("\n正在生成 assets...")
        try:
            graph_obj = GraphFactory.create(graph_config)
            generator = AssetsGenerator(graph_obj, SKILLS_ASSETS_DIR)
            generator.generate_all()
            print(f"✓ Assets 已生成到: {SKILLS_ASSETS_DIR}")
            print("\n✅ Onboard 完成！")
            print(f"\n配置文件: {config_manager.config_path}")
            print(f"Assets 目录: {SKILLS_ASSETS_DIR}")
        except Exception as e:
            print(f"\n❌ 生成 assets 失败: {e}")
            sys.exit(1)
        return

    if new_falkordb:
        csv_dir = Path(new_falkordb).resolve()
        if not validate_csv_directory(csv_dir):
            print(f"❌ CSV 目录无效或缺少必需文件: {csv_dir}")
            sys.exit(1)

        existing_config = {}
        if config_manager.exists():
            try:
                existing_config = config_manager.load()
            except Exception:
                pass

        print("\n=== 跳过 CSV 生成，直接导入 FalkorDB ===")
        print(f"CSV 目录: {csv_dir}\n")

        graph_config = prompt_falkordb_config(csv_dir)
        full_config = {
            **existing_config,
            "metadata": {**existing_config.get("metadata", {}), "csv_dir": str(csv_dir)},
            "graph": graph_config,
        }

        config_manager.save(full_config)
        print(f"✓ 配置已保存到: {config_manager.config_path}")

        print("\n正在生成 assets...")
        try:
            graph_obj = GraphFactory.create(graph_config)
            generator = AssetsGenerator(graph_obj, SKILLS_ASSETS_DIR)
            generator.generate_all()
            print(f"✓ Assets 已生成到: {SKILLS_ASSETS_DIR}")
            print("\n✅ Onboard 完成！")
            print(f"\n配置文件: {config_manager.config_path}")
            print(f"Assets 目录: {SKILLS_ASSETS_DIR}")
        except Exception as e:
            print(f"\n❌ 生成 assets 失败: {e}")
            sys.exit(1)
        return

    if config_manager.exists():
        existing_config = config_manager.load()
        has_backend = "graph" in existing_config and "backend" in existing_config.get("graph", {})

        if has_backend:
            print(f"\n⚠️  检测到已有配置 (backend: {existing_config['graph']['backend']})")
            skip = questionary.confirm(
                "是否跳过 元数据CSV/Graph 配置，仅配置数据源？",
                default=False,
            ).ask()
            if skip:
                full_config = dict(existing_config)
                datasources = prompt_datasource_config(full_config.get("datasources"))
                if datasources is not None:
                    full_config["datasources"] = datasources
                else:
                    full_config.pop("datasources", None)
                config_manager.save(full_config)
                print(f"\n✅ 配置已更新: {config_manager.config_path}")
                return

        print("\n⚠️  配置文件已存在")
        overwrite = questionary.confirm(
            "是否覆盖现有配置？",
            default=False,
        ).ask()
        if not overwrite:
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

    metadata = {
        **csv_config,
        "csv_dir": str(Path(csv_config["csv_dir"]).resolve()),
    }

    backend = prompt_backend_choice()
    csv_dir = Path(csv_config["csv_dir"])

    if backend == "networkx":
        graph_config = prompt_networkx_config()
    else:
        graph_config = prompt_falkordb_config(csv_dir)

    full_config: dict[str, Any] = {"metadata": metadata, "graph": graph_config}

    print("\n正在保存配置...")
    config_manager.save(full_config)
    print(f"✓ 配置已保存到: {config_manager.config_path}")

    datasources = prompt_datasource_config()
    if datasources:
        full_config["datasources"] = datasources
        config_manager.save(full_config)

    print("\n正在生成 assets...")
    try:
        graph_obj = GraphFactory.create(graph_config)
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
