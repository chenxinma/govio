from pathlib import Path
from typing import Any

import questionary

from .config import ConfigManager
from govio.crypto import encrypt_value, parse_password_from_url


# ---------------------------------------------------------------------------
# CSV validation helper (still used internally)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Datasource configuration (used by onboard flow)
# ---------------------------------------------------------------------------

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


def _encrypt_url_password(url: str) -> dict[str, Any]:
    """解析 URL 中的密码并加密，返回存储字段

    Args:
        url: 原始连接 URL

    Returns:
        dict: 包含 url（脱敏）和 encrypted_password（若有密码）
    """
    masked_url, password = parse_password_from_url(url)
    result: dict[str, Any] = {"url": masked_url}
    if password:
        result["encrypted_password"] = encrypt_value(password)
    return result


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
    print("密码将自动加密存储，配置文件可安全分享\n")

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

            # 加密密码并存储脱敏 URL
            ds_entry = _encrypt_url_password(url)
            ds_entry["connect_args"] = connect_args
            datasources[name] = ds_entry
            print(f"  已添加数据源: {name}（密码已加密）")

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


# ---------------------------------------------------------------------------
# Onboard main flow (simplified)
# ---------------------------------------------------------------------------

def onboard():
    """Onboard 向导主函数 — 图数据库后端选择 + 数据源配置"""
    config_manager = ConfigManager()

    if config_manager.exists():
        existing_config = config_manager.load()
        has_backend = "graph" in existing_config and "backend" in existing_config.get("graph", {})

        if has_backend:
            print(f"\n检测到已有配置 (backend: {existing_config['graph']['backend']})")
            skip = questionary.confirm(
                "是否跳过图后端配置，仅配置数据源？",
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
                print(f"\n配置已更新: {config_manager.config_path}")
                return

        print("\n配置文件已存在")
        overwrite = questionary.confirm(
            "是否覆盖现有配置？",
            default=False,
        ).ask()
        if not overwrite:
            print("已取消配置")
            return

    # --- Graph backend selection ---
    print("\n=== Govio Onboard 向导 ===\n")

    backend = questionary.select(
        "请选择图数据库后端：",
        choices=[
            questionary.Choice("networkx - 本地 GML 文件", value="networkx"),
            questionary.Choice("falkordb - FalkorDB 图数据库", value="falkordb"),
            questionary.Choice("ladybug - Ladybug 嵌入式图数据库", value="ladybug"),
        ],
        default="networkx",
    ).ask()

    if backend == "networkx":
        print("\n--- NetworkX 配置 ---\n")
        gml_path_input = questionary.text(
            "请输入 GML 文件路径:",
            validate=lambda v: True if Path(v).exists() else "GML 文件不存在",
        ).ask()
        graph_config = {"backend": "networkx", "networkx": {"gml_path": str(Path(gml_path_input))}}
    elif backend == "falkordb":
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

        graph_config = {"backend": "falkordb", "falkordb": {"host": host, "port": port, "graph": graph_name}}
    else:
        # ladybug
        print("\n--- Ladybug 配置 ---\n")
        default_db = str(Path.home() / ".govio" / "ontology.lbdb")
        db_path_input = questionary.text(
            "请输入 Ladybug 数据库文件路径:",
            default=default_db,
        ).ask() or default_db
        graph_config = {
            "backend": "ladybug",
            "ladybug": {"db_path": str(Path(db_path_input))},
        }

    full_config: dict[str, Any] = {"graph": graph_config}

    # --- Datasource config ---
    datasources = prompt_datasource_config()
    if datasources:
        full_config["datasources"] = datasources

    config_manager.save(full_config)
    print(f"\n配置已保存到: {config_manager.config_path}")
    print("\nOnboard 完成！")


if __name__ == "__main__":
    onboard()
