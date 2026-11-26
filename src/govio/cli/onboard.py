import sys
from pathlib import Path
from typing import Any

import questionary

from govio.crypto import encrypt_value, parse_password_from_url

from .config import ConfigManager

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

    return all((csv_dir / filename).exists() for filename in required_files)


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
        connect_args[key] = _coerce_scalar(value.strip())

    return connect_args


def _coerce_scalar(value: str) -> Any:
    """将字符串值转换为 bool/int/float，失败则保留原字符串

    Args:
        value: 原始字符串

    Returns:
        Any: 转换后的值
    """
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def parse_cli_connect_args(pairs: list[str] | None) -> dict[str, Any]:
    """解析命令行 --connect-args key=value 参数

    Args:
        pairs: key=value 字符串列表

    Returns:
        dict: 连接参数字典（值自动转换 bool/int/float）

    Raises:
        ValueError: 格式错误
    """
    connect_args: dict[str, Any] = {}
    for pair in pairs or []:
        key, sep, value = pair.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"连接参数格式错误: {pair}（应为 key=value）")
        connect_args[key.strip()] = _coerce_scalar(value.strip())
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


def _attach_password(url: str, password: str) -> str:
    """将独立传入的密码嵌入无密码的 URL（scheme://user@host 形式）

    Args:
        url: 不含密码的连接 URL
        password: 数据源密码

    Returns:
        str: 嵌入密码后的完整 URL

    Raises:
        ValueError: URL 中已含密码或格式不支持嵌入
    """
    _, existing = parse_password_from_url(url)
    if existing:
        raise ValueError("URL 中已包含密码，请勿同时使用 --password")

    scheme_sep = url.find("://")
    if scheme_sep == -1:
        raise ValueError(f"URL 格式无效: {url}（需包含协议前缀，如 mysql+pymysql://）")
    at_pos = url.find("@", scheme_sep + 3)
    if at_pos == -1:
        raise ValueError(f"URL 不含用户信息(@)，无法通过 --password 嵌入密码: {url}")

    user_part = url[scheme_sep + 3 : at_pos]
    if ":" in user_part:
        # user: 空密码占位形式，直接在冒号后补密码
        return f"{url[: scheme_sep + 3]}{user_part}{password}{url[at_pos:]}"
    return f"{url[: scheme_sep + 3]}{user_part}:{password}{url[at_pos:]}"


def _apply_add_datasource(
    config: dict[str, Any],
    name: str,
    url: str,
    connect_args: dict[str, Any] | None = None,
    password: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """在配置字典上添加/覆盖数据源（不落盘，供单次 save 组合场景复用）

    Args:
        config: 配置字典（原地修改 datasources 键）
        name: 数据源名称
        url: 连接 URL（可含密码，存储时自动脱敏并加密）
        connect_args: 连接参数
        password: 独立传入的密码（URL 不含密码时嵌入）
        overwrite: 是否允许覆盖同名数据源

    Returns:
        dict: 保存的数据源条目

    Raises:
        ValueError: 参数冲突或同名数据源已存在且未指定 overwrite
    """
    name = name.strip()
    if not name:
        raise ValueError("数据源名称不能为空")
    if "://" not in url:
        raise ValueError(f"URL 格式无效: {url}（需包含协议前缀，如 mysql+pymysql://）")

    if password:
        url = _attach_password(url, password)

    datasources: dict[str, Any] = dict(config.get("datasources") or {})
    if name in datasources and not overwrite:
        raise ValueError(f"数据源 '{name}' 已存在，使用 --overwrite 覆盖")

    ds_entry = _encrypt_url_password(url)
    ds_entry["connect_args"] = dict(connect_args or {})
    datasources[name] = ds_entry
    config["datasources"] = datasources
    return ds_entry


def _apply_remove_datasource(config: dict[str, Any], name: str) -> None:
    """在配置字典上删除数据源（不落盘，供单次 save 组合场景复用）

    Args:
        config: 配置字典（原地修改 datasources 键）
        name: 数据源名称

    Raises:
        ValueError: 数据源不存在
    """
    datasources: dict[str, Any] = dict(config.get("datasources") or {})
    if name not in datasources:
        raise ValueError(f"数据源 '{name}' 不存在")

    del datasources[name]
    if datasources:
        config["datasources"] = datasources
    else:
        config.pop("datasources", None)


def add_datasource(
    name: str,
    url: str,
    connect_args: dict[str, Any] | None = None,
    password: str | None = None,
    overwrite: bool = False,
    config_manager: ConfigManager | None = None,
) -> dict[str, Any]:
    """非交互式添加/覆盖数据源并保存配置

    Args:
        name: 数据源名称
        url: 连接 URL（可含密码，存储时自动脱敏并加密）
        connect_args: 连接参数
        password: 独立传入的密码（URL 不含密码时嵌入）
        overwrite: 是否允许覆盖同名数据源
        config_manager: 配置管理器（默认使用全局配置）

    Returns:
        dict: 保存的数据源条目

    Raises:
        ValueError: 参数冲突或同名数据源已存在且未指定 overwrite
    """
    config_manager = config_manager or ConfigManager()
    config = config_manager.load() if config_manager.exists() else {}
    ds_entry = _apply_add_datasource(
        config,
        name,
        url,
        connect_args=connect_args,
        password=password,
        overwrite=overwrite,
    )
    config_manager.save(config)
    return ds_entry


def remove_datasource(
    name: str,
    config_manager: ConfigManager | None = None,
) -> None:
    """非交互式删除数据源并保存配置

    Args:
        name: 数据源名称
        config_manager: 配置管理器（默认使用全局配置）

    Raises:
        ValueError: 配置文件不存在或数据源不存在
    """
    config_manager = config_manager or ConfigManager()
    if not config_manager.exists():
        raise ValueError("配置文件不存在，无数据源可删除")

    config = config_manager.load()
    _apply_remove_datasource(config, name)
    config_manager.save(config)


# ---------------------------------------------------------------------------
# Non-interactive graph backend configuration (used by onboard --backend)
# ---------------------------------------------------------------------------

# 各后端专属旗标 -> （CLI 旗标名, 所属后端），用于「--xxx 需配合 --backend <后端>」校验
_BACKEND_ONLY_FLAGS: list[tuple[str, str, str]] = [
    ("db_path", "--db-path", "ladybug"),
    ("host", "--host", "falkordb"),
    ("port", "--port", "falkordb"),
    ("graph_name", "--graph", "falkordb"),
    ("gml_path", "--gml-path", "networkx"),
]


def build_graph_config(
    backend: str | None,
    db_path: str | None = None,
    host: str | None = None,
    port: str | None = None,
    graph_name: str | None = None,
    gml_path: str | None = None,
) -> dict[str, Any]:
    """按非交互参数生成 graph 配置段（输出与 prompt_graph_config 严格一致）

    Args:
        backend: 图后端类型（ladybug/falkordb/networkx）
        db_path: ladybug 数据库文件路径（缺省 ~/.govio/ontology.lbdb）
        host: falkordb 主机（缺省 localhost）
        port: falkordb 端口（缺省 6379，须为数字）
        graph_name: falkordb 图名（缺省 ontology）
        gml_path: networkx GML 文件路径（networkx 必填，须已存在）

    Returns:
        dict: graph 配置段

    Raises:
        ValueError: 后端专属参数与 backend 不匹配、必填缺失或取值非法
    """
    given = {
        "db_path": db_path,
        "host": host,
        "port": port,
        "graph_name": graph_name,
        "gml_path": gml_path,
    }
    for param, cli_name, expected in _BACKEND_ONLY_FLAGS:
        if given[param] is not None and backend != expected:
            raise ValueError(f"{cli_name} 需配合 --backend {expected}")

    if backend not in ("ladybug", "falkordb", "networkx"):
        raise ValueError("--backend 需指定图后端（ladybug/falkordb/networkx）")

    if backend == "ladybug":
        default_db = str(Path.home() / ".govio" / "ontology.lbdb")
        db = str(Path(db_path)) if db_path else default_db
        return {"backend": "ladybug", "ladybug": {"db_path": db}}

    if backend == "falkordb":
        if port is not None and not str(port).isdigit():
            raise ValueError("端口必须是数字")
        return {
            "backend": "falkordb",
            "falkordb": {
                "host": host or "localhost",
                "port": int(port) if port is not None else 6379,
                "graph": graph_name or "ontology",
            },
        }

    # networkx
    if not gml_path:
        raise ValueError("--backend networkx 需同时提供 --gml-path")
    if not Path(gml_path).exists():
        raise ValueError(f"GML 文件不存在: {gml_path}")
    return {"backend": "networkx", "networkx": {"gml_path": str(Path(gml_path))}}


def onboard_graph_cli(
    backend: str | None,
    db_path: str | None = None,
    host: str | None = None,
    port: str | None = None,
    graph_name: str | None = None,
    gml_path: str | None = None,
    add_name: str | None = None,
    remove_name: str | None = None,
    url: str | None = None,
    password: str | None = None,
    connect_args: list[str] | None = None,
    overwrite: bool = False,
) -> None:
    """onboard 非交互式图后端配置入口（可与数据源增删组合，单次 save 落盘）

    graph 段 upsert 覆盖，既有 datasources 原样保留；全程不弹 questionary。

    Args:
        backend: 图后端类型（ladybug/falkordb/networkx）
        db_path: ladybug 数据库文件路径（--db-path）
        host: falkordb 主机（--host）
        port: falkordb 端口（--port）
        graph_name: falkordb 图名（--graph）
        gml_path: networkx GML 文件路径（--gml-path）
        add_name: 待添加的数据源名称（--add-datasource）
        remove_name: 待删除的数据源名称（--remove-datasource）
        url: 数据源连接 URL（--url）
        password: 数据源密码（--password）
        connect_args: key=value 参数列表（--connect-args，可重复）
        overwrite: 是否覆盖同名数据源（--overwrite）
    """

    def _fail(message: str) -> None:
        print(f"错误: {message}", file=sys.stderr)
        sys.exit(1)

    # 数据源旗标组合校验（与 onboard_datasource_cli 同款契约）
    if add_name and remove_name:
        _fail("--add-datasource 与 --remove-datasource 不能同时使用")
    if remove_name and (url or password or connect_args):
        _fail("--remove-datasource 不接受 --url/--password/--connect-args 参数")
    if (
        not add_name
        and not remove_name
        and (url or password or connect_args or overwrite)
    ):
        _fail(
            "--url/--password/--connect-args/--overwrite 需配合 --add-datasource 使用"
        )
    if add_name and not url:
        _fail("--add-datasource 需同时提供 --url")

    try:
        graph_config = build_graph_config(
            backend,
            db_path=db_path,
            host=host,
            port=port,
            graph_name=graph_name,
            gml_path=gml_path,
        )
    except ValueError as e:
        _fail(str(e))

    config_manager = ConfigManager()
    config: dict[str, Any] = (
        dict(config_manager.load()) if config_manager.exists() else {}
    )
    # graph 段 upsert：只覆盖 graph 键，datasources 原样保留
    config["graph"] = graph_config

    if remove_name:
        try:
            _apply_remove_datasource(config, remove_name)
        except ValueError as e:
            _fail(str(e))
    if add_name:
        try:
            parsed_args = parse_cli_connect_args(connect_args)
            _apply_add_datasource(
                config,
                add_name,
                url or "",
                connect_args=parsed_args,
                password=password,
                overwrite=overwrite,
            )
        except ValueError as e:
            _fail(str(e))

    # 图后端 + 数据源增删单次 save 落盘，避免两次写文件互相覆盖
    config_manager.save(config)

    print(f"已配置图后端: {graph_config['backend']}")
    if remove_name:
        print(f"已删除数据源: {remove_name}")
    if add_name:
        print(f"已添加数据源: {add_name}")
    print(f"配置文件: {config_manager.config_path}")


def onboard_datasource_cli(
    add_name: str | None = None,
    remove_name: str | None = None,
    url: str | None = None,
    password: str | None = None,
    connect_args: list[str] | None = None,
    overwrite: bool = False,
) -> None:
    """onboard 非交互式数据源管理入口（供 CLI 参数调用）

    Args:
        add_name: 待添加的数据源名称（--add-datasource）
        remove_name: 待删除的数据源名称（--remove-datasource）
        url: 连接 URL（--url）
        password: 独立密码（--password）
        connect_args: key=value 参数列表（--connect-args，可重复）
        overwrite: 是否覆盖同名数据源（--overwrite）
    """

    def _fail(message: str) -> None:
        print(f"错误: {message}", file=sys.stderr)
        sys.exit(1)

    if add_name and remove_name:
        _fail("--add-datasource 与 --remove-datasource 不能同时使用")

    if add_name:
        if not url:
            _fail("--add-datasource 需同时提供 --url")
        try:
            parsed_args = parse_cli_connect_args(connect_args)
            add_datasource(
                add_name,
                url,
                connect_args=parsed_args,
                password=password,
                overwrite=overwrite,
            )
        except ValueError as e:
            _fail(str(e))
        print(f"已添加数据源: {add_name}")
        config_manager = ConfigManager()
        print(f"配置文件: {config_manager.config_path}")
        if not config_manager.exists() or "graph" not in config_manager.load():
            print("提示: 尚未配置图后端，可运行 govio-cli onboard 完成配置")
    elif remove_name:
        if url or password or connect_args:
            _fail("--remove-datasource 不接受 --url/--password/--connect-args 参数")
        try:
            remove_datasource(remove_name)
        except ValueError as e:
            _fail(str(e))
        print(f"已删除数据源: {remove_name}")


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

    return datasources or None


# ---------------------------------------------------------------------------
# Onboard main flow (simplified)
# ---------------------------------------------------------------------------


def prompt_graph_config() -> dict[str, Any]:
    """交互式选择图数据库后端并生成 graph 配置

    Returns:
        dict: graph 配置段
    """
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
        return {
            "backend": "networkx",
            "networkx": {"gml_path": str(Path(gml_path_input))},
        }
    if backend == "falkordb":
        print("\n--- FalkorDB 配置 ---\n")
        host = (
            questionary.text(
                "请输入 FalkorDB 主机地址:",
                default="localhost",
            ).ask()
            or "localhost"
        )

        port_str = (
            questionary.text(
                "请输入 FalkorDB 端口:",
                default="6379",
                validate=lambda v: True if v.isdigit() else "端口必须是数字",
            ).ask()
            or "6379"
        )
        port = int(port_str)

        graph_name = (
            questionary.text(
                "请输入图数据库名称:",
                default="ontology",
            ).ask()
            or "ontology"
        )

        return {
            "backend": "falkordb",
            "falkordb": {"host": host, "port": port, "graph": graph_name},
        }

    # ladybug
    print("\n--- Ladybug 配置 ---\n")
    default_db = str(Path.home() / ".govio" / "ontology.lbdb")
    db_path_input = (
        questionary.text(
            "请输入 Ladybug 数据库文件路径:",
            default=default_db,
        ).ask()
        or default_db
    )
    return {
        "backend": "ladybug",
        "ladybug": {"db_path": str(Path(db_path_input))},
    }


def onboard():
    """Onboard 向导主函数 — 图数据库后端选择 + 数据源配置"""
    config_manager = ConfigManager()

    if config_manager.exists():
        existing_config = config_manager.load()
        has_backend = "graph" in existing_config and "backend" in existing_config.get(
            "graph", {}
        )

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
        elif existing_config.get("datasources"):
            # 配置中只有数据源（如通过 --add-datasource 非交互式创建）：
            # 补充图后端配置，保留已有数据源
            print("\n检测到已有配置（尚未设置图后端，将保留已配置的数据源）")
            full_config = dict(existing_config)
            full_config["graph"] = prompt_graph_config()
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
    graph_config = prompt_graph_config()

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
