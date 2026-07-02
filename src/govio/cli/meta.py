"""meta 命令组 — 知识图库维护

提供元数据同步、导出、数据标准推荐、配置管理等功能。
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import questionary

from .config import MetaConfigManager


def cmd_sync(args: argparse.Namespace) -> None:
    """meta sync — 完整/增量同步管线"""
    from .meta_export import meta_export

    if args.db:
        db_path = args.db
        schemas = args.schemas.split(",") if args.schemas else None
        db_name = args.db_name
        output = Path(args.output)
        dry_run = args.dry_run
    else:
        # 交互模式：从 meta_config.yaml 读取默认值
        meta_cfg = MetaConfigManager()
        try:
            config = meta_cfg.load_or_migrate()
        except FileNotFoundError:
            print("错误: meta 配置文件不存在且无法迁移，请先运行 govio-cli meta config", file=sys.stderr)
            sys.exit(1)

        db_path = config.get("db", "")
        schemas_str = ",".join(config.get("schemas", [])) if config.get("schemas") else ""
        db_name = config.get("db_name", "")
        output_default = config.get("output", "")

        db_path = questionary.text("DuckDB 数据库文件路径:", default=db_path).ask() or db_path
        schemas_input = questionary.text("要导出的 schema 列表（逗号分隔，留空跳过）:", default=schemas_str).ask() or schemas_str
        db_name = questionary.text("单库模式 app 名称（留空使用全量模式）:", default=db_name).ask() or db_name
        output_str = questionary.text("CSV 输出目录:", default=output_default).ask() or output_default

        schemas = [s.strip() for s in schemas_input.split(",") if s.strip()] if schemas_input else None
        output = Path(output_str)
        dry_run = args.dry_run

    if not schemas and not db_name:
        print("错误: 必须指定 --schemas 或 --db-name 之一", file=sys.stderr)
        sys.exit(1)

    meta_export(
        db_path=db_path,
        schemas=schemas,
        db_name=db_name,
        output=output,
        dry_run=dry_run,
    )


def cmd_export(args: argparse.Namespace) -> None:
    """meta export — 等同于 meta sync --dry-run"""
    args.dry_run = True
    cmd_sync(args)


def cmd_recommend(args: argparse.Namespace) -> None:
    """meta recommend — 数据标准推荐"""
    from .std_recommend import std_recommend

    meta_cfg = MetaConfigManager()
    try:
        config = meta_cfg.load_or_migrate()
    except FileNotFoundError:
        print("错误: meta 配置文件不存在且无法迁移，请先运行 govio-cli meta config", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir
    if not output_dir:
        output_dir = config.get("output", "./output")

    std_recommend(Path(output_dir))


def cmd_config(args: argparse.Namespace) -> None:
    """meta config — 交互式配置 meta_config.yaml"""
    meta_cfg = MetaConfigManager()

    if meta_cfg.exists():
        config = meta_cfg.load()
        print("当前 meta 配置:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        print()

        modify = questionary.confirm("是否修改当前配置？", default=False).ask()
        if not modify:
            return
    else:
        print("meta 配置文件不存在，开始创建...")
        config = {}

    fields = [
        ("kundb", "元数据库 URL", ""),
        ("workspace_uuid", "工作区 UUID", "82ee37374b314a938bf28170ab4db7cf"),
        ("app_list", "应用清单 Excel 文件路径", ""),
        ("app_map", "应用数据库映射 JSON 文件路径", ""),
        ("relationship", "表关系 JSON 文件路径（可选，留空跳过）", ""),
        ("metric", "指标定义 JSON 文件路径（可选，留空跳过）", ""),
        ("csv_dir", "CSV 输出目录", ""),
    ]

    for key, prompt, default_val in fields:
        current = config.get(key, default_val)
        value = questionary.text(f"{prompt}:", default=str(current)).ask() or str(current)
        if key in ("relationship", "metric"):
            config[key] = value if value else None
        else:
            config[key] = value

    meta_cfg.save(config)
    print(f"\n配置已保存到: {meta_cfg.config_path}")


def meta():
    """meta 命令入口"""
    parser = argparse.ArgumentParser(
        prog="govio-cli meta",
        description="知识图库维护 — 元数据同步、导出、推荐、配置",
    )
    sub = parser.add_subparsers(dest="action", required=True)

    # meta sync
    p_sync = sub.add_parser("sync", help="完整/增量同步管线：读取元数据源 → 生成 CSV → 更新图数据 → 生成 assets")
    p_sync.add_argument("--db", type=str, help="DuckDB 数据库文件路径")
    p_sync.add_argument("--schemas", type=str, help="要导出的 schema 列表，逗号分隔（与 --db-name 互斥）")
    p_sync.add_argument("--db-name", type=str, help="单库模式：按 app 名导出单个数据库的相关子图")
    p_sync.add_argument("--output", type=str, help="CSV 输出目录")
    p_sync.add_argument("--dry-run", action="store_true", help="仅生成 CSV 并输出状态，不更新图数据和生成 assets")

    # meta export (sync --dry-run 的别名)
    p_export = sub.add_parser("export", help="等同于 meta sync --dry-run")
    p_export.add_argument("--db", type=str, help="DuckDB 数据库文件路径")
    p_export.add_argument("--schemas", type=str, help="要导出的 schema 列表，逗号分隔（与 --db-name 互斥）")
    p_export.add_argument("--db-name", type=str, help="单库模式：按 app 名导出单个数据库的相关子图")
    p_export.add_argument("--output", type=str, help="CSV 输出目录")

    # meta recommend
    p_recommend = sub.add_parser("recommend", help="数据标准推荐")
    p_recommend.add_argument("--output-dir", type=str, help="推荐数据标准的输出目录")

    # meta config
    sub.add_parser("config", help="交互式配置 meta_config.yaml")

    args = parser.parse_args(sys.argv[1:])

    match args.action:
        case "sync":
            cmd_sync(args)
        case "export":
            cmd_export(args)
        case "recommend":
            cmd_recommend(args)
        case "config":
            cmd_config(args)


if __name__ == "__main__":
    meta()
