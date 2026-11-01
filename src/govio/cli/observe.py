"""observe 命令 — 数据表探查 CLI

提供数据表比较、关系探索、数据表加载释放等功能。
DataFrame 持久化到 .govio/observe/ 目录。
所有子命令参数均为关键字参数。
"""

import argparse
import json
import sys
from typing import TYPE_CHECKING, NoReturn

from .config import ConfigManager

if TYPE_CHECKING:
    from ..observe_data.core.database import DatabaseManager
from ..observe_data.core.observe_store import ObserveStore
from ..observe_data.tools.list_dataframes import list_dataframes
from ..observe_data.tools.list_datasources import list_datasources
from ..observe_data.tools.release_dataframe import (
    release_all_dataframes,
    release_dataframe,
)


def get_db_manager(config: dict) -> "DatabaseManager":
    """从配置创建 DatabaseManager"""
    from ..observe_data.config import DataSourceConfig
    from ..observe_data.core.database import DatabaseManager

    datasources = config.get("datasources", {})
    ds_configs = {
        name: DataSourceConfig(
            url=ds["url"],
            connect_args=ds.get("connect_args", {}),
            encrypted_password=ds.get("encrypted_password"),
        )
        for name, ds in datasources.items()
    }
    return DatabaseManager(ds_configs)


def _fail(message: str, output_json: bool = True) -> NoReturn:
    """失败统一出口：stdout 保留结构化错误 JSON，stderr 输出 ❌ 错误并以退出码 1 结束

    Args:
        message: 错误信息（不带 ❌ 前缀）
        output_json: 是否向 stdout 输出 {"success": false, "error": ...} JSON；
            结果 JSON 已经打印过时传 False（仅补 stderr 并退出）

    Returns:
        NoReturn: 总是以退出码 1 结束
    """
    if output_json:
        print(
            json.dumps(
                {"success": False, "error": message}, ensure_ascii=False, indent=2
            )
        )
    print(f"❌ {message}", file=sys.stderr)
    sys.exit(1)


def cmd_show_datasource(config: dict, detail: bool = False) -> None:
    """显示数据源"""
    db_manager = get_db_manager(config)
    result = list_datasources(db_manager)
    if not detail:
        result = [ds["name"] for ds in result]
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_load(
    config: dict,
    name: str,
    sql: str,
    datasource: str | None = None,
    memory: bool = False,
    output: str | None = None,
) -> None:
    """加载 DataFrame"""
    from ..observe_data.tools.load_dataframe import load_dataframe, load_from_memory

    store = ObserveStore()

    if memory:
        result = load_from_memory(store=store, name=name, sql=sql)
    else:
        if datasource is None:
            print("❌ datasource not setted.", file=sys.stderr)
            sys.exit(1)
        
        db_manager = get_db_manager(config)
        result = load_dataframe(
            store=store,
            db_manager=db_manager,
            datasource=datasource,
            name=name,
            sql=sql,
        )

    if output and result.get("success"):
        df = store.get(name)
        if df is not None:
            df.to_json(output, orient="records", force_ascii=False, indent=2)
            result["output_file"] = output

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if not result.get("success"):
        # 加载失败：保留 stdout 的 result JSON（success/error 契约），
        # 错误信息同步到 stderr 并以退出码 1 结束
        print(f"❌ {result.get('error', '加载失败')}", file=sys.stderr)
        sys.exit(1)


def cmd_list(config: dict) -> None:
    """列出 DataFrame"""
    store = ObserveStore()
    result = list_dataframes(store=store)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_release(config: dict, name: str | None, release_all: bool) -> None:
    """释放 DataFrame"""
    store = ObserveStore()
    if release_all:
        result = release_all_dataframes(store=store)
    else:
        result = release_dataframe(store=store, name=name) # pyright:ignore
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result.get("success"):
        _fail(result.get("error", "释放失败"), output_json=False)


def cmd_compare(
    config: dict, source_df: str, target_df: str, join_columns: list[str]
) -> None:
    """比对 DataFrame"""
    store = ObserveStore()

    source = store.get(source_df)
    if source is None:
        _fail(f"DataFrame '{source_df}' 不存在")

    target = store.get(target_df)
    if target is None:
        _fail(f"DataFrame '{target_df}' 不存在")

    from ..observe_data.core.comparator import TableComparator

    try:
        comparator = TableComparator()
        result = comparator.compare(source, target, join_columns)
    except Exception as e:
        _fail(f"比对执行错误: {e!s}")
    print(json.dumps({"success": True, **result}, ensure_ascii=False, indent=2))


def cmd_explore(config: dict, dataframes: list[str] | None = None) -> None:
    """探查关系"""
    store = ObserveStore()

    if dataframes is None:
        infos = store.list()
        dataframes = [info.name for info in infos]

    df_dict = {}
    for name in dataframes:
        df = store.get(name)
        if df is None:
            _fail(f"DataFrame '{name}' 不存在")
        df_dict[name] = df

    from ..observe_data.core.explorer import RelationExplorer

    try:
        explorer = RelationExplorer()
        result = explorer.explore(df_dict)
    except Exception as e:
        _fail(f"关系探查执行错误: {e!s}")
    print(json.dumps({"success": True, **result}, ensure_ascii=False, indent=2))


def cmd_visualize(config: dict, relations_file: str) -> None:
    """可视化关系"""
    try:
        with open(relations_file, encoding="utf-8") as f:
            relations = json.load(f)
    except FileNotFoundError:
        _fail(f"关系文件不存在: {relations_file}")
    except json.JSONDecodeError as e:
        _fail(f"JSON 解析失败: {e!s}")

    from ..observe_data.tools.visualize_relations import visualize_relations

    try:
        result = visualize_relations(relations=relations)
    except Exception as e:
        _fail(f"关系可视化执行错误: {e!s}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result.get("success"):
        _fail(result.get("error", "关系可视化失败"), output_json=False)


def cmd_chart(
    config: dict,
    name: str,
    chart_type: str,
    x: str,
    y: str,
    output: str,
) -> None:
    """生成图表 PNG"""
    store = ObserveStore()
    df = store.get(name)
    if df is None:
        _fail(f"DataFrame '{name}' 不存在")

    from ..observe_data.core.chart import render_chart

    try:
        result = render_chart(
            df=df,
            chart_type=chart_type,
            x_col=x,
            y_col=y,
            output_path=output,
        )
    except Exception as e:
        _fail(f"图表生成执行错误: {e!s}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result.get("success"):
        _fail(result.get("error", "图表生成失败"), output_json=False)


def cmd_info(config: dict, args: argparse.Namespace) -> None:
    """info 子命令分发：合并 show-datasource / list / show"""
    # 无任何标志时，显示概览（数据源名称 + DataFrame 列表）
    if not args.datasource and not args.df and not args.name:
        db_manager = get_db_manager(config)
        ds_result = list_datasources(db_manager)
        ds_names = [ds["name"] for ds in ds_result]

        store = ObserveStore()
        df_result = list_dataframes(store=store)

        print(json.dumps({
            "datasources": ds_names,
            "dataframes": df_result,
        }, ensure_ascii=False, indent=2))
        return

    if args.datasource:
        cmd_show_datasource(config, detail=False)
    elif args.df:
        cmd_list(config)
    elif args.name:
        cmd_show(config, args.name, args.rows)


def cmd_show(config: dict, name: str, rows: int) -> None:
    """显示已加载 DataFrame 的结构和样本数据"""
    store = ObserveStore()
    df = store.get(name)
    if df is None:
        _fail(f"DataFrame '{name}' 不存在")
    if rows < 0:
        _fail(f"样本行数不能为负数: {rows}")

    schema = [{"column": col, "dtype": str(df[col].dtype)} for col in df.columns]
    sample = json.loads(df.head(rows).to_json(orient="records", force_ascii=False))
    result = {
        "success": True,
        "name": name,
        "rows": len(df),
        "columns": len(df.columns),
        "schema": schema,
        "sample": sample,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def observe():
    """observe 命令入口"""
    parser = argparse.ArgumentParser(
        prog="govio-cli observe",
        description="数据表探查命令 — 加载、比较、探索数据表",
    )
    sub = parser.add_subparsers(dest="action", required=True)

    # load --name (--datasource | --memory) --sql [-o output]
    p = sub.add_parser(
        "load",
        help="加载 DataFrame",
        description=(
            "加载 DataFrame 到 ObserveStore。\n\n"
            "两种数据来源（互斥）：\n"
            "  --datasource  从数据库抽取（需先用 onboard 配置数据源）\n"
            "  --memory      不连数据库，把 store 中所有已加载的 DataFrame 注入\n"
            "                内存 DuckDB（按 DataFrame 名注册为同名表），再执行 SQL。\n\n"
            "-o/--output 可在加载/查询成功后把结果 DataFrame 以 JSON（records 方式、\n"
            "UTF-8、缩进 2）写出到磁盘，返回结果会额外带 output_file 字段。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--name", required=True, help="DataFrame 名称（小写字母+下划线）")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--datasource",
        help="数据源名称（从 show-datasource 获取），从数据库抽取数据",
    )
    src.add_argument(
        "--memory",
        action="store_true",
        help=(
            "不连数据库，改为在已加载的 DataFrame 上跑 SQL：把 store 中所有 "
            "DataFrame 注入内存 DuckDB，按 DataFrame 名注册为同名表后执行查询；"
            "SQL 中的表名须与已加载的 DataFrame 名一致"
        ),
    )
    p.add_argument("--sql", required=True, help="查询 SQL 语句")
    p.add_argument(
        "-o",
        "--output",
        help="将结果 DataFrame 内容输出为 JSON 文件（records 方式、UTF-8、缩进 2）",
    )

    # release --name / --all
    p = sub.add_parser("release", help="释放 DataFrame")
    p.add_argument("--name", help="DataFrame 名称")
    p.add_argument("--all", action="store_true", help="释放所有已加载的 DataFrame")

    # compare --source --target --join-columns col1,col2
    p = sub.add_parser("compare", help="比对两个 DataFrame")
    p.add_argument("--source", required=True, help="源 DataFrame 名称")
    p.add_argument("--target", required=True, help="目标 DataFrame 名称")
    p.add_argument("--join-columns", required=True, help="比对列，逗号分隔")

    # explore --dataframes df1 df2 ...
    p = sub.add_parser("explore", help="探查 DataFrame 之间的关系")
    p.add_argument("--dataframes", nargs="*", help="DataFrame 名称列表")

    # visualize-relations --relations-file <path>
    p = sub.add_parser("visualize-relations", help="生成关系图谱")
    p.add_argument(
        "--relations-file",
        required=True,
        help="关系 JSON 文件路径（通常为 explore 命令的输出）",
    )

    # chart --name --type --x --y -o
    p = sub.add_parser("chart", help="从 DataFrame 生成图表 PNG")
    p.add_argument("--name", required=True, help="DataFrame 名称")
    p.add_argument("--type", required=True, choices=["bar", "line"], help="图表类型")
    p.add_argument("--x", required=True, help="X 轴列名")
    p.add_argument("--y", required=True, help="Y 轴列名")
    p.add_argument("-o", "--output", required=True, help="输出 PNG 路径")

    # info: 合并 show-datasource / list / show
    p = sub.add_parser("info", help="数据源与 DataFrame 信息查询")
    p.add_argument("--datasource", action="store_true", help="仅显示已配置的数据源名称")
    p.add_argument("--df", action="store_true", help="仅显示已加载的 DataFrame 列表")
    p.add_argument("--name", help="显示指定 DataFrame 的结构和样本数据")
    p.add_argument("--rows", type=int, default=10, help="样本行数（默认 10，仅与 --name 配合使用）")

    args = parser.parse_args(sys.argv[1:])

    # 加载配置
    config_manager = ConfigManager()
    if not config_manager.exists():
        print("错误: 配置文件不存在，请先运行 govio onboard", file=sys.stderr)
        sys.exit(1)

    config = config_manager.load()

    if not config.get("datasources") and not getattr(args, "memory", False):
        print(
            "警告: 配置中无 datasources，请先用 onboard 配置数据源",
            file=sys.stderr,
        )

    # 分发子命令（失败统一转 ❌ stderr + 退出码 1，避免裸 traceback）
    try:
        match args.action:
            case "load":
                cmd_load(
                    config,
                    args.name,
                    args.sql,
                    datasource=args.datasource,
                    memory=args.memory,
                    output=getattr(args, "output", None),
                )
            case "release":
                if not args.all and not args.name:
                    print("❌ 请指定 --name 或 --all", file=sys.stderr)
                    sys.exit(1)
                if args.all and args.name:
                    print("❌ 请不要同时指定 --name 和 --all", file=sys.stderr)
                    sys.exit(1)
                cmd_release(config, args.name, args.all)
            case "compare":
                cols = [c.strip() for c in args.join_columns.split(",")]
                cmd_compare(config, args.source, args.target, cols)
            case "explore":
                cmd_explore(config, args.dataframes or None)
            case "visualize-relations":
                cmd_visualize(config, args.relations_file)
            case "chart":
                cmd_chart(config, args.name, args.type, args.x, args.y, args.output)
            case "info":
                cmd_info(config, args)
    except Exception as e:
        print(f"❌ 执行失败: {e!s}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    observe()
