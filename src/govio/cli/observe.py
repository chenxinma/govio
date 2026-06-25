"""observe 命令 — 数据表探查 CLI

提供数据表比较、关系探索、数据表加载释放等功能。
DataFrame 持久化到 .govio/observe/ 目录。
所有子命令参数均为关键字参数。
"""

import argparse
import json
import sys

from .config import ConfigManager
from ..observe_data.core.observe_store import ObserveStore
from ..observe_data.core.database import DatabaseManager
from ..observe_data.tools.list_dataframes import list_dataframes
from ..observe_data.tools.load_dataframe import load_dataframe, load_from_memory
from ..observe_data.tools.release_dataframe import release_dataframe, release_all_dataframes
from ..observe_data.tools.visualize_relations import visualize_relations
from ..observe_data.core.chart import render_chart
from ..observe_data.tools.list_datasources import list_datasources


def get_db_manager(config: dict) -> DatabaseManager:
    """从配置创建 DatabaseManager"""
    from ..observe_data.config import DataSourceConfig

    datasources = config.get("datasources", {})
    ds_configs = {
        name: DataSourceConfig(url=ds["url"], connect_args=ds.get("connect_args", {}))
        for name, ds in datasources.items()
    }
    return DatabaseManager(ds_configs)


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
    store = ObserveStore()

    if memory:
        result = load_from_memory(store=store, name=name, sql=sql)
    else:
        if datasource is None:
            raise Exception("datasource not setted.")
        
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


def cmd_compare(
    config: dict, source_df: str, target_df: str, join_columns: list[str]
) -> None:
    """比对 DataFrame"""
    store = ObserveStore()

    source = store.get(source_df)
    if source is None:
        print(json.dumps({"success": False, "error": f"DataFrame '{source_df}' 不存在"}))
        return

    target = store.get(target_df)
    if target is None:
        print(json.dumps({"success": False, "error": f"DataFrame '{target_df}' 不存在"}))
        return

    from ..observe_data.core.comparator import TableComparator

    comparator = TableComparator()
    result = comparator.compare(source, target, join_columns)
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
            print(json.dumps({"success": False, "error": f"DataFrame '{name}' 不存在"}))
            return
        df_dict[name] = df

    from ..observe_data.core.explorer import RelationExplorer

    explorer = RelationExplorer()
    result = explorer.explore(df_dict)
    print(json.dumps({"success": True, **result}, ensure_ascii=False, indent=2))


def cmd_visualize(config: dict, relations_json: str) -> None:
    """可视化关系"""
    try:
        relations = json.loads(relations_json)
    except json.JSONDecodeError as e:
        print(json.dumps({"success": False, "error": f"JSON 解析失败: {e}"}))
        return

    result = visualize_relations(relations=relations)
    print(json.dumps(result, ensure_ascii=False, indent=2))


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
        print(json.dumps({"success": False, "error": f"DataFrame '{name}' 不存在"}))
        return

    result = render_chart(
        df=df,
        chart_type=chart_type,
        x_col=x,
        y_col=y,
        output_path=output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def observe():
    """observe 命令入口"""
    parser = argparse.ArgumentParser(
        description="数据表探查命令 — 加载、比较、探索数据表",
    )
    sub = parser.add_subparsers(dest="action", required=True)

    # show-datasource
    p = sub.add_parser("show-datasource", help="显示已配置的数据源")
    p.add_argument("--detail", action="store_true", help="显示数据源的完整信息（默认仅列出名称）")

    # load --name (--datasource | --memory) --sql [-o output]
    p = sub.add_parser("load", help="加载 DataFrame")
    p.add_argument("--name", required=True, help="DataFrame 名称")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--datasource", help="数据源名称")
    src.add_argument("--memory", action="store_true", help="从已加载的 DataFrame 中查询")
    p.add_argument("--sql", required=True, help="查询 SQL")
    p.add_argument("-o", "--output", help="将数据内容输出到 JSON 文件")

    # list
    sub.add_parser("list", help="列出已加载的 DataFrame")

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

    # visualize-relations --relations <json>
    p = sub.add_parser("visualize-relations", help="生成关系图谱")
    p.add_argument("--relations", required=True, help="关系 JSON")

    # chart --name --type --x --y -o
    p = sub.add_parser("chart", help="从 DataFrame 生成图表 PNG")
    p.add_argument("--name", required=True, help="DataFrame 名称")
    p.add_argument("--type", required=True, choices=["bar", "line"], help="图表类型")
    p.add_argument("--x", required=True, help="X 轴列名")
    p.add_argument("--y", required=True, help="Y 轴列名")
    p.add_argument("-o", "--output", required=True, help="输出 PNG 路径")

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

    # 分发子命令
    match args.action:
        case "show-datasource":
            cmd_show_datasource(config, args.detail)
        case "load":
            cmd_load(
                config,
                args.name,
                args.sql,
                datasource=args.datasource,
                memory=args.memory,
                output=getattr(args, "output", None),
            )
        case "list":
            cmd_list(config)
        case "release":
            if not args.all and not args.name:
                print("请指定 --name 或 --all")
                sys.exit(1)
            cmd_release(config, args.name, args.all)
        case "compare":
            cols = [c.strip() for c in args.join_columns.split(",")]
            cmd_compare(config, args.source, args.target, cols)
        case "explore":
            cmd_explore(config, args.dataframes if args.dataframes else None)
        case "visualize-relations":
            cmd_visualize(config, args.relations)
        case "chart":
            cmd_chart(config, args.name, args.type, args.x, args.y, args.output)


if __name__ == "__main__":
    observe()
