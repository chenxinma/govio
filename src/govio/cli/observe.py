"""observe 命令 — 数据表探查 CLI

提供数据表比较、关系探索、数据表加载释放等功能。
DataFrame 持久化到 .govio/observe/ 目录。
"""

import argparse
import json
import sys

from .config import ConfigManager
from ..observe_data.core.observe_store import ObserveStore
from ..observe_data.core.database import DatabaseManager
from ..observe_data.tools.list_dataframes import list_dataframes
from ..observe_data.tools.load_dataframe import load_dataframe
from ..observe_data.tools.release_dataframe import release_dataframe
from ..observe_data.tools.visualize_relations import visualize_relations
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


def cmd_show_datasource(config: dict) -> None:
    """显示数据源"""
    db_manager = get_db_manager(config)
    result = list_datasources(db_manager)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_load(config: dict, name: str, datasource: str, sql: str) -> None:
    """加载 DataFrame"""
    db_manager = get_db_manager(config)
    store = ObserveStore()
    result = load_dataframe(
        store=store,
        db_manager=db_manager,
        datasource=datasource,
        name=name,
        sql=sql,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_list(config: dict) -> None:
    """列出 DataFrame"""
    store = ObserveStore()
    result = list_dataframes(store=store)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_release(config: dict, name: str) -> None:
    """释放 DataFrame"""
    store = ObserveStore()
    result = release_dataframe(store=store, name=name)
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
    relations = explorer.explore(df_dict)
    print(json.dumps({"success": True, "relations": relations}, ensure_ascii=False, indent=2))


def cmd_visualize(config: dict, relations_json: str) -> None:
    """可视化关系"""
    try:
        relations = json.loads(relations_json)
    except json.JSONDecodeError as e:
        print(json.dumps({"success": False, "error": f"JSON 解析失败: {e}"}))
        return

    result = visualize_relations(relations=relations)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def observe():
    """observe 命令入口"""
    parser = argparse.ArgumentParser(
        description="数据表探查命令 — 加载、比较、探索数据表",
    )
    sub = parser.add_subparsers(dest="action", required=True)

    # show-datasource
    sub.add_parser("show-datasource", help="显示已配置的数据源")

    # load <name> <datasource> <sql>
    p = sub.add_parser("load", help="加载 DataFrame")
    p.add_argument("name", help="DataFrame 名称")
    p.add_argument("datasource", help="数据源名称")
    p.add_argument("sql", help="查询 SQL")

    # list
    sub.add_parser("list", help="列出已加载的 DataFrame")

    # release <name>
    p = sub.add_parser("release", help="释放 DataFrame")
    p.add_argument("name", help="DataFrame 名称")

    # compare <source> <target> --join-columns col1,col2
    p = sub.add_parser("compare", help="比对两个 DataFrame")
    p.add_argument("source", help="源 DataFrame 名称")
    p.add_argument("target", help="目标 DataFrame 名称")
    p.add_argument("--join-columns", required=True, help="比对列，逗号分隔")

    # explore [df1 df2 ...]
    p = sub.add_parser("explore", help="探查 DataFrame 之间的关系")
    p.add_argument("dataframes", nargs="*", help="DataFrame 名称列表")

    # visualize-relations <json>
    p = sub.add_parser("visualize-relations", help="生成关系图谱")
    p.add_argument("relations", help="关系 JSON")

    args = parser.parse_args(sys.argv[1:])

    # 加载配置
    config_manager = ConfigManager()
    if not config_manager.exists():
        print("错误: 配置文件不存在，请先运行 govio onboard", file=sys.stderr)
        sys.exit(1)

    config = config_manager.load()

    if not config.get("datasources"):
        print(
            "警告: 配置中无 datasources，请先用 onboard 配置数据源",
            file=sys.stderr,
        )

    # 分发子命令
    match args.action:
        case "show-datasource":
            cmd_show_datasource(config)
        case "load":
            cmd_load(config, args.name, args.datasource, args.sql)
        case "list":
            cmd_list(config)
        case "release":
            cmd_release(config, args.name)
        case "compare":
            cols = [c.strip() for c in args.join_columns.split(",")]
            cmd_compare(config, args.source, args.target, cols)
        case "explore":
            cmd_explore(config, args.dataframes if args.dataframes else None)
        case "visualize-relations":
            cmd_visualize(config, args.relations)


if __name__ == "__main__":
    observe()
