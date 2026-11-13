"""govio-cli sql 命令组：指标 SQL 组装"""

import argparse
import json
import sys
from pathlib import Path

from govio.core.sql_builder import build_metric_sql

from .config import ConfigManager


def sql():
    """sql 命令组入口"""
    parser = argparse.ArgumentParser(prog="govio-cli sql")
    sub = parser.add_subparsers(dest="sql_action")

    p_build = sub.add_parser("build", help="从 JSON 规格组装 SQL")
    p_build.add_argument("-f", "--file", help="JSON 规格文件路径")
    p_build.add_argument(
        "-o", "--output", help="输出 SQL 文件路径，省略则打印到 stdout"
    )

    p_caliber = sub.add_parser("caliber", help="SQL 口径冲突检测（与图谱指标定义比对）")
    p_caliber.add_argument("--sql", help="SQL 文本")
    p_caliber.add_argument("--sql-file", help="SQL 文件路径（与 --sql 二选一）")
    p_caliber.add_argument(
        "-m",
        "--metric",
        action="append",
        default=[],
        metavar="REF",
        help="指标 code 或节点 ID，可重复指定",
    )

    args = parser.parse_args()

    if args.sql_action == "build":
        _build(args)
    elif args.sql_action == "caliber":
        _caliber(args)
    else:
        parser.print_help()
        sys.exit(1)


def _build(args):
    """sql build 子命令实现"""
    if args.file:
        try:
            text = Path(args.file).read_text(encoding="utf-8")
        except FileNotFoundError:
            print(f"错误：文件不存在：{args.file}", file=sys.stderr)
            sys.exit(1)
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        print("错误：请通过 -f 指定 JSON 文件或通过 stdin 传入", file=sys.stderr)
        sys.exit(1)

    try:
        req = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"错误：JSON 解析失败：{e}", file=sys.stderr)
        sys.exit(1)

    try:
        sql_text = build_metric_sql(
            metrics=req["metrics"],
            dimensions=req.get("dimensions"),
            filters=req.get("filters"),
            order_by=req.get("order_by"),
            limit=req.get("limit", 100),
            cte_refs=req.get("cte_refs"),
        )
    except (ValueError, KeyError) as e:
        print(f"错误：SQL 组装失败：{e}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        Path(args.output).write_text(sql_text + "\n", encoding="utf-8")
    else:
        print(sql_text)


def _caliber(args):
    """sql caliber 子命令实现：SQL 与图谱指标口径比对

    输出契约：stdout 只输出一行机器可读 JSON
    `{"results": [{"metric", "status", "differences"}]}`（ensure_ascii 转义，
    免疫控制台编码差异）；错误只落 stderr。

    退出码契约：比对完成（含 conflict/unknown 结果）退出 0；坏输入（缺 SQL /
    缺 --metric / 文件不存在）、配置或图后端错误退出 1。图谱中查不到的指标
    按单指标 `status: "unknown"` 降级，不视为命令失败。
    """
    from govio.core.graph_factory import GraphFactory
    from govio.metadata.caliber import (
        compare_metric,
        extract_sql_facts,
        read_metric_caliber,
    )

    if args.sql_file:
        try:
            sql_text = Path(args.sql_file).read_text(encoding="utf-8")
        except FileNotFoundError:
            print(f"错误：文件不存在：{args.sql_file}", file=sys.stderr)
            sys.exit(1)
    else:
        sql_text = args.sql

    if not sql_text or not sql_text.strip():
        print("错误：缺少 SQL 语句，请用 --sql 或 --sql-file 指定", file=sys.stderr)
        sys.exit(1)

    if not args.metric:
        print("错误：请至少指定一个 --metric", file=sys.stderr)
        sys.exit(1)

    try:
        config_manager = ConfigManager()
        if not config_manager.exists():
            print("错误：配置文件不存在，请先运行 govio-cli onboard", file=sys.stderr)
            sys.exit(1)
        graph_config = config_manager.load().get("graph") or {}
        if not graph_config.get("backend"):
            print("错误：配置文件缺少 'graph.backend' 字段", file=sys.stderr)
            sys.exit(1)
        graph = GraphFactory.create(graph_config)
    except SystemExit:
        raise
    except Exception as e:
        print(f"错误：连接图库失败：{e!s}", file=sys.stderr)
        sys.exit(1)

    results = []
    facts = extract_sql_facts(sql_text)
    for ref in args.metric:
        metric = read_metric_caliber(graph, ref)
        if metric is None:
            results.append(
                {
                    "metric": {"code": ref},
                    "status": "unknown",
                    "differences": [
                        {
                            "aspect": "metric",
                            "expected": ref,
                            "found": "",
                            "detail": "图谱中未找到该指标",
                        }
                    ],
                }
            )
        else:
            results.append(compare_metric(metric, facts))

    print(json.dumps({"results": results}, ensure_ascii=True))
