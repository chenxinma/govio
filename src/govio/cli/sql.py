"""govio-cli sql 命令组：指标 SQL 组装"""

import argparse
import json
import sys
from pathlib import Path

from govio.core.sql_builder import build_metric_sql


def sql():
    """sql 命令组入口"""
    parser = argparse.ArgumentParser(prog="govio-cli sql")
    sub = parser.add_subparsers(dest="sql_action")

    p_build = sub.add_parser("build", help="从 JSON 规格组装 SQL")
    p_build.add_argument("-f", "--file", help="JSON 规格文件路径")
    p_build.add_argument("-o", "--output", help="输出 SQL 文件路径，省略则打印到 stdout")

    args = parser.parse_args()

    if args.sql_action == "build":
        _build(args)
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
