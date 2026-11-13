"""子命令处理函数（cmd_*）—— 参数校验、退出码与步骤编排

约定（docs/specs/cli.md）：失败 → 结构化错误写 stderr 并 `sys.exit(1)`；
成功（含空结果）→ 退出码 0。真正的数据动作都在 `steps.py` / `graph_ops.py`。
"""

import argparse
import sys
from pathlib import Path

from ..config import ConfigManager
from .common import _describe_duckdb_schemas, _resolve_datasource_schemas
from .constants import DEFAULT_ASSETS_DIR, SCHEMA_HINTS, SOURCE_LABELS
from .graph_ops import (
    _clear_graph,
    _generate_assets,
    _update_graph,
    _write_import_report,
)
from .steps import (
    step_compliance_export,
    step_meta_export,
    step_metric_export,
    step_rel_export,
    step_std_export,
)


def cmd_meta(args: argparse.Namespace) -> None:
    """meta meta — 导入元数据（TDS/DuckDB/MySQL/PostgreSQL/Trino，Datasource, PhysicalTable, Col, HAS_COLUMN, OWNS）"""
    source = args.source
    db_path = args.db or ""
    schemas = [s.strip() for s in args.schemas.split(",")] if args.schemas else []
    schemas = [s for s in schemas if s]
    output = Path(args.output) if args.output else Path("./output")
    kundb = args.kundb or ""
    workspace_uuid = args.workspace_uuid or ""
    datasource_name = args.datasource or ""
    datasource_file = args.datasources_file or ""

    if source == "tds":
        # TDS 抽取范围由声明文件 filter.schemas 决定，不接受 --schemas
        missing = []
        if not kundb:
            missing.append("--kundb")
        if not workspace_uuid:
            missing.append("--workspace-uuid")
        if not datasource_file:
            missing.append("--datasources-file")
        if missing:
            print(f"❌ TDS 模式需要指定: {', '.join(missing)}", file=sys.stderr)
            sys.exit(1)
    else:
        # 非 TDS 模式过滤由 CLI --schemas 执行：省略会静默导出 0 张表
        label = SOURCE_LABELS.get(source, source)
        db_desc = "文件路径" if source == "duckdb" else "连接 URL"
        if not db_path:
            print(f"❌ {label} 模式需要指定 --db（{db_desc}）", file=sys.stderr)
            sys.exit(1)
        if not datasource_name:
            print(f"❌ {label} 模式需要指定 --datasource", file=sys.stderr)
            sys.exit(1)
        if not schemas:
            hint = SCHEMA_HINTS.get(source)
            tail = f"；{hint}" if hint else ""
            print(
                f"❌ 需要指定 --schemas（源库 schema 名，逗号分隔{tail}）",
                file=sys.stderr,
            )
            if source == "duckdb":
                print(f"\n📖 {db_path} 中可导入的 schema:", file=sys.stderr)
                print(f"   {_describe_duckdb_schemas(db_path)}", file=sys.stderr)
            sys.exit(1)

    result = step_meta_export(
        output,
        source=source,
        db_path=db_path,
        schemas=schemas,
        datasource_name=datasource_name,
        datasource_file=datasource_file,
        kundb=kundb,
        workspace_uuid=workspace_uuid,
    )
    if result is None:
        sys.exit(1)


def cmd_rel(args: argparse.Namespace) -> None:
    """meta rel — 导入表关系（RELATES_TO 边）"""
    if not args.file:
        print("❌ 需要指定 --file", file=sys.stderr)
        sys.exit(1)

    output = Path(args.output) if args.output else Path("./output")
    step_rel_export(output, args.file)


def cmd_std(args: argparse.Namespace) -> None:
    """meta std — 导入数据标准（Standard 节点）"""
    if not args.kundb:
        print("❌ 需要指定 --kundb", file=sys.stderr)
        sys.exit(1)

    output = Path(args.output) if args.output else Path("./output")
    step_std_export(output, args.kundb, args.workspace_uuid)


def cmd_compliance(args: argparse.Namespace) -> None:
    """meta compliance — 从 TDS 导出已有标准-字段关联（COMPLIES_WITH 边）"""
    if not args.kundb:
        print("❌ 需要指定 --kundb", file=sys.stderr)
        sys.exit(1)

    output = Path(args.output) if args.output else Path("./output")
    step_compliance_export(
        output, args.kundb, args.workspace_uuid, args.datasources_file
    )


def cmd_metric(args: argparse.Namespace) -> None:
    """meta metric — 导入指标维度定义（Metric, Dimension + 边）"""
    if not args.file:
        print("❌ 需要指定 --file", file=sys.stderr)
        sys.exit(1)

    output = Path(args.output) if args.output else Path("./output")
    if not step_metric_export(output, args.file):
        sys.exit(1)


def cmd_graph(args: argparse.Namespace) -> None:
    """meta graph — 更新/重建/清空图数据库 + 生成 assets"""
    output = Path(args.output) if args.output else Path("./output")
    graph_mode = args.mode if hasattr(args, "mode") and args.mode else "update"
    assets_dir = Path(args.assets_dir) if args.assets_dir else DEFAULT_ASSETS_DIR

    if graph_mode == "clear":
        _clear_graph(assets_dir)
        print("\n✅ graph 清空完成！")
        return

    if not output.exists():
        print(f"❌ 输出目录不存在: {output}", file=sys.stderr)
        sys.exit(1)

    if not _update_graph(output, graph_mode, assets_dir):
        print("❌ graph 更新失败（未生成校验报告）", file=sys.stderr)
        sys.exit(1)
    _generate_assets(assets_dir)
    _write_import_report(output, graph_mode, getattr(args, "report", None))

    print("\n✅ graph 更新完成！")


def cmd_recommend(args: argparse.Namespace) -> None:
    """meta recommend — 数据标准推荐"""
    from ..std_recommend import std_recommend

    if not args.kundb:
        print("❌ 需要指定 --kundb", file=sys.stderr)
        sys.exit(1)
    if not args.datasource:
        print("❌ 需要指定 --datasource", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else Path("./output")
    csv_dir = Path(args.csv_dir) if args.csv_dir else output_dir

    schemas = _resolve_datasource_schemas(
        args.datasource, args.datasources_file or "", csv_dir
    )
    if not schemas:
        print(
            f"❌ 无法解析数据源 '{args.datasource}' 的 filter.schemas："
            "请提供 --datasources-file，或先运行 meta meta 生成 Datasource.csv",
            file=sys.stderr,
        )
        sys.exit(1)

    std_recommend(
        output_dir=output_dir,
        kundb=args.kundb,
        workspace_uuid=args.workspace_uuid,
        datasource_name=args.datasource,
        schemas=schemas,
        csv_dir=csv_dir,
    )


def cmd_schema(args: argparse.Namespace) -> None:
    """meta schema — 输出标准 JSON Schema（供外部 agent 生成标准数据）"""
    from govio.metadata.datasource import SCHEMA_PATH as DATASOURCE_SCHEMA_PATH
    from govio.metadata.metric import SCHEMA_PATH as METRIC_SCHEMA_PATH
    from govio.metadata.relationship import SCHEMA_PATH as RELATIONSHIP_SCHEMA_PATH

    schema_files = {
        "metric": METRIC_SCHEMA_PATH,
        "relationship": RELATIONSHIP_SCHEMA_PATH,
        "datasource": DATASOURCE_SCHEMA_PATH,
    }
    text = schema_files[args.schema_name].read_text(encoding="utf-8")
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"✓ Schema 已写入: {args.output}")
    else:
        print(text, end="")


def cmd_import_schema(args: argparse.Namespace) -> None:
    """meta import-schema — 从已配置数据源导入元数据到图库（仅 DuckDB）"""
    ds_name = args.datasource

    config = ConfigManager().load()
    datasources = config.get("datasources", {})
    if ds_name not in datasources:
        print(f"❌ 数据源 '{ds_name}' 不存在，请先用 onboard 配置", file=sys.stderr)
        sys.exit(1)

    url = datasources[ds_name].get("url", "")
    if not url.startswith("duckdb://"):
        print(
            f"❌ import-schema 仅支持 DuckDB 数据源，'{ds_name}' 的 URL 不是 duckdb:// 开头",
            file=sys.stderr,
        )
        sys.exit(1)

    db_path = url[len("duckdb://") :]
    schemas = [s.strip() for s in args.schemas.split(",") if s.strip()]
    if not schemas:
        print("❌ 需要指定 --schemas", file=sys.stderr)
        sys.exit(1)

    output = Path(args.output) if args.output else Path("./output")
    assets_dir = Path(args.assets_dir) if args.assets_dir else DEFAULT_ASSETS_DIR

    result = step_meta_export(
        output,
        source="duckdb",
        db_path=db_path,
        schemas=schemas,
        datasource_name=ds_name,
    )
    if result is None:
        sys.exit(1)

    if not _update_graph(output, args.mode, assets_dir):
        print("❌ import-schema 图数据更新失败（未生成校验报告）", file=sys.stderr)
        sys.exit(1)
    _generate_assets(assets_dir)
    _write_import_report(output, args.mode, getattr(args, "report", None))
    print("\n✅ import-schema 完成！")


# ---------------------------------------------------------------------------
# CLI entry point
