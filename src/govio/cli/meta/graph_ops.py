"""图与 assets 操作 —— `meta graph` / `meta import-schema` 的后端装载步骤

按配置的 graph backend（falkordb / networkx / ladybug）把 CSV 装载进图库，
并在成功后生成 assets 与导入校验报告。CSV→GML 由 `govio.graph.networkx_loader` 负责。
"""

from pathlib import Path

from govio.core.assets_generator import AssetsGenerator
from govio.core.graph_factory import GraphFactory
from govio.graph.falkordb_loader import import_csv_to_falkordb, upsert_csv_to_falkordb
from govio.graph.ladybug_loader import import_csv_to_ladybug, upsert_csv_to_ladybug
from govio.metadata.import_report import (
    REPORT_FILE,
    build_import_report,
    take_merge_delta,
    write_import_report,
)

from ..config import ConfigManager


def _update_graph(output: Path, graph_mode: str, assets_dir: Path) -> bool:
    """更新图数据库。返回是否成功。"""
    from govio.graph.networkx_loader import build_graph

    graph_config = ConfigManager().load()
    graph = graph_config.get("graph") or {}
    backend = graph.get("backend")
    incremental = graph_mode == "update"

    if backend == "falkordb":
        falkordb_cfg = graph.get("falkordb", {})
        host = falkordb_cfg.get("host", "localhost")
        port = falkordb_cfg.get("port", 6379)
        graph_name = falkordb_cfg.get("graph", "ontology")
        try:
            if incremental:
                upsert_csv_to_falkordb(output, host, port, graph_name)
                print("✓ FalkorDB 数据已更新")
            else:
                import_csv_to_falkordb(output, host, port, graph_name)
                print("✓ FalkorDB 数据已重建")
        except Exception as e:
            print(f"❌ 导入 FalkorDB 失败: {e}")
            return False
    elif backend == "networkx":
        networkx_cfg = graph.get("networkx", {})
        gml_path = networkx_cfg.get("gml_path", str(assets_dir / "ontology.gml"))
        label = "更新" if incremental else "重建"
        print(f"\n正在从 CSV {label} GML 文件 ({gml_path})...")
        try:
            build_graph(str(output), gml_path, incremental=incremental)
            print(f"✓ GML 文件已{label}")
        except Exception as e:
            print(f"❌ {label} GML 失败: {e}")
            return False
    elif backend == "ladybug":
        ladybug_cfg = graph.get("ladybug", {})
        db_path_val = ladybug_cfg.get("db_path")
        bp = ladybug_cfg.get("buffer_pool_size", 256 * 1024 * 1024)
        maxdb = ladybug_cfg.get("max_db_size", 1 * 1024 * 1024 * 1024)
        if not db_path_val:
            print("❌ Ladybug 配置缺少 db_path，跳过图数据更新")
            return False
        try:
            if incremental:
                upsert_csv_to_ladybug(
                    output, db_path_val, buffer_pool_size=bp, max_db_size=maxdb
                )
                print("✓ Ladybug 数据已更新")
            else:
                # rebuild 先删旧文件，避免版本不兼容导致无法打开
                db_file = Path(db_path_val)
                if db_file.exists():
                    db_file.unlink()
                import_csv_to_ladybug(
                    output, db_path_val, buffer_pool_size=bp, max_db_size=maxdb
                )
                print("✓ Ladybug 数据已重建")
        except Exception as e:
            print(f"❌ 导入 Ladybug 失败: {e}")
            return False
    else:
        print("提示: 未配置 graph backend，跳过图数据更新")
    return True


def _clear_graph(assets_dir: Path) -> bool:
    """清空图数据库。返回是否成功。"""
    graph_config = ConfigManager().load()
    graph = graph_config.get("graph") or {}
    backend = graph.get("backend")

    if backend == "falkordb":
        from govio.graph.falkordb_loader import delete_falkordb_graph

        falkordb_cfg = graph.get("falkordb", {})
        host = falkordb_cfg.get("host", "localhost")
        port = falkordb_cfg.get("port", 6379)
        graph_name = falkordb_cfg.get("graph", "ontology")
        try:
            delete_falkordb_graph(host, port, graph_name)
            print(f"✓ FalkorDB 图 '{graph_name}' 已删除")
        except Exception as e:
            print(f"❌ 删除 FalkorDB 图失败: {e}")
            return False
    elif backend == "networkx":
        networkx_cfg = graph.get("networkx", {})
        gml_path = networkx_cfg.get("gml_path", str(assets_dir / "ontology.gml"))
        gml_file = Path(gml_path)
        if gml_file.exists():
            gml_file.unlink()
            print(f"✓ GML 文件已删除: {gml_path}")
        else:
            print("提示: GML 文件不存在，无需删除")
    elif backend == "ladybug":
        ladybug_cfg = graph.get("ladybug", {})
        db_path_val = ladybug_cfg.get("db_path")
        if not db_path_val:
            print("❌ Ladybug 配置缺少 db_path")
            return False
        db_file = Path(db_path_val)
        if db_file.exists():
            db_file.unlink()
            print(f"✓ Ladybug 数据库已删除: {db_path_val}")
        else:
            print("提示: Ladybug 数据库文件不存在，无需删除")
    else:
        print("提示: 未配置 graph backend，跳过")
    return True


def _write_import_report(output: Path, graph_mode: str, report_arg: str | None) -> None:
    """生成并落盘校验报告，stdout 输出报告路径与校验结果。

    默认写入 <output>/import-report.json，`--report` 可指定别处。
    增量计数取自 import-delta.json（自上次 graph 同步以来的 merge 分类），
    取走即清零。

    Args:
        output: 导入任务的 CSV 目录
        graph_mode: graph 更新模式（update / rebuild）
        report_arg: --report 指定的报告路径（缺省写入 output 目录）
    """
    graph_config = ConfigManager().load()
    backend = (graph_config.get("graph") or {}).get("backend") or "none"
    changes = take_merge_delta(output)
    report = build_import_report(
        output, mode=graph_mode, backend=backend, changes=changes
    )
    path = Path(report_arg) if report_arg else output / REPORT_FILE
    write_import_report(report, path)
    print(f"\n✓ 校验报告已写入: {path}（校验结果: {report['summary']['status']}）")


def _generate_assets(assets_dir: Path) -> None:
    """生成 schema.md、name 索引、metrics_index.md 等 assets。"""
    print("\n正在生成 assets...")
    try:
        graph_config = ConfigManager().load()
        graph = graph_config.get("graph") or {}
        graph_obj = GraphFactory.create(graph)
        resolved = assets_dir.resolve()
        generator = AssetsGenerator(graph_obj, resolved)
        generator.generate_all()
        print(f"✓ Assets 已生成到: {resolved}")
    except Exception as e:
        print(f"❌ 生成 assets 失败: {e}")


# ---------------------------------------------------------------------------
# Step functions — 可独立调用的管线步骤
