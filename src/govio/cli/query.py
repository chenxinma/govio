"""Govio 知识图谱统一查询入口

支持通过 CLI 或作为包内模块调用。
用法: govio query --query "查询语句"
"""
import json
import logging
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import NoReturn

import pandas as pd

from govio import FalkorDBGraph, LadybugGraph, NetworkXGraph

from .config import ConfigManager

log_dir = Path.home() / ".govio" / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

log_file = log_dir / f"query_{datetime.now().strftime('%Y%m%d')}.log"
logging.basicConfig(
    filename=str(log_file),
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)

logger = logging.getLogger(__name__)

# 内联阈值：小于等于该行数时 stdout 额外内联一份结果
INLINE_LIMIT = 50


def _fail(message: str) -> NoReturn:
    """打印错误并以退出码 1 结束进程

    报错文案双路输出：原文（无前缀）落 stdout，供脚本/回归契约直接读取；
    同时以 ❌ 前缀落 stderr（stdout 不出现 ❌ 前缀行）。

    Args:
        message: 错误信息（stderr 侧自动加 ❌ 前缀）
    """
    print(message)
    print(f"❌ {message}", file=sys.stderr)
    sys.exit(1)


def output_result(data, columns: list[str] | None = None):
    """输出查询结果

    输出契约：
    - 结果永远落 UTF-8 JSONL 文件（records 带字段名），文件是权威结果；
    - stdout 摘要行 Result output: <path> rows: N columns: [...] 全 ASCII
      （columns 经 ensure_ascii 转义，免疫控制台编码差异）；
    - 行数 <= INLINE_LIMIT 时 stdout 额外内联一行 JSON（ensure_ascii=True，
      同样编码安全）；
    - 空结果只输出 Data not found.，不写文件。

    Args:
        data: 查询结果（list[行] 或单行/标量）
        columns: 列名（RETURN 别名）；None 时用位置键 "0"/"1" 兑底
    """
    if not data:
        print("Data not found.")
        logger.info("data not found.")
        return

    if not isinstance(data, list):
        data = [data]

    # to_json 往返统一 NaN→null，保证 stdout / 文件的 JSON 都合法
    # pandas 会把 Decimal 直接序列化为 float（822 -> 822.0），与「Decimal->str」契约不符，
    # 故在往返前显式转 str
    # shortcut: RETURN 别名重复时 records 同名键会合并，出现真实需求再处理
    df = pd.DataFrame(data, columns=columns)
    df = df.map(lambda v: str(v) if isinstance(v, Decimal) else v)
    records = json.loads(
        df.to_json(
            orient="records",
            force_ascii=False,
            date_format="iso",
            default_handler=str,
        )
    )

    output_dir = Path(".") / ".govio"
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = (
        output_dir / f"output-{datetime.now().strftime('%Y%m%d%H%M%S%f')}.json"
    )
    with open(fname, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)

    _size = len(records)
    print(
        "Result output:",
        fname,
        "rows:",
        _size,
        "columns:",
        json.dumps([str(c) for c in df.columns], ensure_ascii=True),
    )
    if _size <= INLINE_LIMIT:
        print(json.dumps(records, ensure_ascii=True))
    logger.info("result file: %s rows: %s", str(fname), str(_size))


def cmd_networkx(code: str, gml_path: str):
    """NetworkX 查询处理"""
    gml_file = Path(gml_path)
    if not gml_file.exists():
        _fail(f"GML file not found: {gml_file}")

    try:
        gf = NetworkXGraph(gml_file) # pyright: ignore[reportArgumentType]
        g = gf.G
    except Exception as e:
        _fail(f"连接图库失败: {e!s}")

    logger.info("NetworkX Code: " + code)

    local_vars = {"g": g}
    try:
        exec(code, locals=local_vars)
    except Exception as e:
        _fail(f"查询执行失败: {e!s}")
    # Python exec 的 result 无列名元信息，由 output_result 按位置键兑底
    output_result(local_vars.get("result"))


def cmd_falkordb(cypher: str, host: str = "localhost", port: int = 6379, graph_name: str = "ontology"):
    """FalkorDB 查询处理"""
    if not cypher.upper().startswith("MATCH"):
        _fail("Please write a MATCH query.")

    logger.info("Cypher: " + cypher)

    try:
        g = FalkorDBGraph(graph=graph_name, host=host, port=port)
    except Exception as e:
        _fail(f"连接图库失败: {e!s}")

    try:
        columns, data = g.query_with_columns(cypher)
    except ValueError:
        # 非法 Cypher 由图层包装为 ValueError，按当前契约向上传播
        raise
    except Exception as e:
        _fail(f"查询执行失败: {e!s}")
    output_result(data, columns)


def cmd_ladybug(cypher: str, db_path: str):
    """Ladybug 查询处理（Cypher，与 FalkorDB 语法兼容）"""
    if not cypher.upper().startswith("MATCH"):
        _fail("Please write a MATCH query.")

    logger.info("Cypher: " + cypher)

    try:
        g = LadybugGraph(db_path, read_only=True)
    except Exception as e:
        _fail(f"连接图库失败: {e!s}")

    try:
        columns, data = g.query_with_columns(cypher)
    except ValueError:
        # 非法 Cypher 由图层包装为 ValueError，按当前契约向上传播
        raise
    except Exception as e:
        _fail(f"查询执行失败: {e!s}")
    output_result(data, columns)


def query(query_text: str | None):
    """Query CLI 主函数

    退出码契约：成功（含空结果）退出 0；失败（参数缺失、配置/后端错误、
    连接失败、查询执行失败）报错文案双路输出（stdout 原文 + stderr ❌）并退出 1。
    例外：非法 Cypher 由图层包装为 ValueError，按当前契约向上传播（不转 SystemExit）。
    """
    if not query_text or not query_text.strip():
        _fail("缺少查询语句，请用 -c/--code 指定查询语句")

    try:
        config_manager = ConfigManager()
        if not config_manager.exists():
            _fail("配置文件不存在，请先运行 govio-cli onboard")

        config = config_manager.load()
        graph = config.get("graph") or {}
        backend = graph.get("backend")
        if not backend:
            _fail("配置文件缺少 'graph.backend' 字段，请重新运行 govio-cli onboard")

        if backend == "networkx":
            gml_path = graph.get("networkx", {}).get("gml_path")
            if not gml_path:
                _fail("配置文件缺少 graph.networkx.gml_path 字段")
            cmd_networkx(query_text, gml_path)
        elif backend == "falkordb":
            falkordb_config = graph.get("falkordb", {})
            cmd_falkordb(
                query_text,
                host=falkordb_config.get("host", "localhost"),
                port=falkordb_config.get("port", 6379),
                graph_name=falkordb_config.get("graph", "ontology"),
            )
        elif backend == "ladybug":
            ladybug_config = graph.get("ladybug", {})
            db_path = ladybug_config.get("db_path")
            if not db_path:
                _fail("配置文件缺少 graph.ladybug.db_path 字段")
            cmd_ladybug(query_text, db_path)
        else:
            _fail(f"不支持的 backend: {backend}")
    except ValueError:
        # 图层包装的非法 Cypher ValueError 按当前契约向上传播（不转 SystemExit）
        raise
    except Exception as e:
        # 兜底：未预期异常转简洁错误，避免裸 traceback
        _fail(f"查询执行失败: {e!s}")


