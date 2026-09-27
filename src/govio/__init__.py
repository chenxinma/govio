"""Govio 公共 API — 惰性导入，避免 CLI 启动时加载图后端等重依赖"""

import importlib
from typing import Any

__all__ = [
    "FalkorDBGraph",
    "LadybugGraph",
    "NetworkXGraph",
    "main",
    "build_metric_sql",
]

_LAZY_ATTRS = {
    "FalkorDBGraph": ".graph.falkordb_graph",
    "LadybugGraph": ".graph.ladybug_graph",
    "NetworkXGraph": ".graph.networkx_graph",
    "main": ".cli",
    "build_metric_sql": ".core.sql_builder",
}


def __getattr__(name: str) -> Any:
    try:
        module = _LAZY_ATTRS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    return getattr(importlib.import_module(module, __name__), name)


def __dir__() -> list[str]:
    return sorted(__all__)
