"""数据治理元数据模块

包含数据标准、数据库元数据和推荐器功能。
惰性导出，避免非推荐路径加载 sklearn。
"""

import importlib
from typing import Any

__all__ = [
    "DEFAULT_K_NEIGHBORS",
    "DEFAULT_TOP_N",
    "DEFAULT_WEIGHTS",
    "MIN_SIMILARITY",
    "RelationshipLoader",
    "StandardRecommender",
    "create_recommender",
    "load_relationships",
]

_LAZY_ATTRS = {
    "StandardRecommender": ".recommender",
    "create_recommender": ".recommender",
    "DEFAULT_WEIGHTS": ".recommender",
    "DEFAULT_K_NEIGHBORS": ".recommender",
    "DEFAULT_TOP_N": ".recommender",
    "MIN_SIMILARITY": ".recommender",
    "RelationshipLoader": ".relationship",
    "load_relationships": ".relationship",
}


def __getattr__(name: str) -> Any:
    try:
        module = _LAZY_ATTRS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    return getattr(importlib.import_module(module, __name__), name)


def __dir__() -> list[str]:
    return sorted(__all__)
