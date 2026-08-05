"""迭代因子搜索（对标 AlphaBench ``searcher/``）。

当前提供链式精炼（CoT）单路径搜索：``FactorSearcher.cot_search``。
后续可在此目录扩展进化算法（EA）、树状思维（ToT）等 ``algo``。
"""
from .base import SearchResult, SearchStep, mutate_expressions
from .cot import FactorSearcher, map_rank
from .prompts import parse_expression_response, build_chain_prompt, SEARCH_SYSTEM

__all__ = [
    "SearchResult",
    "SearchStep",
    "mutate_expressions",
    "FactorSearcher",
    "map_rank",
    "parse_expression_response",
    "build_chain_prompt",
    "SEARCH_SYSTEM",
]
