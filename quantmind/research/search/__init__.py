"""迭代因子搜索（对标 AlphaBench ``searcher/``）。

提供三种可插拔的 LLM 引导搜索算法（经 :func:`register_algo` / :func:`create_algo`
按名调用）：
  - ``co``  ：链式精炼（Chain-of-Thought / Chain-of-Experience），单路径迭代；
  - ``ea``  ：进化算法（Evolutionary），种群变异 + 交叉 + 选择；
  - ``tot`` ：树状思维（Tree-of-Thought），递归分支 + 剪枝。

三者遵循统一 :class:`BaseAlgo.run` 契约，可在同一面板/数据集上横向对比
（对标 AlphaBench ``searcher/algo/``，论文 4.4 节三类范式）。
"""
from .base import (
    SearchResult,
    SearchStep,
    mutate_expressions,
    BaseAlgo,
    register_algo,
    list_algos,
    create_algo,
)
from .cot import FactorSearcher, map_rank
from .ea import EASearcher
from .tot import ToTSearcher
from .prompts import parse_expression_response, build_chain_prompt, build_kb_block, SEARCH_SYSTEM

__all__ = [
    "SearchResult",
    "SearchStep",
    "mutate_expressions",
    "BaseAlgo",
    "register_algo",
    "list_algos",
    "create_algo",
    "FactorSearcher",
    "EASearcher",
    "ToTSearcher",
    "map_rank",
    "parse_expression_response",
    "build_chain_prompt",
    "build_kb_block",
    "SEARCH_SYSTEM",
]
