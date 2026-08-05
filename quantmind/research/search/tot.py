"""树状思维（ToT）因子搜索 —— 递归分支 + 剪枝（对标 AlphaBench ``searcher/algo/tot.py``）。

ToT 把搜索表示为**树**：根为 seed，每个深度并行展开多个候选分支，评估后
只保留指标最优的 K 个幸存者（剪枝）进入下一深度，最终选全树最优。

相比 CoT 的单路径精炼，ToT 在每层做宽度优先的探索，能覆盖更多候选空间，
代价是更多评估调用（token/计算开销）。离线/无 LLM 时回落到
:func:`quantmind.research.search.base.mutate_expressions` 的确定性变异。
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from ...ai.provider import LLMProvider
from ..factors.alpha_cs import Panel
from .base import (
    BaseAlgo,
    EvalFn,
    SearchFn,
    SearchResult,
    SearchStep,
    mutate_expressions,
    register_algo,
)

_logger = logging.getLogger("quantmind.research.search.tot")

_TOT_SYSTEM = (
    "You are an expert quantitative researcher exploring a tree of alpha factor "
    "candidates. You are given the best factor found so far (parent node) with its "
    "Rank IC. Propose ONE promising child factor expression that explores a nearby "
    "but meaningfully different direction, aiming to improve predictive signal.\n"
    "Use only these variables: $close, $open, $high, $low, $volume, $amount.\n"
    "Use QLib-style operators: Mean(x,n), Std(x,n), Sum(x,n), Rank(x), TsRank(x,n), "
    "Min(x,n), Max(x,n), Delay(x,n), Delta(x,n), Corr(a,b,n), Cov(a,b,n), TsZscore(x,n).\n"
    "Return ONLY a JSON object: {\"expression\": \"...\"}"
)


@register_algo("tot")
class ToTSearcher(BaseAlgo):
    """树状思维因子搜索器。"""

    name = "tot"

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        evaluate_fn: Optional[EvalFn] = None,
        depth: int = 3,
        branch: int = 3,
        survivors: int = 2,
    ) -> None:
        super().__init__(evaluate_fn=evaluate_fn)
        self.provider = provider
        self.depth = depth
        self.branch = branch
        self.survivors = survivors

    async def run(
        self,
        seed: str,
        panel: Panel,
        val_panel: Optional[Panel] = None,
        forward_periods: int = 1,
        market: str = "",
        instruction: str = "",
        search_fn: Optional[SearchFn] = None,
        **kwargs,
    ) -> SearchResult:
        evaluate_fn = self.evaluate_fn or (await self._make_default_eval(panel, forward_periods, market))
        search_fn = search_fn or self._build_default_search_fn(instruction)

        result = SearchResult(seed=seed, _best_expression=seed)
        history: List[dict] = []

        seed_metrics = await evaluate_fn(seed)
        result.seed_rank_ic = _map(seed_metrics.get("rank_ic"))
        result.best_rank_ic = result.seed_rank_ic
        result.best_ic = seed_metrics.get("ic", float("nan"))
        history.append(_step(0, seed, result.seed_rank_ic, result.best_ic, is_best=True))

        # 当前深度待展开的节点（表达式 + 指标）
        frontier: List[dict] = [{"expression": seed, "rank_ic": result.seed_rank_ic}]
        expanded_rounds = 0

        for depth in range(1, self.depth + 1):
            expanded_rounds = depth
            children: List[dict] = []
            for node in frontier:
                cands = await search_fn(seed=seed, best=node, depth=depth)
                for cand in (cands or [])[: self.branch]:
                    if not cand or cand in {c["expression"] for c in children}:
                        continue
                    try:
                        m = await evaluate_fn(cand)
                    except Exception as exc:  # noqa: BLE001
                        _logger.debug("ToT 候选 %s 评估失败: %s", cand, exc)
                        m = {"rank_ic": float("nan"), "ic": float("nan")}
                    ric = _map(m.get("rank_ic"))
                    children.append({"expression": cand, "rank_ic": ric, "ic": m.get("ic")})
                    history.append(_step(depth, cand, ric, m.get("ic")))

            if not children:
                _logger.info("ToT depth %d 无新候选，提前终止", depth)
                expanded_rounds = depth - 1
                break

            # 剪枝：保留最优 survivors 个不同表达式
            children = _prune(children, self.survivors)
            frontier = children

            # 更新全局 best
            for child in children:
                if _safe_ric(child["rank_ic"]) > _safe_ric(result.best_rank_ic):
                    result._best_expression = child["expression"]
                    result.best_rank_ic = _map(child["rank_ic"])
                    result.best_ic = child.get("ic", float("nan"))
            _mark_best(history, result._best_expression)
        else:
            expanded_rounds = self.depth

        result.rounds = expanded_rounds
        result.history = [SearchStep(**h) for h in history]
        result.improved = bool(
            result.best_rank_ic == result.best_rank_ic
            and result.seed_rank_ic == result.seed_rank_ic
            and result.best_rank_ic > result.seed_rank_ic
        )
        if val_panel is not None:
            result.val_rank_ic, result.val_ic = await self._val_eval(
                result._best_expression or seed, val_panel, forward_periods, market)
        return result

    @staticmethod
    async def _make_default_eval(panel, forward_periods, market):
        from ..eval import evaluate_expression

        async def _ev(expr):
            rep = evaluate_expression(expr, panel, forward_periods=forward_periods,
                                      market=market, use_cache=False)
            return {"rank_ic": rep.ic_mean if rep.ic_mean == rep.ic_mean else float("nan"),
                    "ic": rep.ic_pearson if rep.ic_pearson == rep.ic_pearson else float("nan")}
        return _ev

    def _build_default_search_fn(self, instruction: str) -> SearchFn:
        async def _search(seed: str, best: dict, **kw):
            if self.provider is not None:
                try:
                    resp = await self.provider.chat(_TOT_SYSTEM, f"{instruction}\n"
                                                     f"Parent: {best.get('expression')}  "
                                                     f"RankIC={best.get('rank_ic')}\n")
                    from .prompts import parse_expression_response
                    parsed = parse_expression_response(resp)
                    if parsed:
                        return [parsed]
                except Exception as exc:  # noqa: BLE001
                    _logger.warning("ToT LLM 展开失败: %s", exc)
            # 回落到确定性变异：每节点产出 branch 个候选
            return mutate_expressions(best.get("expression") or seed, n=max(self.branch, 3))
        return _search

    async def _val_eval(self, expr, val_panel, forward_periods, market):
        try:
            ev = await self._make_default_eval(val_panel, forward_periods, market)
            m = await ev(expr)
            r = _map(m.get("rank_ic"))
            return (r if r == r else None), m.get("ic")
        except Exception as exc:  # noqa: BLE001
            _logger.warning("ToT val 评估失败: %s", exc)
            return None, None


# -- 工具 ---------------------------------------------------------------
def _map(x) -> float:
    try:
        f = float(x)
        return f if f == f else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _safe_ric(x) -> float:
    f = _map(x)
    return f if f == f else float("-inf")


def _prune(children: List[dict], k: int) -> List[dict]:
    ranked = sorted(children, key=lambda x: _safe_ric(x.get("rank_ic")), reverse=True)
    out: List[dict] = []
    for c in ranked:
        if c["expression"] not in {d["expression"] for d in out}:
            out.append(c)
        if len(out) >= k:
            break
    return out


def _mark_best(history: List[dict], best_expr: str) -> None:
    for h in history:
        if h["expression"] == best_expr:
            h["is_best"] = True


def _step(round_: int, expression: str, rank_ic, ic, is_best: bool = False) -> dict:
    r4 = lambda x: round(float(x), 4) if x == x else None  # noqa: E731
    return {"round": round_, "expression": expression,
            "rank_ic": r4(rank_ic), "ic": r4(ic), "is_best": is_best}
