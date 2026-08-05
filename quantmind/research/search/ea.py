"""进化算法（EA）因子搜索 —— 种群迭代（对标 AlphaBench ``searcher/algo/ea.py``）。

EA 以**种群**为单位迭代：每代对当前最优/精选个体做变异与交叉产生一批新候选，
评估后按指标选出幸存者进入下一代。相比 CoT 的单路径精炼，EA 更好利用 LLM 的
探索能力（论文 4.4 节：population 方法通常优于单路径搜索）。

离线/无 LLM 时回落到 :func:`quantmind.research.search.base.mutate_expressions`
的确定性变异，保证流程可跑通与可测试。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
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

_logger = logging.getLogger("quantmind.research.search.ea")

# 逻辑/算术算子提示（用于 LLM 生成变异/交叉）
_EA_SYSTEM = (
    "You are an expert quantitative researcher running an evolutionary search over "
    "alpha factor expressions. You are given the current population of factors with "
    "their Rank IC metrics. Propose a NEW candidate factor expression that recombines "
    "or mutates promising existing ones to improve predictive signal and robustness.\n"
    "Use only these variables: $close, $open, $high, $low, $volume, $amount.\n"
    "Use QLib-style operators: Mean(x,n), Std(x,n), Sum(x,n), Rank(x), TsRank(x,n), "
    "Min(x,n), Max(x,n), Delay(x,n), Delta(x,n), Corr(a,b,n), Cov(a,b,n), TsZscore(x,n), "
    "Sign(x), Abs(x), Log(x), Power(x,a).\n"
    "Return ONLY a JSON object: {\"expression\": \"...\"}"
)


@register_algo("ea")
class EASearcher(BaseAlgo):
    """进化算法因子搜索器。"""

    name = "ea"

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        evaluate_fn: Optional[EvalFn] = None,
        generations: int = 4,
        pop_size: int = 6,
        survivors: int = 3,
        seed_count: int = 2,
    ) -> None:
        super().__init__(evaluate_fn=evaluate_fn)
        self.provider = provider
        self.generations = generations
        self.pop_size = pop_size
        self.survivors = survivors
        self.seed_count = seed_count

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

        # gen 0：评估 seed + 变异出的初始种群
        seed_metrics = await evaluate_fn(seed)
        result.seed_rank_ic = _map(seed_metrics.get("rank_ic"))
        result.best_rank_ic = result.seed_rank_ic
        result.best_ic = seed_metrics.get("ic", float("nan"))

        population: List[dict] = [{"expression": seed, "rank_ic": result.seed_rank_ic}]
        history.append(_step(0, seed, result.seed_rank_ic, seed_metrics.get("ic"), is_best=True))

        for gen in range(1, self.generations + 1):
            # 1) 从种群中选父母，产生子代
            parents = sorted(population, key=lambda x: _safe_ric(x.get("rank_ic")), reverse=True)[:self.seed_count]
            candidates: List[str] = []
            for p in parents:
                cands = await search_fn(seed=seed, best=p)
                candidates.extend(cands or [])
            # 去重、剔除已在种群中的
            seen_expr = {c["expression"] for c in population}
            candidates = [c for c in candidates if c and c not in seen_expr][:self.pop_size]

            if not candidates:
                _logger.info("EA gen %d 无新候选，提前终止", gen)
                result.rounds = gen - 1
                break

            # 2) 评估子代
            for cand in candidates:
                try:
                    m = await evaluate_fn(cand)
                except Exception as exc:  # noqa: BLE001
                    _logger.debug("EA 候选 %s 评估失败: %s", cand, exc)
                    m = {"rank_ic": float("nan"), "ic": float("nan")}
                ric = _map(m.get("rank_ic"))
                step = _step(gen, cand, ric, m.get("ic"))
                history.append(step)
                population.append({"expression": cand, "rank_ic": ric})

            # 3) 选择幸存者（保留指标最优且表达式各异）
            population = _select(population, self.survivors + self.seed_count)

            # 4) 更新 best
            for step in history:
                if step["expression"] == population[0]["expression"]:
                    step["is_best"] = True
            if _safe_ric(population[0]["rank_ic"]) > _safe_ric(result.best_rank_ic):
                result._best_expression = population[0]["expression"]
                result.best_rank_ic = _map(population[0]["rank_ic"])
                m = await evaluate_fn(population[0]["expression"])
                result.best_ic = m.get("ic", float("nan"))
        else:
            result.rounds = self.generations

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
                    resp = await self.provider.chat(_EA_SYSTEM, f"{instruction}\n"
                                                     f"Population best: {best.get('expression')}  "
                                                     f"RankIC={best.get('rank_ic')}\n")
                    from .prompts import parse_expression_response
                    parsed = parse_expression_response(resp)
                    if parsed:
                        return [parsed]
                except Exception as exc:  # noqa: BLE001
                    _logger.warning("EA LLM 变异失败: %s", exc)
            return mutate_expressions(best.get("expression") or seed, n=self.pop_size)
        return _search

    async def _val_eval(self, expr, val_panel, forward_periods, market):
        try:
            ev = await self._make_default_eval(val_panel, forward_periods, market)
            m = await ev(expr)
            r = _map(m.get("rank_ic"))
            return (r if r == r else None), m.get("ic")
        except Exception as exc:  # noqa: BLE001
            _logger.warning("EA val 评估失败: %s", exc)
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


def _step(round_: int, expression: str, rank_ic, ic, is_best: bool = False) -> dict:
    r4 = lambda x: round(float(x), 4) if x == x else None  # noqa: E731
    return {"round": round_, "expression": expression,
            "rank_ic": r4(rank_ic), "ic": r4(ic), "is_best": is_best}


def _select(population: List[dict], k: int) -> List[dict]:
    """按 Rank IC 降序选前 k 个（保留最优多样化的前 k 名）。"""
    ranked = sorted(population, key=lambda x: _safe_ric(x.get("rank_ic")), reverse=True)
    out: List[dict] = []
    for p in ranked:
        if p["expression"] not in {q["expression"] for q in out}:
            out.append(p)
        if len(out) >= k:
            break
    return out
