"""链式精炼（Chain-of-Thought）因子搜索 —— 对标 AlphaBench ``searcher/algo/cot.py``。

``FactorSearcher.cot_search`` 实现单条路径的迭代精炼：
  1. 评估 seed 得到基线指标；
  2. 每轮把**完整链历史**（seed → 既往候选 → 各指标）喂给 LLM，请求其基于此
     变异/精炼出 1 个改进候选；离线或解析失败时回落到确定性变异器；
  3. 用 P0 的统一评估入口对候选做截面 IC 评估；
  4. 保留相较当前最优有改进的候选，记录轨迹；
  5. 可选：在独立的 val 面板上评估最终 best（防泄漏——val 指标不参与搜索决策）。

与 AlphaBench ``ValEvalTracker`` 一致：搜索期的一切决策只使用 search 期指标，
val 期仅在最终报告时评估，避免对搜索期过拟合。
"""
from __future__ import annotations

import logging
from typing import Awaitable, Callable, Dict, List, Optional

from ...ai.provider import LLMProvider
from ..factors.alpha_cs import Panel
from .base import SearchResult, SearchStep, mutate_expressions
from .prompts import SEARCH_SYSTEM, build_chain_prompt, parse_expression_response

_logger = logging.getLogger("quantmind.research.search.cot")


# 评估函数签名：async (expr) -> {"rank_ic": float, "ic": float}（search 期度量）
EvaluateFn = Callable[[str], Awaitable[Dict[str, float]]]
# 搜索函数签名：async (chain: str, best: dict, **kw) -> List[str]（候选表达式）
SearchFn = Callable[..., Awaitable[List[str]]]


async def _default_evaluate(panel: Panel, forward_periods: int, market: str):
    """闭包构造基于 P0 统一评估入口的默认评估函数（search 期）。"""
    from ..eval import evaluate_expression

    async def _ev(expr: str) -> Dict[str, float]:
        rep = evaluate_expression(expr, panel, forward_periods=forward_periods,
                                  market=market, use_cache=False)
        return {"rank_ic": rep.ic_mean if rep.ic_mean == rep.ic_mean else float("nan"),
                "ic": rep.ic_pearson if rep.ic_pearson == rep.ic_pearson else float("nan")}

    return _ev


class FactorSearcher:
    """LLM 引导的因子迭代搜索执行器（CoT 单路径）。"""

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        evaluate_fn: Optional[EvaluateFn] = None,
        rounds: int = 6,
        cold_start: bool = False,
    ) -> None:
        """初始化搜索器。

        Args:
            provider: 可插拔 ``LLMProvider``（真实 provider 时用 LLM 变异，
                      否则回落到确定性变异器）。
            evaluate_fn: 注入的候选评估函数（默认基于 P0 ``evaluate_expression``）。
            rounds: 迭代轮数。
            cold_start: True 时首轮不评估 seed（直接当作上一轮上下文）。
        """
        self.provider = provider
        self._evaluate_fn = evaluate_fn
        self.rounds = rounds
        self.cold_start = cold_start

    # -- 对外主入口 ----------------------------------------------------------
    async def cot_search(
        self,
        seed_expr: str,
        panel: Panel,
        val_panel: Optional[Panel] = None,
        forward_periods: int = 1,
        market: str = "",
        instruction: str = "",
        search_fn: Optional[SearchFn] = None,
    ) -> SearchResult:
        """执行链式精炼搜索。

        Args:
            seed_expr: 初始因子表达式。
            panel: 搜索期面板（决策依据）。
            val_panel: 可选的独立验证期面板（仅评估最终 best，防泄漏）。
            forward_periods: 前向收益周期。
            market: 市场标识（透传评估/缓存键）。
            instruction: 附加变异方向提示。
            search_fn: 注入的候选生成函数（默认：真实 LLM 或 mock 变异器）。

        Returns:
            ``SearchResult``（含轨迹 history、best 与可选 val 指标）。
        """
        evaluate_fn = self._evaluate_fn or await _default_evaluate(panel, forward_periods, market)
        search_fn = search_fn or self._build_default_search_fn(instruction)

        result = SearchResult(seed=seed_expr, _best_expression=seed_expr)

        # 1) 评估 seed 作为基线
        seed_metrics = await evaluate_fn(seed_expr)
        result.seed_rank_ic = seed_metrics.get("rank_ic", float("nan"))
        rseed = map_rank(seed_metrics.get("rank_ic"))
        result.best_rank_ic = rseed
        result.best_ic = seed_metrics.get("ic", float("nan"))
        _logger.info("seed=%s seed_rank_ic=%.4f", seed_expr, rseed)

        history: List[dict] = []

        # 2) 逐步精炼
        for rnd in range(1, self.rounds + 1):
            best_so_far = result._best_expression or seed_expr
            best_ric = result.best_rank_ic

            chain = build_chain_prompt(
                seed=seed_expr,
                history=history,
                best_expression=best_so_far,
                best_rank_ic=best_ric,
                instruction=instruction,
            )

            # 候选生成
            try:
                candidates = await search_fn(
                    chain=chain,
                    seed=seed_expr,
                    best={"expression": best_so_far, "rank_ic": best_ric},
                )
            except Exception as exc:  # noqa: BLE001
                _logger.warning("候选生成失败: %s", exc)
                candidates = []

            candidates = [c for c in (candidates or []) if c and c != best_so_far]

            if not candidates:
                _logger.info("round %d 无新候选，提前终止", rnd)
                result.rounds = rnd - 1
                break

            # 评估全部候选，选最优改进者
            best_cand = None
            best_cand_ric = float("nan")
            for cand in candidates:
                try:
                    m = await evaluate_fn(cand)
                except Exception as exc:  # noqa: BLE001
                    m = {"rank_ic": float("nan"), "ic": float("nan")}
                    _logger.debug("候选 %s 评估失败: %s", cand, exc)
                ric = map_rank(m.get("rank_ic"))
                step = SearchStep(round=rnd, expression=cand, rank_ic=ric, ic=m.get("ic"))
                step_dict = step.to_dict()
                step_dict.pop("improved", None)
                step_dict.pop("is_best", None)
                step_dict.pop("note", None)
                history.append(step_dict)

                if ric == ric and (best_cand_ric != best_cand_ric or ric > best_cand_ric):
                    best_cand_ric = ric
                    best_cand = cand

            # 若最佳候选优于当前 best，采纳
            if best_cand is not None and best_cand_ric == best_cand_ric and \
               (result.best_rank_ic != result.best_rank_ic or best_cand_ric > result.best_rank_ic):
                result._best_expression = best_cand
                result.best_rank_ic = best_cand_ric
                result.best_ic = await _eval_ic(evaluate_fn, best_cand)

            # 记录每一步的 is_best 标记
            for st in history[-len(candidates):]:
                if st["expression"] == result._best_expression:
                    st["is_best"] = True
        else:
            # 未提前 break：正常完成全部 rounds
            result.rounds = self.rounds

        result.history = [SearchStep(**h) for h in history]
        _logger.info("round %d 采纳=%s best_rank_ic=%.4f", result.rounds, result._best_expression, result.best_rank_ic)

        result.improved = bool(result.best_rank_ic == result.best_rank_ic and
                               result.seed_rank_ic == result.seed_rank_ic and
                               result.best_rank_ic > result.seed_rank_ic)

        # 3) 可选 val 防泄漏评估
        if val_panel is not None:
            result.val_rank_ic, result.val_ic = await self._val_eval(
                result._best_expression or seed_expr, val_panel, forward_periods, market)

        return result

    # -- val 独立评估 ----------------------------------------------------------
    async def _val_eval(self, expr: str, val_panel: Panel, forward_periods: int, market: str):
        """在独立验证期评估（不参与搜索决策）。"""
        try:
            ev = await _default_evaluate(val_panel, forward_periods, market)
            m = await ev(expr)
            r = map_rank(m.get("rank_ic"))
            return (r if r == r else None), m.get("ic")
        except Exception as exc:  # noqa: BLE001
            _logger.warning("val 评估失败: %s", exc)
            return None, None

    # -- 默认候选生成（真实 LLM 或 mock 变异器） ----------------------------
    def _build_default_search_fn(self, instruction: str) -> SearchFn:
        async def _search(chain: str, seed: str, best: dict):
            # 优先真实 LLM
            if self.provider is not None:
                resp = await self.provider.chat(SEARCH_SYSTEM, chain)
                parsed = parse_expression_response(resp)
                if parsed:
                    return [parsed]
            # 回落：确定性变异器（离线 / 解析失败）
            return mutate_expressions(best.get("expression") or seed, n=6)

        return _search


def map_rank(x) -> float:
    """把可能是 None / NaN 的 rank_ic 规整为 float。"""
    try:
        f = float(x)
        return f if f == f else float("nan")
    except (TypeError, ValueError):
        return float("nan")


async def _eval_ic(evaluate_fn: EvaluateFn, expr: str) -> Optional[float]:
    try:
        m = await evaluate_fn(expr)
        ic = m.get("ic")
        return float(ic) if ic == ic else None
    except Exception:  # noqa: BLE001
        return None
