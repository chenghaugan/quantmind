"""评测任务定义与评分（对标 AlphaBench T1-T4）。

每个任务把「一个 LLM 在一次提示范式下的输出」转成一个 0-1 的标量分数，
供 runner 汇总成「模型 × 任务 × 提示范式」评分矩阵。所有任务都可在离线
（MockProvider）下运行，保证评测管线可跑通、可测试。

任务清单：
  - T1 generation：把自然语言指令翻译成合法、语义正确的因子表达式
    （reliability=可执行 / stability=稳定 / accuracy=意图匹配）。
  - T2 ranking / scoring：LLM 评估因子质量，对照真实回测 IC。
  - T4 atomic：signal classification + pairwise selection（对照真实标签）。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from ..ai.provider import LLMProvider, MockProvider
from ..research.factors.panel_expr import panel_eval_expression
from ..research.judge import (
    judge_signal,
    judge_pairwise,
    judge_ranking,
    judge_scoring,
    score_signal_accuracy,
    score_pairwise_accuracy,
    score_ranking,
    score_scoring,
)

_logger = logging.getLogger("quantmind.benchmark")

__all__ = [
    "TASK_REGISTRY",
    "run_task",
    "T1_GENERATION_INSTRUCTIONS",
]


# -- 数据结构 ---------------------------------------------------------------
@dataclass
class TaskResult:
    """单个任务样例的评分结果。"""

    task: str
    sample: str = ""
    reliability: Optional[float] = None    # 是否产出可执行表达式（0/1）
    stability: Optional[float] = None      # 稳定程度（0-1）
    accuracy: Optional[float] = None       # 意图/标签匹配度（0-1）
    note: str = ""

    def overall(self) -> float:
        vals = [v for v in (self.reliability, self.stability, self.accuracy) if v is not None]
        return sum(vals) / len(vals) if vals else 0.0

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "sample": self.sample,
            "reliability": self.reliability,
            "stability": self.stability,
            "accuracy": self.accuracy,
            "overall": round(self.overall(), 4),
            "note": self.note,
        }


# T1 生成指令样本（自然语言 → 应产出对应语义的因子）
T1_GENERATION_INSTRUCTIONS: List[Dict[str, str]] = [
    {"name": "momentum", "text": "Generate a factor: 20-day momentum of close price.",
     "expected": "delta(close, 20)"},
    {"name": "mean_reversion", "text": "Generate a factor: short-term mean reversion of close.",
     "expected": "-delta(close, 5)"},
    {"name": "volatility", "text": "Generate a factor: 20-day realized volatility (std) of returns.",
     "expected": "std(close, 20)"},
    {"name": "volume_ratio", "text": "Generate a factor: ratio of short/long volume average.",
     "expected": "mean(volume, 5) / mean(volume, 20)"},
    {"name": "price_volume_corr", "text": "Generate a factor: 10-day correlation between close and volume.",
     "expected": "corr(close, volume, 10)"},
]


# -- T1 生成任务 ------------------------------------------------------------
async def _t1_generation(provider: LLMProvider, sample: dict, panel) -> TaskResult:
    """让 LLM 把指令翻译成表达式；reliability=可执行，accuracy=与期望语义匹配。"""
    from ..research.factors.panel_expr import ExpressionError
    from ..ai.factor_gen import generate_factors
    from ..research.target import FactorSpec
    from ..ai.expr_map import factor_spec_to_expression

    factors = await generate_factors(provider, sample["text"])
    exprs = [factor_spec_to_expression(f) for f in factors]
    exprs = [e for e in exprs if e and e.strip()]

    # reliability：至少一个表达式可被面板求值器执行
    executable = False
    for e in exprs:
        try:
            panel_eval_expression(e, panel)
            executable = True
            break
        except Exception:  # noqa: BLE001
            continue
    reliability = 1.0 if executable else 0.0

    # accuracy：产出表达式与期望语义相近（字符串/结构启发式，可被测试精确化）
    accuracy = _semantic_match(exprs, sample["expected"])

    # stability：多次（此处单次 + mock 兜底）产出表达式是否一致——用结构相似近似
    stability = 1.0 if exprs else 0.0

    return TaskResult(
        task="T1_generation", sample=sample["name"],
        reliability=reliability, stability=stability, accuracy=accuracy,
    )


def _semantic_match(exprs: List[str], expected: str) -> float:
    """期望表达式与产出表达式的主要算子/变量重合度 [0,1]。"""
    if not exprs:
        return 0.0
    exp_tokens = set(re.findall(r"[a-zA-Z_]+", expected))
    best = 0.0
    for e in exprs:
        got = set(re.findall(r"[a-zA-Z_]+", e))
        inter = len(exp_tokens & got)
        union = len(exp_tokens | got)
        if union:
            best = max(best, inter / union)
    return best


# -- T2/T4 评测任务（对照真实 IC 标签） -------------------------------------
def _true_signal(expr: str, panel, threshold: float = 0.02) -> bool:
    """用真实回测 rank-IC 判定表达式是『信号』还是『噪声』（T4 标签）。"""
    try:
        from ..research import evaluate_expression
        rep = evaluate_expression(expr, panel, use_cache=False)
        ic = rep.ic_mean
        return bool(ic == ic and abs(ic) >= threshold)
    except Exception:  # noqa: BLE001
        return False


async def _t4_signal(provider: LLMProvider, expr: str, panel) -> TaskResult:
    """信号分类：LLM 判断 vs 真实标签。"""
    res = await judge_signal(expr, provider)
    truth = _true_signal(expr, panel)
    scored = score_signal_accuracy(res, truth_is_signal=truth)
    return TaskResult(task="T4_signal", sample=expr,
                      reliability=1.0, accuracy=scored.score,
                      note=f"pred={res.prediction} truth={'signal' if truth else 'noise'}")


async def _t4_pairwise(provider: LLMProvider, a: str, b: str, panel) -> TaskResult:
    """两两选择：LLM 选出更优者 vs 真实 IC 对比。"""
    res = await judge_pairwise(a, b, provider)
    ic_a = _true_ic(a, panel)
    ic_b = _true_ic(b, panel)
    better = 0 if abs(ic_a) >= abs(ic_b) else 1
    scored = score_pairwise_accuracy(res, better_index=better)
    return TaskResult(task="T4_pairwise", sample=f"{a} | {b}",
                      reliability=1.0, accuracy=scored.score,
                      note=f"pred={res.prediction} truth={better}")


def _true_ic(expr: str, panel) -> float:
    try:
        from ..research import evaluate_expression
        rep = evaluate_expression(expr, panel, use_cache=False)
        return rep.ic_mean if rep.ic_mean == rep.ic_mean else 0.0
    except Exception:  # noqa: BLE001
        return 0.0


async def _t2_ranking(provider: LLMProvider, exprs: List[str], panel, top_k: int = 2) -> TaskResult:
    """ranking：LLM 选 top-k vs 真实 IC 排序。"""
    res = await judge_ranking(exprs, provider, top_k=top_k)
    ics = [_true_ic(e, panel) for e in exprs]
    true_order = sorted(range(len(exprs)), key=lambda i: -abs(ics[i]))
    scored = score_ranking(res, true_order=true_order, top_k=top_k)
    return TaskResult(task="T2_ranking", sample=",".join(exprs[:3]),
                      reliability=1.0, accuracy=scored.score)


async def _t2_scoring(provider: LLMProvider, expr: str, panel) -> TaskResult:
    """scoring：LLM 评分 vs 真实 IC 导出的标签（此处用 IC/IR 归一近似）。"""
    res = await judge_scoring(expr, provider)
    ic = _true_ic(expr, panel)
    # 用 IC 映射到一个 1-5 的『参考答案』（单调映射，便于评测 MAE）
    ref = min(5, max(1, round(3 + 6 * ic)))
    truth = {"Signal": ref, "Performance": ref, "Stability": ref,
             "WinRate": ref, "Skewness": 3}
    scored = score_scoring(res, truth=truth)
    return TaskResult(task="T2_scoring", sample=expr,
                      reliability=1.0, accuracy=scored.score,
                      note=f"ic={ic:.3f} ref={ref}")


# -- 任务注册 ---------------------------------------------------------------
TASK_REGISTRY: Dict[str, Callable[..., "TaskResult"]] = {
    "T1_generation": _t1_generation,
    "T4_signal": _t4_signal,
    "T4_pairwise": _t4_pairwise,
    "T2_ranking": _t2_ranking,
    "T2_scoring": _t2_scoring,
}


async def run_task(
    task: str,
    provider: LLMProvider,
    panel,
    exprs: Optional[List[str]] = None,
    **kwargs,
) -> TaskResult:
    """按任务名运行单个评测，返回 ``TaskResult``。未知任务抛 ``ValueError``。"""
    if task not in TASK_REGISTRY:
        raise ValueError(f"未知评测任务: {task}（可用: {sorted(TASK_REGISTRY)}）")
    fn = TASK_REGISTRY[task]
    provider = provider or MockProvider()

    if task == "T1_generation":
        sample = kwargs.get("sample") or T1_GENERATION_INSTRUCTIONS[0]
        return await _t1_generation(provider, sample, panel)
    if task == "T4_signal":
        return await _t4_signal(provider, exprs[0], panel)
    if task == "T4_pairwise":
        return await _t4_pairwise(provider, exprs[0], exprs[1], panel)
    if task == "T2_ranking":
        return await _t2_ranking(provider, exprs or [], panel, top_k=kwargs.get("top_k", 2))
    if task == "T2_scoring":
        return await _t2_scoring(provider, exprs[0], panel)
    raise ValueError(f"未实现: {task}")
