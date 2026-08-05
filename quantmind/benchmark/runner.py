"""Benchmark 评测编排（对标 AlphaBench ``benchmark/run_benchmark.py``）。

产出「模型 × 任务」评分矩阵：对给定 Provider（可选不同模型/提示范式），在
一组评测任务（T1 generation / T2 ranking+scoring / T4 signal+pairwise）上
给出 0-1 的 reliability / stability / accuracy / overall 分数。

核心用途：告诉研究者「哪个 LLM 配置在因子挖掘的哪个环节可靠/不可靠」，
把 QuantMind 从『能做 IC 计算的工具』升级为『能横向评测 LLM 的框架』。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..ai.provider import LLMProvider, MockProvider
from ..research.factors.alpha_cs import Panel
from .tasks import TaskResult, run_task, T1_GENERATION_INSTRUCTIONS

_logger = logging.getLogger("quantmind.benchmark")

__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "run_benchmark",
    "summarize_matrix",
]


# 默认评测用因子表达式池（T2/T4 样本）
_DEFAULT_EXPRS = [
    "delta(close, 20)",
    "mean(close, 5) / mean(close, 20)",
    "corr(close, volume, 10)",
    "std(close, 20)",
    "-delta(close, 5)",
    "rank(close, 20)",
    "ts_zscore(close, 30)",
    "sma(close, 5)",
]


@dataclass
class BenchmarkConfig:
    """单次评测配置。"""

    tasks: List[str] = field(
        default_factory=lambda: ["T1_generation", "T4_signal", "T4_pairwise",
                                 "T2_ranking", "T2_scoring"])
    t1_instructions: List[dict] = field(default_factory=lambda: T1_GENERATION_INSTRUCTIONS)
    exprs: List[str] = field(default_factory=lambda: list(_DEFAULT_EXPRS))
    top_k: int = 2


@dataclass
class BenchmarkResult:
    """一次评测的完整结果。"""

    label: str = ""                      # 模型/配置标签
    config: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    results: List[TaskResult] = field(default_factory=list)

    def scores_by_task(self) -> Dict[str, Dict[str, float]]:
        """按任务聚合 overall / accuracy。"""
        out: Dict[str, Dict[str, float]] = {}
        for r in self.results:
            item = out.setdefault(r.task, {"overall": [], "accuracy": [], "reliability": []})
            if r.overall():
                item["overall"].append(r.overall())
            if r.accuracy is not None:
                item["accuracy"].append(r.accuracy)
            if r.reliability is not None:
                item["reliability"].append(r.reliability)

        def _mean(v):
            return float(sum(v) / len(v)) if v else 0.0

        return {t: {k: round(_mean(v), 4) for k, v in d.items()} for t, d in out.items()}

    def to_matrix_dict(self) -> Dict[str, object]:
        """供 runner 汇总成矩阵的一行。"""
        by_task = self.scores_by_task()
        row: Dict[str, object] = {"label": self.label}
        for t, scores in by_task.items():
            for k, v in scores.items():
                row[f"{t}.{k}"] = v
        overalls = [s.get("overall", 0.0) for s in by_task.values()]
        row["overall"] = round(sum(overalls) / len(overalls), 4) if overalls else 0.0
        return row


async def run_benchmark(
    provider: Optional[LLMProvider] = None,
    panel: Optional[Panel] = None,
    label: str = "mock",
    config: Optional[BenchmarkConfig] = None,
) -> BenchmarkResult:
    """运行一次完整评测，返回 ``BenchmarkResult``。

    Args:
        provider: 被测 LLM provider；None → MockProvider（离线可跑）。
        panel: 评估用面板（T2/T4 需要真实 IC 标签；缺省会退化，accuracy 受限于无法对照）。
        label: 结果标签（如 "gpt-4.1" / "deepseek-v3" / "mock"）。
        config: 评测配置（任务/样本）。

    Returns:
        ``BenchmarkResult``，可用 :meth:`~BenchmarkResult.to_matrix_dict` 汇总。
    """
    provider = provider or MockProvider()
    config = config or BenchmarkConfig()
    out = BenchmarkResult(label=label, config=config)

    # T1：对每组指令跑一次生成
    for inst in config.t1_instructions:
        r = await run_task("T1_generation", provider, panel, sample=inst)
        out.results.append(r)

    # T2/T4：需要面板做真实 IC 对照；无面板时跳过（避免无意义分数）
    if panel is None or panel.close.shape[1] < 2:
        _logger.warning("未提供 ≥2 标的面板，跳过 T2/T4 评测（无法对照真实 IC）")
        return out

    exprs = config.exprs[: len(config.exprs)]
    if len(exprs) < 2:
        return out

    # T4 信号分类：取前几个表达式
    for e in exprs[:4]:
        out.results.append(await run_task("T4_signal", provider, panel, exprs=[e]))

    # T4 两两选择：相邻两两
    for i in range(0, min(len(exprs) - 1, 4)):
        out.results.append(await run_task("T4_pairwise", provider, panel,
                                          exprs=[exprs[i], exprs[i + 1]]))

    # T2 ranking
    out.results.append(await run_task("T2_ranking", provider, panel,
                                      exprs=exprs[:5], top_k=config.top_k))

    # T2 scoring
    for e in exprs[:3]:
        out.results.append(await run_task("T2_scoring", provider, panel, exprs=[e]))

    return out


def summarize_matrix(results: List[BenchmarkResult]) -> List[Dict[str, object]]:
    """把多个 ``BenchmarkResult`` 汇总成「模型 × 任务」评分矩阵。"""
    return [r.to_matrix_dict() for r in results]
