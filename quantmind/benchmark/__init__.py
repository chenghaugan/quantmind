"""Benchmark 评测包：对 LLM 在因子挖掘各环节做可复现的横向评测。

对标 AlphaBench（T1-T4）：
  - T1 generation：自然语言 → 可执行因子表达式（reliability/stability/accuracy）
  - T2 ranking / scoring：LLM 评估因子质量（对照真实回测 IC）
  - T4 atomic：signal classification + pairwise selection

用法::

    from quantmind.benchmark import run_benchmark, summarize_matrix
    from quantmind.ai.provider import mock provider...

    res = await run_benchmark(provider=p, panel=panel, label="deepseek")
    row = res.to_matrix_dict()
"""
from .tasks import (
    TaskResult,
    TASK_REGISTRY,
    run_task,
    T1_GENERATION_INSTRUCTIONS,
)
from .runner import (
    BenchmarkConfig,
    BenchmarkResult,
    run_benchmark,
    summarize_matrix,
)

__all__ = [
    "TaskResult",
    "TASK_REGISTRY",
    "run_task",
    "T1_GENERATION_INSTRUCTIONS",
    "BenchmarkConfig",
    "BenchmarkResult",
    "run_benchmark",
    "summarize_matrix",
]
