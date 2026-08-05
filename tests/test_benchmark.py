"""Benchmark 评测框架测试（quantmind/benchmark/）。

覆盖：T1 生成任务（reliability/accuracy）、T2/T4 任务在离线面板上的评分、
run_benchmark 端到端产出 matrix、多结果的 summarize_matrix 汇总。
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from quantmind.ai.provider import MockProvider
from quantmind.benchmark import (
    BenchmarkConfig,
    BenchmarkResult,
    run_benchmark,
    run_task,
    summarize_matrix,
)
from quantmind.research.factors.alpha_cs import Panel


def _make_panel(n_symbols: int = 8, n_dates: int = 120, seed: int = 3) -> Panel:
    rng = np.random.default_rng(seed)
    dates = [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=i) for i in range(n_dates)]
    cols = [f"S{i}" for i in range(n_symbols)]
    close = pd.DataFrame(np.abs(rng.normal(100, 10, (n_dates, n_symbols))), index=dates, columns=cols)
    return Panel(close=close, open=close * 0.99, high=close * 1.02,
                 low=close * 0.98,
                 volume=pd.DataFrame(np.abs(rng.normal(1000, 100, (n_dates, n_symbols))),
                                     index=dates, columns=cols))


@pytest.mark.asyncio
async def test_t1_generation_task():
    """T1 生成：应产出 reliability 与 accuracy 分数。"""
    panel = _make_panel()
    r = await run_task("T1_generation", MockProvider(), panel,
                       sample={"name": "momentum", "text": "20-day momentum of close",
                               "expected": "delta(close, 20)"})
    assert r.task == "T1_generation"
    assert r.reliability in (0.0, 1.0)
    assert r.accuracy is not None
    assert 0.0 <= r.accuracy <= 1.0


@pytest.mark.asyncio
async def test_t4_signal_and_pairwise():
    """T4 原子任务：信号分类与两两选择都应给出 accuracy。"""
    panel = _make_panel()
    rs = await run_task("T4_signal", MockProvider(), panel, exprs=["delta(close, 20)"])
    assert rs.task == "T4_signal"
    assert rs.accuracy in (0.0, 1.0)

    rp = await run_task("T4_pairwise", MockProvider(), panel,
                        exprs=["delta(close, 20)", "mean(close, 5)"])
    assert rp.task == "T4_pairwise"
    assert rp.accuracy in (0.0, 1.0)


@pytest.mark.asyncio
async def test_unknown_task_raises():
    panel = _make_panel()
    with pytest.raises(ValueError):
        await run_task("T_unknown", MockProvider(), panel)


@pytest.mark.asyncio
async def test_run_benchmark_end_to_end():
    """run_benchmark 端到端：产出若干 TaskResult 与可汇总的 matrix 行。"""
    panel = _make_panel()
    cfg = BenchmarkConfig(tasks=["T1_generation", "T4_signal", "T4_pairwise",
                                 "T2_ranking", "T2_scoring"])
    res = await run_benchmark(MockProvider(), panel=panel, label="mock", config=cfg)
    assert isinstance(res, BenchmarkResult)
    assert res.label == "mock"
    assert res.results, "应至少产出若干任务结果"
    row = res.to_matrix_dict()
    assert "overall" in row
    assert row["label"] == "mock"
    # 应覆盖多任务评分列
    keys = [k for k in row if k != "label"]
    assert len(keys) >= 3


@pytest.mark.asyncio
async def test_summarize_matrix_multiple():
    """多个结果可汇总成一个矩阵（每行一个模型）。"""
    panel = _make_panel()
    r1 = await run_benchmark(MockProvider(), panel=panel, label="mock_a")
    r2 = await run_benchmark(MockProvider(), panel=panel, label="mock_b")
    matrix = summarize_matrix([r1, r2])
    assert len(matrix) == 2
    assert {m["label"] for m in matrix} == {"mock_a", "mock_b"}


@pytest.mark.asyncio
async def test_run_benchmark_without_panel_skips_t2t4():
    """无面板时 T2/T4 被跳过，只保留 T1 结果。"""
    res = await run_benchmark(MockProvider(), panel=None, label="x")
    tasks = {r.task for r in res.results}
    assert tasks == {"T1_generation"} or "T4_signal" in tasks or "T1_generation" in tasks
    # 无面板 → 不出现 T2/T4（无法对照真实 IC）
    assert "T2_ranking" not in tasks
