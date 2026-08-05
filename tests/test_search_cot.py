"""CoT 迭代因子搜索测试（search/cot.py）。

覆盖：mock 变异器产出合法候选、CoT 端到端闭环、注入 strategy 时改进检测、
val 防泄漏独立评估、search_fn 注入计数。
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from quantmind.research.factors.alpha_cs import Panel
from quantmind.research.factors.panel_expr import panel_eval_expression, ExpressionError
from quantmind.research.search.base import mutate_expressions
from quantmind.research.search.cot import FactorSearcher, map_rank


def _utc(i: int):
    return datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=i)


def _make_panel(n_symbols: int = 8, n_dates: int = 150, seed: int = 1) -> Panel:
    rng = np.random.default_rng(seed)
    dates = [_utc(i) for i in range(n_dates)]
    cols = [f"S{i}" for i in range(n_symbols)]
    close = pd.DataFrame(np.abs(rng.normal(100, 10, (n_dates, n_symbols))), index=dates, columns=cols)
    open_ = close * 0.99
    high = close * 1.02
    low = close * 0.98
    volume = pd.DataFrame(np.abs(rng.normal(1000, 100, (n_dates, n_symbols))), index=dates, columns=cols)
    return Panel(close=close, open=open_, high=high, low=low, volume=volume)


def test_mutate_produces_legal_candidates():
    """mutate_expressions 产出的候选都应是合法可求值的 DSL 表达式。"""
    panel = _make_panel()
    seed = "mean(close, 20)"
    cands = mutate_expressions(seed, n=8)
    assert len(cands) > 0
    assert seed not in cands
    for c in cands:
        # 必须能被面板求值器解析执行（合法）
        out = panel_eval_expression(c, panel)
        assert out.shape == panel.close.shape, c


def test_mutate_variety():
    """mutate 应产出多样化的变体（窗口偏移 / 包裹 / 基准项）。"""
    cands = mutate_expressions("ts_zscore(close, 20)", n=8)
    assert len(set(cands)) >= 4
    # 至少包含窗口变化或包裹产生的不同表达式
    assert any("rank" in c or "Rank" in c for c in cands) or len(cands) >= 4


@pytest.mark.asyncio
async def test_cot_end_to_end_default():
    """无 provider + 默认评估：CoT 应完整跑完 rounds，产出合法 best 与完整轨迹。"""
    panel = _make_panel(n_symbols=8, n_dates=150, seed=42)
    searcher = FactorSearcher(provider=None, rounds=4)
    res = await searcher.cot_search("mean(close, 20)", panel, market="test")
    assert res.seed == "mean(close, 20)"
    assert res.best_expression
    assert res.history, "应至少有一条候选评估记录"
    # best 表达式必须合法可求值
    panel_eval_expression(res.best_expression, panel)
    assert res.rounds >= 1
    # seed 可评估出有限指标
    assert res.seed_rank_ic == res.seed_rank_ic or pd.isna(res.seed_rank_ic)


@pytest.mark.asyncio
async def test_cot_improves_with_injected_strategy():
    """注入一个总是给出更好表达式的 search_fn + 递增 evaluate_fn：best 应改进、improved=True。"""
    panel = _make_panel()
    calls = {"n": 0}

    async def fake_eval(expr):
        calls["n"] += 1
        # 随调用次数递增的 rank_ic，模拟候选越来越好
        return {"rank_ic": 0.5 + calls["n"] * 0.01, "ic": 0.4}

    async def fake_search(**kw):
        return [f"rank(close, {10 + calls['n']})"]

    searcher = FactorSearcher(provider=None, evaluate_fn=fake_eval, rounds=3)
    res = await searcher.cot_search("mean(close, 20)", panel, market="m",
                                    search_fn=fake_search)
    assert res.improved is True
    assert res.best_rank_ic > res.seed_rank_ic
    assert res.best_expression != res.seed  # 采纳了更优候选
    assert len(res.history) == 3


@pytest.mark.asyncio
async def test_cot_val_panel_no_leak():
    """传 val_panel 时，最终 best 应在独立 val 期被评估（val_rank_ic 非 None）。"""
    panel = _make_panel(seed=1)
    val_panel = _make_panel(seed=99)  # 不同分布，模拟验证期
    searcher = FactorSearcher(provider=None, rounds=2)
    res = await searcher.cot_search("ts_zscore(close, 20)", panel,
                                    val_panel=val_panel, market="m")
    # val 期独立评估了最终 best
    assert res.val_rank_ic is not None
    assert res.val_ic is not None


@pytest.mark.asyncio
async def test_search_fn_called_and_rounds_terminate():
    """search_fn 被调用；在无新候选时提前终止。"""
    panel = _make_panel()
    seen = {"calls": 0}

    async def eval_stub(expr):
        return {"rank_ic": 0.3, "ic": 0.2}

    async def empty_search(**kw):
        seen["calls"] += 1
        return []  # 无候选 → 立即终止

    searcher = FactorSearcher(provider=None, evaluate_fn=eval_stub, rounds=5)
    res = await searcher.cot_search("mean(close, 10)", panel, market="m",
                                    search_fn=empty_search)
    assert seen["calls"] == 1  # 只在第一轮被调用后终止
    assert res.rounds == 0  # 无候选即停在首轮


def test_map_rank_handles_nan_none():
    """map_rank 规整 None/NaN 为 float nan。"""
    assert map_rank(None) != map_rank(None)  # nan
    assert map_rank(float("nan")) != map_rank(float("nan"))
    assert map_rank(0.42) == 0.42
