"""进化（EA）与树状思维（ToT）因子搜索测试（search/ea.py + search/tot.py）。

覆盖：算法注册表按名创建、EA 种群迭代闭环产出合法 best、ToT 递归剪枝闭环、
与 CoT 统一 BaseAlgo 契约（可注入 evaluate_fn / search_fn 做确定性控制）。
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from quantmind.research.factors.alpha_cs import Panel
from quantmind.research.factors.panel_expr import panel_eval_expression
from quantmind.research.search.base import list_algos, create_algo
from quantmind.research.search.ea import EASearcher
from quantmind.research.search.tot import ToTSearcher


def _utc(i: int):
    return datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=i)


def _make_panel(n_symbols: int = 8, n_dates: int = 120, seed: int = 7) -> Panel:
    rng = np.random.default_rng(seed)
    dates = [_utc(i) for i in range(n_dates)]
    cols = [f"S{i}" for i in range(n_symbols)]
    close = pd.DataFrame(np.abs(rng.normal(100, 10, (n_dates, n_symbols))), index=dates, columns=cols)
    open_ = close * 0.99
    high = close * 1.02
    low = close * 0.98
    volume = pd.DataFrame(np.abs(rng.normal(1000, 100, (n_dates, n_symbols))), index=dates, columns=cols)
    return Panel(close=close, open=open_, high=high, low=low, volume=volume)


def _inc_eval(calls: dict):
    async def _ev(expr):
        calls["n"] += 1
        return {"rank_ic": 0.3 + calls["n"] * 0.005, "ic": 0.2}
    return _ev


def test_registry_shape():
    """注册表应含 co/ea/tot，且能按名创建。"""
    assert {"co", "ea", "tot"} <= set(list_algos())
    assert isinstance(create_algo("ea"), EASearcher)
    assert isinstance(create_algo("tot"), ToTSearcher)
    with pytest.raises(ValueError):
        create_algo("nonexistent")


@pytest.mark.asyncio
async def test_ea_end_to_end_default():
    """EA 无 provider + 默认评估：种群迭代完整跑完，best 合法，轨迹齐全。"""
    panel = _make_panel()
    searcher = EASearcher(provider=None, generations=2, pop_size=4, seed_count=1)
    res = await searcher.run("mean(close, 20)", panel, market="m")
    assert res.seed == "mean(close, 20)"
    assert res.best_expression
    panel_eval_expression(res.best_expression, panel)  # 必须合法可求值
    assert res.history, "应至少有一条候选评估记录"
    assert res.rounds >= 1


@pytest.mark.asyncio
async def test_ea_improves_with_injected_eval():
    """注入递增评估 + 默认变异器：best 应优于 seed（improved=True）。"""
    panel = _make_panel()
    calls = {"n": 0}
    searcher = EASearcher(provider=None, evaluate_fn=_inc_eval(calls), generations=2, pop_size=4)
    res = await searcher.run("ts_zscore(close, 20)", panel, market="m")
    assert res.improved is True
    assert res.best_rank_ic >= res.seed_rank_ic
    assert res.history


@pytest.mark.asyncio
async def test_tot_end_to_end_default():
    """ToT 无 provider + 默认评估：分支剪枝闭环，best 合法。"""
    panel = _make_panel()
    searcher = ToTSearcher(provider=None, depth=2, branch=2, survivors=1)
    res = await searcher.run("delta(close, 5)", panel, market="m")
    assert res.seed == "delta(close, 5)"
    assert res.best_expression
    panel_eval_expression(res.best_expression, panel)
    assert res.history


@pytest.mark.asyncio
async def test_tot_improves_with_injected_eval():
    """ToT 注入递增评估：best 应改进。"""
    panel = _make_panel()
    calls = {"n": 0}
    searcher = ToTSearcher(provider=None, evaluate_fn=_inc_eval(calls), depth=2, branch=2, survivors=1)
    res = await searcher.run("delta(close, 5)", panel, market="m")
    assert res.improved is True
    assert res.best_rank_ic >= res.seed_rank_ic


@pytest.mark.asyncio
async def test_algo_uniform_contract():
    """三种算法都遵循 BaseAlgo.run 契约，返回 SearchResult。"""
    panel = _make_panel()
    kw = {
        "co": {"rounds": 1},
        "ea": {"generations": 1, "pop_size": 4},
        "tot": {"depth": 1, "branch": 2},
    }
    for name in ["co", "ea", "tot"]:
        algo = create_algo(name, provider=None, **kw[name])
        res = await algo.run("mean(close, 10)", panel, market="m")
        assert hasattr(res, "best_expression")
        assert hasattr(res, "history")
        assert hasattr(res, "improved")
        # to_dict 可序列化（前端/API 用）
        d = res.to_dict()
        assert "best_expression" in d and "history" in d
