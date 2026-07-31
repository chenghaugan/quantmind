"""截面 Alpha 因子（alpha_cs）与截面评估的测试。"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import pytest

from quantmind.research.factors.alpha_cs import (
    Panel, compute_alpha_cross_sectional, list_alpha_cs, a093,
)
from quantmind.research.factors.wq import _rank_cs
from quantmind.research.evaluator import FactorEvaluator
from quantmind.core.object import BarData
from quantmind.core.constant import Exchange, Interval


def _utc(d: int):
    return datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=d)


def test_rank_cs_cross_sectional():
    """_rank_cs 应对每行跨列做百分位排名（最大→1，最小→最小分位，且单调）。"""
    df = pd.DataFrame({"A": [1.0], "B": [2.0], "C": [3.0]}, index=[_utc(0)])
    out = _rank_cs(df)
    row = out.loc[_utc(0)]
    assert row["A"] < row["B"] < row["C"]   # 跨标单调
    assert row["C"] == 1.0                   # 最大值分位=1


def _make_panel(n_symbols: int = 4, n_dates: int = 60, seed: int = 0) -> Panel:
    rng = np.random.default_rng(seed)
    dates = [_utc(i) for i in range(n_dates)]
    cols = [f"S{i}" for i in range(n_symbols)]
    close = pd.DataFrame(rng.normal(0, 1, (n_dates, n_symbols)), index=dates, columns=cols)
    open_ = close * 0.99
    high = close * 1.02
    low = close * 0.98
    volume = pd.DataFrame(rng.normal(1000, 100, (n_dates, n_symbols)), index=dates, columns=cols)
    return Panel(close=close, open=open_, high=high, low=low, volume=volume)


def test_compute_alpha_cs_shapes_and_fillna():
    """面板因子输出形状与面板一致，且已 fillna（无 NaN）。"""
    panel = _make_panel()
    names = ["alpha021", "alpha002", "alpha191_007"]
    res = compute_alpha_cross_sectional(names, panel)
    for n in names:
        assert n in res
        assert res[n].shape == panel.close.shape
        assert res[n].notna().all().all()


def test_a093_matches_rank_of_return():
    """a093 = rank(delay(close,1)-close)；应与手动截面 rank 一致（首行 NaN→0）。"""
    panel = _make_panel()
    out = a093(panel)
    expected = _rank_cs(panel.close.shift(1) - panel.close)
    assert np.allclose(out.values[1:], expected.values[1:], atol=1e-9)


def test_evaluate_panel_returns_finite_ic():
    """随机面板下截面评估应给出有限的 IC 与足够截面样本数。"""
    panel = _make_panel(n_symbols=8, n_dates=120, seed=3)
    ev = FactorEvaluator()
    reports = ev.evaluate_cross_sectional_panel(["alpha093"], panel)
    rep = reports["alpha093"]
    assert rep.n_samples > 50
    assert pd.notna(rep.ic_mean)
    assert abs(rep.ic_mean) < 1.0


def test_panel_lt2_symbols_note():
    """单标的面板应给出“需 ≥2 标的”提示，而非崩溃。"""
    panel = _make_panel(n_symbols=1)
    ev = FactorEvaluator()
    reports = ev.evaluate_cross_sectional_panel(["alpha021"], panel)
    assert reports["alpha021"].note


def test_panel_from_bars():
    """由合成 BarData 构建面板：对齐日期、字段、取值正确。"""
    dates = [_utc(i) for i in range(20)]
    bars_by_symbol = {}
    for s in ["A", "B"]:
        bars = [
            BarData(symbol=s, exchange=Exchange.SHFE, datetime=d, interval=Interval.DAILY,
                    open_price=10 + i, high_price=11 + i, low_price=9 + i,
                    close_price=10.5 + i, volume=1000 + i)
            for i, d in enumerate(dates)
        ]
        bars_by_symbol[s] = bars
    panel = Panel.from_bars(bars_by_symbol)
    assert set(panel.symbols) == {"A", "B"}
    assert panel.close.shape[0] == 20
    assert abs(panel.close.loc[dates[5], "A"] - 15.5) < 1e-9


def test_list_alpha_cs_nonempty():
    names = list_alpha_cs()
    assert "alpha021" in names and "alpha191_007" in names
    assert len(names) >= 40
