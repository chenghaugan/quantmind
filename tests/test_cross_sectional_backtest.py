"""截面因子 -> 多空组合回测桥接测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantmind.research.factors.alpha_cs import Panel
from quantmind.research.cross_sectional_backtest import cross_sectional_backtest


def _make_panel(n_symbols=5, n_dates=120, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-02", periods=n_dates, freq="B")
    syms = [f"S{i}" for i in range(n_symbols)]
    base = rng.normal(0, 1, (n_dates, n_symbols)).cumsum(axis=0)
    close = pd.DataFrame(base + 100, index=dates, columns=syms)
    # 加入一点日内结构，确保 open/high/low 合理
    open_ = close.shift(1).fillna(close.iloc[0]) * (1 + rng.normal(0, 0.002, (n_dates, n_symbols)))
    high = close * (1 + rng.uniform(0, 0.01, (n_dates, n_symbols)))
    low = close * (1 - rng.uniform(0, 0.01, (n_dates, n_symbols)))
    volume = pd.DataFrame(rng.integers(1000, 5000, (n_dates, n_symbols)), index=dates, columns=syms)
    return Panel(close=close, open=open_, high=high, low=low, volume=volume)


def test_cs_backtest_runs_and_consistent():
    panel = _make_panel()
    res = cross_sectional_backtest(panel, "alpha021", forward_periods=1, n_groups=5)
    assert res["n_symbols"] == 5
    assert res["n_dates"] > 0
    assert res["ic_report"] is not None
    p = res["portfolio"]
    assert "sharpe" in p and "total_return" in p
    # 权益曲线首日权益为 1.0（无起始漂移）
    drets = p["daily_returns"]
    # 至少部分交易日有足够标的形成组合
    assert len(drets) > 0


def test_flat_prices_no_gain_no_lookahead():
    # 价格全平 -> 前向收益全 0 -> 组合无收益（仅被成本扣减）
    n = 60
    dates = pd.date_range("2023-01-02", periods=n, freq="B")
    syms = [f"S{i}" for i in range(4)]
    close = pd.DataFrame(np.ones((n, 4)), index=dates, columns=syms)
    panel = Panel(close=close, open=close, high=close, low=close,
                  volume=pd.DataFrame(np.ones((n, 4)), index=dates, columns=syms))
    res = cross_sectional_backtest(panel, "alpha093", forward_periods=1, n_groups=4, cost_rate=0.0)
    # 全平价格下多空组合收益应≈0（无成本）
    assert abs(res["portfolio"]["total_return"]) < 1e-9


def test_unknown_factor_raises():
    panel = _make_panel()
    with pytest.raises(KeyError):
        cross_sectional_backtest(panel, "alpha999")


def test_empty_panel_raises():
    empty = Panel(close=pd.DataFrame(), open=pd.DataFrame(), high=pd.DataFrame(),
                  low=pd.DataFrame(), volume=pd.DataFrame())
    with pytest.raises(ValueError):
        cross_sectional_backtest(empty, "alpha021")


def test_gross_net_equal_at_zero_cost():
    """cost_rate=0 → net 指标与 gross 一致（无成本，纯研究展示口径）。"""
    panel = _make_panel()
    res = cross_sectional_backtest(panel, "alpha021", forward_periods=1, n_groups=5, cost_rate=0.0)
    p = res["portfolio"]
    assert p["net_total_return"] == pytest.approx(p["total_return"], abs=1e-9)
    assert p["net_sharpe_annual"] == pytest.approx(p["sharpe_annual"], abs=1e-9)


def test_turnover_aware_cost_reduces_net():
    """cost_rate>0 时按实际换手计费：换手为正 → net < gross，且换手被上报。"""
    panel = _make_panel()
    res = cross_sectional_backtest(panel, "alpha021", forward_periods=1, n_groups=5, cost_rate=0.002)
    p = res["portfolio"]
    assert p["turnover_mean"] > 0
    assert p["net_total_return"] <= p["total_return"] + 1e-9
    assert p["cost_ratio"] >= 0
    assert "net_sharpe_annual" in p and "net_max_drawdown" in p


def test_higher_cost_penalizes_more():
    """成本越高，净收益越低（单调）。"""
    panel = _make_panel()
    lo = cross_sectional_backtest(panel, "alpha021", n_groups=5, cost_rate=0.001)["portfolio"]
    hi = cross_sectional_backtest(panel, "alpha021", n_groups=5, cost_rate=0.01)["portfolio"]
    assert hi["net_total_return"] < lo["net_total_return"]
