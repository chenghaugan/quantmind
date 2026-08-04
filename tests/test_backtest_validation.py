"""回测验证套件测试。"""
from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd
import pytest

from quantmind.backtest.validation import (
    monte_carlo,
    shuffle_noise_test,
    cost_sensitivity,
    regime_split_test,
    static_lookahead_scan,
)
from quantmind.strategy.dual_ma import DualMaStrategy

from .helpers import load_bars


def _bars():
    return asyncio.run(load_bars())


def _vt():
    return "rb0.SHFE"


# ---- 蒙特卡洛 ----
def test_monte_carlo_reproducible():
    rets = pd.Series(np.random.default_rng(1).normal(0.0002, 0.01, 500))
    a = monte_carlo(rets, n_simulations=200, seed=7, method="bootstrap")
    b = monte_carlo(rets, n_simulations=200, seed=7, method="bootstrap")
    assert a == b  # 同 seed 完全一致


def test_monte_carlo_fields():
    rets = pd.Series(np.random.default_rng(1).normal(0.0002, 0.01, 500))
    r = monte_carlo(rets, n_simulations=100, seed=1)
    for k in ("n_simulations", "mean_total_return", "pct5_total_return", "pct95_total_return",
              "mean_sharpe", "mean_max_drawdown", "prob_positive", "seed"):
        assert k in r
    assert r["n_simulations"] == 100
    assert r["pct5_total_return"] <= r["pct95_total_return"]


def test_monte_carlo_equity_curve_input():
    curve = [{"date": f"2024-01-{i+1:02d}", "equity": 1_000_000 * (1 + i * 0.001)}
             for i in range(30)]
    r = monte_carlo(curve, n_simulations=50, seed=3)
    assert r["n_simulations"] == 50


# ---- 置乱噪声测试 ----
def test_shuffle_noise_runs():
    bars = _bars()
    if len(bars) < 50:
        pytest.skip("样本不足")
    vt = _vt()
    r = shuffle_noise_test(bars, DualMaStrategy, {"size": 1, "max_pos": 1.0},
                           vt, n_shuffles=30, seed=42, sizes={vt: 1.0})
    assert "real_total_return" in r
    assert "no_lookahead" in r
    assert r["n_shuffles"] > 0


# ---- 成本敏感性 ----
def test_cost_sensitivity_monotonic():
    bars = _bars()
    vt = _vt()
    if len(bars) < 50:
        pytest.skip("样本不足")
    data = {vt: bars}
    r = cost_sensitivity(DualMaStrategy, data, vt,
                         cost_multipliers=(0.0, 1.0, 4.0),
                         base_commission=0.0002, sizes={vt: 1.0})
    rows = r["rows"]
    # 成本越高，Sharpe 应不升（单调非递增的稳健检查：至少 4x 不高于 0x 太多）
    assert rows[0]["multiplier"] == 0.0
    assert rows[-1]["multiplier"] == 4.0
    assert "robust_to_cost" in r
    # 成本 0 档的 Sharpe 应 >= 高成本档（允许相等）
    assert rows[0]["sharpe"] >= rows[-1]["sharpe"] - 1e-9


# ---- 状态分割 ----
def test_regime_split_structure():
    bars = _bars()
    vt = _vt()
    data = {vt: bars}
    r = regime_split_test(bars, DualMaStrategy, {"size": 1, "max_pos": 1.0},
                          vt, sizes={vt: 1.0})
    assert "regimes" in r
    if r["regimes"]:
        for seg in r["regimes"]:
            for k in ("regime", "sharpe", "total_return", "drawdown", "n_bars"):
                assert k in seg
        assert "robust_across_regimes" in r


# ---- 前视静态扫描 ----
def test_static_lookahead_scan_detects():
    bad = (
        "def sig(close):\n"
        "    return close.shift(-1)\n"
    )
    violations = static_lookahead_scan(bad)
    assert any("shift" in v for v in violations)


def test_static_lookahead_scan_detects_pct_change():
    bad = (
        "def sig(close):\n"
        "    fwd = close.pct_change().shift(-5)\n"
        "    return fwd\n"
    )
    violations = static_lookahead_scan(bad)
    assert any("pct_change" in v or "shift" in v for v in violations)


def test_static_lookahead_scan_clean():
    good = (
        "def sig(close):\n"
        "    return close.rolling(20).mean()\n"
    )
    # rolling(20) 不引用未来
    violations = static_lookahead_scan(good)
    assert violations == []
