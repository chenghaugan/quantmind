"""Barra 式多因子风险归因测试（research/barra.py）。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantmind.research.barra import (
    estimate_factor_returns,
    portfolio_weights_from_signal,
    barra_factor_risk_attribution,
)
from quantmind.research.combine import composite_backtest
from quantmind.research.factors.alpha_cs import Panel


def _synth_inputs(n_dates: int = 90, n_sym: int = 12, seed: int = 7):
    """构造有真实因子结构的合成面板：收益由 f1/f2 驱动，f3 弱。"""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=n_dates, freq="B")
    assets = [f"s{i}" for i in range(n_sym)]
    X1 = pd.DataFrame(rng.normal(size=(n_dates, n_sym)), index=dates, columns=assets)
    X2 = pd.DataFrame(rng.normal(size=(n_dates, n_sym)), index=dates, columns=assets)
    X3 = pd.DataFrame(rng.normal(size=(n_dates, n_sym)), index=dates, columns=assets)
    exposures = {"f1": X1, "f2": X2, "f3": X3}
    b1 = pd.Series(rng.normal(0.0005, 0.01, n_dates), index=dates)
    b2 = pd.Series(rng.normal(-0.0002, 0.008, n_dates), index=dates)
    b3 = pd.Series(rng.normal(0.0001, 0.006, n_dates), index=dates)
    fwd_vals = (0.5 * X1.mul(b1, axis=0) + 0.8 * X2.mul(b2, axis=0)
                + 0.3 * X3.mul(b3, axis=0) + rng.normal(0, 0.001, (n_dates, n_sym)))
    fwd = pd.DataFrame(fwd_vals, index=dates, columns=assets)
    sig = 0.7 * X1 + 0.7 * X2 + 0.1 * X3
    return sig, fwd, exposures, dates, assets


class TestEstimateFactorReturns:
    def test_regression_shape(self):
        sig, fwd, expos, dates, assets = _synth_inputs()
        fr, resid, r2, names = estimate_factor_returns(fwd, expos)
        assert fr.shape[0] == len(dates)
        assert set(names) == {"f1", "f2", "f3"}
        assert set(fr.columns) == {"f1", "f2", "f3", "_market"}
        assert resid.shape == fwd.shape
        assert r2.notna().any()

    def test_drops_constant_exposure(self):
        sig, fwd, expos, dates, assets = _synth_inputs()
        expos["flat"] = pd.DataFrame(5.0, index=dates, columns=assets)  # 无横截面信息
        fr, resid, r2, names = estimate_factor_returns(fwd, expos)
        assert "flat" not in names


class TestPortfolioWeights:
    def test_long_short_leverage_one(self):
        sig, fwd, expos, dates, assets = _synth_inputs()
        w = portfolio_weights_from_signal(sig, n_groups=3, long_short=True)
        # 多空净暴露 ≈ 0（多+空抵消），总杠杆波动约 1
        net = w.sum(axis=1).abs().mean()
        assert net < 0.05
        day = w.iloc[0]
        assert abs(day[day > 0].sum()) > 0.99
        assert abs(day[day < 0].sum() + 1.0) < 1e-9

    def test_long_only(self):
        sig, fwd, expos, dates, assets = _synth_inputs()
        w = portfolio_weights_from_signal(sig, n_groups=3, long_short=False)
        assert (w >= -1e-12).all().all()
        assert abs(w.iloc[0].sum() - 1.0) < 1e-9


class TestBarraAttribution:
    def test_additivity_exact(self):
        """MCTR 之和必须精确等于组合波动（可加性）。"""
        sig, fwd, expos, dates, assets = _synth_inputs()
        res = barra_factor_risk_attribution(sig, fwd, expos)
        s = (sum(f["mctr_vol"] for f in res["factors"])
             + res["specific"]["mctr_vol"] + res["market"]["mctr_vol"])
        assert abs(s - res["total"]["vol"]) < 1e-6
        assert abs(res["additivity"]["closure"]) < 1e-6

    def test_report_fields(self):
        sig, fwd, expos, dates, assets = _synth_inputs()
        res = barra_factor_risk_attribution(sig, fwd, expos)
        assert res["total"]["ann_vol"] is not None
        assert res["diagnostics"]["n_factors"] == 3
        assert len(res["factors"]) == 3
        for f in res["factors"]:
            assert "mctr_vol" in f and "risk_pct" in f and "exposure_mean" in f

    def test_stronger_factor_drives_more_risk(self):
        """f2 收益波动最大且信号权重大 → 通常 MCTR 最大（结构约束下）。"""
        sig, fwd, expos, dates, assets = _synth_inputs()
        res = barra_factor_risk_attribution(sig, fwd, expos)
        mctr = {f["name"]: f["mctr_vol"] for f in res["factors"]}
        assert mctr["f2"] > mctr["f1"]  # f2 真实收益波动远大于 f1

    def test_short_inputs_raises(self):
        sig, fwd, expos, _, _ = _synth_inputs(n_dates=30, n_sym=5)
        sig2 = sig.iloc[:4]  # <5 个公共日 → 报错
        fwd2 = fwd.iloc[:4]
        with pytest.raises(ValueError):
            barra_factor_risk_attribution(sig2, fwd2, expos)


def test_composite_backtest_includes_risk_attribution():
    idx = pd.date_range("2020-01-01", periods=120, freq="D")
    cols = [f"S{i}" for i in range(8)]
    rng = np.random.default_rng(3)
    close = pd.DataFrame(rng.standard_normal((120, 8)).cumsum(0) + 50,
                         index=idx, columns=cols)
    vol = pd.DataFrame(np.abs(rng.standard_normal((120, 8))) * 1000 + 1e4,
                       index=idx, columns=cols)
    panel = Panel(close=close, open=close * 0.99, high=close * 1.01,
                  low=close * 0.99, volume=vol, amount=vol * close)
    r = composite_backtest(
        ["delta(close,10)", "ts_zscore(close,20)", "rank(close,5)"],
        panel, scheme="equal", forward_periods=1)
    assert "risk_attribution" in r
    ra = r["risk_attribution"]
    assert ra is not None
    if "error" not in ra:
        # 保证可加性
        s = (sum(f["mctr_vol"] for f in ra["factors"])
             + ra["specific"]["mctr_vol"] + ra["market"]["mctr_vol"])
        assert abs(s - ra["total"]["vol"]) < 1e-5
        assert ra["total"]["ann_vol"] is not None
