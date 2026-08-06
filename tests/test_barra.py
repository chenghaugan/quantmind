"""Barra 式多因子风险归因测试（research/barra.py）。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantmind.research.barra import (
    estimate_factor_returns,
    portfolio_weights_from_signal,
    orthogonalize_exposures,
    newey_west_cov,
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
        """MCTR 之和必须精确等于组合波动（可加性）。

        用 ``additivity["closure"]``（基于原始未取整值）做严格校验；逐个取整后
        的 ``mctr_vol`` 求和会因各自 round 产生 ~1e-6 量级的取整误差，故不用于严格断。
        """
        sig, fwd, expos, dates, assets = _synth_inputs()
        res = barra_factor_risk_attribution(sig, fwd, expos)
        assert res["additivity"]["closure"] is not None
        assert abs(res["additivity"]["closure"]) < 1e-8
        assert abs(res["additivity"]["recon_total"] - res["additivity"]["port_vol"]) < 1e-6

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
        # 保证可加性（用原始未取整的 closure 字段，避免逐项 round 引入 ~1e-6 误差）
        assert ra["additivity"]["closure"] is not None
        assert abs(ra["additivity"]["closure"]) < 1e-6
        assert ra["total"]["ann_vol"] is not None


class TestOrthogonalization:
    """业界 Barra 风格正交化：因子横截面互不相关、单位方差。"""

    def test_orthogonalized_exposures_decorrelated(self):
        sig, fwd, expos, dates, assets = _synth_inputs()
        ort = orthogonalize_exposures(expos, dates, assets)
        # 逐日横截面相关应趋于 0（对残差化 + 归一化后的因子）
        names = list(ort.keys())
        max_corr = 0.0
        for d in dates:
            vals = np.column_stack([ort[n].loc[d].values for n in names])
            # 用 OLS 残差相关（正交化后 Gram 近单位阵，相关应极小）
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a = vals[:, i]; b = vals[:, j]
                    if np.std(a) > 1e-12 and np.std(b) > 1e-12:
                        c = np.corrcoef(a, b)[0, 1]
                        if c == c:
                            max_corr = max(max_corr, abs(c))
        assert max_corr < 0.2  # 正交化后因子间相关性应显著低于原始

    def test_orthogonalization_flag_diagnostics(self):
        sig, fwd, expos, dates, assets = _synth_inputs()
        res = barra_factor_risk_attribution(sig, fwd, expos)  # 默认 open
        assert res["diagnostics"]["orthogonalized"] is True
        res_off = barra_factor_risk_attribution(
            sig, fwd, expos, orthogonalize_style=False)
        assert res_off["diagnostics"]["orthogonalized"] is False


class TestNeweyWest:
    """业界 Barra Newey-West（HAC）稳健协方差。"""

    def test_nw_cov_matches_sample_when_independent(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=400)
        y = rng.normal(size=400)
        nw = newey_west_cov(x, y, lags=1)
        sample = np.cov(x, y, ddof=1)[0, 1]
        assert abs(nw - sample) < 0.05  # 独立序列时 HAC≈样本

    def test_default_uses_newey_west(self):
        sig, fwd, expos, dates, assets = _synth_inputs()
        res = barra_factor_risk_attribution(sig, fwd, expos)
        assert res["diagnostics"]["covariance"] == "newey_west"
        assert res["diagnostics"]["nw_lags"] is not None
        # HAC 口径下总方差 + 各分量用同一估计器 → closure 仍精确
        assert abs(res["additivity"]["closure"]) < 1e-8

    def test_sample_covariance_option(self):
        sig, fwd, expos, dates, assets = _synth_inputs()
        res = barra_factor_risk_attribution(sig, fwd, expos, newey_west=False)
        assert res["diagnostics"]["covariance"] == "sample"
        assert res["diagnostics"]["nw_lags"] is None
        assert abs(res["additivity"]["closure"]) < 1e-8


class TestWLS:
    """业界 Barra WLS（市值加权）截面回归。"""

    def test_wls_path_runs_and_closes(self):
        sig, fwd, expos, dates, assets = _synth_inputs()
        capw = pd.DataFrame(np.abs(np.random.default_rng(2).normal(size=fwd.shape)),
                            index=fwd.index, columns=fwd.columns) + 1.0
        res = barra_factor_risk_attribution(
            sig, fwd, expos, newey_west=False, cap_weights=capw)
        assert res["diagnostics"]["wls"] is True
        assert abs(res["additivity"]["closure"]) < 1e-8
        assert res["total"]["ann_vol"] is not None


class TestTimeSeriesPayload:
    """前端展示所需的时序 payload：JSON 安全 + 形状正确。"""

    def test_all_ts_fields_present(self):
        sig, fwd, expos, dates, assets = _synth_inputs()
        res = barra_factor_risk_attribution(sig, fwd, expos)
        assert set(res["factor_returns_ts"]["series"].keys()) == {"f1", "f2", "f3"}
        assert len(res["factor_returns_ts"]["dates"]) == len(res["factor_returns_ts"]["series"]["f1"])
        assert "r2" in res["r2_ts"]
        assert len(res["r2_ts"]["dates"]) == len(res["r2_ts"]["r2"])
        assert set(res["exposure_ts"]["series"].keys()) == {"f1", "f2", "f3"}
        assert res["return_attribution"]["factors"]  # 每因子有累计收益贡献
        assert res["rolling_risk"] is not None
        assert res["rolling_risk"]["dates"]
        assert len(res["rolling_risk"]["dates"]) == len(res["rolling_risk"]["portfolio_vol"])

    def test_raw_factor_returns_ts_when_orthogonalized(self):
        sig, fwd, expos, dates, assets = _synth_inputs()
        res = barra_factor_risk_attribution(sig, fwd, expos)  # 默认正交化
        # 正交化开启时，应返回原始（未正交化）因子收益时序供对比
        assert "factor_returns_raw_ts" in res
        assert res["factor_returns_raw_ts"] is not None
        assert set(res["factor_returns_raw_ts"]["series"].keys()) == {"f1", "f2", "f3"}

    def test_ts_payload_json_safe(self):
        """时序 payload 中不得含 NaN/inf（前端经 API JSON 序列化）。"""
        import math
        import json
        sig, fwd, expos, dates, assets = _synth_inputs()
        res = barra_factor_risk_attribution(sig, fwd, expos)
        blob = {
            "fr": res["factor_returns_ts"],
            "raw": res["factor_returns_raw_ts"],
            "r2": res["r2_ts"],
            "expo": res["exposure_ts"],
            "ra": res["return_attribution"]["ts"],
            "rr": res["rolling_risk"],
        }
        json.dumps(blob)  # 必须可 JSON 序列化
        def _walk(o):
            if isinstance(o, float):
                assert not math.isnan(o) and not math.isinf(o), f"NaN/inf in {o}"
            elif isinstance(o, dict):
                for v in o.values():
                    _walk(v)
            elif isinstance(o, (list, tuple)):
                for v in o:
                    _walk(v)
        _walk(blob)

    def test_rolling_risk_closes_per_slice(self):
        """每个滚动窗口切片：Σ(因子+特异+市场)MCTR ≈ 组合σ。"""
        sig, fwd, expos, dates, assets = _synth_inputs()
        res = barra_factor_risk_attribution(sig, fwd, expos, newey_west=False)
        rr = res["rolling_risk"]
        fr = rr["factors"]
        keys = [c for c in fr.keys()]
        spec_key = "_specific" if "_specific" in keys else "_specific"
        mkt_key = "_market" if "_market" in keys else "_market"
        for i in range(len(rr["dates"])):
            s = sum(fr.get(c, [None] * len(rr["dates"]))[i]
                    for c in keys if c not in ("_specific", "_market"))
            s += fr.get(spec_key, [None] * len(rr["dates"]))[i]
            s += fr.get(mkt_key, [None] * len(rr["dates"]))[i]
            pv = rr["portfolio_vol"][i]
            if s is not None and pv is not None:
                assert abs(s - pv) < 1e-6  # 可加性（样本协方差滚动）

