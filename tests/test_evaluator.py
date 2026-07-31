"""因子评估 / 多因子加权 / 中性化 升级测试。"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from quantmind.core.constant import Exchange, Interval
from quantmind.core.object import BarData
from quantmind.research import (
    FactorEvaluator, FactorReport, MomentumFactor,
    MultiFactorModel, FactorSpec, build_model_from_specs,
    icir_weights, winsorize, cross_sectional_neutralize, orthogonalize_factors,
)
from quantmind.research.factors.base import Factor, FactorMeta
from tests.helpers import load_bars


def make_bars(closes, start=datetime(2024, 1, 1)):
    bars = []
    for i, c in enumerate(closes):
        bars.append(BarData(
            symbol="rb0", exchange=Exchange.SHFE,
            datetime=start + timedelta(days=i), interval=Interval.DAILY,
            open_price=c, high_price=c, low_price=c, close_price=c,
            volume=1000.0, open_interest=0.0,
        ))
    return bars


def _predictive_bars(n=300, seed=0):
    """构造『因子 = 下一期收益』的强预测数据（IC 应≈1）。"""
    rng = np.random.default_rng(seed)
    ret = np.sin(np.linspace(0, 20, n)) * 0.01 + rng.normal(0, 0.002, n)
    closes = [100.0]
    for r in ret[1:]:
        closes.append(closes[-1] * (1.0 + r))
    bars = make_bars(closes)
    factor = pd.Series([ret[i + 1] if i + 1 < len(ret) else 0.0 for i in range(len(ret))], dtype=float)
    factor.name = "predictive"
    return bars, factor


@pytest.mark.asyncio
async def test_double_ic_strong_predictive():
    bars, factor = _predictive_bars()
    rep = FactorEvaluator().evaluate(factor, bars)
    assert rep.n_samples == len(bars)
    # rank IC 与 pearson IC 都应接近 1
    assert rep.ic_mean > 0.9
    assert rep.ic_pearson > 0.9
    assert len(rep.ic_decay) == 5
    # 多空组合应盈利、单调性高、综合主分高
    assert rep.ls_portfolio_return > 0
    assert rep.monotonicity_5 > 0.5
    assert rep.composite_score > 0.7


@pytest.mark.asyncio
async def test_turnover_positive_and_bootstrap_ci():
    bars, factor = _predictive_bars()
    rep = FactorEvaluator().evaluate(factor, bars)
    assert rep.turnover_annual > 0
    # bootstrap CI 应包含 IC 均值
    assert pd.notna(rep.ic_ci_low) and pd.notna(rep.ic_ci_high)
    assert rep.ic_ci_low <= rep.ic_mean <= rep.ic_ci_high


def test_ic_decay_half_life():
    # 指数衰减 0.9 -> 半衰期约 1
    hl = FactorEvaluator._ic_decay_half_life([0.9, 0.45, 0.225, 0.11, 0.055])
    assert abs(hl - 1.0) < 0.2
    # 无衰减信息 -> nan
    assert pd.isna(FactorEvaluator._ic_decay_half_life([0.0, 0.0, 0.0]))


@pytest.mark.asyncio
async def test_composite_in_range():
    bars, factor = _predictive_bars()
    rep = FactorEvaluator().evaluate(factor, bars)
    assert 0.0 <= rep.composite_score <= 1.0
    # 弱因子综合分应低于强因子
    rng = np.random.default_rng(1)
    weak = pd.Series(rng.normal(0, 1, len(bars)), dtype=float)
    weak.name = "weak"
    repw = FactorEvaluator().evaluate(weak, bars)
    assert repw.composite_score < rep.composite_score


@pytest.mark.asyncio
async def test_cross_sectional_evaluate():
    rng = np.random.default_rng(2)
    dates = [datetime(2023, 1, 1, tzinfo=timezone.utc) + timedelta(days=i)
             for i in range(150)]
    syms = [f"S{i}" for i in range(15)]
    factor_by_symbol, bars_by_symbol = {}, {}
    for s in syms:
        ret = rng.normal(0, 0.01, len(dates))
        closes = [100.0]
        for r in ret[1:]:
            closes.append(closes[-1] * (1 + r))
        bars = [BarData(symbol=s, exchange=Exchange.SSE, datetime=d,
                        interval=Interval.DAILY, open_price=c, high_price=c,
                        low_price=c, close_price=c, volume=1.0, open_interest=0.0)
                 for d, c in zip(dates, closes)]
        bars_by_symbol[s] = bars
        # 因子 = 下一期收益 + 噪声（弱预测）
        fv = pd.Series([ret[i + 1] if i + 1 < len(ret) else 0.0 for i in range(len(ret))],
                       index=dates, dtype=float)
        factor_by_symbol[s] = fv
    rep = FactorEvaluator().evaluate_cross_sectional(factor_by_symbol, bars_by_symbol)
    assert pd.notna(rep.ic_mean)
    assert rep.ic_mean > 0  # 因子含下一期收益信息
    assert rep.n_samples >= 10


# ---- 多因子加权 ----
def test_icir_weights():
    # 正 ICIR 加权、负 ICIR 置 0
    w = icir_weights([0.1, -0.05, 0.2], [0.05, 0.05, 0.1])
    assert all(x >= 0 for x in w)
    assert abs(sum(w) - 1.0) < 1e-9
    # 全 0 -> 等权
    w0 = icir_weights([0.0, 0.0], [0.0, 0.0])
    assert w0 == [0.5, 0.5]


class _ConstFactor(Factor):
    def __init__(self, values, name="c"):
        self._v = list(values)
        self.meta = FactorMeta(name=name)
    def compute(self, bars):
        return pd.Series(self._v, dtype=float)


@pytest.mark.asyncio
async def test_fit_weights_from_ics():
    bars = await load_bars()
    n = len(bars)
    rng = np.random.default_rng(3)
    a = rng.normal(0, 1, n)
    fa = _ConstFactor(a, name="fa")
    fb = _ConstFactor(rng.normal(0, 1, n), name="fb")
    model = MultiFactorModel([fa, fb])
    model.fit_weights_from_ics({"fa": {"ic_mean": 0.1, "ic_std": 0.05},
                                "fb": {"ic_mean": -0.02, "ic_std": 0.05}})
    assert model.weights[0] > 0 and abs(model.weights[1]) < 1e-9


@pytest.mark.asyncio
async def test_dedup_correlated():
    bars = await load_bars()
    n = len(bars)
    rng = np.random.default_rng(4)
    base = rng.normal(0, 1, n)
    redundant = base + rng.normal(0, 0.01, n)        # 与 base 高度相关
    indep = rng.normal(0, 1, n)
    model = MultiFactorModel([
        _ConstFactor(base, name="base"),
        _ConstFactor(redundant, name="redundant"),
        _ConstFactor(indep, name="indep"),
    ])
    kept = model.dedup_correlated(bars, threshold=0.7)
    assert len(model.factors) == 2
    names = {f.meta.name for f in model.factors}
    assert "redundant" not in names and "base" in names and "indep" in names


@pytest.mark.asyncio
async def test_target_position_dedup_runs():
    bars = await load_bars()
    specs = [FactorSpec(name="mom", kind="momentum", window=20, weight=1.0),
             FactorSpec(name="rev", kind="mean_reversion", window=60, weight=-0.5)]
    model = build_model_from_specs(specs, bars)
    tgt = model.target_position(bars, size=10, max_pos=1.0, dedup=True)
    assert len(tgt) == len(bars)


# ---- 中性化 / 正交化 ----
def test_winsorize():
    s = pd.Series([1, 2, 3, 4, 5, 100, -100])
    w = winsorize(s, 0.1)
    assert w.max() < 100 and w.min() > -100
    assert w.iloc[1] == 2


def test_orthogonalize_reduces_correlation():
    rng = np.random.default_rng(5)
    x = rng.normal(0, 1, 500)
    y = 0.9 * x + rng.normal(0, 0.3, 500)   # 与 x 高度相关
    z = rng.normal(0, 1, 500)
    out = orthogonalize_factors([pd.Series(x), pd.Series(y), pd.Series(z)])
    corr_xy_after = out[0].corr(out[1])
    # 正交化后 x 与 y 残差相关应大幅降低
    assert abs(corr_xy_after) < 0.3


def test_cross_sectional_neutralize_removes_mktcap():
    rng = np.random.default_rng(6)
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(60)]
    syms = [f"S{i}" for i in range(20)]
    mcv = rng.uniform(1e9, 1e10, (60, 20))
    market_cap = pd.DataFrame(mcv, index=dates, columns=syms)
    # 因子 = 2*log(mktcap) + 噪声
    factor = np.log(market_cap.values) * 2 + rng.normal(0, 0.1, (60, 20))
    panel = pd.DataFrame(factor, index=dates, columns=syms)
    resid = cross_sectional_neutralize(panel, market_cap=market_cap)
    # 残差与市值的相关性应大幅降低
    before = abs(panel.corrwith(market_cap).mean())
    after = abs(resid.corrwith(market_cap).mean())
    assert after < before * 0.5
