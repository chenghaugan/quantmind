"""回测严谨性诊断测试：涨跌停 / 前视 / 过拟合 / 健康度。"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from quantmind.core.constant import Direction, Exchange, Interval, Offset
from quantmind.core.object import BarData
from quantmind.core.gateway import OrderRequest
from quantmind.backtest import (
    BacktestEngine, limit_day_mask, detect_lookahead, diagnose_overfitting, health_checks,
)
from quantmind.backtest.analyzer import PerformanceReport
from quantmind.research.evaluator import FactorReport, FactorEvaluator
from quantmind.strategy.multifactor import MultiFactorStrategy
from quantmind.core.contracts import default_size
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


def test_limit_day_mask():
    closes = [100, 110, 100, 90, 100]   # bar1 +10% 涨停, bar3 -10% 跌停
    bars = make_bars(closes)
    mask = limit_day_mask(bars, limit_pct=0.10)
    assert mask[1] == "up"
    assert mask[3] == "down"
    assert mask[0] is None
    # 关闭时不标记
    assert all(m is None for m in limit_day_mask(bars, limit_pct=None))


@pytest.mark.asyncio
async def test_detect_lookahead():
    rng = np.random.default_rng(7)
    n = 300
    ret = pd.Series(rng.normal(0, 0.01, n))
    fwd_ret = ret.shift(-1)   # 未来一期收益
    # 因子 = 同期收益（泄露未来信息）
    res = detect_lookahead(ret, ret, fwd_ret)
    assert res["lookahead_suspected"] is True
    # 因子 = 未来收益（正常，无泄露）
    res2 = detect_lookahead(fwd_ret, ret, fwd_ret)
    assert res2["lookahead_suspected"] is False


def test_diagnose_overfitting():
    good_train = PerformanceReport(sharpe=2.0, total_return=0.3)
    bad_test = PerformanceReport(sharpe=0.2, total_return=-0.05)
    r = diagnose_overfitting(good_train, bad_test)
    assert r["overfit_suspected"] is True
    # 样本内外都好 -> 不过拟合
    good_test = PerformanceReport(sharpe=1.8, total_return=0.25)
    r2 = diagnose_overfitting(good_train, good_test)
    assert r2["overfit_suspected"] is False


def test_health_checks_pass_and_fail():
    # 健康因子
    fr = FactorReport(n_samples=300, ic_mean=0.06, ir=0.8, turnover_annual=5.0)
    perf = PerformanceReport(max_drawdown=-0.2)
    fv = pd.Series(np.random.default_rng(8).normal(0, 1, 300))
    hr = health_checks(factor_values=fv, report=fr, perf=perf)
    assert hr.passed is True
    # 失败：IC 不显著 + 超高换手
    fr_bad = FactorReport(n_samples=300, ic_mean=0.005, ir=0.2, turnover_annual=99.0)
    hr_bad = health_checks(factor_values=fv, report=fr_bad, perf=perf)
    assert hr_bad.passed is False


@pytest.mark.asyncio
async def test_backtest_exclude_limit_gate():
    """涨停日无法买入：_execute_fill 应返回 False（不成交）。"""
    closes = [100, 110, 100, 100, 100]   # bar1 涨停
    bars = make_bars(closes)
    vt = "rb0.SHFE"
    eng = BacktestEngine({vt: bars}, sizes={vt: 1})
    eng.exclude_limit = True
    eng.limit_pct = 0.10
    eng._limit_flag = {vt: {bars[1].datetime: "up"}}
    pending = {
        "vt_symbol": vt,
        "req": OrderRequest(symbol="rb0", exchange=Exchange.SHFE,
                            direction=Direction.LONG, offset=Offset.OPEN, volume=1.0),
        "fill_date": bars[1].datetime,
    }
    ok = eng._execute_fill(pending, bars[1].datetime)
    assert ok is False
    # 非涨停日应可成交
    eng._limit_flag = {vt: {bars[2].datetime: None}}
    pending2 = {
        "vt_symbol": vt,
        "req": OrderRequest(symbol="rb0", exchange=Exchange.SHFE,
                            direction=Direction.LONG, offset=Offset.OPEN, volume=1.0),
        "fill_date": bars[2].datetime,
    }
    ok2 = eng._execute_fill(pending2, bars[2].datetime)
    assert ok2 is True


@pytest.mark.asyncio
async def test_backtest_runs_with_exclude_limit_on_random():
    bars = await load_bars()
    vt = "rb0.SHFE"
    eng = BacktestEngine({vt: bars}, capital=1_000_000, sizes={vt: default_size(vt)},
                         exclude_limit=True, limit_pct=0.10)
    eng.add_strategy(MultiFactorStrategy, vt, {"size": default_size(vt), "max_pos": 1.0})
    rep = eng.run()
    assert rep.trade_count >= 0
    assert len(rep.equity_curve) == len(bars)
