"""组合级风控测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantmind.risk.portfolio import (
    PortfolioRiskEngine,
    PortfolioLimits,
    PortfolioRiskState,
    compute_strategy_correlation,
)


def test_state_exposure():
    st = PortfolioRiskState()
    st.set_position("a", "rb0.SHFE", volume=10, value=100_000)
    st.set_position("a", "hc0.SHFE", volume=-5, value=-50_000)
    st.set_position("b", "rb0.SHFE", volume=3, value=30_000)
    st.set_equity("a", 500_000)
    st.set_equity("b", 500_000)
    assert st.long_exposure() == 130_000
    assert st.short_exposure() == 50_000
    assert st.gross_exposure() == 180_000
    assert st.net_exposure() == 80_000
    assert st.total_equity() == 1_000_000
    assert st.position_value("rb0.SHFE") == 130_000


def test_engine_concentration():
    st = PortfolioRiskState()
    eng = PortfolioRiskEngine(PortfolioLimits(max_position_concentration=0.3)).attach(st)
    st.set_equity("a", 1_000_000)
    st.set_position("a", "rb0.SHFE", value=200_000)
    # 加仓到 400k / 1000k = 40% > 30% → 拒
    dec = eng.check_position("a", "rb0.SHFE", value=200_000, total_equity=1_000_000,
                             existing_value=200_000)
    assert not dec.passed
    assert dec.code.value == "POSITION_CONCENTRATION"


def test_engine_exposure():
    st = PortfolioRiskState()
    eng = PortfolioRiskEngine(PortfolioLimits(max_gross_exposure=0.8, max_net_exposure=0.5)).attach(st)
    eng.state.set_equity("a", 1_000_000)
    # 总敞口 90% > 80% → 拒
    dec = eng.check_exposure(gross_exposure=900_000, net_exposure=100_000, total_equity=1_000_000)
    assert not dec.passed
    assert dec.code.value == "EXPOSURE_LIMIT"


def test_engine_correlation():
    rng = np.random.default_rng(1)
    a = pd.Series(rng.normal(0.001, 0.01, 300))
    b = a * 1.0 + rng.normal(0.001, 0.011, 300) * 0.0  # 完全同向
    c = pd.Series(rng.normal(0.001, 0.05, 300))        # 独立
    mat = compute_strategy_correlation({"a": a, "b": b, "c": c})
    eng = PortfolioRiskEngine(PortfolioLimits(max_strategy_correlation=0.8))
    eng.update_correlation_matrix(mat)
    viol = eng.check_strategy_correlation()
    # a/b 相关 ~1 应触发
    assert len(viol) >= 1
    assert viol[0].code.value == "STRATEGY_CORRELATION"


def test_engine_pass_when_ok():
    st = PortfolioRiskState()
    eng = PortfolioRiskEngine(PortfolioLimits(max_position_concentration=0.5)).attach(st)
    st.set_equity("a", 1_000_000)
    dec = eng.check_position("a", "rb0.SHFE", value=200_000, total_equity=1_000_000, existing_value=0)
    assert dec.passed
