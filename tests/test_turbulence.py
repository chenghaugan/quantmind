"""Turbulence 市场状态检测测试。"""
from __future__ import annotations

import numpy as np
import pytest

from quantmind.risk.turbulence import (
    Regime,
    TurbulenceConfig,
    TurbulenceDetector,
    TurbulenceRiskAdapter,
)


def test_quiet_market_low_regime():
    rng = np.random.default_rng(0)
    # 平稳低波动序列 → turbulence 应较低，多为 LOW
    returns = rng.normal(0.0, 0.005, 400)
    det = TurbulenceDetector(TurbulenceConfig(lookback=100, quantile_hi=0.95, quantile_extreme=0.99))
    series = det.compute(returns)
    last = series[-1]
    regime = det.get_regime(last)
    assert regime == Regime.LOW
    assert det.suggested_scale(last) == 1.0


def test_spiky_market_high_regime():
    rng = np.random.default_rng(1)
    returns = rng.normal(0.0, 0.005, 400)
    # 末尾注入极端尖峰
    returns[-1] = 0.30
    det = TurbulenceDetector(TurbulenceConfig(lookback=100, quantile_hi=0.95, quantile_extreme=0.99))
    series = det.compute(returns)
    last = series[-1]
    regime = det.get_regime(last)
    # 尖峰应使 turbulence 显著高于阈值
    assert regime in (Regime.HIGH, Regime.EXTREME)
    assert det.suggested_scale(last) <= 0.5


def test_extreme_detection():
    rng = np.random.default_rng(2)
    returns = rng.normal(0.0, 0.005, 400)
    returns[-1] = 0.60   # 极极端
    det = TurbulenceDetector(TurbulenceConfig(lookback=100, quantile_hi=0.95, quantile_extreme=0.99))
    series = det.compute(returns)
    assert det.get_regime(series[-1]) == Regime.EXTREME
    assert det.suggested_scale(series[-1]) == 0.0


def test_adapter_decision():
    rng = np.random.default_rng(3)
    returns = rng.normal(0.0, 0.005, 400)
    returns[-1] = 0.60
    adapter = TurbulenceRiskAdapter()
    dec = adapter.decision(returns, vt_symbol="rb0.SHFE")
    assert not dec.passed
    assert dec.code.value == "TURBULENCE_EXTREME"


def test_adapter_scale_quiet():
    rng = np.random.default_rng(4)
    returns = rng.normal(0.0, 0.005, 400)
    adapter = TurbulenceRiskAdapter()
    assert adapter.check_scale(returns) == 1.0


def test_detector_reproducible():
    rng = np.random.default_rng(5)
    returns = rng.normal(0.0, 0.005, 400)
    d1 = TurbulenceDetector().compute(returns)
    rng2 = np.random.default_rng(5)
    returns2 = rng2.normal(0.0, 0.005, 400)
    d2 = TurbulenceDetector().compute(returns2)
    np.testing.assert_allclose(d1[-10:], d2[-10:])
