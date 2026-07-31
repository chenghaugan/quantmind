"""Tier2 测试：Alpha101 / Alpha191 因子族计算正确性与稳定性。"""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

from quantmind.research import list_alpha101, list_alpha191, build_alpha_factor, build_alpha191_factor
from tests.helpers import load_bars


@pytest.mark.asyncio
async def test_alpha101_all_compute():
    bars = await load_bars("rb0", years=2)
    names = list_alpha101()
    assert len(names) >= 20
    for name in names:
        f = build_alpha_factor(name)
        s = f.compute(bars)
        assert len(s) == len(bars), f"{name} 长度不匹配"
        # 计算后已 fillna(0)，应无 NaN/inf
        assert np.isfinite(s).all(), f"{name} 含非有限值"
        assert s.abs().max() < 1e9, f"{name} 数值疑似爆炸"


@pytest.mark.asyncio
async def test_alpha191_all_compute():
    bars = await load_bars("rb0", years=2)
    names = list_alpha191()
    assert len(names) >= 5
    for name in names:
        f = build_alpha191_factor(name)
        s = f.compute(bars)
        assert len(s) == len(bars)
        assert np.isfinite(s).all()


@pytest.mark.asyncio
async def test_alpha_with_evaluator():
    """Alpha 因子可进入评估器产出完整报告。"""
    from quantmind.research import FactorEvaluator
    bars = await load_bars("rb0", years=2)
    f = build_alpha_factor("alpha101")
    s = f.compute(bars)
    rep = FactorEvaluator().evaluate(s, bars)
    assert rep.n_samples > 0
    assert np.isfinite(rep.ic_mean)
    assert np.isfinite(rep.composite_score)
