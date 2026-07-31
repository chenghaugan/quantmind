"""Tier2 测试：全天候(VolTarget) 与 配对交易(Pair) 策略模板端到端运行。"""
from __future__ import annotations

import asyncio

import pytest

from quantmind.core.engine import EventEngine
from quantmind.core.contracts import default_size
from quantmind.strategy import run_strategy, VolTargetStrategy, PairTradingStrategy, build_spread_bars
from tests.helpers import load_bars


@pytest.mark.asyncio
async def test_vol_target_backtest():
    bars = await load_bars("rb0", years=2)
    vt = "rb0.SHFE"
    sizes = {vt: default_size(vt)}
    setting = {"size": sizes[vt], "max_pos": 1.0}
    ee = EventEngine(); await ee.start()
    res = run_strategy("backtest", VolTargetStrategy, vt, setting, bars, ee, sizes, gateway_name="ctp")
    await ee.stop()
    assert res["mode"] == "backtest"
    assert "equity_curve" in res
    assert isinstance(res["trades"], int)
    assert res["equity_curve"]  # 非空曲线


@pytest.mark.asyncio
async def test_vol_target_paper():
    bars = await load_bars("rb0", years=2)
    vt = "rb0.SHFE"
    sizes = {vt: default_size(vt)}
    setting = {"size": sizes[vt], "max_pos": 1.0}
    ee = EventEngine(); await ee.start()
    res = run_strategy("paper", VolTargetStrategy, vt, setting, bars, ee, sizes, gateway_name="ctp")
    await ee.stop()
    assert res["mode"] == "paper"
    assert "summary" in res


@pytest.mark.asyncio
async def test_pair_backtest():
    bars_a = await load_bars("rb0", years=2)
    bars_b = await load_bars("hc0", years=2)
    spread = build_spread_bars(bars_a, bars_b)
    assert spread, "价差合成标的应为非空"
    vt = f"SPREAD.{spread[0].exchange.value}"
    sizes = {vt: 1}
    setting = {"size": 1, "max_pos": 1.0}
    ee = EventEngine(); await ee.start()
    res = run_strategy("backtest", PairTradingStrategy, vt, setting, spread, ee, sizes, gateway_name="ctp")
    await ee.stop()
    assert res["mode"] == "backtest"
    assert "equity_curve" in res


@pytest.mark.asyncio
async def test_pair_live_route():
    """配对交易可切换至实盘路线（CTP 桩），无异常。"""
    bars_a = await load_bars("rb0", years=1)
    bars_b = await load_bars("hc0", years=1)
    spread = build_spread_bars(bars_a, bars_b)
    vt = f"SPREAD.{spread[0].exchange.value}"
    sizes = {vt: 1}
    setting = {"size": 1, "max_pos": 1.0}
    res = run_strategy("live", PairTradingStrategy, vt, setting, spread, None, sizes, gateway_name="ctp")
    assert res["routed"] is True
