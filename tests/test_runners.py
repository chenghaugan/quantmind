"""切路线测试：回测 / 模拟 / 实盘 同策略共用。"""
from __future__ import annotations

import pytest

from quantmind.core.engine import EventEngine
from quantmind.strategy import run_strategy, MultiFactorStrategy
from quantmind.core.event import EventType
from tests.helpers import load_bars


@pytest.mark.asyncio
async def test_switch_route_all_modes():
    bars = await load_bars()
    vt = "rb0.SHFE"
    sizes = {vt: 10.0}
    setting = {"size": 10, "max_pos": 1.0}

    ee = EventEngine()
    captured = []
    async def cap(e):
        captured.append(e.type)
    ee.register_general(cap)
    await ee.start()

    r_bt = run_strategy("backtest", MultiFactorStrategy, vt, setting, bars, ee, sizes)
    await __import__("asyncio").sleep(0.02)
    r_paper = run_strategy("paper", MultiFactorStrategy, vt, setting, bars, ee, sizes)
    await __import__("asyncio").sleep(0.02)
    r_live = run_strategy("live", MultiFactorStrategy, vt, setting, bars, ee, sizes, gateway_name="ctp")
    await __import__("asyncio").sleep(0.02)

    await ee.stop()

    assert r_bt["mode"] == "backtest"
    assert r_bt["trades"] > 0
    assert r_paper["mode"] == "paper"
    assert r_paper["summary"]["trade_count"] > 0
    assert r_live["mode"] == "live" and r_live["routed"] is True
    # 模拟/实盘应产生事件（bar/signal/order/trade/position/account）
    types = {e.value for e in captured}
    assert EventType.EVENT_TRADE.value in types
    assert EventType.EVENT_ORDER.value in types
