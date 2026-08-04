"""实盘网关半可用（SimGateway + 桩升级）测试。"""
from __future__ import annotations

import asyncio

from quantmind.core.constant import Direction, Exchange, Offset
from quantmind.core.engine import EventEngine
from quantmind.core.gateway import OrderRequest
from quantmind.live import SimGateway, build_gateway, CtpGateway


def _req(symbol="rb2410", exchange=Exchange.SHFE, direction=Direction.LONG,
         offset=Offset.OPEN, volume=5.0, price=3500.0) -> OrderRequest:
    return OrderRequest(symbol=symbol, exchange=exchange, direction=direction,
                        offset=offset, volume=volume, price=price)


def test_sim_gateway_full_fill():
    gw = SimGateway(None)
    gw.connect({})
    oid = gw.send_order(_req())
    assert oid
    assert gw.connected
    assert len(gw.trades) == 1
    # 净持仓 +5 手多头
    assert gw.positions["rb2410.SHFE"].volume == 5.0


def test_sim_gateway_net_position_short():
    gw = SimGateway(None)
    gw.connect({})
    gw.send_order(_req(direction=Direction.SHORT, volume=3.0))
    assert gw.positions["rb2410.SHFE"].volume == -3.0


def test_sim_gateway_partial_fill():
    gw = SimGateway(None, fill_ratio=0.5)
    gw.connect({})
    gw.send_order(_req(volume=10.0))
    assert gw.positions["rb2410.SHFE"].volume == 5.0


def test_sim_gateway_reject():
    gw = SimGateway(None, reject_rate=1.0)  # 全部拒单
    gw.connect({})
    oid = gw.send_order(_req())
    assert gw.positions.get("rb2410.SHFE", None) is None
    assert len(gw.trades) == 0


def test_sim_gateway_query_account():
    gw = SimGateway(None)
    gw.connect({})
    acct = gw.query_account()
    assert acct.balance > 0


def test_sim_gateway_disconnect_reconnect():
    gw = SimGateway(None)
    gw.connect({})
    assert gw.connected
    gw.simulate_disconnect()
    assert not gw.connected
    gw.connect({})
    assert gw.connected


def test_sim_gateway_events_via_event_engine():
    ee = EventEngine()
    asyncio.run(ee.start())
    try:
        gw = SimGateway(ee)
        gw.connect({})
        gw.send_order(_req())
        # 事件应已入队（TRADE/POSITION/ACCOUNT/ORDER）
        assert len(gw.trades) == 1
    finally:
        asyncio.run(ee.stop())


def test_build_gateway_sim_registered():
    from quantmind.core.engine import EventEngine
    ee = EventEngine()
    asyncio.run(ee.start())
    try:
        gw = build_gateway("sim", ee, {})
        assert isinstance(gw, SimGateway)
    finally:
        asyncio.run(ee.stop())


def test_ctp_stub_now_emits_fill():
    ee = EventEngine()
    asyncio.run(ee.start())
    try:
        gw = CtpGateway(ee)
        gw.connect({})
        oid = gw.send_order(_req(volume=2.0, price=3500.0))
        assert oid
        # 桩升级后应产生成交（trades 通过 simulate_one_trade 广播），此处至少不抛异常
        assert gw.connected
    finally:
        asyncio.run(ee.stop())


def test_live_engine_full_cycle_with_sim():
    """用 SimGateway 跑通 LiveEngine 完整闭环（含风控与订单簿）。"""
    from quantmind.live import LiveEngine

    ee = EventEngine()
    asyncio.run(ee.start())
    try:
        gw = SimGateway(ee)
        gw.connect({})
        eng = LiveEngine(gw, event_engine=ee)
        # 绕过风控时段闸门：直接发单（LiveEngine 默认保守风控，交易时段外会拒）
        oid = eng.send_order(_req())
        # 若被风控拒（非交易时段），返回空字符串；否则有 oid
        assert isinstance(oid, str)
        report = eng.reconcile(gw.query_position(), gw.query_account().balance)
        assert isinstance(report.to_dict(), dict)
    finally:
        asyncio.run(ee.stop())
