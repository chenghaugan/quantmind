"""本地订单簿、委托状态机与持仓对账测试。

重点覆盖实盘特有的「脏场景」：乱序回报、重复成交回报、部分成交、
挂单超时撤单、对账不一致触发熔断。
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from quantmind.core.constant import Direction, Exchange, Offset, Status
from quantmind.core.gateway import CancelRequest, OrderRequest
from quantmind.core.object import PositionData, TradeData
from quantmind.live.order_manager import OrderManager
from quantmind.live.reconcile import reconcile, reconcile_positions
from quantmind.risk import RiskEngine, RiskLimits

UTC = timezone.utc
T0 = datetime(2026, 8, 3, 2, 0, tzinfo=UTC)


def req(symbol="rb2410", direction=Direction.LONG, offset=Offset.OPEN,
        volume=10.0, price=3500.0) -> OrderRequest:
    return OrderRequest(symbol=symbol, exchange=Exchange.SHFE, direction=direction,
                        offset=offset, volume=volume, price=price)


def trade(order_id, trade_id, volume, price=3500.0, direction=Direction.LONG,
          symbol="rb2410", dt=None) -> TradeData:
    return TradeData(symbol=symbol, exchange=Exchange.SHFE, order_id=order_id,
                     trade_id=trade_id, direction=direction, offset=Offset.OPEN,
                     price=price, volume=volume, datetime=dt or T0)


def position(vt: str, volume: float) -> PositionData:
    sym, exch = vt.rsplit(".", 1)
    return PositionData(symbol=sym, exchange=Exchange(exch),
                        direction=Direction.NET, volume=volume)


# ---------------------------------------------------------------- 状态机
class TestOrderStateMachine:
    def setup_method(self):
        self.om = OrderManager(timeout_seconds=60.0)

    def test_add_order_is_active(self):
        o = self.om.add_order(req(), "O1", now=T0)
        assert o.status is Status.SUBMITTING
        assert o.is_active and o.remaining == 10.0
        assert len(self.om.active_orders) == 1

    def test_partial_then_full_fill(self):
        self.om.add_order(req(volume=10), "O1", now=T0)
        self.om.on_trade(trade("O1", "T1", 4))
        o = self.om.orders["O1"]
        assert o.status is Status.PARTTRADED and o.traded == 4 and o.remaining == 6
        self.om.on_trade(trade("O1", "T2", 6))
        assert o.status is Status.ALLTRADED and o.remaining == 0
        assert not o.is_active

    def test_duplicate_trade_report_is_idempotent(self):
        """实盘常见：同一成交回报重复推送，不能重复累加。"""
        self.om.add_order(req(volume=10), "O1", now=T0)
        self.om.on_trade(trade("O1", "T1", 4))
        self.om.on_trade(trade("O1", "T1", 4))   # 重复
        assert self.om.orders["O1"].traded == 4
        assert len(self.om.trades) == 1

    def test_out_of_order_report_ignored(self):
        """已成交的委托不能被后到的旧状态回报改回 SUBMITTED。"""
        self.om.add_order(req(volume=10), "O1", now=T0)
        self.om.on_trade(trade("O1", "T1", 10))
        assert self.om.orders["O1"].status is Status.ALLTRADED
        self.om.update_status("O1", Status.SUBMITTED)
        assert self.om.orders["O1"].status is Status.ALLTRADED

    def test_rejected_is_final(self):
        self.om.add_order(req(), "O1", now=T0)
        self.om.update_status("O1", Status.REJECTED, reason="资金不足")
        o = self.om.orders["O1"]
        assert o.status is Status.REJECTED and not o.is_active
        assert o.reject_reason == "资金不足"

    def test_unknown_order_report_does_not_crash(self):
        assert self.om.update_status("NOPE", Status.SUBMITTED) is None
        assert self.om.on_trade(trade("NOPE", "T9", 1)) is None

    def test_frozen_volume_signed(self):
        self.om.add_order(req(direction=Direction.LONG, volume=10), "O1", now=T0)
        self.om.add_order(req(direction=Direction.SHORT, volume=4), "O2", now=T0)
        assert self.om.frozen_volume("rb2410.SHFE") == pytest.approx(6.0)


# ---------------------------------------------------------------- 超时撤单
class TestTimeout:
    def test_timeout_orders_detected_and_cancelled(self):
        om = OrderManager(timeout_seconds=60.0)
        om.add_order(req(), "O1", now=T0)
        assert om.timeout_orders(T0 + timedelta(seconds=30)) == []
        late = T0 + timedelta(seconds=90)
        assert len(om.timeout_orders(late)) == 1

        sent = []

        class FakeGw:
            def cancel_order(self, r: CancelRequest):
                sent.append(r)

        reqs = om.cancel_timeouts(FakeGw(), late)
        assert len(reqs) == 1 and len(sent) == 1
        assert om.orders["O1"].status is Status.CANCELLING
        # 已发过撤单的不会重复发
        assert om.cancel_timeouts(FakeGw(), late + timedelta(seconds=10)) == []

    def test_timeout_disabled(self):
        om = OrderManager(timeout_seconds=None)
        om.add_order(req(), "O1", now=T0)
        assert om.timeout_orders(T0 + timedelta(days=1)) == []


# ---------------------------------------------------------------- 持仓推算
class TestNetPositions:
    def test_net_position_from_trades(self):
        om = OrderManager()
        om.add_order(req(volume=10), "O1", now=T0)
        om.on_trade(trade("O1", "T1", 10, price=3500))
        om.add_order(req(direction=Direction.SHORT, volume=4), "O2", now=T0)
        om.on_trade(trade("O2", "T2", 4, price=3600, direction=Direction.SHORT))
        pos = om.net_positions()["rb2410.SHFE"]
        assert pos.volume == pytest.approx(6.0)
        assert pos.price == pytest.approx(3500.0)   # 减仓不改均价

    def test_avg_price_on_add(self):
        om = OrderManager()
        om.add_order(req(volume=10), "O1", now=T0)
        om.on_trade(trade("O1", "T1", 10, price=3000))
        om.add_order(req(volume=10), "O2", now=T0)
        om.on_trade(trade("O2", "T2", 10, price=4000))
        pos = om.net_positions()["rb2410.SHFE"]
        assert pos.volume == pytest.approx(20.0)
        assert pos.price == pytest.approx(3500.0)

    def test_stats(self):
        om = OrderManager()
        om.add_order(req(), "O1", now=T0)
        om.on_trade(trade("O1", "T1", 10))
        s = om.stats()
        assert s["total_orders"] == 1 and s["total_trades"] == 1
        assert s["by_status"]["全部成交"] == 1


# ---------------------------------------------------------------- 对账
class TestReconcile:
    def test_match(self):
        local = {"rb2410.SHFE": position("rb2410.SHFE", 5)}
        remote = {"rb2410.SHFE": position("rb2410.SHFE", 5)}
        rep = reconcile_positions(local, remote)
        assert rep.ok and rep.checked == 1 and not rep.diffs

    def test_zero_positions_ignored(self):
        local = {"rb2410.SHFE": position("rb2410.SHFE", 0)}
        rep = reconcile_positions(local, {})
        assert rep.ok and rep.checked == 0

    def test_mismatch_kinds(self):
        local = {
            "rb2410.SHFE": position("rb2410.SHFE", 5),
            "hc2410.SHFE": position("hc2410.SHFE", 3),
        }
        remote = {
            "rb2410.SHFE": position("rb2410.SHFE", 7),
            "i2409.DCE": position("i2409.DCE", 2),
        }
        rep = reconcile_positions(local, remote)
        kinds = {d.vt_symbol: d.kind for d in rep.diffs}
        assert kinds["rb2410.SHFE"] == "MISMATCH"
        assert kinds["hc2410.SHFE"] == "MISSING_REMOTE"
        assert kinds["i2409.DCE"] == "MISSING_LOCAL"
        assert not rep.ok

    def test_account_tolerance(self):
        rep = reconcile({}, {}, local_equity=1_000_000, remote_equity=1_000_500)
        assert rep.account_ok        # 差 500 < 0.1% 容差
        rep2 = reconcile({}, {}, local_equity=1_000_000, remote_equity=1_050_000)
        assert not rep2.account_ok

    def test_mismatch_triggers_soft_halt(self):
        risk = RiskEngine(RiskLimits(check_trading_session=False))
        local = {"rb2410.SHFE": position("rb2410.SHFE", 5)}
        remote = {"rb2410.SHFE": position("rb2410.SHFE", 9)}
        rep = reconcile(local, remote, risk_engine=risk)
        assert not rep.ok
        assert risk.state.halted and risk.state.halt_level == "SOFT"
        assert "SOFT 熔断" in rep.note
        assert "对账失败" in rep.to_dict()["summary"]

    def test_halt_can_be_disabled(self):
        risk = RiskEngine(RiskLimits(check_trading_session=False))
        rep = reconcile({"a.SHFE": position("a.SHFE", 1)}, {},
                        risk_engine=risk, halt_on_mismatch=False)
        assert not rep.ok and not risk.state.halted


# ---------------------------------------------------------------- Live 集成
class TestLiveEngineIntegration:
    def _engine(self, **kw):
        from quantmind.live.runner import LiveEngine

        class FakeGw:
            gateway_name = "fake"

            def __init__(self):
                self.sent = []
                self.cancelled = []

            def send_order(self, r):
                self.sent.append(r)
                return f"GW-{len(self.sent)}"

            def cancel_order(self, r):
                self.cancelled.append(r)

            def close(self):
                pass

        gw = FakeGw()
        eng = LiveEngine(gw, initial_equity=1_000_000.0, **kw)
        return eng, gw

    def test_risk_blocks_order_before_gateway(self):
        eng, gw = self._engine(
            risk_engine=RiskEngine(
                RiskLimits(check_trading_session=False, max_order_volume=5),
                initial_equity=1_000_000.0,
            )
        )
        eng.last_prices["rb2410.SHFE"] = 3500.0
        assert eng.send_order(req(volume=100)) == ""
        assert gw.sent == []                     # 根本没有到达网关
        oid = eng.send_order(req(volume=2))
        assert oid == "GW-1" and len(gw.sent) == 1

    def test_default_risk_is_conservative_not_none(self):
        """未显式传风控时必须自动启用保守档，绝不能裸奔。"""
        eng, _ = self._engine()
        assert eng.risk is not None
        assert eng.risk.limits.max_order_volume == 10.0

    def test_trade_updates_local_position(self):
        eng, gw = self._engine(
            risk_engine=RiskEngine(RiskLimits.unlimited(), initial_equity=1_000_000.0)
        )
        oid = eng.send_order(req(volume=10))
        eng.on_trade(trade(oid, "T1", 10))
        assert eng.get_position("rb2410.SHFE").volume == pytest.approx(10.0)
        assert eng.status()["orders"]["total_trades"] == 1

    def test_reconcile_via_engine(self):
        eng, _ = self._engine(
            risk_engine=RiskEngine(
                RiskLimits(check_trading_session=False), initial_equity=1_000_000.0
            )
        )
        oid = eng.send_order(req(volume=10))
        eng.on_trade(trade(oid, "T1", 10))
        rep = eng.reconcile({"rb2410.SHFE": position("rb2410.SHFE", 8)})
        assert not rep.ok and eng.risk.state.halted
