"""模拟券商网关（SimGateway）：实盘链路的离线可测完全实现。

CTP/XTP/IB 桩只能打日志、模拟即时回执，无法在无网络/无凭证时验证
``live/runner`` + ``OrderManager`` + ``reconcile`` 的完整闭环。本模块提供
**真正的在进程内撮合券商**：对一个 OrderRequest 做撮合（可全部/部分成交、
可拒绝、可挂单超时），并把 ``OrderData``/``TradeData``/``PositionData``/
``AccountData`` 事件广播到 EventEngine——让实盘引擎的所有状态逻辑（委托状态机、
理论持仓、对账）在纯本地闭环里被真正压测。

同时提供 :func:`simulate_one_trade`，供 CTP/XTP/IB 桩在"模拟成交"时复用，
把它们的发单从"只打日志"升级为"产出真实回报事件"（半可用）。

**注意**：SimGateway 仍是模拟撮合，不连接任何真实交易所；真实成交路径
仍由各 provider 网关（simnow/openctp/IB）接替。
"""
from __future__ import annotations

import itertools
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

from ..core.constant import Direction, Offset, Status
from ..core.event import EventType
from ..core.gateway import (
    BaseGateway,
    CancelRequest,
    OrderRequest,
    SubscribeRequest,
)
from ..core.object import AccountData, OrderData, PositionData, TradeData

_logger = logging.getLogger("quantmind.live.sim")

# 全局单调序列号：保证同毫秒内多笔成交的 trade_id 唯一
_trade_seq = itertools.count()
UTC = timezone.utc


def simulate_one_trade(
    event_engine,
    order_id: str,
    req: OrderRequest,
    price: Optional[float] = None,
    fill_volume: Optional[float] = None,
    gateway_name: str = "SIM",
    status: Optional[Status] = None,
) -> TradeData:
    """构造并广播一笔模拟成交回报（也可由桩网关复用）。

    :param event_engine: 事件引擎（None 则只返回不广播）。
    :param req: 原始委托。
    :param price: 成交价；默认用委托价（市价则用 1.0 占位）。
    :param fill_volume: 成交手数；默认全部成交。
    :param status: 广播的订单终态，默认 ``ALLTRADED``。
    :returns: 构造出的 TradeData。
    """
    px = price if price is not None else (req.price if req.price > 0 else 1.0)
    vol = fill_volume if fill_volume is not None else req.volume
    trade = TradeData(
        gateway_name=gateway_name,
        symbol=req.symbol, exchange=req.exchange,
        order_id=order_id,
        # 同毫秒多笔成交（不同 order）会撞 id 被 order_manager 幂等去重丢弃，
        # 追加全局单调序列号保证唯一
        trade_id=f"{gateway_name}-{int(time.time() * 1000)}-{next(_trade_seq)}",
        direction=req.direction, offset=req.offset,
        price=px, volume=vol,
        datetime=datetime.now(UTC),
    )
    # 广播成交 + 订单终态
    if event_engine is not None:
        from ..core.object import OrderData as OD
        od = OD(
            gateway_name=gateway_name, symbol=req.symbol, exchange=req.exchange,
            order_id=order_id, type=req.type, direction=req.direction, offset=req.offset,
            price=req.price, volume=req.volume, traded=vol,
            status=status or Status.ALLTRADED, datetime=datetime.now(UTC),
        )
        event_engine.put_event(EventType.EVENT_ORDER, od)
        event_engine.put_event(EventType.EVENT_TRADE, trade)
    return trade


class SimGateway(BaseGateway):
    """在进程内模拟撮合的券商网关。

    撮合规则（可配置）：
      - ``fill_ratio``：单笔委托成交比例（<=1）。``1.0``=全成，``0.5``=半成。
      - ``partial_fills``：是否拆成多次部分成交回报。
      - ``reject_rate``：0~1，发单被拒比例（模拟极端行情拒单/风控拒单）。
      - ``latency``：模拟成交延迟秒数（仅影响状态时间戳，离线即时返回）。
    持仓/资金由成交回报累计，``query_position``/``query_account`` 返回结果，
    并能在 ``connect``/``query_*`` 时模拟重连与状态自检。
    """

    def __init__(self, event_engine, gateway_name: str = "SIM",
                 fill_ratio: float = 1.0, partial_fills: bool = False,
                 reject_rate: float = 0.0, latency: float = 0.0) -> None:
        super().__init__(event_engine, gateway_name)
        self.fill_ratio = fill_ratio
        self.partial_fills = partial_fills
        self.reject_rate = reject_rate
        self.latency = latency
        self.connected = False
        self._seq = 0
        self._orders: Dict[str, OrderRequest] = {}
        self.trades: List[TradeData] = []
        # 持仓：vt_symbol -> PositionData
        self.positions: Dict[str, PositionData] = {}
        self.balance: float = 1_000_000.0
        self._reconnect_count = 0

    # ---- BaseGateway ----
    def connect(self, settings: dict) -> None:
        self.connected = True
        self._reconnect_count += 1
        _logger.info("[%s] 连接成功 (connect #%d)", self.gateway_name, self._reconnect_count)
        # 连接成功即广播一次账户快照
        self._push_account()

    def subscribe(self, req: SubscribeRequest) -> None:
        _logger.debug("[%s] 订阅 %s.%s", self.gateway_name, req.symbol, req.exchange.value)

    def send_order(self, req: OrderRequest) -> str:
        self._seq += 1
        oid = f"{self.gateway_name}-{self._seq}"
        self._orders[oid] = req
        # 模拟风控拒单
        if self.reject_rate > 0 and (self._seq % max(1, int(1 / max(self.reject_rate, 1e-9)))) == 0:
            self._emit_order(oid, req, status=Status.REJECTED)
            _logger.warning("[%s] 模拟拒单 %s", self.gateway_name, oid)
            return oid
        # 撮合
        volume = req.volume
        if self.fill_ratio < 1.0:
            volume = round(volume * self.fill_ratio, 4)
        if self.partial_fills and volume > 0 and volume < req.volume:
            self._emit_order(oid, req, status=Status.PARTTRADED, traded=volume)
            # 剩余部分在最后全额成交
            rem = req.volume - volume
            self._fill(oid, req, price=req.price, vol=rem)
            self._emit_order(oid, req, status=Status.ALLTRADED, traded=req.volume)
        else:
            self._fill(oid, req, price=req.price if req.price > 0 else 1.0,
                       vol=volume)
            self._emit_order(oid, req, status=Status.ALLTRADED, traded=volume)
        self._push_account()
        return oid

    def cancel_order(self, req: CancelRequest) -> None:
        if req.order_id in self._orders:
            self._emit_order(req.order_id, self._orders[req.order_id], status=Status.CANCELLED)
        _logger.info("[%s] 撤单 %s", self.gateway_name, req.order_id)

    def query_position(self) -> Dict[str, PositionData]:
        if self.event_engine is not None:
            for vt, p in self.positions.items():
                self.event_engine.put_event(EventType.EVENT_POSITION, p)
        return dict(self.positions)

    def query_account(self) -> AccountData:
        acct = AccountData(account_id=f"{self.gateway_name}_ACCOUNT",
                           balance=self.balance, available=self.balance)
        if self.event_engine is not None:
            self.event_engine.put_event(EventType.EVENT_ACCOUNT, acct)
        return acct

    def close(self) -> None:
        self.connected = False
        _logger.info("[%s] 断开", self.gateway_name)

    def simulate_disconnect(self) -> None:
        """模拟断线（用于重连/状态恢复测试）。"""
        self.connected = False
        _logger.warning("[%s] 模拟断线", self.gateway_name)

    # ---- 内部 ----
    def _fill(self, oid: str, req: OrderRequest, price: float, vol: float) -> None:
        if vol <= 0:
            return
        vt = f"{req.symbol}.{req.exchange.value}"
        trade = TradeData(
            gateway_name=self.gateway_name, symbol=req.symbol, exchange=req.exchange,
            order_id=oid, trade_id=f"{self.gateway_name}-{self._seq}-{len(self.trades)}",
            direction=req.direction, offset=req.offset, price=price, volume=vol,
            datetime=datetime.now(UTC),
        )
        self.trades.append(trade)
        # 累加净持仓与资金
        pos = self.positions.get(vt, PositionData(
            symbol=req.symbol, exchange=req.exchange, direction=Direction.NET,
            volume=0.0, price=0.0))
        signed = vol if req.direction == Direction.LONG else -vol
        cur = pos.volume
        new = cur + signed
        if new == 0:
            pos.price = 0.0
        elif cur == 0 or cur * new < 0:
            # 反手（穿越零仓）：新方向以最新成交价计成本
            pos.price = price
        elif abs(new) > abs(cur):
            added = abs(new) - abs(cur)
            pos.price = (pos.price * abs(cur) + price * added) / abs(new)
        pos.volume = new
        pos.pnl = 0.0
        self.positions[vt] = pos
        self.balance -= price * vol  # 简化：全额占用资金
        if self.event_engine is not None:
            self.event_engine.put_event(EventType.EVENT_TRADE, trade)
            self.event_engine.put_event(EventType.EVENT_POSITION, pos)

    def _emit_order(self, oid: str, req: OrderRequest, status: Status,
                    traded: Optional[float] = None) -> None:
        if self.event_engine is None:
            return
        od = OrderData(
            gateway_name=self.gateway_name, symbol=req.symbol, exchange=req.exchange,
            order_id=oid, type=req.type, direction=req.direction, offset=req.offset,
            price=req.price, volume=req.volume, traded=traded or 0.0, status=status,
            datetime=datetime.now(UTC),
        )
        self.event_engine.put_event(EventType.EVENT_ORDER, od)

    def _push_account(self) -> None:
        if self.event_engine is not None:
            acct = AccountData(account_id=f"{self.gateway_name}_ACCOUNT",
                               balance=self.balance, available=self.balance)
            self.event_engine.put_event(EventType.EVENT_ACCOUNT, acct)

    def stats(self) -> dict:
        return {
            "connected": self.connected,
            "orders": len(self._orders),
            "trades": len(self.trades),
            "balance": round(self.balance, 2),
            "positions": {vt: {"volume": p.volume, "avg": round(p.price, 4)}
                          for vt, p in self.positions.items() if p.volume != 0},
        }


__all__ = ["SimGateway", "simulate_one_trade"]
