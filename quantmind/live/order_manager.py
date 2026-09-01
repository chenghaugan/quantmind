"""本地订单簿与委托状态机（实盘化 P0）。

回测里「发单即成交」，实盘不是：一笔委托要经历
``SUBMITTING → SUBMITTED → PARTTRADED → ALLTRADED``（或 ``CANCELLED`` / ``REJECTED``），
中间可能收到乱序回报、重复回报、超时未回报。没有本地订单簿，就无法回答
三个实盘必答问题：

  1. 我现在**挂着几笔单**？（防重复下单、防自成交）
  2. 哪些单**挂了太久**没成交？（需要撤单重发，否则策略状态与实际持仓脱节）
  3. 我的**理论持仓**是多少？（用于与网关持仓对账，见 :mod:`quantmind.live.reconcile`）

状态机
------
::

    SUBMITTING ──(网关受理)──> SUBMITTED ──(部分成交)──> PARTTRADED
         │                        │                        │
         │                        └──(全部成交)────────────>┴──> ALLTRADED[终态]
         ├──(拒单)──> REJECTED[终态]
         └──(撤单)──> CANCELLING ──> CANCELLED[终态]

状态只允许**向前**流转：已到终态的委托不会被后到的旧回报覆盖
（实盘回报乱序是常态，这条规则能挡掉大部分「幽灵持仓」）。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from ..core.constant import Direction, Exchange, Offset, Status
from ..core.event import EventType
from ..core.gateway import CancelRequest, OrderRequest
from ..core.object import OrderData, PositionData, TradeData

_logger = logging.getLogger("quantmind.live.order")
UTC = timezone.utc

#: 活动状态（尚未终结，占用挂单额度）
ACTIVE_STATUSES = {
    Status.SUBMITTING,
    Status.SUBMITTED,
    Status.PARTTRADED,
    Status.CANCELLING,
}
#: 终态（不可再流转）
FINAL_STATUSES = {Status.ALLTRADED, Status.CANCELLED, Status.REJECTED}

#: 状态优先级：数值大的不可被数值小的覆盖
_STATUS_RANK = {
    Status.SUBMITTING: 0,
    Status.SUBMITTED: 1,
    Status.PARTTRADED: 2,
    Status.CANCELLING: 3,
    Status.CANCELLED: 10,
    Status.REJECTED: 10,
    Status.ALLTRADED: 10,
}


@dataclass
class ManagedOrder:
    """本地跟踪的一笔委托。"""

    order_id: str
    request: OrderRequest
    status: Status = Status.SUBMITTING
    traded: float = 0.0
    submit_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    update_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    cancel_sent: bool = False
    reject_reason: str = ""
    trade_ids: List[str] = field(default_factory=list)

    @property
    def vt_symbol(self) -> str:
        return f"{self.request.symbol}.{self.request.exchange.value}"

    @property
    def volume(self) -> float:
        return self.request.volume

    @property
    def remaining(self) -> float:
        return max(0.0, self.request.volume - self.traded)

    @property
    def is_active(self) -> bool:
        return self.status in ACTIVE_STATUSES

    def age_seconds(self, now: Optional[datetime] = None) -> float:
        now = now or datetime.now(UTC)
        return (now - self.submit_time).total_seconds()

    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "vt_symbol": self.vt_symbol,
            "direction": self.request.direction.value,
            "offset": self.request.offset.value,
            "price": self.request.price,
            "volume": self.request.volume,
            "traded": self.traded,
            "remaining": self.remaining,
            "status": self.status.value,
            "submit_time": self.submit_time.isoformat(),
            "update_time": self.update_time.isoformat(),
            "cancel_sent": self.cancel_sent,
            "reject_reason": self.reject_reason,
        }


class OrderManager:
    """本地订单簿 + 成交归集 + 超时撤单。

    参数
    ----
    timeout_seconds
        委托挂单超时（秒）；超过则进入 :meth:`timeout_orders` 返回列表。
        ``None`` 表示不做超时管理。
    event_engine
        事件引擎，状态变更时广播 ``EVENT_ORDER`` / ``EVENT_POSITION``。
    """

    def __init__(
        self,
        timeout_seconds: Optional[float] = 300.0,
        event_engine=None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.event_engine = event_engine
        self.orders: Dict[str, ManagedOrder] = {}
        self.trades: List[TradeData] = []
        self._seen_trade_ids: set = set()
        self._local_seq = 0

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    def add_order(
        self,
        req: OrderRequest,
        order_id: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> ManagedOrder:
        """登记一笔已发出的委托。"""
        now = now or datetime.now(UTC)
        if not order_id:
            self._local_seq += 1
            order_id = f"LOCAL-{self._local_seq}"
        order = ManagedOrder(order_id=order_id, request=req, submit_time=now, update_time=now)
        self.orders[order_id] = order
        self._emit_order(order)
        return order

    def update_status(
        self,
        order_id: str,
        status: Status,
        traded: Optional[float] = None,
        reason: str = "",
        now: Optional[datetime] = None,
    ) -> Optional[ManagedOrder]:
        """处理网关委托回报（只允许状态向前流转）。"""
        order = self.orders.get(order_id)
        if order is None:
            _logger.warning("[ORDER] 收到未知委托回报 order_id=%s", order_id)
            return None
        if order.status in FINAL_STATUSES and _STATUS_RANK[status] <= _STATUS_RANK[order.status]:
            _logger.debug("[ORDER] 忽略过期回报 %s: %s -> %s", order_id, order.status, status)
            return order
        if _STATUS_RANK[status] < _STATUS_RANK[order.status]:
            _logger.debug("[ORDER] 忽略乱序回报 %s: %s -> %s", order_id, order.status, status)
            return order
        order.status = status
        if traded is not None:
            order.traded = max(order.traded, traded)
        if reason:
            order.reject_reason = reason
        order.update_time = now or datetime.now(UTC)
        self._emit_order(order)
        return order

    def on_trade(self, trade: TradeData) -> Optional[ManagedOrder]:
        """处理成交回报（幂等：同一 ``trade_id`` 只累计一次）。"""
        if trade.trade_id and trade.trade_id in self._seen_trade_ids:
            _logger.debug("[ORDER] 忽略重复成交回报 %s", trade.trade_id)
            return self.orders.get(trade.order_id)
        if trade.trade_id:
            self._seen_trade_ids.add(trade.trade_id)
        self.trades.append(trade)

        order = self.orders.get(trade.order_id)
        if order is None:
            _logger.warning("[ORDER] 成交回报无对应本地委托 order_id=%s", trade.order_id)
            return None
        order.traded += trade.volume
        order.trade_ids.append(trade.trade_id)
        order.update_time = trade.datetime
        if order.traded >= order.volume - 1e-9:
            order.status = Status.ALLTRADED
        elif order.status not in FINAL_STATUSES:
            order.status = Status.PARTTRADED
        self._emit_order(order)
        return order

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    @property
    def active_orders(self) -> List[ManagedOrder]:
        return [o for o in self.orders.values() if o.is_active]

    def active_requests(self) -> List[OrderRequest]:
        """活动委托的原始请求（供风控自成交检查）。"""
        return [o.request for o in self.active_orders]

    def active_for(self, vt_symbol: str) -> List[ManagedOrder]:
        return [o for o in self.active_orders if o.vt_symbol == vt_symbol]

    def frozen_volume(self, vt_symbol: str) -> float:
        """该合约挂单中尚未成交的手数（带符号：多为正、空为负）。"""
        total = 0.0
        for o in self.active_for(vt_symbol):
            sign = 1.0 if o.request.direction == Direction.LONG else -1.0
            total += sign * o.remaining
        return total

    def timeout_orders(self, now: Optional[datetime] = None) -> List[ManagedOrder]:
        """返回挂单超时且尚未发出撤单的委托。"""
        if self.timeout_seconds is None:
            return []
        now = now or datetime.now(UTC)
        return [
            o for o in self.active_orders
            if not o.cancel_sent and o.age_seconds(now) >= self.timeout_seconds
        ]

    def cancel_timeouts(self, gateway=None, now: Optional[datetime] = None) -> List[CancelRequest]:
        """对超时委托生成（并可选发送）撤单请求。"""
        reqs: List[CancelRequest] = []
        for o in self.timeout_orders(now):
            req = CancelRequest(
                order_id=o.order_id,
                symbol=o.request.symbol,
                exchange=o.request.exchange,
            )
            o.cancel_sent = True
            o.status = Status.CANCELLING
            o.update_time = now or datetime.now(UTC)
            reqs.append(req)
            _logger.warning(
                "[ORDER] 委托 %s 挂单 %.0fs 超时，发出撤单", o.order_id, o.age_seconds(now)
            )
            if gateway is not None:
                try:
                    gateway.cancel_order(req)
                except Exception:  # pragma: no cover
                    _logger.exception("[ORDER] 撤单失败 %s", o.order_id)
            self._emit_order(o)
        return reqs

    def net_positions(self) -> Dict[str, PositionData]:
        """由**成交回报**推算的本地净持仓（对账基准）。"""
        acc: Dict[str, List[float]] = {}   # vt -> [净手数, 成本额]
        for t in self.trades:
            vt = f"{t.symbol}.{t.exchange.value}"
            signed = t.volume if t.direction == Direction.LONG else -t.volume
            cur = acc.setdefault(vt, [0.0, 0.0])
            prev_vol = cur[0]
            new_vol = prev_vol + signed
            if prev_vol != 0 and prev_vol * new_vol < 0:
                # 反手（穿越零仓）：新方向成本以最新成交价计（与 sim/backtest 同口径）
                cur[1] = t.price
            elif abs(new_vol) > abs(prev_vol):          # 加仓 → 更新均价
                added = abs(new_vol) - abs(prev_vol)
                cur[1] = (cur[1] * abs(prev_vol) + t.price * added) / abs(new_vol)
            elif new_vol == 0:
                cur[1] = 0.0
            cur[0] = new_vol
        out: Dict[str, PositionData] = {}
        for vt, (vol, price) in acc.items():
            sym, exch = vt.rsplit(".", 1)
            out[vt] = PositionData(
                symbol=sym, exchange=Exchange(exch), direction=Direction.NET,
                volume=vol, price=price,
            )
        return out

    def stats(self) -> dict:
        by_status: Dict[str, int] = {}
        for o in self.orders.values():
            by_status[o.status.value] = by_status.get(o.status.value, 0) + 1
        return {
            "total_orders": len(self.orders),
            "active_orders": len(self.active_orders),
            "total_trades": len(self.trades),
            "by_status": by_status,
            "positions": {
                vt: {"volume": p.volume, "avg_price": round(p.price, 4)}
                for vt, p in self.net_positions().items() if p.volume != 0
            },
        }

    # ------------------------------------------------------------------
    def _emit_order(self, order: ManagedOrder) -> None:
        if self.event_engine is None:
            return
        data = OrderData(
            symbol=order.request.symbol,
            exchange=order.request.exchange,
            order_id=order.order_id,
            type=order.request.type,
            direction=order.request.direction,
            offset=order.request.offset,
            price=order.request.price,
            volume=order.request.volume,
            traded=order.traded,
            status=order.status,
            datetime=order.update_time,
        )
        try:
            self.event_engine.put_event(EventType.EVENT_ORDER, data)
        except Exception:  # pragma: no cover
            _logger.exception("[ORDER] 事件广播失败")
