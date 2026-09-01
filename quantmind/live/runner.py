"""实盘路由引擎（继承 StrategyContext）。

与回测/模拟**共用同一份策略代码**：只需把策略绑定到 ``LiveEngine``，
其 ``send_order`` 就会把委托路由到真实网关（CTP/XTP/IB 桩），实现「切换路线即可跑实盘」。

与回测路径的关键差异（实盘化 P0 补齐）
--------------------------------------
1. **风控前置**：委托先过 :class:`~quantmind.risk.engine.RiskEngine`，
   不通过直接拒单并广播 ``EVENT_RISK``，策略无法绕过。
2. **本地订单簿**：所有委托由 :class:`~quantmind.live.order_manager.OrderManager`
   跟踪状态，支持超时撤单、部分成交、乱序回报去重。
3. **对账**：``reconcile()`` 比对本地推算持仓与网关查询持仓，
   不一致自动触发 SOFT 熔断（禁开仓）。

发单被拒时 ``send_order`` 返回空字符串 ``""``（而不是抛异常），
策略循环不会因风控中断——这是刻意设计：风控是闸门，不是崩溃源。
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from ..core.constant import Direction, Exchange
from ..core.gateway import BaseGateway, OrderRequest
from ..core.object import AccountData, BarData, PositionData, TickData, TradeData
from ..risk.engine import RiskEngine
from ..risk.limits import RiskLimits
from ..strategy.context import StrategyContext
from .order_manager import OrderManager
from .reconcile import ReconcileReport, reconcile

_logger = logging.getLogger("quantmind.live")
UTC = timezone.utc


class LiveEngine(StrategyContext):
    """实盘引擎：把策略委托经风控闸门路由到网关。

    参数
    ----
    gateway
        已连接的网关实例。
    risk_engine
        风控引擎；``None`` 时**自动创建保守档**（``RiskLimits.conservative()``）。
        显式传 ``False`` 可关闭风控（**仅限内部测试，禁止实盘**）。
    order_manager
        本地订单簿；``None`` 时自动创建（默认 300s 挂单超时）。
    """

    mode = "live"

    def __init__(
        self,
        gateway: BaseGateway,
        event_engine=None,
        history: Optional[Dict[str, List[BarData]]] = None,
        risk_engine: Optional[RiskEngine] = None,
        order_manager: Optional[OrderManager] = None,
        initial_equity: float = 0.0,
    ) -> None:
        self.gateway = gateway
        self.event_engine = event_engine
        self._history = history or {}
        self.positions: Dict[str, PositionData] = {}
        self.last_prices: Dict[str, float] = {}
        self.equity: float = initial_equity

        if risk_engine is False:  # type: ignore[comparison-overlap]
            self.risk: Optional[RiskEngine] = None
            _logger.warning("[LIVE] 风控已被显式关闭——严禁用于真实账户")
        else:
            self.risk = risk_engine or RiskEngine(
                RiskLimits.conservative(),
                event_engine=event_engine,
                initial_equity=initial_equity,
            )
        self.order_manager = order_manager or OrderManager(event_engine=event_engine)

    # ------------------------------------------------------------------
    # StrategyContext 接口
    # ------------------------------------------------------------------
    def send_order(self, req: OrderRequest) -> str:
        vt_symbol = f"{req.symbol}.{req.exchange.value}"
        if self.risk is not None:
            decision = self.risk.check_order(
                req,
                position=self.get_position(vt_symbol),
                last_price=self.last_prices.get(vt_symbol, 0.0),
                equity=self.equity or None,
                active_orders=self.order_manager.active_requests(),
            )
            if not decision.passed:
                _logger.warning("[LIVE] 风控拒单 %s: %s", decision.code.value, decision.reason)
                return ""
        _logger.info("[LIVE] 路由委托至网关 %s: %s", self.gateway.gateway_name, vt_symbol)
        order_id = self.gateway.send_order(req)
        self.order_manager.add_order(req, order_id)
        return order_id

    def get_position(self, vt_symbol: str, include_frozen: bool = True) -> PositionData:
        """返回持仓；默认计入在途挂单（预期净持仓）。

        策略 ``set_target`` 的 delta 基于该值计算，可避免在途单未成交时
        重复发出同向全量委托；风控/对账需要纯已成交持仓时传 ``include_frozen=False``。
        """
        pos = self.positions.get(vt_symbol)
        if pos is None:
            sym, exch = vt_symbol.rsplit(".", 1)
            base = PositionData(symbol=sym, exchange=Exchange(exch),
                                direction=Direction.NET, volume=0.0)
        else:
            base = pos
        if not include_frozen:
            return base
        frozen = self.order_manager.frozen_volume(vt_symbol)
        if abs(frozen) < 1e-9:
            return base
        # 已成交 + 在途 = 预期净持仓（不修改存储的持仓对象）
        return PositionData(
            symbol=base.symbol, exchange=base.exchange, direction=base.direction,
            volume=base.volume + frozen, price=base.price,
        )

    def get_history(self, vt_symbol: str, count: int) -> List[BarData]:
        return self._history.get(vt_symbol, [])[-count:]

    # ------------------------------------------------------------------
    # 网关回调（由 gateway 或 EventEngine 驱动）
    # ------------------------------------------------------------------
    def on_bar(self, bar: BarData) -> None:
        self.last_prices[bar.vt_symbol] = bar.close_price
        self._history.setdefault(bar.vt_symbol, []).append(bar)

    def on_tick(self, tick: TickData) -> None:
        self.last_prices[tick.vt_symbol] = tick.last_price

    def on_trade(self, trade: TradeData) -> None:
        self.order_manager.on_trade(trade)
        self.positions = self.order_manager.net_positions()
        if self.risk is not None:
            self.risk.on_trade(trade.volume, trade.datetime)

    def on_account(self, account: AccountData) -> None:
        self.equity = account.balance
        if self.risk is not None:
            self.risk.update_equity(account.balance)

    # ------------------------------------------------------------------
    # 运维动作
    # ------------------------------------------------------------------
    def check_timeouts(self, now: Optional[datetime] = None) -> int:
        """扫描挂单超时并撤单，返回撤单笔数。应在行情心跳中周期调用。"""
        reqs = self.order_manager.cancel_timeouts(self.gateway, now)
        return len(reqs)

    def reconcile(
        self,
        remote_positions: Dict[str, PositionData],
        remote_equity: Optional[float] = None,
        halt_on_mismatch: bool = True,
    ) -> ReconcileReport:
        """与网关持仓/资金对账；不一致时触发 SOFT 熔断。"""
        report = reconcile(
            local_positions=self.order_manager.net_positions(),
            remote_positions=remote_positions,
            local_equity=self.equity or None,
            remote_equity=remote_equity,
            risk_engine=self.risk,
            halt_on_mismatch=halt_on_mismatch,
        )
        _logger.info("[LIVE] %s", report.summary())
        return report

    def status(self) -> dict:
        """运行状态快照（供 Web/API 展示）。"""
        return {
            "mode": self.mode,
            "gateway": self.gateway.gateway_name,
            "equity": self.equity,
            "orders": self.order_manager.stats(),
            "risk": self.risk.stats() if self.risk is not None else {"enabled": False},
        }

    def connect(self, settings: dict) -> None:
        self.gateway.connect(settings)

    def close(self) -> None:
        self.gateway.close()
