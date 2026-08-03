"""模拟交易引擎（继承 StrategyContext，可作为策略运行上下文）。

与 BacktestEngine 共用同一套策略代码：区别在于 PaperEngine 面向「实时/回放」，
维护实时持仓与资金，并通过 EventEngine 广播 bar/signal/position/trade/account 事件
（驱动 Web 实时监控与告警）。

- 回放模式（``run_replay``）：把历史 K 线逐根喂给策略，模拟实时撮合。
- 实时模式（配合网关）：订阅行情后逐根推入，撮合逻辑一致。
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional

from ..core.constant import Direction, Exchange
from ..core.event import EventType
from ..core.gateway import OrderRequest
from ..core.object import AccountData, BarData, PositionData, TradeData
from ..strategy.context import StrategyContext
from ..backtest.broker import fill_price

_logger = logging.getLogger("quantmind.paper")


class PaperContext(StrategyContext):
    """单个策略在模拟引擎中的上下文（委托路由到共享引擎）。"""

    mode = "paper"

    def __init__(self, engine: "PaperEngine", vt_symbol: str) -> None:
        self.engine = engine
        self.vt_symbol = vt_symbol
        self.event_engine = engine.event_engine
        self.strategy = None

    def send_order(self, req: OrderRequest) -> str:
        return self.engine.send_order(req)

    def get_position(self, vt_symbol: str) -> PositionData:
        return self.engine.get_position(vt_symbol)

    def get_history(self, vt_symbol: str, count: int) -> List[BarData]:
        return self.engine.get_history(vt_symbol, count)


class PaperEngine:
    """模拟交易引擎（支持多策略组合）。"""

    def __init__(
        self,
        event_engine=None,
        capital: float = 1_000_000.0,
        commission: float = 0.0002,
        slippage: float = 0.0,
        sizes: Optional[Dict[str, float]] = None,
        risk_engine=None,
    ) -> None:
        self.event_engine = event_engine
        self.capital = capital
        self.cash = capital
        self.commission = commission
        self.slippage = slippage
        self.sizes = sizes or {}
        # 风控引擎（可选）。模拟盘默认不启用，以免历史回放被交易时段闸门拦下；
        # 做「实盘前演练」时应显式传入与实盘同一份 RiskLimits，验证限额是否合理。
        self.risk = risk_engine
        self.risk_rejected: List[dict] = []
        self.equity: float = capital
        self.positions: Dict[str, PositionData] = {}
        self.pending: List[dict] = []
        self.trades: List[TradeData] = []
        self.contexts: List[PaperContext] = []
        self._order_seq = 0
        self._current_date: Optional[datetime] = None
        self._data: Dict[str, List[BarData]] = {}
        self._lookup: Dict[str, Dict[datetime, BarData]] = {}
        self.dates: List[datetime] = []
        self._next_date: Dict[datetime, Optional[datetime]] = {}

    # ---- 装配 ----
    def add_strategy(self, strategy_class, vt_symbol: str, setting: Optional[dict] = None) -> PaperContext:
        ctx = PaperContext(self, vt_symbol)
        strat = strategy_class(ctx, setting)
        strat.vt_symbols = [vt_symbol]
        ctx.strategy = strat
        self.contexts.append(ctx)
        return ctx

    # ---- StrategyContext 接口（供 PaperContext 委托） ----
    def send_order(self, req: OrderRequest) -> str:
        vt_symbol = f"{req.symbol}.{req.exchange.value}"
        if self.risk is not None:
            decision = self.risk.check_order(
                req,
                position=self.get_position(vt_symbol),
                last_price=self._last_price(vt_symbol),
                equity=self.equity,
                now=self._current_date,
            )
            if not decision.passed:
                self.risk_rejected.append(decision.to_dict())
                _logger.warning("[PAPER] 风控拒单 %s: %s", decision.code.value, decision.reason)
                return ""
        self._order_seq += 1
        order_id = f"PAPER-{self._order_seq}"
        fill_date = self._next_date.get(self._current_date)
        self.pending.append({
            "vt_symbol": f"{req.symbol}.{req.exchange.value}",
            "req": req,
            "fill_date": fill_date,
        })
        self._emit(EventType.EVENT_ORDER, {"order_id": order_id, "req": req})
        return order_id

    def get_position(self, vt_symbol: str) -> PositionData:
        return self.positions.get(
            vt_symbol, PositionData(symbol=vt_symbol.split(".")[0],
                                    exchange=Exchange(vt_symbol.split(".")[1]),
                                    direction=Direction.NET, volume=0.0)
        )

    def get_history(self, vt_symbol: str, count: int) -> List[BarData]:
        return self._data.get(vt_symbol, [])[-count:]

    # ---- 运行（回放） ----
    def run_replay(self, data: Dict[str, List[BarData]], strategies_init: bool = True) -> dict:
        self._data = data
        self._lookup = {vt: {b.datetime: b for b in bars} for vt, bars in data.items()}
        self.dates = sorted({b.datetime for bars in data.values() for b in bars})
        self._next_date = {}
        for i, d in enumerate(self.dates):
            self._next_date[d] = self.dates[i + 1] if i + 1 < len(self.dates) else None

        if strategies_init:
            for ctx in self.contexts:
                ctx.strategy.on_init()
                ctx.strategy.on_start()

        for d in self.dates:
            self._current_date = d
            self._fill_pending(d)
            for vt, bars in data.items():
                bar = self._lookup[vt].get(d)
                if bar is not None:
                    self._emit(EventType.EVENT_BAR, bar)
                    for ctx in self.contexts:
                        if ctx.vt_symbol == vt:
                            ctx.strategy.on_bar(bar)
            self._mark_to_market(d)
        self._force_fill_remaining()
        self._mark_to_market(self.dates[-1] if self.dates else datetime.now())
        for ctx in self.contexts:
            ctx.strategy.on_stop()
        return self.summary()

    # ---- 撮合/持仓（与回测一致的净持仓模型） ----
    def _fill_pending(self, date: datetime) -> None:
        remaining = []
        for p in self.pending:
            if p["fill_date"] is not None and p["fill_date"] <= date:
                self._execute_fill(p, date)
            else:
                remaining.append(p)
        self.pending = remaining

    def _execute_fill(self, p: dict, date: datetime) -> None:
        vt = p["vt_symbol"]
        req = p["req"]
        bar = self._bar_at_or_after(vt, p["fill_date"] or date)
        if bar is None:
            return
        px = fill_price(bar, req.direction, self.slippage)
        self._apply_fill(vt, req.direction, req.volume, px)
        self._order_seq += 1
        trade = TradeData(
            symbol=req.symbol, exchange=req.exchange, order_id=f"PAPER-{self._order_seq}",
            trade_id=f"PT-{self._order_seq}", direction=req.direction, offset=req.offset,
            price=px, volume=req.volume, datetime=bar.datetime,
        )
        self.trades.append(trade)
        if self.risk is not None:
            self.risk.on_trade(req.volume, bar.datetime)
        self._emit(EventType.EVENT_TRADE, trade)
        self._emit(EventType.EVENT_POSITION, self.get_position(vt))

    def _apply_fill(self, vt_symbol: str, direction: Direction, volume: float, price: float) -> None:
        size = self.sizes.get(vt_symbol, 1.0)
        pos = self.get_position(vt_symbol)
        sym, exch = vt_symbol.rsplit(".", 1)
        if pos.volume == 0:
            pos = PositionData(symbol=sym, exchange=Exchange(exch), direction=Direction.NET, volume=0.0, price=0.0)
        signed_vol = volume if direction == Direction.LONG else -volume
        cur_vol = pos.volume
        new_vol = cur_vol + signed_vol
        if cur_vol != 0 and signed_vol != 0 and (cur_vol * signed_vol < 0 or abs(new_vol) < abs(cur_vol)):
            closed = min(abs(signed_vol), abs(cur_vol))
            realized = (price - pos.price) * closed * size * (1 if cur_vol > 0 else -1)
            self.cash += realized
        if abs(new_vol) > abs(cur_vol):
            added = abs(new_vol) - abs(cur_vol)
            pos.price = (pos.price * abs(cur_vol) + price * added) / abs(new_vol)
        elif new_vol == 0:
            pos.price = 0.0
        pos.volume = new_vol
        self.cash -= price * volume * size * self.commission
        self.positions[vt_symbol] = pos

    def _bar_at_or_after(self, vt_symbol: str, date: datetime) -> Optional[BarData]:
        lookup = self._lookup.get(vt_symbol, {})
        if date in lookup:
            return lookup[date]
        for d in self.dates:
            if d >= date and d in lookup:
                return lookup[d]
        return None

    def _force_fill_remaining(self) -> None:
        for p in list(self.pending):
            vt = p["vt_symbol"]
            bar = self._bar_at_or_after(vt, self.dates[-1]) if self.dates else None
            if bar is not None:
                px = fill_price(bar, p["req"].direction, self.slippage)
                self._apply_fill(vt, p["req"].direction, p["req"].volume, px)

    def _mark_to_market(self, date: datetime) -> None:
        equity = self.cash
        for vt, pos in self.positions.items():
            if pos.volume == 0:
                continue
            size = self.sizes.get(vt, 1.0)
            bar = self._bar_at_or_after(vt, date)
            if bar is not None:
                equity += pos.volume * (bar.close_price - pos.price) * size
        self.equity = equity
        if self.risk is not None:
            self.risk.update_equity(equity, date)
        acct = AccountData(account_id="PAPER", balance=equity, available=equity)
        self._emit(EventType.EVENT_ACCOUNT, acct)

    def _last_price(self, vt_symbol: str) -> float:
        """当前日期该合约收盘价（供风控价格偏离/市值检查）。"""
        if self._current_date is None:
            return 0.0
        bar = self._lookup.get(vt_symbol, {}).get(self._current_date)
        return bar.close_price if bar is not None else 0.0

    def _emit(self, etype, data) -> None:
        if self.event_engine is not None:
            self.event_engine.put_event(etype, data)

    def summary(self) -> dict:
        out = {
            "mode": "paper",
            "cash": round(self.cash, 2),
            "positions": {
                vt: {"volume": p.volume, "avg_price": round(p.price, 2)}
                for vt, p in self.positions.items() if p.volume != 0
            },
            "trade_count": len(self.trades),
        }
        if self.risk is not None:
            out["risk"] = {
                "rejected": len(self.risk_rejected),
                "state": self.risk.state.to_dict(),
            }
        return out
