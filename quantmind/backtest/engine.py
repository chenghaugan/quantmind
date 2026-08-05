"""回测引擎（继承 StrategyContext，可作为策略运行上下文）。

设计要点：
  - 无前视：策略在 bar t 发出的委托，于 **下一根 K 线开盘价** 撮合。
  - 净持仓模型：按合约维护带符号净仓，开平自动计算已实现盈亏与手续费。
  - 与模拟/实盘共用同一套策略代码：引擎本身即 ``StrategyContext`` 实现。
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

from ..core.constant import Direction, Exchange, Offset
from ..core.event import EventType
from ..core.gateway import OrderRequest
from ..core.object import BarData, PositionData, TradeData
from ..strategy.context import StrategyContext
from .analyzer import PerformanceAnalyzer, PerformanceReport
from .broker import fill_price
from .cost import default_cost_table, lookup_cost, compute_commission, apply_slippage
from .diagnostics import limit_day_mask

_logger = logging.getLogger("quantmind.backtest")


class BacktestEngine(StrategyContext):
    """批量历史回测引擎。

    成本模型：``cost_table`` 为 ``None``/``False`` 时退回旧式单一费率
    （``commission``/``slippage``）；为 ``True`` 时用内置默认成本预设表
    （按品种差异化：平今、最低手续费、印花税、tick 滑点、保证金）；为 ``dict``
    时按用户自定义表（key 为 vt_symbol 或品种前缀）。
    """

    mode = "backtest"

    def __init__(
        self,
        data: Dict[str, List[BarData]],
        capital: float = 1_000_000.0,
        commission: float = 0.0002,
        slippage: float = 0.0,
        sizes: Optional[Dict[str, float]] = None,
        event_engine=None,
        exclude_limit: bool = False,
        limit_pct: Optional[float] = None,
        cost_table: Union[None, bool, Dict[str, object]] = None,
        enforce_margin: bool = False,
    ) -> None:
        self.data = data
        self.capital = capital
        self.cash = capital
        self.commission = commission
        self.slippage = slippage
        self.sizes = sizes or {}
        self.event_engine = event_engine
        self.exclude_limit = exclude_limit
        self.limit_pct = limit_pct
        self.enforce_margin = enforce_margin
        # 成本表解析
        if cost_table is True:
            self._cost_table: Optional[Dict[str, object]] = default_cost_table()
        elif isinstance(cost_table, dict):
            self._cost_table = cost_table
        else:
            self._cost_table = None
        # 成本统计
        self._open_lots: Dict[str, List[Tuple[datetime, float, float]]] = defaultdict(list)
        self._margin_by_vt: Dict[str, float] = {}
        self.total_commission = 0.0
        self.total_stamp = 0.0
        self.total_impact = 0.0
        self.total_slippage_cost = 0.0
        self.margin_used = 0.0
        self._limit_flag: Dict[str, Dict[datetime, Optional[str]]] = {}

        self.positions: Dict[str, PositionData] = {}  # vt_symbol -> 净持仓(NET)
        self.pending: List[dict] = []                  # {vt_symbol, req, fill_date}
        self.trades: List[TradeData] = []
        self.equity_curve: List[dict] = []
        self.strategy = None
        self._primary: Optional[str] = None
        self._current_date: Optional[datetime] = None
        self._order_seq = 0

        # 时间轴（所有合约交易日并集，升序）
        self.dates = sorted({b.datetime for bars in data.values() for b in bars})
        self._next_date: Dict[datetime, Optional[datetime]] = {}
        for i, d in enumerate(self.dates):
            self._next_date[d] = self.dates[i + 1] if i + 1 < len(self.dates) else None
        # vt_symbol -> {datetime: bar}
        self._lookup: Dict[str, Dict[datetime, BarData]] = {
            vt: {b.datetime: b for b in bars} for vt, bars in data.items()
        }

    # ---- 策略装配 ----
    def add_strategy(self, strategy_class, vt_symbol: str, setting: Optional[dict] = None) -> None:
        self.strategy = strategy_class(self, setting)
        self.strategy.vt_symbols = [vt_symbol]
        self._primary = vt_symbol

    def set_universe(self, vt_symbols: List[str]) -> None:
        """为已装配的策略指定完整标的池（多标的 M5）。

        引擎的回测循环本就会逐标的驱动 ``on_bar``（遍历 ``self.data``），
        这里把候选标的写入 ``strategy.vt_symbols``，供 Universe/Alpha 组件感知全部标的。
        """
        if self.strategy is None:
            raise RuntimeError("先调用 add_strategy 再设置标的池")
        kept = [vt for vt in vt_symbols if vt in self.data]
        self.strategy.vt_symbols = kept or list(self.data.keys())

    # ---- StrategyContext 接口 ----
    def get_history(self, vt_symbol: str, count: int) -> List[BarData]:
        bars = self.data.get(vt_symbol, [])
        return bars[-count:]

    def get_position(self, vt_symbol: str) -> PositionData:
        return self.positions.get(
            vt_symbol, PositionData(symbol=vt_symbol.split(".")[0],
                                    exchange=Exchange(vt_symbol.split(".")[1]),
                                    direction=Direction.NET, volume=0.0)
        )

    def _cost_for(self, vt_symbol: str):
        """返回该合约的成本模型；未启用成本表时返回 None（退回旧式费率）。"""
        if self._cost_table is None:
            return None
        return lookup_cost(vt_symbol, self._cost_table)

    def send_order(self, req: OrderRequest) -> str:
        # 可选：保证金占用约束（默认关闭，避免破坏旧行为）
        if self.enforce_margin and self._cost_table is not None and req.offset in (Offset.OPEN, Offset.NONE):
            cost = self._cost_for(f"{req.symbol}.{req.exchange.value}")
            if cost is not None and req.price and req.price > 0:
                size = self.sizes.get(f"{req.symbol}.{req.exchange.value}", 1.0)
                need = cost.margin_for(req.volume, req.price, size)
                available = self.cash - self.margin_used
                if available < need:
                    _logger.info("保证金不足，拒单 %s x%d (可用%.0f < 需%.0f)",
                                 f"{req.symbol}.{req.exchange.value}", req.volume, available, need)
                    return ""
        self._order_seq += 1
        order_id = f"BT-{self._order_seq}"
        fill_date = self._next_date.get(self._current_date)
        self.pending.append({
            "vt_symbol": f"{req.symbol}.{req.exchange.value}",
            "req": req,
            "fill_date": fill_date,
        })
        if self.event_engine is not None:
            self.event_engine.put_event(EventType.EVENT_ORDER, {"order_id": order_id, "req": req})
        return order_id

    # ---- 运行 ----
    def run(self) -> PerformanceReport:
        if self.strategy is None:
            raise RuntimeError("未添加策略")
        # 预计算涨跌停标记（仅当启用时）
        if self.exclude_limit and self.limit_pct is not None:
            for vt, bars in self.data.items():
                mask = limit_day_mask(bars, self.limit_pct)
                self._limit_flag[vt] = {
                    b.datetime: m for b, m in zip(bars, mask)
                }
        self.strategy.on_init()
        self.strategy.on_start()
        for d in self.dates:
            self._current_date = d
            self._fill_pending(d)
            for vt, bars in self.data.items():
                bar = self._lookup[vt].get(d)
                if bar is not None:
                    self.strategy.on_bar(bar)
            self._mark_to_market(d)
        # 收尾：强制平掉剩余挂单（用最后可得价），并重算末日权益（含强平成交）
        self._force_fill_remaining()
        if self.dates:
            self._mark_to_market(self.dates[-1], replace_last=True)
        self.strategy.on_stop()
        return self.analyze()

    def _fill_pending(self, date: datetime) -> None:
        remaining = []
        for p in self.pending:
            if p["fill_date"] is not None and p["fill_date"] <= date:
                executed = self._execute_fill(p, date)
                if not executed and self.exclude_limit:
                    # 因涨跌停未成交：推到下一交易日重试
                    bar = self._bar_at_or_after(p["vt_symbol"], p["fill_date"] or date)
                    if bar is not None:
                        nxt = self._next_date.get(bar.datetime)
                        if nxt is not None:
                            p["fill_date"] = nxt
                    remaining.append(p)
            else:
                remaining.append(p)
        self.pending = remaining

    def _execute_fill(self, p: dict, date: datetime) -> bool:
        vt = p["vt_symbol"]
        req = p["req"]
        bar = self._bar_at_or_after(vt, p["fill_date"] or date)
        if bar is None:
            return False
        # 涨跌停剔除：涨停日无法买入（开多/平空），跌停日无法卖出（开空/平多）
        if self.exclude_limit:
            flag = self._limit_flag.get(vt, {}).get(bar.datetime)
            if flag == "up" and req.direction == Direction.LONG:
                return False
            if flag == "down" and req.direction == Direction.SHORT:
                return False
        cost = self._cost_for(vt)
        px = apply_slippage(cost, bar, req.direction, self.slippage)
        self._apply_fill(vt, req.direction, req.volume, px, req.offset, bar.datetime)
        self._order_seq += 1
        trade = TradeData(
            symbol=req.symbol, exchange=req.exchange, order_id=f"BT-{self._order_seq}",
            trade_id=f"T-{self._order_seq}", direction=req.direction, offset=req.offset,
            price=px, volume=req.volume, datetime=bar.datetime,
        )
        self.trades.append(trade)
        if self.event_engine is not None:
            self.event_engine.put_event(EventType.EVENT_TRADE, trade)
            self.event_engine.put_event(EventType.EVENT_POSITION, self.get_position(vt))
        return True

    def _apply_fill(self, vt_symbol: str, direction: Direction, volume: float, price: float,
                     offset: Offset = Offset.OPEN, fill_dt: Optional[datetime] = None) -> None:
        cost = self._cost_for(vt_symbol)
        size = self.sizes.get(vt_symbol, 1.0)
        pos = self.get_position(vt_symbol)
        sym, exch = vt_symbol.rsplit(".", 1)
        if pos.volume == 0:
            pos = PositionData(symbol=sym, exchange=Exchange(exch), direction=Direction.NET,
                               volume=0.0, price=0.0)
        signed_vol = volume if direction == Direction.LONG else -volume
        cur_vol = pos.volume
        new_vol = cur_vol + signed_vol

        # 已实现盈亏（减仓/反手时）
        if cur_vol != 0 and signed_vol != 0 and (cur_vol * signed_vol < 0 or abs(new_vol) < abs(cur_vol)):
            closed = min(abs(signed_vol), abs(cur_vol))
            realized = (price - pos.price) * closed * size * (1 if cur_vol > 0 else -1)
            self.cash += realized

        # ---- 开仓批次记账（用于平今判定）----
        fill_dt = fill_dt or self._current_date
        close_today_vol = 0.0
        if cur_vol == 0 or cur_vol * signed_vol > 0:
            # 纯开仓 / 加仓
            self._open_lots[vt_symbol].append((fill_dt, signed_vol, price))
        else:
            # 含平仓：FIFO 从批次弹出，判定平今量
            to_close = min(abs(signed_vol), abs(cur_vol))
            remaining = to_close
            lots = self._open_lots[vt_symbol]
            while remaining > 1e-9 and lots:
                od, ov, op = lots[0]
                take = min(abs(ov), remaining)
                if fill_dt.date() == od.date():
                    close_today_vol += take
                remaining -= take
                if take >= abs(ov) - 1e-9:
                    lots.pop(0)
                else:
                    lots[0] = (od, ov - (take if ov > 0 else -take), op)
            # 反手剩余部分作为新开仓
            new_open = abs(signed_vol) - to_close
            if new_open > 1e-9:
                self._open_lots[vt_symbol].append(
                    (fill_dt, (signed_vol / abs(signed_vol)) * new_open, price))

        # 加权均价（加仓时更新）
        if abs(new_vol) > abs(cur_vol):
            added = abs(new_vol) - abs(cur_vol)
            pos.price = (pos.price * abs(cur_vol) + price * added) / abs(new_vol)
        elif new_vol == 0:
            pos.price = 0.0
        pos.volume = new_vol

        # ---- 成本 ----
        if cost is not None:
            fee, tax, impact = compute_commission(
                cost, volume, price, size, direction, offset, close_today_vol)
            total_cost = fee + tax + impact
            self.cash -= total_cost
            self.total_commission += fee
            self.total_stamp += tax
            self.total_impact += impact
            # 保证金占用重算
            new_margin = cost.margin_for(abs(new_vol), pos.price, size)
            delta = new_margin - self._margin_by_vt.get(vt_symbol, 0.0)
            self.margin_used += delta
            self._margin_by_vt[vt_symbol] = new_margin
        else:
            # 旧式单一费率
            comm = price * volume * size * self.commission
            self.cash -= comm
            self.total_commission += comm

        # 隐性滑点成本：用成交价相对当根开盘价的偏离累计（买高卖低为正的成本拖累）
        open_px = self._lookup[vt_symbol].get(fill_dt, None)
        if open_px is not None:
            self.total_slippage_cost += (price - open_px.open_price) * signed_vol * size

        self.positions[vt_symbol] = pos

    def _bar_at_or_after(self, vt_symbol: str, date: datetime) -> Optional[BarData]:
        lookup = self._lookup.get(vt_symbol, {})
        if date in lookup:
            return lookup[date]
        # 向后找最近一根
        for d in self.dates:
            if d >= date and d in lookup:
                return lookup[d]
        return None

    def _force_fill_remaining(self) -> None:
        for p in list(self.pending):
            vt = p["vt_symbol"]
            bar = self._bar_at_or_after(vt, self.dates[-1]) if self.dates else None
            if bar is not None:
                cost = self._cost_for(vt)
                px = apply_slippage(cost, bar, p["req"].direction, self.slippage)
                self._apply_fill(vt, p["req"].direction, p["req"].volume, px,
                                 p["req"].offset, bar.datetime)
                self.trades.append(TradeData(
                    symbol=p["req"].symbol, exchange=p["req"].exchange,
                    order_id="BT-close", trade_id="T-close",
                    direction=p["req"].direction, offset=p["req"].offset,
                    price=px, volume=p["req"].volume, datetime=bar.datetime,
                ))

    def _mark_to_market(self, date: datetime, replace_last: bool = False) -> None:
        equity = self.cash
        for vt, pos in self.positions.items():
            if pos.volume == 0:
                continue
            size = self.sizes.get(vt, 1.0)
            bar = self._bar_at_or_after(vt, date)
            if bar is not None:
                equity += pos.volume * (bar.close_price - pos.price) * size
        entry = {"date": date.isoformat(), "equity": equity}
        if replace_last and self.equity_curve:
            self.equity_curve[-1] = entry
        else:
            self.equity_curve.append(entry)

    def analyze(self) -> PerformanceReport:
        rep = PerformanceAnalyzer().analyze(self.equity_curve, self.trades)
        rep.total_commission = self.total_commission
        rep.total_stamp_tax = self.total_stamp
        rep.total_impact = self.total_impact
        rep.total_slippage = self.total_slippage_cost
        rep.total_cost = self.total_commission + self.total_stamp + self.total_impact + self.total_slippage_cost
        rep.margin_used = self.margin_used
        net_pnl = rep.final_equity - self.capital
        rep.cost_ratio = (rep.total_cost / abs(net_pnl)) if net_pnl not in (0.0, -self.capital) and net_pnl != 0 else 0.0
        return rep
