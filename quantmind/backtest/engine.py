"""回测引擎（继承 StrategyContext，可作为策略运行上下文）。

设计要点：
  - 无前视：策略在 bar t 发出的委托，于 **下一根 K 线开盘价** 撮合。
  - 净持仓模型：按合约维护带符号净仓，开平自动计算已实现盈亏与手续费。
  - 与模拟/实盘共用同一套策略代码：引擎本身即 ``StrategyContext`` 实现。
"""
from __future__ import annotations

import logging
from bisect import bisect_right
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
from .diagnostics import limit_day_mask, limit_price_range

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
        warmup_bars: int = 0,
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
        # 预热期：前 warmup_bars 根只更新指标、拒单、不记入净值曲线（供 OOS 预热）
        self.warmup_bars = max(0, int(warmup_bars))
        self._in_warmup = False
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
        # vt_symbol -> [datetime]（升序，供 get_history 因果截断二分查找）
        self._bar_dates: Dict[str, List[datetime]] = {
            vt: [b.datetime for b in bars] for vt, bars in data.items()
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

    # ---- 辅助方法 ----
    def _is_a_share(self, vt_symbol: str) -> bool:
        """判断是否为 A 股标的（SSE/SZSE）。"""
        _, exch = vt_symbol.rsplit(".", 1)
        return exch in ("SSE", "SZSE")

    def _find_bar_index(self, bars: List[BarData]) -> Optional[int]:
        """在 bars 中找当前日期的索引（兼容 tz-aware/naive）。"""
        if not bars or self._current_date is None:
            return None
        for i, bar in enumerate(bars):
            # 尝试直接比较
            if bar.datetime == self._current_date:
                return i
            # 兜底：归一化比较（去掉时区信息）
            bar_dt = bar.datetime.replace(tzinfo=None) if bar.datetime.tzinfo else bar.datetime
            cur_dt = self._current_date.replace(tzinfo=None) if self._current_date.tzinfo else self._current_date
            if bar_dt == cur_dt:
                return i
        return None

    def _check_t1_sell(self, vt_symbol: str, volume: float, fill_date: datetime) -> bool:
        """A 股 T+1 检查：当日买入的股票不可当日卖出。

        利用已有的 _open_lots 记账（FIFO 批次），检查今日买入量。
        仅对 SSE/SZSE 生效。

        :return: True 表示允许卖出，False 表示 T+1 限制
        """
        if not self._is_a_share(vt_symbol):
            return True  # 非 A 股不限制

        # 计算今日买入量（_open_lots 中今日日期、正数量的批次）
        today_bought = sum(
            abs(vol) for dt, vol, px in self._open_lots.get(vt_symbol, [])
            if dt.date() == fill_date.date() and vol > 0
        )

        # 可卖量 = 当前持仓 - 今日买入量（今日买入的不可卖）
        pos = self.get_position(vt_symbol)
        available_to_sell = max(0, pos.volume - today_bought) if pos.volume > 0 else 0

        if volume > available_to_sell + 1e-9:
            _logger.info("T+1 限制：%s 可卖 %.0f < 委托 %.0f（今日买入 %.0f 锁定）",
                         vt_symbol, available_to_sell, volume, today_bought)
            return False
        return True

    # ---- StrategyContext 接口 ----
    def get_history(self, vt_symbol: str, count: int) -> List[BarData]:
        """返回截至回测当前日期的最近 ``count`` 根 K 线（无前视）。

        - 回放期间（``_current_date`` 非空）：只返回当前日期及之前的 bar；
        - 初始化阶段（``_current_date`` 为空）：返回全量历史（供 on_init
          因果预计算使用，调用方须保证计算本身无前视）。
        """
        bars = self.data.get(vt_symbol, [])
        if self._current_date is not None:
            dates = self._bar_dates.get(vt_symbol)
            if dates is not None:
                i = bisect_right(dates, self._current_date)
                return bars[max(0, i - count):i]
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
        vt_symbol = f"{req.symbol}.{req.exchange.value}"

        if self._in_warmup:
            # OOS 预热期：只热身指标，不下单（否则 IS 期交易污染样本外绩效）
            _logger.debug("预热期拒单: %s x%s", vt_symbol, req.volume)
            return ""

        # A股整手校验：买入必须是100股整数倍
        if self._is_a_share(vt_symbol) and req.direction == Direction.LONG:
            lot_size = 100
            if req.volume % lot_size != 0:
                rounded_vol = (req.volume // lot_size) * lot_size
                if rounded_vol <= 0:
                    _logger.warning(f"A股整手校验失败: {vt_symbol} 委托数量 {req.volume} 不足1手({lot_size}股)")
                    return ""
                _logger.info(f"A股整手调整: {vt_symbol} {req.volume} → {rounded_vol}")
                req.volume = rounded_vol

        # A股涨跌停价格限制
        if self._is_a_share(vt_symbol) and self.exclude_limit and req.price > 0:
            bars = self.data.get(vt_symbol, [])
            # 在 bars 中找当前日期的索引（兼容 tz-aware/naive 比较）
            current_idx = self._find_bar_index(bars)

            if current_idx is not None:
                # 委托在下一根 K 线撮合，应以 t+1 的涨跌停价（基于当根收盘）校验
                limit_down, limit_up = limit_price_range(bars, current_idx + 1, self.limit_pct or 0.10)
                if limit_up is not None and req.price > limit_up:
                    _logger.warning(f"A股涨停价限制: {vt_symbol} 委托价 {req.price} > 涨停价 {limit_up}")
                    return ""
                if limit_down is not None and req.price < limit_down:
                    _logger.warning(f"A股跌停价限制: {vt_symbol} 委托价 {req.price} < 跌停价 {limit_down}")
                    return ""

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
        if fill_date is None:
            # 无下一交易日可撮合（on_init 期间或末日发单）：拒单，避免被
            # 收尾强平以同根 bar 价格伪造成交（前视）。
            _logger.warning("拒单（无下一交易日可撮合）: %s x%s", vt_symbol, req.volume)
            return ""
        self.pending.append({
            "vt_symbol": vt_symbol,
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
        for i, d in enumerate(self.dates):
            self._current_date = d
            self._in_warmup = i < self.warmup_bars
            self._fill_pending(d)
            for vt, bars in self.data.items():
                bar = self._lookup[vt].get(d)
                if bar is not None:
                    self.strategy.on_bar(bar)
            if not self._in_warmup:
                self._mark_to_market(d)
        # 收尾：取消剩余未成交挂单（不再以末日价格伪造成交），并重算末日权益
        self._cancel_remaining()
        if self.dates:
            self._mark_to_market(self.dates[-1], replace_last=True)
        self.strategy.on_stop()
        return self.analyze()

    def _fill_pending(self, date: datetime) -> None:
        remaining = []
        for p in self.pending:
            if p["fill_date"] is not None and p["fill_date"] <= date:
                prev_fill = p["fill_date"]
                executed = self._execute_fill(p, date)
                if executed:
                    continue
                if p["fill_date"] != prev_fill:
                    # 因 T+1 等原因已推迟到下一交易日：保留挂单（丢单→推迟）
                    remaining.append(p)
                else:
                    # 涨跌停拒单或数据缺失：保留待后续重试（收尾统一取消）
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
        if bar.datetime > date:
            # 停牌/数据缺口：恢复日在未来——不在当前日提前以未来价格成交（前视），
            # 保留挂单待引擎走到恢复日再撮合
            return False
        # 涨跌停剔除：涨停日无法买入（开多/平空），跌停日无法卖出（开空/平多）
        if self.exclude_limit:
            flag = self._limit_flag.get(vt, {}).get(bar.datetime)
            if flag == "up" and req.direction == Direction.LONG:
                return False
            if flag == "down" and req.direction == Direction.SHORT:
                return False
        # A股T+1限制：当日买入的股票当日不能卖出
        if req.direction == Direction.SHORT and self._is_a_share(vt):
            if not self._check_t1_sell(vt, req.volume, bar.datetime):
                # T+1限制，推迟到下一交易日
                nxt = self._next_date.get(bar.datetime)
                if nxt is not None:
                    p["fill_date"] = nxt
                    return False  # 不成交但保留挂单
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

        # 均价更新：反手（穿越零仓）时新方向以最新成交价计成本；同向加仓用加权均价
        if new_vol == 0:
            pos.price = 0.0
        elif cur_vol == 0 or cur_vol * new_vol < 0:
            pos.price = price
        elif abs(new_vol) > abs(cur_vol):
            added = abs(new_vol) - abs(cur_vol)
            pos.price = (pos.price * abs(cur_vol) + price * added) / abs(new_vol)
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

    def _bar_at_or_before(self, vt_symbol: str, date: datetime) -> Optional[BarData]:
        """返回该合约在 date 当日或之前最近一根 bar（停牌盯市沿用停牌前价格）。"""
        bars = self.data.get(vt_symbol, [])
        dates = self._bar_dates.get(vt_symbol)
        if not bars or not dates:
            return None
        i = bisect_right(dates, date)
        if i == 0:
            return None
        return bars[i - 1]

    def _cancel_remaining(self) -> None:
        """收尾：取消剩余未成交挂单（不伪造成交，持仓保持现状并按末日盯市）。"""
        for p in self.pending:
            _logger.info("收尾取消未成交挂单: %s x%s",
                         p["vt_symbol"], p["req"].volume)
        self.pending = []

    def _mark_to_market(self, date: datetime, replace_last: bool = False) -> None:
        equity = self.cash
        for vt, pos in self.positions.items():
            if pos.volume == 0:
                continue
            size = self.sizes.get(vt, 1.0)
            # 停牌/数据缺口时用当日或之前最近收盘价，避免用未来价格盯市
            bar = self._bar_at_or_before(vt, date)
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
        rep.cost_ratio = rep.total_cost / abs(net_pnl) if abs(net_pnl) > 1e-9 else 0.0
        return rep
