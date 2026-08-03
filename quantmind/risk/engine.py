"""风控引擎：委托的硬闸门 + 熔断开关。

调用契约
--------
::

    risk = RiskEngine(RiskLimits.conservative(), event_engine=ee)
    decision = risk.check_order(req, position=pos, last_price=px, equity=eq)
    if not decision:
        return None          # 拒单，已广播 EVENT_RISK
    order_id = gateway.send_order(req)

两级熔断
--------
  - ``SOFT``（默认）：触发日亏损 / 最大回撤 → **只允许减仓与平仓**，禁止开仓。
    这是实盘最常用的形态：出事先止血，但不强制在坏价位清仓。
  - ``HARD``：人工或严重异常触发 → **所有委托一律拒绝**（含平仓），
    交由人工介入。

**熔断状态不会因为策略重启而自动清除**：``resume()`` 必须显式调用，
且会记录操作理由。这是为了杜绝「策略自己把风控关掉继续跑」。
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Deque, Dict, Iterable, List, Optional

from ..core.constant import Direction, Offset
from ..core.event import EventType
from ..core.gateway import OrderRequest
from ..core.object import PositionData
from .calendar import TradingCalendar, beijing_time
from .limits import RiskCode, RiskDecision, RiskLimits

_logger = logging.getLogger("quantmind.risk")
UTC = timezone.utc

_CLOSE_OFFSETS = {Offset.CLOSE, Offset.CLOSE_TODAY, Offset.CLOSE_YESTERDAY}


@dataclass
class RiskState:
    """风控运行时状态（可序列化，便于 Web 面板展示与落盘恢复）。"""

    trading_day: Optional[date] = None
    orders_today: int = 0
    rejected_today: int = 0
    traded_volume_today: float = 0.0
    day_start_equity: float = 0.0
    equity: float = 0.0
    equity_peak: float = 0.0
    margin_used: float = 0.0
    halted: bool = False
    halt_level: str = ""           # "SOFT" | "HARD"
    halt_reason: str = ""
    halt_code: str = ""
    halt_time: Optional[datetime] = None
    reject_counts: Dict[str, int] = field(default_factory=dict)

    @property
    def daily_pnl(self) -> float:
        if not self.day_start_equity:
            return 0.0
        return self.equity - self.day_start_equity

    @property
    def daily_pnl_ratio(self) -> float:
        if not self.day_start_equity:
            return 0.0
        return self.daily_pnl / self.day_start_equity

    @property
    def drawdown_ratio(self) -> float:
        if not self.equity_peak:
            return 0.0
        return (self.equity - self.equity_peak) / self.equity_peak

    def to_dict(self) -> dict:
        return {
            "trading_day": self.trading_day.isoformat() if self.trading_day else None,
            "orders_today": self.orders_today,
            "rejected_today": self.rejected_today,
            "traded_volume_today": self.traded_volume_today,
            "day_start_equity": round(self.day_start_equity, 2),
            "equity": round(self.equity, 2),
            "equity_peak": round(self.equity_peak, 2),
            "margin_used": round(self.margin_used, 2),
            "daily_pnl": round(self.daily_pnl, 2),
            "daily_pnl_ratio": round(self.daily_pnl_ratio, 6),
            "drawdown_ratio": round(self.drawdown_ratio, 6),
            "halted": self.halted,
            "halt_level": self.halt_level,
            "halt_reason": self.halt_reason,
            "halt_code": self.halt_code,
            "halt_time": self.halt_time.isoformat() if self.halt_time else None,
            "reject_counts": dict(self.reject_counts),
        }


class RiskEngine:
    """委托风控引擎。

    参数
    ----
    limits
        限额集合；``None`` 时使用 :class:`RiskLimits` 默认档。
    event_engine
        事件引擎；拒单/熔断时广播 ``EVENT_RISK``。
    calendar
        交易日历；``None`` 时使用默认日历。
    sizes
        ``vt_symbol -> 合约乘数``；缺省时回落到 ``core.contracts.default_size``。
    margin_rates
        ``vt_symbol -> 保证金率``；缺省时回落到 ``backtest.cost.lookup_cost``。
    """

    def __init__(
        self,
        limits: Optional[RiskLimits] = None,
        event_engine=None,
        calendar: Optional[TradingCalendar] = None,
        sizes: Optional[Dict[str, float]] = None,
        margin_rates: Optional[Dict[str, float]] = None,
        initial_equity: float = 0.0,
    ) -> None:
        self.limits = limits or RiskLimits()
        self.event_engine = event_engine
        self.calendar = calendar or TradingCalendar()
        self.sizes = sizes or {}
        self.margin_rates = margin_rates or {}
        self.state = RiskState(
            day_start_equity=initial_equity,
            equity=initial_equity,
            equity_peak=initial_equity,
        )
        self._order_times: Deque[datetime] = deque(maxlen=10000)
        self.log: List[dict] = []   # 拒单/熔断审计日志

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def check_order(
        self,
        req: OrderRequest,
        *,
        position: Optional[PositionData] = None,
        last_price: float = 0.0,
        equity: Optional[float] = None,
        margin_used: Optional[float] = None,
        now: Optional[datetime] = None,
        active_orders: Optional[Iterable[OrderRequest]] = None,
        record: bool = True,
    ) -> RiskDecision:
        """检查一笔委托是否放行。

        ``record=True`` 时，通过的委托会计入频率计数器；拒绝的会计入审计日志
        并广播 ``EVENT_RISK``。
        """
        now = now or datetime.now(UTC)
        vt_symbol = f"{req.symbol}.{req.exchange.value}"
        lim = self.limits

        if equity is not None:
            self._touch_equity(equity, now)
        if margin_used is not None:
            self.state.margin_used = margin_used
        self._roll_day(now)

        cur_vol = position.volume if position is not None else 0.0
        signed = req.volume if req.direction == Direction.LONG else -req.volume
        projected = cur_vol + signed
        is_increasing = abs(projected) > abs(cur_vol) + 1e-12
        is_close_intent = req.offset in _CLOSE_OFFSETS or not is_increasing

        checks = (
            self._check_halt(is_close_intent, vt_symbol),
            self._check_symbol(vt_symbol, req.symbol),
            self._check_session(req, now, vt_symbol),
            self._check_open_allowed(is_increasing, vt_symbol),
            self._check_order_volume(req, vt_symbol),
            self._check_price(req, last_price, vt_symbol),
            self._check_self_trade(req, active_orders, vt_symbol),
            self._check_close_volume(req, cur_vol, vt_symbol),
            self._check_position(projected, cur_vol, last_price, req, vt_symbol),
            self._check_margin(projected, cur_vol, last_price, req, vt_symbol),
            self._check_rate(now, vt_symbol),
        )
        for decision in checks:
            if decision is not None and not decision.passed:
                if record:
                    self._on_reject(decision, req, now)
                return decision

        if record:
            self.state.orders_today += 1
            self._order_times.append(now)
        return RiskDecision.ok(vt_symbol)

    # ------------------------------------------------------------------
    # 各项检查（返回 None 表示该项不适用/通过）
    # ------------------------------------------------------------------
    def _check_halt(self, is_close_intent: bool, vt: str) -> Optional[RiskDecision]:
        st = self.state
        if not st.halted:
            return None
        if st.halt_level == "HARD":
            return RiskDecision.reject(
                RiskCode.KILL_SWITCH,
                f"全局熔断中（{st.halt_reason}），所有委托拒绝",
                vt,
            )
        if not is_close_intent:
            return RiskDecision.reject(
                RiskCode.OPEN_FORBIDDEN,
                f"熔断中（{st.halt_reason}），仅允许平仓/减仓",
                vt,
            )
        return None

    def _check_symbol(self, vt: str, symbol: str) -> Optional[RiskDecision]:
        lim = self.limits
        prefix = "".join(ch for ch in symbol.upper() if ch.isalpha())
        if lim.forbidden_symbols:
            if vt in lim.forbidden_symbols or symbol in lim.forbidden_symbols \
                    or (prefix and prefix in lim.forbidden_symbols):
                return RiskDecision.reject(
                    RiskCode.SYMBOL_FORBIDDEN, f"{vt} 在禁止交易清单中", vt
                )
        if lim.allowed_symbols:
            ok = (vt in lim.allowed_symbols or symbol in lim.allowed_symbols
                  or (prefix and prefix in lim.allowed_symbols))
            if not ok:
                return RiskDecision.reject(
                    RiskCode.SYMBOL_NOT_ALLOWED, f"{vt} 不在允许交易白名单中", vt
                )
        return None

    def _check_session(self, req: OrderRequest, now: datetime, vt: str) -> Optional[RiskDecision]:
        if not self.limits.check_trading_session:
            return None
        if self.calendar.is_trading_time(now, req.symbol, req.exchange.value):
            return None
        return RiskDecision.reject(
            RiskCode.NOT_TRADING_TIME,
            f"{vt} 非交易时段（北京时间 {beijing_time(now):%Y-%m-%d %H:%M:%S}）",
            vt,
        )

    def _check_open_allowed(self, is_increasing: bool, vt: str) -> Optional[RiskDecision]:
        if is_increasing and not self.limits.allow_open:
            return RiskDecision.reject(
                RiskCode.OPEN_FORBIDDEN, "当前配置禁止开仓（allow_open=False）", vt
            )
        return None

    def _check_order_volume(self, req: OrderRequest, vt: str) -> Optional[RiskDecision]:
        lim = self.limits
        v = req.volume
        if v <= 0:
            return RiskDecision.reject(
                RiskCode.ORDER_VOLUME_TOO_SMALL, f"委托手数非正数：{v}", vt
            )
        if lim.max_order_volume is not None and v > lim.max_order_volume:
            return RiskDecision.reject(
                RiskCode.ORDER_VOLUME_TOO_LARGE,
                f"单笔手数 {v} 超上限 {lim.max_order_volume}",
                vt,
            )
        if lim.min_order_volume is not None and v < lim.min_order_volume:
            return RiskDecision.reject(
                RiskCode.ORDER_VOLUME_TOO_SMALL,
                f"单笔手数 {v} 低于下限 {lim.min_order_volume}",
                vt,
            )
        if lim.volume_tick:
            ratio = v / lim.volume_tick
            if abs(ratio - round(ratio)) > 1e-6:
                return RiskDecision.reject(
                    RiskCode.VOLUME_TICK_INVALID,
                    f"委托手数 {v} 不是步进 {lim.volume_tick} 的整数倍",
                    vt,
                )
        return None

    def _check_price(self, req: OrderRequest, last_price: float, vt: str) -> Optional[RiskDecision]:
        if req.price < 0:
            return RiskDecision.reject(RiskCode.PRICE_INVALID, f"委托价为负：{req.price}", vt)
        lim = self.limits
        # price == 0 视为市价单（框架约定），不做偏离检查
        if lim.max_price_deviation is None or req.price <= 0 or last_price <= 0:
            return None
        dev = abs(req.price - last_price) / last_price
        if dev > lim.max_price_deviation:
            return RiskDecision.reject(
                RiskCode.PRICE_DEVIATION,
                f"委托价 {req.price} 偏离最新价 {last_price} 达 {dev:.2%}，"
                f"超上限 {lim.max_price_deviation:.2%}",
                vt,
            )
        return None

    def _check_self_trade(
        self, req: OrderRequest, active_orders: Optional[Iterable[OrderRequest]], vt: str
    ) -> Optional[RiskDecision]:
        if not self.limits.self_trade_guard or not active_orders:
            return None
        for o in active_orders:
            o_vt = f"{o.symbol}.{o.exchange.value}"
            if o_vt == vt and o.direction != req.direction:
                return RiskDecision.reject(
                    RiskCode.SELF_TRADE,
                    f"{vt} 已存在反向活动委托（{o.direction.value}），可能自成交",
                    vt,
                )
        return None

    def _check_close_volume(self, req: OrderRequest, cur_vol: float, vt: str) -> Optional[RiskDecision]:
        if self.limits.allow_close_exceed_position:
            return None
        if req.offset not in _CLOSE_OFFSETS:
            return None
        if req.volume > abs(cur_vol) + 1e-9:
            return RiskDecision.reject(
                RiskCode.CLOSE_EXCEEDS_POSITION,
                f"{vt} 平仓 {req.volume} 手超过持仓 {abs(cur_vol)} 手",
                vt,
            )
        return None

    def _check_position(
        self, projected: float, cur_vol: float, last_price: float,
        req: OrderRequest, vt: str,
    ) -> Optional[RiskDecision]:
        lim = self.limits
        # 只约束「增加风险敞口」的方向：已超限时减仓必须放行，
        # 否则风控会把仓位锁死，反而制造更大风险。
        if abs(projected) <= abs(cur_vol) + 1e-12:
            return None
        if lim.max_position_volume is not None and abs(projected) > lim.max_position_volume + 1e-9:
            return RiskDecision.reject(
                RiskCode.POSITION_LIMIT,
                f"{vt} 成交后净持仓 {projected} 手超上限 {lim.max_position_volume}",
                vt,
            )
        if lim.max_position_value is not None:
            px = last_price or req.price
            if px > 0:
                value = abs(projected) * px * self._size(vt)
                if value > lim.max_position_value:
                    return RiskDecision.reject(
                        RiskCode.POSITION_VALUE_LIMIT,
                        f"{vt} 成交后名义市值 {value:,.0f} 元超上限 {lim.max_position_value:,.0f}",
                        vt,
                    )
        return None

    def _check_margin(
        self, projected: float, cur_vol: float, last_price: float,
        req: OrderRequest, vt: str,
    ) -> Optional[RiskDecision]:
        lim = self.limits
        if lim.max_margin_ratio is None:
            return None
        equity = self.state.equity
        if equity <= 0:
            return None
        added_lots = abs(projected) - abs(cur_vol)
        if added_lots <= 0:
            return None
        px = last_price or req.price
        if px <= 0:
            return None
        added_margin = added_lots * px * self._size(vt) * self._margin_rate(vt)
        ratio = (self.state.margin_used + added_margin) / equity
        if ratio > lim.max_margin_ratio:
            return RiskDecision.reject(
                RiskCode.MARGIN_LIMIT,
                f"{vt} 成交后保证金占用率 {ratio:.1%} 超上限 {lim.max_margin_ratio:.1%}",
                vt,
            )
        return None

    def _check_rate(self, now: datetime, vt: str) -> Optional[RiskDecision]:
        lim = self.limits
        st = self.state
        if lim.max_orders_per_day is not None and st.orders_today >= lim.max_orders_per_day:
            return RiskDecision.reject(
                RiskCode.ORDER_COUNT_DAILY,
                f"当日下单数 {st.orders_today} 已达上限 {lim.max_orders_per_day}",
                vt,
            )
        if lim.max_orders_per_minute is not None:
            cutoff = now - timedelta(seconds=60)
            recent = sum(1 for t in reversed(self._order_times) if t > cutoff)
            if recent >= lim.max_orders_per_minute:
                return RiskDecision.reject(
                    RiskCode.ORDER_RATE_LIMIT,
                    f"近 60 秒下单 {recent} 笔，已达频率上限 {lim.max_orders_per_minute}",
                    vt,
                )
        if lim.max_trade_volume_per_day is not None and \
                st.traded_volume_today >= lim.max_trade_volume_per_day:
            return RiskDecision.reject(
                RiskCode.TRADE_VOLUME_DAILY,
                f"当日成交 {st.traded_volume_today} 手已达上限 {lim.max_trade_volume_per_day}",
                vt,
            )
        return None

    # ------------------------------------------------------------------
    # 状态更新
    # ------------------------------------------------------------------
    def update_equity(self, equity: float, now: Optional[datetime] = None) -> Optional[RiskDecision]:
        """更新权益并检查日亏损 / 最大回撤，必要时触发熔断。

        返回触发熔断的 :class:`RiskDecision`；未触发返回 ``None``。
        """
        now = now or datetime.now(UTC)
        self._roll_day(now)
        self._touch_equity(equity, now)
        lim = self.limits
        st = self.state

        if lim.max_daily_loss is not None and st.daily_pnl <= -abs(lim.max_daily_loss):
            return self._trigger(
                RiskCode.DAILY_LOSS_LIMIT,
                f"当日亏损 {st.daily_pnl:,.0f} 元触及上限 {abs(lim.max_daily_loss):,.0f} 元",
                now,
                hard=not lim.halt_on_daily_loss,
            )
        if lim.max_daily_loss_ratio is not None and st.day_start_equity > 0 and \
                st.daily_pnl_ratio <= -abs(lim.max_daily_loss_ratio):
            return self._trigger(
                RiskCode.DAILY_LOSS_LIMIT,
                f"当日亏损率 {st.daily_pnl_ratio:.2%} 触及上限 "
                f"{-abs(lim.max_daily_loss_ratio):.2%}",
                now,
                hard=not lim.halt_on_daily_loss,
            )
        if lim.max_drawdown_ratio is not None and st.equity_peak > 0 and \
                st.drawdown_ratio <= -abs(lim.max_drawdown_ratio):
            return self._trigger(
                RiskCode.DRAWDOWN_LIMIT,
                f"回撤 {st.drawdown_ratio:.2%} 触及熔断线 {-abs(lim.max_drawdown_ratio):.2%}",
                now,
                hard=not lim.halt_on_drawdown,
            )
        return None

    def on_trade(self, volume: float, now: Optional[datetime] = None) -> None:
        """登记一笔成交（累计当日成交手数）。"""
        self._roll_day(now or datetime.now(UTC))
        self.state.traded_volume_today += abs(volume)

    def halt(self, reason: str, level: str = "HARD", now: Optional[datetime] = None) -> RiskDecision:
        """人工熔断。``level`` 取 ``"SOFT"``（禁开仓）或 ``"HARD"``（全禁）。"""
        code = RiskCode.KILL_SWITCH if level == "HARD" else RiskCode.OPEN_FORBIDDEN
        return self._trigger(code, reason, now or datetime.now(UTC), hard=(level == "HARD"))

    def resume(self, operator: str = "manual", note: str = "") -> None:
        """解除熔断（**必须显式调用**，并留痕）。"""
        st = self.state
        prev = st.halt_reason
        st.halted = False
        st.halt_level = ""
        st.halt_reason = ""
        st.halt_code = ""
        st.halt_time = None
        entry = {
            "type": "RESUME",
            "operator": operator,
            "note": note,
            "prev_reason": prev,
            "time": datetime.now(UTC).isoformat(),
        }
        self.log.append(entry)
        _logger.warning("[RISK] 熔断解除 by %s：%s（原因：%s）", operator, note, prev)
        self._emit(entry)

    def reset_day(self, now: Optional[datetime] = None, equity: Optional[float] = None) -> None:
        """日切：重置当日计数器与日初权益（**不会自动解除熔断**）。"""
        now = now or datetime.now(UTC)
        st = self.state
        st.trading_day = beijing_time(now).date()
        st.orders_today = 0
        st.rejected_today = 0
        st.traded_volume_today = 0.0
        st.day_start_equity = equity if equity is not None else st.equity
        self._order_times.clear()

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _roll_day(self, now: datetime) -> None:
        today = beijing_time(now).date()
        if self.state.trading_day is None:
            self.state.trading_day = today
            if not self.state.day_start_equity:
                self.state.day_start_equity = self.state.equity
        elif today != self.state.trading_day:
            self.reset_day(now)

    def _touch_equity(self, equity: float, now: datetime) -> None:
        st = self.state
        st.equity = equity
        if not st.day_start_equity:
            st.day_start_equity = equity
        if equity > st.equity_peak:
            st.equity_peak = equity

    def _trigger(self, code: RiskCode, reason: str, now: datetime, hard: bool = False) -> RiskDecision:
        st = self.state
        level = "HARD" if hard else "SOFT"
        # 已处于更严格状态则不降级
        if st.halted and st.halt_level == "HARD":
            level = "HARD"
        st.halted = True
        st.halt_level = level
        st.halt_reason = reason
        st.halt_code = code.value
        st.halt_time = now
        entry = {
            "type": "HALT",
            "level": level,
            "code": code.value,
            "reason": reason,
            "time": now.isoformat(),
            "equity": round(st.equity, 2),
        }
        self.log.append(entry)
        _logger.error("[RISK] 熔断触发（%s）%s：%s", level, code.value, reason)
        self._emit(entry)
        return RiskDecision.reject(code, reason)

    def _on_reject(self, decision: RiskDecision, req: OrderRequest, now: datetime) -> None:
        st = self.state
        st.rejected_today += 1
        st.reject_counts[decision.code.value] = st.reject_counts.get(decision.code.value, 0) + 1
        entry = {
            "type": "REJECT",
            "code": decision.code.value,
            "reason": decision.reason,
            "vt_symbol": decision.vt_symbol,
            "direction": req.direction.value,
            "offset": req.offset.value,
            "volume": req.volume,
            "price": req.price,
            "time": now.isoformat(),
        }
        self.log.append(entry)
        _logger.warning("[RISK] 拒单 %s：%s", decision.code.value, decision.reason)
        self._emit(entry)

    def _emit(self, payload: dict) -> None:
        if self.event_engine is not None:
            try:
                self.event_engine.put_event(EventType.EVENT_RISK, payload)
            except Exception:  # pragma: no cover - 事件失败不得影响交易主链路
                _logger.exception("[RISK] 事件广播失败")

    def _size(self, vt_symbol: str) -> float:
        if vt_symbol in self.sizes:
            return self.sizes[vt_symbol]
        from ..core.contracts import default_size

        return default_size(vt_symbol)

    def _margin_rate(self, vt_symbol: str) -> float:
        if vt_symbol in self.margin_rates:
            return self.margin_rates[vt_symbol]
        try:
            from ..backtest.cost import lookup_cost

            return lookup_cost(vt_symbol).margin_rate
        except Exception:
            return 0.1

    # ---- 展示 ----
    def stats(self) -> dict:
        return {
            "limits": self.limits.to_dict(),
            "state": self.state.to_dict(),
            "recent_log": self.log[-20:],
        }
