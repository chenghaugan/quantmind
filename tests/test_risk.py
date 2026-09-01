"""风控引擎与交易日历测试。

覆盖：交易时段（日盘/夜盘/跨零点/节假日）、各类限额拒单、
两级熔断（SOFT 只禁开仓 / HARD 全禁）、日切重置、频率限制。
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from quantmind.core.constant import Direction, Exchange, Offset
from quantmind.core.gateway import OrderRequest
from quantmind.core.object import PositionData
from quantmind.risk import RiskEngine, RiskLimits, TradingCalendar
from quantmind.risk.limits import RiskCode

CST = timezone(timedelta(hours=8))


def bj(y, m, d, H=0, M=0) -> datetime:
    """构造北京时间。"""
    return datetime(y, m, d, H, M, tzinfo=CST)


def order(symbol="rb2410", exchange=Exchange.SHFE, direction=Direction.LONG,
          offset=Offset.OPEN, volume=1.0, price=3500.0) -> OrderRequest:
    return OrderRequest(symbol=symbol, exchange=exchange, direction=direction,
                        offset=offset, volume=volume, price=price)


def pos(volume: float, symbol="rb2410", exchange=Exchange.SHFE) -> PositionData:
    return PositionData(symbol=symbol, exchange=exchange,
                        direction=Direction.NET, volume=volume)


# ---------------------------------------------------------------- 交易日历
class TestTradingCalendar:
    def setup_method(self):
        self.cal = TradingCalendar()

    def test_weekend_is_not_trading_day(self):
        assert not self.cal.is_trading_day(date(2026, 8, 8))   # 周六
        assert not self.cal.is_trading_day(date(2026, 8, 9))   # 周日
        assert self.cal.is_trading_day(date(2026, 8, 3))       # 周一

    def test_holiday_is_not_trading_day(self):
        assert not self.cal.is_trading_day(date(2026, 10, 1))
        assert not self.cal.is_trading_day(date(2025, 1, 1))

    def test_commodity_day_sessions(self):
        assert self.cal.is_trading_time(bj(2026, 8, 3, 9, 30), "rb2410", "SHFE")
        assert self.cal.is_trading_time(bj(2026, 8, 3, 14, 0), "rb2410", "SHFE")
        # 10:15-10:30 小节休
        assert not self.cal.is_trading_time(bj(2026, 8, 3, 10, 20), "rb2410", "SHFE")
        # 午休
        assert not self.cal.is_trading_time(bj(2026, 8, 3, 12, 0), "rb2410", "SHFE")
        # 收盘后
        assert not self.cal.is_trading_time(bj(2026, 8, 3, 15, 30), "rb2410", "SHFE")

    def test_night_session_2300(self):
        """RB 夜盘 21:00-23:00，不跨零点。"""
        assert self.cal.is_trading_time(bj(2026, 8, 3, 22, 0), "rb2410", "SHFE")
        assert not self.cal.is_trading_time(bj(2026, 8, 3, 23, 30), "rb2410", "SHFE")
        # 次日凌晨不属于 RB 夜盘
        assert not self.cal.is_trading_time(bj(2026, 8, 4, 1, 30), "rb2410", "SHFE")

    def test_night_session_crossing_midnight(self):
        """CU 夜盘 21:00-01:00；AU 到 02:30。"""
        assert self.cal.is_trading_time(bj(2026, 8, 3, 23, 30), "cu2410", "SHFE")
        assert self.cal.is_trading_time(bj(2026, 8, 4, 0, 30), "cu2410", "SHFE")
        assert not self.cal.is_trading_time(bj(2026, 8, 4, 1, 30), "cu2410", "SHFE")
        assert self.cal.is_trading_time(bj(2026, 8, 4, 2, 0), "au2412", "SHFE")
        assert not self.cal.is_trading_time(bj(2026, 8, 4, 3, 0), "au2412", "SHFE")

    def test_no_night_before_holiday(self):
        """节假日前最后一个交易日晚上不开夜盘（夜盘归属下一交易日）。"""
        # 2026-09-30 是国庆前最后交易日，10-01 休市 → 9/30 晚无夜盘
        assert self.cal.is_trading_day(date(2026, 9, 30))
        assert not self.cal.has_night_session(date(2026, 9, 30))
        assert not self.cal.is_trading_time(bj(2026, 9, 30, 22, 0), "rb2410", "SHFE")

    def test_night_on_friday(self):
        """周五晚有夜盘（归属下周一，间隔 3 天）。"""
        assert self.cal.has_night_session(date(2026, 8, 7))  # 周五 → 下周一开市

    def test_stock_sessions(self):
        assert not self.cal.is_trading_time(bj(2026, 8, 3, 9, 20), "600000", "SSE")
        assert self.cal.is_trading_time(bj(2026, 8, 3, 10, 0), "600000", "SSE")
        assert self.cal.is_trading_time(bj(2026, 8, 3, 14, 30), "600000", "SSE")
        assert not self.cal.is_trading_time(bj(2026, 8, 3, 11, 45), "600000", "SSE")
        # A 股无夜盘
        assert not self.cal.is_trading_time(bj(2026, 8, 3, 21, 30), "600000", "SSE")

    def test_cffex_bond_extra_15min(self):
        """国债期货 15:15 收盘，股指 15:00。"""
        assert self.cal.is_trading_time(bj(2026, 8, 3, 15, 5), "T2409", "CFFEX")
        assert not self.cal.is_trading_time(bj(2026, 8, 3, 15, 5), "IF2409", "CFFEX")

    def test_utc_input_is_converted(self):
        """传入 UTC 时间应正确换算到北京时间判断。"""
        utc_dt = datetime(2026, 8, 3, 2, 0, tzinfo=timezone.utc)  # = 北京 10:00
        assert self.cal.is_trading_time(utc_dt, "rb2410", "SHFE")

    def test_custom_holidays_override(self):
        cal = TradingCalendar(holidays={date(2026, 8, 3)})
        assert not cal.is_trading_day(date(2026, 8, 3))
        assert cal.is_trading_day(date(2026, 10, 1))  # 自定义集合里没有 → 视为交易日


# ---------------------------------------------------------------- 单笔限额
class TestOrderLimits:
    def setup_method(self):
        self.risk = RiskEngine(
            RiskLimits(check_trading_session=False), initial_equity=1_000_000.0
        )

    def test_normal_order_passes(self):
        d = self.risk.check_order(order(volume=5), last_price=3500)
        assert d.passed and d.code is RiskCode.PASS

    def test_max_order_volume(self):
        d = self.risk.check_order(order(volume=500), last_price=3500)
        assert not d
        assert d.code is RiskCode.ORDER_VOLUME_TOO_LARGE

    def test_volume_tick(self):
        d = self.risk.check_order(order(volume=1.5), last_price=3500)
        assert d.code is RiskCode.VOLUME_TICK_INVALID

    def test_non_positive_volume(self):
        d = self.risk.check_order(order(volume=0))
        assert d.code is RiskCode.ORDER_VOLUME_TOO_SMALL

    def test_price_deviation(self):
        d = self.risk.check_order(order(price=4000), last_price=3500)
        assert d.code is RiskCode.PRICE_DEVIATION
        # 市价单（price=0）不检查偏离
        assert self.risk.check_order(order(price=0.0), last_price=3500).passed

    def test_negative_price(self):
        d = self.risk.check_order(order(price=-1))
        assert d.code is RiskCode.PRICE_INVALID


# ---------------------------------------------------------------- 持仓/组合
class TestPositionLimits:
    def setup_method(self):
        self.risk = RiskEngine(
            RiskLimits(check_trading_session=False, max_position_volume=10),
            initial_equity=1_000_000.0,
        )

    def test_position_limit(self):
        d = self.risk.check_order(order(volume=5), position=pos(8), last_price=3500)
        assert d.code is RiskCode.POSITION_LIMIT

    def test_reduce_position_allowed_beyond_limit(self):
        """超限状态下减仓必须放行，否则风控会把仓位锁死。"""
        d = self.risk.check_order(
            order(direction=Direction.SHORT, offset=Offset.CLOSE, volume=5),
            position=pos(20), last_price=3500,
        )
        assert d.passed

    def test_close_exceeds_position(self):
        d = self.risk.check_order(
            order(direction=Direction.SHORT, offset=Offset.CLOSE, volume=5),
            position=pos(2), last_price=3500,
        )
        assert d.code is RiskCode.CLOSE_EXCEEDS_POSITION

    def test_position_value_limit(self):
        risk = RiskEngine(
            RiskLimits(check_trading_session=False, max_position_volume=None,
                       max_position_value=100_000, max_margin_ratio=None),
            initial_equity=1_000_000.0,
        )
        # rb 乘数 10，3500 * 10 * 5 = 175,000 > 100,000
        d = risk.check_order(order(volume=5), last_price=3500)
        assert d.code is RiskCode.POSITION_VALUE_LIMIT

    def test_margin_limit(self):
        risk = RiskEngine(
            RiskLimits(check_trading_session=False, max_position_volume=None,
                       max_order_volume=None, max_margin_ratio=0.1),
            initial_equity=100_000.0,
        )
        # 20 手 rb：3500*10*20*margin_rate 明显超过 1 万
        d = risk.check_order(order(volume=20), last_price=3500)
        assert d.code is RiskCode.MARGIN_LIMIT


# ---------------------------------------------------------------- 准入
class TestAdmission:
    def test_forbidden_symbol_by_prefix(self):
        risk = RiskEngine(RiskLimits(check_trading_session=False,
                                     forbidden_symbols={"RB"}))
        d = risk.check_order(order())
        assert d.code is RiskCode.SYMBOL_FORBIDDEN

    def test_allowed_whitelist(self):
        risk = RiskEngine(RiskLimits(check_trading_session=False,
                                     allowed_symbols={"IF"}))
        assert risk.check_order(order()).code is RiskCode.SYMBOL_NOT_ALLOWED
        assert risk.check_order(order(symbol="IF2409", exchange=Exchange.CFFEX)).passed

    def test_trading_session_gate(self):
        risk = RiskEngine(RiskLimits(check_trading_session=True))
        # 周日凌晨，任何品种都不可交易
        d = risk.check_order(order(), now=bj(2026, 8, 9, 3, 0))
        assert d.code is RiskCode.NOT_TRADING_TIME
        assert risk.check_order(order(), now=bj(2026, 8, 3, 10, 0), last_price=3500).passed

    def test_self_trade_guard(self):
        risk = RiskEngine(RiskLimits(check_trading_session=False))
        active = [order(direction=Direction.SHORT)]
        d = risk.check_order(order(direction=Direction.LONG), active_orders=active)
        assert d.code is RiskCode.SELF_TRADE
        # 同向挂单不算自成交
        assert risk.check_order(order(direction=Direction.SHORT),
                                active_orders=active).passed


# ---------------------------------------------------------------- 熔断
class TestHalt:
    def setup_method(self):
        self.risk = RiskEngine(
            RiskLimits(check_trading_session=False, max_daily_loss_ratio=0.05,
                       max_drawdown_ratio=0.2),
            initial_equity=1_000_000.0,
        )

    def test_daily_loss_triggers_soft_halt(self):
        decision = self.risk.update_equity(940_000.0)
        assert decision is not None and decision.code is RiskCode.DAILY_LOSS_LIMIT
        assert self.risk.state.halted and self.risk.state.halt_level == "SOFT"

    def test_soft_halt_blocks_open_allows_close(self):
        self.risk.update_equity(900_000.0)
        opened = self.risk.check_order(order(volume=1), last_price=3500)
        assert opened.code is RiskCode.OPEN_FORBIDDEN
        closed = self.risk.check_order(
            order(direction=Direction.SHORT, offset=Offset.CLOSE, volume=1),
            position=pos(3), last_price=3500,
        )
        assert closed.passed

    def test_hard_halt_blocks_everything(self):
        self.risk.halt("人工紧急停机", level="HARD")
        closed = self.risk.check_order(
            order(direction=Direction.SHORT, offset=Offset.CLOSE, volume=1),
            position=pos(3), last_price=3500,
        )
        assert closed.code is RiskCode.KILL_SWITCH

    def test_resume_requires_explicit_call(self):
        self.risk.update_equity(900_000.0)
        assert self.risk.state.halted
        # 日切不解除熔断
        self.risk.reset_day(equity=900_000.0)
        assert self.risk.state.halted
        self.risk.resume(operator="tester", note="确认为数据错误")
        assert not self.risk.state.halted
        assert any(e["type"] == "RESUME" for e in self.risk.log)

    def test_drawdown_halt(self):
        risk = RiskEngine(
            RiskLimits(check_trading_session=False, max_daily_loss_ratio=None,
                       max_drawdown_ratio=0.1),
            initial_equity=1_000_000.0,
        )
        risk.update_equity(1_200_000.0)          # 新高
        assert not risk.state.halted
        d = risk.update_equity(1_000_000.0)      # 回撤 16.7% > 10%
        assert d is not None and d.code is RiskCode.DRAWDOWN_LIMIT

    def test_soft_halt_not_downgraded_by_later_soft(self):
        self.risk.halt("人工", level="HARD")
        self.risk.update_equity(900_000.0)
        assert self.risk.state.halt_level == "HARD"


# ---------------------------------------------------------------- 频率与日切
class TestRateAndRollover:
    def test_max_orders_per_day(self):
        risk = RiskEngine(RiskLimits(check_trading_session=False,
                                     max_orders_per_day=3,
                                     max_orders_per_minute=None))
        now = bj(2026, 8, 3, 10, 0)
        for _ in range(3):
            assert risk.check_order(order(), now=now, last_price=3500).passed
        d = risk.check_order(order(), now=now, last_price=3500)
        assert d.code is RiskCode.ORDER_COUNT_DAILY

    def test_max_orders_per_minute(self):
        risk = RiskEngine(RiskLimits(check_trading_session=False,
                                     max_orders_per_day=None,
                                     max_orders_per_minute=2))
        base = bj(2026, 8, 3, 10, 0)
        assert risk.check_order(order(), now=base, last_price=3500).passed
        assert risk.check_order(order(), now=base, last_price=3500).passed
        assert risk.check_order(order(), now=base, last_price=3500).code \
            is RiskCode.ORDER_RATE_LIMIT
        # 61 秒后窗口滑出
        assert risk.check_order(order(), now=base + timedelta(seconds=61),
                                last_price=3500).passed

    def test_day_rollover_resets_counters(self):
        risk = RiskEngine(RiskLimits(check_trading_session=False,
                                     max_orders_per_day=2,
                                     max_orders_per_minute=None))
        d1 = bj(2026, 8, 3, 10, 0)
        risk.check_order(order(), now=d1, last_price=3500)
        risk.check_order(order(), now=d1, last_price=3500)
        assert risk.check_order(order(), now=d1, last_price=3500).code \
            is RiskCode.ORDER_COUNT_DAILY
        d2 = bj(2026, 8, 4, 10, 0)
        assert risk.check_order(order(), now=d2, last_price=3500).passed
        assert risk.state.orders_today == 1

    def test_reject_counts_recorded(self):
        risk = RiskEngine(RiskLimits(check_trading_session=False))
        risk.check_order(order(volume=999))
        risk.check_order(order(volume=999))
        assert risk.state.reject_counts["ORDER_VOLUME_TOO_LARGE"] == 2
        assert risk.state.rejected_today == 2
        assert "limits" in risk.stats()

    def test_unlimited_profile_passes_everything(self):
        risk = RiskEngine(RiskLimits.unlimited())
        d = risk.check_order(order(volume=99999, price=1.0), last_price=3500,
                             now=bj(2026, 8, 9, 3, 0))
        assert d.passed
