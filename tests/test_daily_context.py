# -*- coding: utf-8 -*-
"""日线级上下文（DailyContext）与 BarData 兼容别名测试。"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from quantmind.core.object import BarData
from quantmind.backtest.interval_check import check_strategy_interval_compatibility
from quantmind.strategy.daily_context import DailyContext

UTC = timezone.utc


def _mk_daily_bars(n: int = 10, start: datetime = datetime(2026, 1, 1, tzinfo=UTC)):
    """构造 n 根日线：close=i+1，high=close+1，low=close-1，open=close。"""
    bars = []
    for i in range(n):
        dt = start + timedelta(days=i)
        c = float(i + 1)
        bars.append(BarData(
            symbol="IC0", datetime=dt,
            open_price=c, high_price=c + 1, low_price=c - 1, close_price=c,
        ))
    return bars


def test_daily_context_prev_values_no_lookahead():
    """在 D 日分钟bar上查询，只看 D-1 及之前日线：无前视。"""
    bars = _mk_daily_bars(10)
    dc = DailyContext(bars)
    # 当前时刻是第5天（index=4，close=5）的 10:00
    cur = datetime(2026, 1, 5, 10, 0, tzinfo=UTC)
    # 前一交易日 = index 3（close=4）
    assert dc.prev_close(cur) == 4.0
    assert dc.prev_high(cur) == 5.0
    assert dc.prev_low(cur) == 3.0
    assert dc.prev_open(cur) == 4.0
    # offset=1 → 前两交易日（index 2）
    assert dc.prev_close(cur, offset=1) == 3.0


def test_daily_context_no_bar_for_current_day():
    """即使当前日期没有日线bar（盘中），也绝不使用当日数据。"""
    bars = _mk_daily_bars(10)  # 日线只到 2026-01-10
    dc = DailyContext(bars)
    # 分钟bar在 2026-01-10 当天 10:00 → 当日日线未完成，应取 01-09
    cur = datetime(2026, 1, 10, 10, 0, tzinfo=UTC)
    assert dc.prev_close(cur) == 9.0
    # 未来日期（超出日线范围）→ 取最后一根
    fut = datetime(2026, 1, 20, 10, 0, tzinfo=UTC)
    assert dc.prev_close(fut) == 10.0


def test_daily_context_sma_atr():
    bars = _mk_daily_bars(10)
    dc = DailyContext(bars)
    cur = datetime(2026, 1, 10, 10, 0, tzinfo=UTC)
    # 截至前一日（close 1..9）的 3 日均线 = (7+8+9)/3 = 8.0
    assert dc.sma(3, cur) == 8.0
    # 最高/最低：截至前一日（close 1..9 的最后3根：7,8,9），high=close+1
    assert dc.highest(3, cur) == 10.0
    assert dc.lowest(3, cur) == 6.0
    # ATR：TR = max(high-low, |high-prev_close|, |low-prev_close|) = 2（等差数列）
    assert dc.atr(3, cur) == 2.0
    # 数据不足 → None
    assert dc.sma(50, cur) is None


def test_daily_context_before_first_bar():
    bars = _mk_daily_bars(3)
    dc = DailyContext(bars)
    early = datetime(2025, 12, 1, 10, 0, tzinfo=UTC)
    assert dc.prev_close(early) is None
    assert dc.sma(3, early) is None


def test_bardata_compat_aliases():
    """BarData 兼容别名：bar.high → bar.high_price 等（LLM 生成安全网）。"""
    b = BarData(symbol="IC0", high_price=5.0, low_price=3.0,
                open_price=4.0, close_price=4.5)
    assert b.high == 5.0 and b.low == 3.0
    assert b.open == 4.0 and b.close == 4.5


def test_interval_check_daily_context_patterns():
    """使用 self.daily 的策略在分钟周期下视为兼容（日线级上下文已注入）。"""
    code = (
        "from quantmind.strategy.base import CtaTemplate\n"
        "class XStrategy(CtaTemplate):\n"
        "    def on_bar(self, bar):\n"
        "        ph = self.daily.prev_high(bar.datetime)\n"
        "        if bar.close_price > ph:\n"
        "            self.set_target(bar.vt_symbol, 1)\n"
    )
    out = check_strategy_interval_compatibility(code, "15m")
    assert out["compatible"] is True
    # 1d 周期 + self.daily → 无意义，报 issue
    out2 = check_strategy_interval_compatibility(code, "1d")
    assert out2["compatible"] is False
