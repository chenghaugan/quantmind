# -*- coding: utf-8 -*-
"""多周期上下文（MultiTFContext）测试：重采样、防前视、失败显式。"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from quantmind.core.object import BarData
from quantmind.backtest.interval_check import check_strategy_interval_compatibility
from quantmind.strategy.multi_tf import MultiTFContext, resample_bars

UTC = timezone.utc
BASE = datetime(2026, 1, 5, 9, 0, tzinfo=UTC)  # 周一 09:00 UTC


def _15m_bars(n: int, start: datetime = BASE, price0: float = 10.0):
    """n 根 15m bar：close = price0 + i（等差），时间逐根 +15min。"""
    bars = []
    for i in range(n):
        c = price0 + i
        bars.append(BarData(
            symbol="X", datetime=start + timedelta(minutes=15 * i),
            open_price=c, high_price=c + 1, low_price=c - 1, close_price=c,
        ))
    return bars


def test_resample_1h_ohlcv():
    """15m→1h：4根聚合，OHLC 正确（open=首开盘 high=max low=min close=末收盘）。"""
    bars = _15m_bars(8)  # close: 10..17, high=close+1, low=close-1
    out, _ct = resample_bars(bars, "1h", "15m")
    assert len(out) == 2
    b0, b1 = out
    assert b0.open_price == 10.0 and b0.close_price == 13.0
    assert b0.high_price == 14.0 and b0.low_price == 9.0
    assert b1.close_price == 17.0


def test_resample_gap_no_phantom_bucket():
    """基础数据含大 gap（隔夜）：gap 不产生幻影空桶，锚定推进正确。"""
    bars = _15m_bars(4)  # 09:00..09:45
    # 跳到次日 09:00（隔夜 gap 21h+）
    bars += _15m_bars(4, start=BASE + timedelta(days=1), price0=20.0)
    out, _ct = resample_bars(bars, "1h", "15m")
    # 每天各 1 根 1h（4×15m=1h），共 2 根
    assert len(out) == 2
    assert out[0].close_price == 13.0
    assert out[1].close_price == 23.0


def test_tfview_no_lookahead():
    """锚定时刻 t 只见 close_time ≤ t 的 bar。"""
    bars = _15m_bars(8)
    mtf = MultiTFContext()
    mtf.add("1h", *resample_bars(bars, "1h", "15m"))
    # 1h bar[0] 覆盖 09:00-10:00，close_time=10:00
    # 锚定 09:45（第4根15m收盘时刻）→ 1h bar 尚未完成 → 不可见
    v = mtf.tf("1h", datetime(2026, 1, 5, 9, 45, tzinfo=UTC))
    assert v.prev_close() is None
    # 锚定 10:00（1h bar 收盘时刻）→ 可见
    v = mtf.tf("1h", datetime(2026, 1, 5, 10, 0, tzinfo=UTC))
    assert v.prev_close() == 13.0


def test_multi_tf_unknown_key():
    bars = _15m_bars(4)
    mtf = MultiTFContext()
    mtf.add("1d", bars, close_times=[b.datetime for b in bars])
    with pytest.raises(KeyError):
        mtf.tf("4h", datetime(2026, 1, 5, 9, 0, tzinfo=UTC))


def test_tfview_insufficient_depth():
    bars = _15m_bars(3)
    mtf = MultiTFContext()
    mtf.add("1h", *resample_bars(bars, "1h", "15m"))
    v = mtf.tf("1h", datetime(2026, 1, 5, 9, 45, tzinfo=UTC))
    # 仅1根1h → sma(20) 数据不足 → None（预热语义）
    assert v.sma(20) is None


def test_interval_check_mtf_intraday_ok():
    """分钟周期 + self.mtf → 兼容（多周期上下文已注入）。"""
    code = (
        "from quantmind.strategy.base import CtaTemplate\n"
        "class XStrategy(CtaTemplate):\n"
        "    def on_bar(self, bar):\n"
        "        c = self.mtf.tf('1h', bar.datetime).prev_close()\n"
        "        if bar.close_price > c:\n"
        "            self.set_target(bar.vt_symbol, 1)\n"
    )
    out = check_strategy_interval_compatibility(code, "15m")
    assert out["compatible"] is True


def test_interval_check_mtf_on_daily():
    """1d 周期 + self.mtf → 不兼容（无多周期数据源）。"""
    code = (
        "from quantmind.strategy.base import CtaTemplate\n"
        "class XStrategy(CtaTemplate):\n"
        "    def on_bar(self, bar):\n"
        "        c = self.mtf.tf('1w', bar.datetime).prev_close()\n"
    )
    out = check_strategy_interval_compatibility(code, "1d")
    assert out["compatible"] is False
