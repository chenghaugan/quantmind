"""K 线合成与序列管理工具（移植自 vnpy.trader.utility）。"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

from .constant import Exchange, Interval
from .event import Event, EventType
from .object import BarData, TickData

_INTERVAL_DELTA: Dict[Interval, timedelta] = {
    Interval.MINUTE: timedelta(minutes=1),
    Interval.MINUTE_3: timedelta(minutes=3),
    Interval.MINUTE_5: timedelta(minutes=5),
    Interval.MINUTE_15: timedelta(minutes=15),
    Interval.MINUTE_30: timedelta(minutes=30),
    Interval.HOUR: timedelta(hours=1),
    Interval.HOUR_2: timedelta(hours=2),
    Interval.HOUR_4: timedelta(hours=4),
    Interval.DAILY: timedelta(days=1),
    Interval.WEEKLY: timedelta(weeks=1),
}

# 需要区分平今/平昨的交易所
DISTINGUISH_CLOSE_EXCHANGES = {Exchange.SHFE, Exchange.CFFEX}


class BarGenerator:
    """把 Tick 合成为指定周期的 Bar（参考 vnpy BarGenerator，简化为单周期）。"""

    def __init__(
        self,
        on_bar: Callable[[BarData], None],
        interval: Interval = Interval.MINUTE,
        exchange: Exchange = Exchange.CFFEX,
    ) -> None:
        self.on_bar = on_bar
        self.interval = interval
        self.exchange = exchange
        self.bar: Optional[BarData] = None

    def update_tick(self, tick: TickData) -> None:
        """用 tick 更新当前 bar；跨过周期边界时回调 on_bar 并新建 bar。"""
        if self.bar is None:
            self.bar = BarData(
                symbol=tick.symbol,
                exchange=tick.exchange,
                interval=self.interval,
                datetime=tick.datetime,
                open_price=tick.last_price,
                high_price=tick.last_price,
                low_price=tick.last_price,
                close_price=tick.last_price,
                volume=tick.volume,
                open_interest=tick.open_interest,
            )
            return

        # 周期边界判断
        delta = _INTERVAL_DELTA[self.interval]
        if tick.datetime - self.bar.datetime >= delta:
            finished = self.bar
            self.bar = BarData(
                symbol=tick.symbol,
                exchange=tick.exchange,
                interval=self.interval,
                datetime=tick.datetime,
                open_price=tick.last_price,
                high_price=tick.last_price,
                low_price=tick.last_price,
                close_price=tick.last_price,
                volume=tick.volume,
                open_interest=tick.open_interest,
            )
            self.on_bar(finished)
        else:
            self.bar.high_price = max(self.bar.high_price, tick.last_price)
            self.bar.low_price = min(self.bar.low_price, tick.last_price)
            self.bar.close_price = tick.last_price
            self.bar.volume = tick.volume
            self.bar.open_interest = tick.open_interest

    def generate(self) -> None:
        """强制产出当前未完成 bar。"""
        if self.bar is not None:
            self.on_bar(self.bar)
            self.bar = None


class ArrayManager:
    """固定长度序列管理（技术指标计算的底层容器，参考 vnpy ArrayManager）。"""

    def __init__(self, size: int = 100) -> None:
        self.size = size
        self.count = 0
        self.open: List[float] = []
        self.high: List[float] = []
        self.low: List[float] = []
        self.close: List[float] = []
        self.volume: List[float] = []
        self.open_interest: List[float] = []

    def update_bar(self, bar: BarData) -> None:
        self.open.append(bar.open_price)
        self.high.append(bar.high_price)
        self.low.append(bar.low_price)
        self.close.append(bar.close_price)
        self.volume.append(bar.volume)
        self.open_interest.append(bar.open_interest)
        self.count = min(self.count + 1, self.size)
        if len(self.close) > self.size:
            self.open.pop(0)
            self.high.pop(0)
            self.low.pop(0)
            self.close.pop(0)
            self.volume.pop(0)
            self.open_interest.pop(0)

    @property
    def inited(self) -> bool:
        return self.count >= self.size

    def sma(self, n: int) -> float:
        if len(self.close) < n:
            return 0.0
        return sum(self.close[-n:]) / n
