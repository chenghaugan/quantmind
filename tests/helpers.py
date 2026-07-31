"""测试辅助：离线加载 Mock 数据。"""
from __future__ import annotations

from datetime import datetime

from quantmind.data.feed.mock import MockFeed
from quantmind.data.feed.base import HistoryRequest
from quantmind.core.constant import Exchange, Interval


async def load_bars(symbol: str = "rb0", exchange=Exchange.SHFE, years: int = 1):
    feed = MockFeed()
    end = datetime(2024, 12, 31)
    start = datetime(2024, 1, 1)
    return await feed.fetch_bar_data(
        HistoryRequest(symbol=symbol, exchange=exchange, interval=Interval.DAILY, start=start, end=end)
    )
