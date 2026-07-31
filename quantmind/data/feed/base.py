"""数据馈送适配器的抽象基类（参考 vnpy.datafeed）。

所有数据源（AKShare 期货/期权、mootdx A股、东财·Yahoo 港股）实现
``BaseDataFeed.fetch_bar_data``，统一返回内部 ``BarData`` 列表。
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from ...core.constant import Exchange, Interval
from ...core.object import BarData


@dataclass
class HistoryRequest:
    """历史数据请求。"""

    symbol: str
    exchange: Exchange
    interval: Interval = Interval.DAILY
    start: Optional[datetime] = None
    end: Optional[datetime] = None


class BaseDataFeed(ABC):
    """数据源基类。"""

    name: str = "base"

    @abstractmethod
    async def fetch_bar_data(self, req: HistoryRequest) -> List[BarData]:
        """拉取历史 K 线，返回 ``BarData`` 列表（按时间升序）。"""
        raise NotImplementedError

    @staticmethod
    def _make_bar(
        symbol: str,
        exchange: Exchange,
        dt: datetime,
        interval: Interval,
        o: float,
        h: float,
        l: float,
        c: float,
        v: float = 0.0,
        oi: float = 0.0,
        turnover: float = 0.0,
    ) -> BarData:
        return BarData(
            symbol=symbol,
            exchange=exchange,
            datetime=dt,
            interval=interval,
            open_price=o,
            high_price=h,
            low_price=l,
            close_price=c,
            volume=v,
            open_interest=oi,
            turnover=turnover,
        )
