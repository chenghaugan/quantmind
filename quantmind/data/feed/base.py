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


# 中英文列名别名 -> 规范字段名。AKShare 不同接口（stock_zh_a_hist / option_*/futures_*）
# 返回中文或英文列头，统一在此映射，避免 _df_to_bars 按固定英文列名取数导致 KeyError。
_OHLC_ALIASES = {
    "date": ["date", "datetime", "trade_date", "日期", "时间"],
    "open": ["open", "开盘", "开盘价"],
    "high": ["high", "最高", "最高价"],
    "low": ["low", "最低", "最低价"],
    "close": ["close", "收盘", "收盘价"],
    "volume": ["volume", "成交量", "成交量_手"],
    "turnover": ["turnover", "amount", "成交额", "成交金额"],
    "open_interest": ["open_interest", "hold", "持仓量", "持仓"],
}


def resolve_ohlc_columns(df) -> Dict[str, str]:
    """把 DataFrame 的列解析为规范字段名 -> 实际列名。

    大小写不敏感，兼容中文（如 ``日期``）与英文（如 ``date``）两种表头。
    缺失的字段不会出现在返回 dict 中（调用方据此给默认值）。
    """
    lower = {c.lower(): c for c in df.columns}
    out: Dict[str, str] = {}
    for canon, aliases in _OHLC_ALIASES.items():
        for alias in aliases:
            if alias in lower:
                out[canon] = lower[alias]
                break
    return out
