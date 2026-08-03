"""国内期货数据源：AKShare（商品 + 金融期货 主连/指数/具体合约）。

底层函数：futures_main_sina / futures_zh_daily_sina / futures_zh_minute_sina /
futures_daily_{shfe,dce,czce,cffex,ine} / futures_main_mapping_em。
akshare 为同步库，用 asyncio.to_thread 包裹避免阻塞事件循环；导入延迟化，
未安装 akshare 时本模块仍可导入（仅运行时报缺依赖）。
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import List

from .base import BaseDataFeed, HistoryRequest, resolve_ohlc_columns
from ...core.constant import Exchange, Interval
from ...core.object import BarData

_logger = logging.getLogger("quantmind.data.akshare_future")


def _parse_dt(value) -> datetime:
    if isinstance(value, datetime):
        return value
    s = str(value)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(s[: len(fmt) + 2], fmt)
        except ValueError:
            continue
    return datetime.fromisoformat(s)


class AkShareFuturesFeed(BaseDataFeed):
    name = "akshare_future"

    async def fetch_bar_data(self, req: HistoryRequest) -> List[BarData]:
        import akshare as ak

        symbol = req.symbol
        if req.interval == Interval.DAILY:
            # AKShare 1.18 起 futures_daily_<exchange> 系列已移除；
            # futures_zh_daily_sina 对主连（rb0/IF0）与具体合约（rb2401）均可用，统一走它。
            df = await asyncio.to_thread(ak.futures_zh_daily_sina, symbol=symbol)
        else:
            df = await asyncio.to_thread(
                ak.futures_zh_minute_sina, symbol=symbol, period=req.interval.value.replace("m", "")
            )

        return self._df_to_bars(df, symbol, req.exchange, req.interval)

    @staticmethod
    def _df_to_bars(df, symbol, exchange, interval) -> List[BarData]:
        bars: List[BarData] = []
        if df is None or len(df) == 0:
            return bars
        cols = resolve_ohlc_columns(df)
        date_col = cols.get("date")
        o = cols.get("open")
        h = cols.get("high")
        l = cols.get("low")
        c = cols.get("close")
        v = cols.get("volume")
        oi = cols.get("open_interest")
        to = cols.get("turnover")
        if not (date_col and o and h and l and c):
            raise ValueError(f"期货数据缺少关键列，实际列名={list(df.columns)}")
        for _, row in df.iterrows():
            bars.append(
                BarData(
                    symbol=symbol,
                    exchange=exchange,
                    datetime=_parse_dt(row[date_col]),
                    interval=interval,
                    open_price=float(row[o]),
                    high_price=float(row[h]),
                    low_price=float(row[l]),
                    close_price=float(row[c]),
                    volume=float(row[v]) if v else 0.0,
                    open_interest=float(row[oi]) if oi else 0.0,
                    turnover=float(row[to]) if to else 0.0,
                )
            )
        return bars
