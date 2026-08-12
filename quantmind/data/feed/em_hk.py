"""港股数据源：东财 push2 / Yahoo（来自你提供的 global-stock-data 仓库思路）。

主用 yfinance（覆盖港股代码如 00700.HK），东财 push2 作为兜底。延迟导入。
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import List

from .base import BaseDataFeed, HistoryRequest, resolve_ohlc_columns
from ...core.constant import Exchange, Interval
from ...core.object import BarData

_logger = logging.getLogger("quantmind.data.em_hk")


def _parse_dt(value) -> datetime:
    if isinstance(value, datetime):
        return value
    s = str(value)
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d")
    except ValueError:
        return datetime.fromisoformat(s)


class EmHkFeed(BaseDataFeed):
    name = "em_hk"

    async def fetch_bar_data(self, req: HistoryRequest) -> List[BarData]:
        # 分层降级：新浪(stock_hk_daily, GET 不封 IP、本网络稳定) → yfinance → 东财。
        # 任一源失败不中断整链。
        try:
            return await self._fetch_sina(req)
        except Exception as exc:  # noqa: BLE001
            _logger.warning("新浪港股源失败，尝试 yfinance: %s", exc)
        try:
            return await self._fetch_yfinance(req)
        except Exception as exc:  # noqa: BLE001
            _logger.warning("yfinance 失败，回退东财: %s", exc)
            return await self._fetch_em(req)

    async def _fetch_sina(self, req: HistoryRequest) -> List[BarData]:
        """新浪港股日线（akshare ``stock_hk_daily``）：5 位代码，GET 不封 IP。"""
        import akshare as ak

        df = await asyncio.to_thread(ak.stock_hk_daily, symbol=req.symbol)
        return self._df_to_bars(df, req)

    async def _fetch_yfinance(self, req: HistoryRequest) -> List[BarData]:
        import yfinance as yf

        ticker = f"{req.symbol}.HK"
        period_map = {
            Interval.DAILY: "1d",
            Interval.WEEKLY: "1wk",
            Interval.MINUTE: "1m",
        }
        yf_interval = period_map.get(req.interval, "1d")
        end = req.end or datetime.now()
        start = req.start or (end - __import__("datetime").timedelta(days=365))
        df = await asyncio.to_thread(
            yf.Ticker(ticker).history,
            start=start,
            end=end,
            interval=yf_interval,
        )
        return self._df_to_bars(df, req)

    async def _fetch_em(self, req: HistoryRequest) -> List[BarData]:
        import akshare as ak

        df = await asyncio.to_thread(
            ak.stock_hk_hist, symbol=req.symbol, period="daily", adjust=""
        )
        return self._df_to_bars(df, req)

    @staticmethod
    def _df_to_bars(df, req: HistoryRequest) -> List[BarData]:
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
        to = cols.get("turnover")
        if not (date_col and o and h and l and c):
            raise ValueError(f"港股数据缺少关键列，实际列名={list(df.columns)}")
        for _, row in df.iterrows():
            bars.append(
                BarData(
                    symbol=req.symbol,
                    exchange=req.exchange,
                    datetime=_parse_dt(row[date_col]),
                    interval=req.interval,
                    open_price=float(row[o]),
                    high_price=float(row[h]),
                    low_price=float(row[l]),
                    close_price=float(row[c]),
                    volume=float(row[v]) if v else 0.0,
                    turnover=float(row[to]) if to else 0.0,
                )
            )
        return bars
