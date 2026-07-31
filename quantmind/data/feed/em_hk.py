"""港股数据源：东财 push2 / Yahoo（来自你提供的 global-stock-data 仓库思路）。

主用 yfinance（覆盖港股代码如 00700.HK），东财 push2 作为兜底。延迟导入。
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import List

from .base import BaseDataFeed, HistoryRequest
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
        try:
            return await self._fetch_yfinance(req)
        except Exception as exc:  # noqa: BLE001
            _logger.warning("yfinance 失败，回退东财: %s", exc)
            return await self._fetch_em(req)

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
        cols = {c.lower(): c for c in df.columns}
        date_col = cols.get("date") or cols.get("datetime") or cols.get("trade_date")
        o = cols.get("open") or cols.get("开盘")
        h = cols.get("high") or cols.get("最高")
        l = cols.get("low") or cols.get("最低")
        c = cols.get("close") or cols.get("收盘")
        v = cols.get("volume") or cols.get("成交量")
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
                )
            )
        return bars
