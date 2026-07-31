"""期权数据源：AKShare（ETF/指数/商品期权）。

底层函数：option_sina_cffex_hs300（股指期权）、option_sse_*（上交所 ETF 期权）、
option_szse_*（深交所 ETF 期权）、option_sina_* 等。延迟导入。
MVP 先接入期权标的的 OHLCV 行情序列；期权链（expiry/strike）建模留作 Phase 2。
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import List, Optional

from .base import BaseDataFeed, HistoryRequest
from ...core.constant import Exchange, Interval
from ...core.object import BarData

_logger = logging.getLogger("quantmind.data.akshare_option")


def _parse_dt(value) -> datetime:
    if isinstance(value, datetime):
        return value
    s = str(value)
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d")
    except ValueError:
        return datetime.fromisoformat(s)


class AkShareOptionFeed(BaseDataFeed):
    name = "akshare_option"

    async def fetch_bar_data(self, req: HistoryRequest) -> List[BarData]:
        import akshare as ak

        # 期权代码形如 10004230（上交所 ETF 期权）或 IO2409-C-3900（中金所）
        code = req.symbol
        if code.upper().startswith(("IO", "MO", "HO")) or "C" in code or "P" in code:
            func = ak.option_sina_cffex_hs300
        elif code.startswith("1"):
            func = ak.option_sse_50etf_daily if code.startswith("100") else ak.option_sse_300etf_daily
        else:
            func = ak.option_szse_50etf_daily
        df = await asyncio.to_thread(func, symbol=code)
        return self._df_to_bars(df, req)

    @staticmethod
    def _df_to_bars(df, req: HistoryRequest) -> List[BarData]:
        bars: List[BarData] = []
        if df is None or len(df) == 0:
            return bars
        cols = {c.lower(): c for c in df.columns}
        date_col = cols.get("date") or cols.get("datetime") or cols.get("trade_date")
        o = cols.get("open")
        h = cols.get("high")
        l = cols.get("low")
        c = cols.get("close")
        v = cols.get("volume")
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
