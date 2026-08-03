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

from .base import BaseDataFeed, HistoryRequest, resolve_ohlc_columns
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
            funcs = ["option_sina_cffex_hs300"]
        elif code.startswith("1000"):
            funcs = ["option_sse_50etf_daily"]
        elif code.startswith("100"):  # 510300 等沪深300ETF期权
            funcs = ["option_sse_300etf_daily", "option_sse_50etf_daily"]
        else:  # 深交所 ETF 期权（如 9000 开头）
            funcs = ["option_szse_50etf_daily", "option_szse_300etf_daily"]

        last_err = None
        for fname in funcs:
            fn = getattr(ak, fname, None)
            if fn is None:
                continue  # akshare 版本已移除该接口
            try:
                df = await asyncio.to_thread(fn, symbol=code)
                if df is not None and len(df):
                    return self._df_to_bars(df, req)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                _logger.warning("期权接口 %s 失败: %s", fname, exc)
                continue
        if last_err is not None:
            raise last_err
        raise ValueError(f"无可用期权数据源（代码 {code}）")

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
        if not (date_col and o and h and l and c):
            raise ValueError(f"期权数据缺少关键列，实际列名={list(df.columns)}")
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
