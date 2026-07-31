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

from .base import BaseDataFeed, HistoryRequest
from ...core.constant import Exchange, Interval
from ...core.object import BarData

_logger = logging.getLogger("quantmind.data.akshare_future")

# 各交易所对应的具体合约日线函数
_DAILY_BY_EXCHANGE = {
    Exchange.SHFE: "futures_daily_shfe",
    Exchange.DCE: "futures_daily_dce",
    Exchange.CZCE: "futures_daily_czce",
    Exchange.INE: "futures_daily_ine",
    Exchange.CFFEX: "futures_daily_cffex",
}


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
        # 主连/指数约定：symbol 以 '0' 结尾（如 rb0 / IF0）走 Sina 主连接口
        is_main = symbol.endswith("0")
        if req.interval == Interval.DAILY:
            if is_main:
                df = await asyncio.to_thread(ak.futures_zh_daily_sina, symbol=symbol)
            else:
                func = _DAILY_BY_EXCHANGE.get(req.exchange)
                if func is None:
                    raise ValueError(f"不支持的期货交易所: {req.exchange}")
                df = await asyncio.to_thread(getattr(ak, func), symbol=symbol)
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
        cols = {c.lower(): c for c in df.columns}
        date_col = cols.get("date") or cols.get("datetime") or cols.get("trade_date")
        o = cols.get("open")
        h = cols.get("high")
        l = cols.get("low")
        c = cols.get("close")
        v = cols.get("volume")
        oi = cols.get("open_interest") or cols.get("hold")
        to = cols.get("turnover") or cols.get("amount")
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
