"""TimescaleDB 存储（异步，sqlalchemy + asyncpg）。

提供 bars 超表的写入（UPSERT）与按区间读取。连接延迟建立，导入本模块无需数据库。
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from ...core.constant import Exchange, Interval
from ...core.object import BarData

_logger = logging.getLogger("quantmind.data.timescale")


class TimescaleStore:
    """TimescaleDB 持久化存储。"""

    def __init__(self, url: str) -> None:
        self.url = url
        self._engine = None
        self._sessionmaker: Optional[async_sessionmaker] = None

    async def connect(self) -> None:
        self._engine = create_async_engine(self.url, pool_pre_ping=True)
        self._sessionmaker = async_sessionmaker(self._engine, expire_on_commit=False)
        _logger.info("TimescaleDB 连接已建立: %s", _mask(self.url))

    async def close(self) -> None:
        if self._engine:
            await self._engine.dispose()

    async def save_bars(self, bars: List[BarData]) -> int:
        if not bars or self._sessionmaker is None:
            return 0
        rows = [
            {
                "symbol": b.symbol,
                "exchange": b.exchange.value,
                "interval": b.interval.value,
                "ts": b.datetime,
                "open": b.open_price,
                "high": b.high_price,
                "low": b.low_price,
                "close": b.close_price,
                "volume": b.volume,
                "open_interest": b.open_interest,
                "turnover": b.turnover,
            }
            for b in bars
        ]
        stmt = text(
            """
            INSERT INTO bars (symbol, exchange, interval, ts, open, high, low, close, volume, open_interest, turnover)
            VALUES (:symbol, :exchange, :interval, :ts, :open, :high, :low, :close, :volume, :open_interest, :turnover)
            ON CONFLICT (symbol, exchange, interval, ts) DO UPDATE SET
                open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low, close=EXCLUDED.close,
                volume=EXCLUDED.volume, open_interest=EXCLUDED.open_interest, turnover=EXCLUDED.turnover
            """
        )
        async with self._sessionmaker() as session:  # type: AsyncSession
            await session.execute(stmt, rows)
            await session.commit()
        return len(rows)

    async def load_bars(
        self,
        symbol: str,
        exchange: Exchange,
        interval: Interval,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 10000,
    ) -> List[BarData]:
        if self._sessionmaker is None:
            return []
        conds = ["symbol=:symbol", "exchange=:exchange", "interval=:interval"]
        params = {
            "symbol": symbol,
            "exchange": exchange.value,
            "interval": interval.value,
        }
        if start:
            conds.append("ts >= :start")
            params["start"] = start
        if end:
            conds.append("ts <= :end")
            params["end"] = end
        stmt = text(
            f"SELECT symbol, exchange, interval, ts, open, high, low, close, volume, open_interest, turnover "
            f"FROM bars WHERE {' AND '.join(conds)} ORDER BY ts ASC LIMIT :limit"
        )
        params["limit"] = limit
        async with self._sessionmaker() as session:
            result = await session.execute(stmt, params)
            rows = result.mappings().all()
        bars: List[BarData] = []
        for r in rows:
            bars.append(
                BarData(
                    symbol=r["symbol"],
                    exchange=Exchange(r["exchange"]),
                    datetime=r["ts"],
                    interval=Interval(r["interval"]),
                    open_price=r["open"],
                    high_price=r["high"],
                    low_price=r["low"],
                    close_price=r["close"],
                    volume=r["volume"] or 0.0,
                    open_interest=r["open_interest"] or 0.0,
                    turnover=r["turnover"] or 0.0,
                )
            )
        return bars


def _mask(url: str) -> str:
    # 隐藏密码
    return url.replace("://", "://***@", 1) if "://" in url else url
