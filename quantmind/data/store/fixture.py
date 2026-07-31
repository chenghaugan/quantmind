"""测试/轻量存储夹具：内存实现（无外部依赖，CI 与单测使用）。

实现与 TimescaleStore 一致的接口（save_bars / load_bars），便于在缺少
TimescaleDB 的环境跑通 DataManager 全链路。DuckDB/Parquet 可作为持久化夹具另接。
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Tuple

from ...core.constant import Exchange, Interval
from ...core.object import BarData


class InMemoryStore:
    """进程内 bars 存储。"""

    def __init__(self) -> None:
        # (symbol, exchange, interval) -> list of BarData (按时间升序)
        self._data: Dict[Tuple[str, str, str], List[BarData]] = {}

    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def save_bars(self, bars: List[BarData]) -> int:
        if not bars:
            return 0
        for b in bars:
            key = (b.symbol, b.exchange.value, b.interval.value)
            existing = self._data.setdefault(key, [])
            # 简单去重（按时间）
            seen = {x.datetime for x in existing}
            for nb in bars:
                if nb.datetime not in seen:
                    existing.append(nb)
                    seen.add(nb.datetime)
            existing.sort(key=lambda x: x.datetime)
        return len(bars)

    async def load_bars(
        self,
        symbol: str,
        exchange: Exchange,
        interval: Interval,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 10000,
    ) -> List[BarData]:
        key = (symbol, exchange.value, interval.value)
        bars = self._data.get(key, [])
        out = [
            b
            for b in bars
            if (start is None or b.datetime >= start)
            and (end is None or b.datetime <= end)
        ]
        return out[:limit]
