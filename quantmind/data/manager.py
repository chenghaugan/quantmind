"""数据管理器：统一查询入口（缓存 → 持久库 → 数据源回退 → 回写）。

查询链：Redis 缓存（最新）→ TimescaleDB（区间）→ 数据源 Fallback Chain 下载
并回写 DB + 缓存。对应 vnpy 把 datafeed 与 database 分开、由统一入口调度的设计。
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from .feed.base import BaseDataFeed, HistoryRequest
from .feed.registry import DataFeedRegistry
from .store.timescale import TimescaleStore
from .store.cache import RedisStore
from .store.fixture import InMemoryStore
from ..core.object import BarData

_logger = logging.getLogger("quantmind.data.manager")


class DataManager:
    """统一数据访问。"""

    def __init__(
        self,
        registry: DataFeedRegistry,
        store: TimescaleStore | InMemoryStore,
        cache: Optional[RedisStore] = None,
    ) -> None:
        self.registry = registry
        self.store = store
        self.cache = cache

    async def connect(self) -> None:
        await self.store.connect()
        if self.cache is not None:
            await self.cache.connect()

    async def close(self) -> None:
        await self.store.close()
        if self.cache is not None:
            await self.cache.close()

    async def get_bar_data(
        self,
        req: HistoryRequest,
        source_sink: Optional[Dict[str, str]] = None,
    ) -> List[BarData]:
        """统一数据访问。

        ``source_sink``（可选）：dict，实际数据源成功时写入
        ``source_sink[req.symbol] = feed.name``（真实 / mock）。
        """
        # 1) 持久库（失败容错：离线/未连 DB 时回退到数据源）
        bars: List[BarData] = []
        try:
            bars = await self.store.load_bars(
                symbol=req.symbol,
                exchange=req.exchange,
                interval=req.interval,
                start=req.start,
                end=req.end,
            )
        except Exception as exc:  # noqa: BLE001
            _logger.warning("持久库读取失败，回退数据源: %s", exc)
            bars = []
        if bars:
            _logger.debug("命中持久库: %s.%s (%d)", req.symbol, req.exchange.value, len(bars))
            if source_sink is not None:
                source_sink[req.symbol] = "persistent_store"
            return bars

        # 2) 数据源 Fallback Chain
        bars = await self.registry.get_bar_data(req, source_sink=source_sink)

        # 3) 回写（容错）
        if bars:
            try:
                await self.store.save_bars(bars)
                if self.cache is not None:
                    await self.cache.set_latest_bar(bars[-1])
            except Exception as exc:  # noqa: BLE001
                _logger.warning("回写存储失败（忽略）: %s", exc)
        return bars
