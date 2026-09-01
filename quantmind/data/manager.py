"""数据管理器：统一查询入口（缓存 → 持久库 → 数据源回退 → 回写）。

查询链：本地行情仓库(Parquet，秒级) → Redis 缓存（最新）→ TimescaleDB（区间）
→ 数据源 Fallback Chain 下载并回写 DB + 缓存 + 本地仓库。
对应 vnpy 把 datafeed 与 database 分开、由统一入口调度的设计。
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from .feed.base import BaseDataFeed, HistoryRequest
from .feed.registry import DataFeedRegistry
from .store.timescale import TimescaleStore
from .store.cache import RedisStore
from .store.fixture import InMemoryStore
from .store.disk_cache import DiskBarCache
from ..core.object import BarData

_logger = logging.getLogger("quantmind.data.manager")

# 真实数据源标识一组（用于「是否可回写本地仓库」判定）。mock 除外，其余皆真实。
_MOCK_SOURCE_NAMES = {"mock", "persistent_store"}


class DataManager:
    """统一数据访问。"""

    def __init__(
        self,
        registry: DataFeedRegistry,
        store: TimescaleStore | InMemoryStore,
        cache: Optional[RedisStore] = None,
        disk_cache: Optional[DiskBarCache] = None,
    ) -> None:
        self.registry = registry
        self.store = store
        self.cache = cache
        self.disk_cache = disk_cache

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

        查询链（缓存未命中时逐级下探）：
            1) 本地行情仓库（Parquet，秒级，仅当配置且非 refresh）
            2) 持久库（失败容错：离线/未连 DB 时回退到数据源）
            3) 数据源 Fallback Chain（真实源 / mock）
        命中真实源（非 mock / 非持久库）后回写本地行情仓库与持久库。

        ``source_sink``（可选）：dict，实际数据源成功时写入
        ``source_sink[req.symbol] = feed.name``（真实 / mock）。
        """
        # 1) 本地行情仓库（最快，秒级）
        if self.disk_cache is not None:
            try:
                hit = self.disk_cache.load(req)
            except Exception as exc:  # noqa: BLE001
                _logger.warning("本地行情仓库读取异常（忽略）: %s", exc)
                hit = []
            if hit:
                _logger.debug("命中本地行情仓库: %s.%s (%d)", req.symbol, req.exchange.value, len(hit))
                if source_sink is not None:
                    source_sink[req.symbol] = "disk_cache"
                return hit

        # 2) 持久库（失败容错：离线/未连 DB 时回退到数据源）
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

        # 3) 数据源 Fallback Chain
        _sink: Dict[str, str] = {}
        bars = await self.registry.get_bar_data(req, source_sink=_sink)
        src_name = _sink.get(req.symbol, "")
        # 部分数据源（新浪/腾讯 fallback 等）无视 start/end 返回全量历史，
        # 统一在此按请求窗口裁剪，保证各层返回口径一致（防前视/防越界）。
        bars = self._clip_bars(bars, req)

        # 4) 回写（容错）：真实源数据落本地仓库 + 持久库
        if bars:
            # 仅真实源可回写本地仓库（mock 合成数据不入库，避免污染）
            if src_name and src_name not in _MOCK_SOURCE_NAMES:
                if self.disk_cache is not None:
                    try:
                        self.disk_cache.save(bars)
                    except Exception as exc:  # noqa: BLE001
                        _logger.warning("回写本地行情仓库失败（忽略）: %s", exc)
            try:
                await self.store.save_bars(bars)
                if self.cache is not None:
                    await self.cache.set_latest_bar(bars[-1])
            except Exception as exc:  # noqa: BLE001
                _logger.warning("回写存储失败（忽略）: %s", exc)

        if source_sink is not None:
            source_sink[req.symbol] = src_name or "unknown"
        return bars

    @staticmethod
    def _clip_bars(bars: List[BarData], req: HistoryRequest) -> List[BarData]:
        """按请求窗口裁剪 bars（兼容 tz-aware/naive 混合比较）。"""
        if not bars:
            return bars

        def _norm(dt):
            return dt.replace(tzinfo=None) if dt.tzinfo else dt

        start = _norm(req.start) if req.start is not None else None
        end = _norm(req.end) if req.end is not None else None
        out = bars
        if start is not None:
            out = [b for b in out if _norm(b.datetime) >= start]
        if end is not None:
            out = [b for b in out if _norm(b.datetime) <= end]
        return out
