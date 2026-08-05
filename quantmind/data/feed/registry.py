"""数据源注册表 + Fallback Chain（借鉴 Vibe-Trading 的 Loader Registry + Fallback）。

按优先级尝试各数据源；单源失败（网络/限流/接口变更）时自动降级到下一个，
实现"失败闭合"而非静默返回残缺数据。
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from .base import BaseDataFeed, HistoryRequest
from ...core.object import BarData

_logger = logging.getLogger("quantmind.data.registry")


class DataUnavailable(RuntimeError):
    """所有数据源均不可用。"""


class DataFeedRegistry:
    """数据源注册表（带优先级与回退链）。"""

    def __init__(self) -> None:
        # name -> (feed, priority)
        self._feeds: Dict[str, Tuple[BaseDataFeed, int]] = {}

    def register(self, feed: BaseDataFeed, priority: int = 10) -> None:
        self._feeds[feed.name] = (feed, priority)
        _logger.info("注册数据源 %s (优先级 %d)", feed.name, priority)

    def ordered(self) -> List[Tuple[BaseDataFeed, int]]:
        return sorted(self._feeds.values(), key=lambda x: x[1])

    def list_feeds(self) -> List[str]:
        return [name for name in self._feeds]

    async def get_bar_data(self, req: HistoryRequest, source_sink: Dict[str, str] | None = None) -> List[BarData]:
        """按优先级回退拉取；任一源成功即返回，全部失败抛 ``DataUnavailable``。

        ``source_sink``（可选）：dict，成功时写入 ``source_sink[req.symbol] = feed.name``，
        供调用方知道该标的实际由哪个数据源（真实 / mock）提供。
        """
        errors: List[str] = []
        for feed, _priority in self.ordered():
            try:
                bars = await feed.fetch_bar_data(req)
                if bars:
                    _logger.info(
                        "数据源 %s 返回 %d 根 %s.%s",
                        feed.name,
                        len(bars),
                        req.symbol,
                        req.exchange.value,
                    )
                    if source_sink is not None:
                        source_sink[req.symbol] = feed.name
                    return bars
                errors.append(f"{feed.name}: 空结果")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{feed.name}: {exc!r}")
                _logger.warning("数据源 %s 失败，尝试下一个: %s", feed.name, exc)
                continue
        raise DataUnavailable("; ".join(errors))
