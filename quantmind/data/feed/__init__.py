"""数据馈送包。"""
from .base import BaseDataFeed, HistoryRequest
from .registry import DataFeedRegistry, DataUnavailable
from .akshare_future import AkShareFuturesFeed
from .mootdx_astock import MootdxAStockFeed
from .em_hk import EmHkFeed
from .akshare_option import AkShareOptionFeed
from .mock import MockFeed

__all__ = [
    "BaseDataFeed",
    "HistoryRequest",
    "DataFeedRegistry",
    "DataUnavailable",
    "AkShareFuturesFeed",
    "MootdxAStockFeed",
    "EmHkFeed",
    "AkShareOptionFeed",
    "MockFeed",
]


def build_default_registry() -> DataFeedRegistry:
    """按优先级注册数据源（Fallback Chain）。

    顺序：期货(AKShare) → A股(mootdx) → 港股(东财/Yahoo) → 期权(AKShare) → mock(离线兜底)。
    单源失败自动降级；mock 为最后一级，保证离线也能跑通全链路演示。
    """
    reg = DataFeedRegistry()
    reg.register(AkShareFuturesFeed(), priority=10)
    reg.register(MootdxAStockFeed(), priority=20)
    reg.register(EmHkFeed(), priority=30)
    reg.register(AkShareOptionFeed(), priority=40)
    reg.register(MockFeed(), priority=100)
    return reg
