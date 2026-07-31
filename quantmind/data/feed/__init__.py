"""数据馈送包。"""
from .base import BaseDataFeed, HistoryRequest
from .registry import DataFeedRegistry, DataUnavailable
from .akshare_future import AkShareFuturesFeed
from .mootdx_astock import MootdxAStockFeed
from .em_hk import EmHkFeed
from .akshare_option import AkShareOptionFeed
from .mock import MockFeed
from .local_file import LocalFileFeed
from .china_futures_csv import ChinaFuturesCSVFeed

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
    "LocalFileFeed",
    "ChinaFuturesCSVFeed",
]


def build_default_registry(local_data_root: str | None = None) -> DataFeedRegistry:
    """按优先级注册数据源（Fallback Chain）。

    默认顺序：期货(AKShare) → A股(mootdx) → 港股(东财) → 期权(AKShare) → mock(兜底)。
    若 ``local_data_root`` 指向已克隆的本地数据根目录（如 china-futures CSV），则注册
    ``ChinaFuturesCSVFeed`` 且优先级最高（5）：期货请求优先吃本地真实文件；文件缺失时自动
    降级到 AKShare，再降级到 mock。单源失败自动降级，保证全链路可跑。
    """
    import logging
    from pathlib import Path

    _logger = logging.getLogger("quantmind.data.registry")
    reg = DataFeedRegistry()
    if local_data_root:
        if Path(local_data_root).exists():
            reg.register(ChinaFuturesCSVFeed(local_data_root), priority=5)
            _logger.info("已注册本地期货源: %s", local_data_root)
        else:
            _logger.warning("local_data_root 不存在，跳过本地源: %s", local_data_root)
    reg.register(AkShareFuturesFeed(), priority=10)
    reg.register(MootdxAStockFeed(), priority=20)
    reg.register(EmHkFeed(), priority=30)
    reg.register(AkShareOptionFeed(), priority=40)
    reg.register(MockFeed(), priority=100)
    return reg
