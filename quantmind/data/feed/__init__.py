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
from .astock_parquet import ChinaAStockParquetFeed

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
    "ChinaAStockParquetFeed",
]


def build_default_registry(
    local_data_root: str | None = None,
    local_stock_root: str | None = None,
    continuous_method: str = "back_adjusted",
) -> DataFeedRegistry:
    """按优先级注册数据源（Fallback Chain）。

    默认顺序：期货(AKShare) → A股(mootdx) → 港股(东财) → 期权(AKShare) → mock(兜底)。
    若 ``local_data_root`` 指向已克隆的本地数据根目录（如 china-futures CSV），则注册
    ``ChinaFuturesCSVFeed`` 且优先级最高（5）：期货请求优先吃本地真实文件；文件缺失时自动
    降级到 AKShare，再降级到 mock。单源失败自动降级，保证全链路可跑。

    若 ``local_stock_root`` 指向已落地的 A 股 Parquet/CSV 根目录（如 astock-data-toolkit），
    则注册 ``ChinaAStockParquetFeed``（优先级 15，高于 mootdx 的 20）：A 股(SSE/SZSE) 请求
    优先吃本地真实文件，缺失时降级到 mootdx / mock。

    :param continuous_method: 主连构造方式，透传给 ``ChinaFuturesCSVFeed``
        （"back_adjusted" 向后复权，默认/推荐；"simple" 旧式窗口拼接）。
    """
    import logging
    from pathlib import Path

    _logger = logging.getLogger("quantmind.data.registry")
    reg = DataFeedRegistry()
    if local_data_root:
        if Path(local_data_root).exists():
            reg.register(
                ChinaFuturesCSVFeed(local_data_root, continuous_method=continuous_method),
                priority=5,
            )
            _logger.info("已注册本地期货源: %s (主连=%s)", local_data_root, continuous_method)
        else:
            _logger.warning("local_data_root 不存在，跳过本地源: %s", local_data_root)
    reg.register(AkShareFuturesFeed(), priority=10)
    if local_stock_root:
        if Path(local_stock_root).exists():
            reg.register(ChinaAStockParquetFeed(local_stock_root), priority=15)
            _logger.info("已注册本地 A 股源: %s", local_stock_root)
        else:
            _logger.warning("local_stock_root 不存在，跳过本地 A 股源: %s", local_stock_root)
    reg.register(MootdxAStockFeed(), priority=20)
    reg.register(EmHkFeed(), priority=30)
    reg.register(AkShareOptionFeed(), priority=40)
    reg.register(MockFeed(), priority=100)
    return reg
