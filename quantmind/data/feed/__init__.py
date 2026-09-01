"""数据馈送包。"""
from .base import BaseDataFeed, HistoryRequest
from .registry import DataFeedRegistry, DataUnavailable
from .akshare_future import AkShareFuturesFeed
from .efinance_feed import EfinanceFeed
from .mootdx_astock import MootdxAStockFeed
from .em_hk import EmHkFeed
from .akshare_option import AkShareOptionFeed
from .mock import MockFeed
from .local_file import LocalFileFeed
from .china_futures_csv import ChinaFuturesCSVFeed
from .tqsdk_feed import TqSdkFeed
from .local_daily import (
    LocalDailyParquetFeed,
    ChinaAStockParquetFeed,
    ChinaHKAStockParquetFeed,
    ChinaOptionParquetFeed,
)

__all__ = [
    "BaseDataFeed",
    "HistoryRequest",
    "DataFeedRegistry",
    "DataUnavailable",
    "AkShareFuturesFeed",
    "EfinanceFeed",
    "MootdxAStockFeed",
    "EmHkFeed",
    "AkShareOptionFeed",
    "MockFeed",
    "LocalFileFeed",
    "ChinaFuturesCSVFeed",
    "TqSdkFeed",
    "LocalDailyParquetFeed",
    "ChinaAStockParquetFeed",
    "ChinaHKAStockParquetFeed",
    "ChinaOptionParquetFeed",
]


def build_default_registry(
    local_data_root: str | None = None,
    local_stock_root: str | None = None,
    local_hk_root: str | None = None,
    local_option_root: str | None = None,
    continuous_method: str = "back_adjusted",
) -> DataFeedRegistry:
    """按优先级注册数据源（Fallback Chain）。

    默认顺序：期货(AKShare) → A股(mootdx) → 港股(东财) → 期权(AKShare) → mock(兜底)。
    若传入本地根目录，则在对应位置插入本地真实文件源（优先级高于对应实时源、低于期货本地源）：

      - ``local_data_root``   -> ChinaFuturesCSVFeed（priority=5，期货真实数据优先）
      - ``local_stock_root``  -> ChinaAStockParquetFeed（priority=15，高于 mootdx 20）
      - ``local_hk_root``     -> ChinaHKAStockParquetFeed（priority=25，高于 em_hk 30）
      - ``local_option_root`` -> ChinaOptionParquetFeed（priority=35，高于 akshare_option 40）

    文件缺失/未配置时自动降级到实时源或 mock，单源失败不影响全链路。

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
    # TqSdk 提供 8000 根 K 线（1m≈47天，5m≈251天，15m≈2年，30m≈4年），远超新浪 1023 根
    # 需要环境变量 QM_TQSDK_USER 和 QM_TQSDK_PASS，未配置时自动跳过
    import os
    if os.environ.get("QM_TQSDK_USER") and os.environ.get("QM_TQSDK_PASS"):
        reg.register(TqSdkFeed(), priority=8)
        _logger.info("已注册 TqSdk 数据源（8000根K线）")
    reg.register(AkShareFuturesFeed(), priority=10)
    reg.register(EfinanceFeed(), priority=11)  # efinance 作为期货数据备选源
    if local_stock_root:
        if Path(local_stock_root).exists():
            reg.register(ChinaAStockParquetFeed(local_stock_root), priority=15)
            _logger.info("已注册本地 A 股源: %s", local_stock_root)
        else:
            _logger.warning("local_stock_root 不存在，跳过本地 A 股源: %s", local_stock_root)
    reg.register(MootdxAStockFeed(), priority=20)
    if local_hk_root:
        if Path(local_hk_root).exists():
            reg.register(ChinaHKAStockParquetFeed(local_hk_root), priority=25)
            _logger.info("已注册本地港股源: %s", local_hk_root)
        else:
            _logger.warning("local_hk_root 不存在，跳过本地港股源: %s", local_hk_root)
    reg.register(EmHkFeed(), priority=30)
    if local_option_root:
        if Path(local_option_root).exists():
            reg.register(ChinaOptionParquetFeed(local_option_root), priority=35)
            _logger.info("已注册本地期权源: %s", local_option_root)
        else:
            _logger.warning("local_option_root 不存在，跳过本地期权源: %s", local_option_root)
    reg.register(AkShareOptionFeed(), priority=40)
    reg.register(MockFeed(), priority=100)
    return reg
