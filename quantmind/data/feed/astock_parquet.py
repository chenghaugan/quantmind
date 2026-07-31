"""A 股（沪深）本地 Parquet/CSV 适配器。

已重构：``ChinaAStockParquetFeed`` 现定义于 :mod:`quantmind.data.feed.local_daily`
（通用本地日频源基类），本模块仅做向后兼容再导出。A 股源仅接管 SSE/SZSE，非 A 股请求
（期货/港股/期权）不被接走，正确降级到对应数据源。

时间约定：文件时间为**交易日期**（日频，无具体时刻）时，保持交易日期不变；
若文件带具体时刻（如盘后 15:00），则按北京时间减 8h 转 UTC（与体系一致）。
A 股无主力连续概念，恒返回 is_main=False。

注意：读取 .parquet 需 pyarrow 或 fastparquet；未安装时该源静默降级到下一数据源。
"""
from __future__ import annotations

from .local_daily import ChinaAStockParquetFeed

__all__ = ["ChinaAStockParquetFeed"]
