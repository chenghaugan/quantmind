"""美股数据源（yfinance）。

提供美股日线/分钟线数据，支持 NYSE/NASDAQ 交易所。
依赖：yfinance（可选，未安装时跳过注册）。
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional

from .base import BarData, DataFeed, HistoryRequest
from ..core.constant import Exchange, Interval

_logger = logging.getLogger("quantmind.data.feed.yfinance_us")

try:
    import yfinance as yf
    _YFINANCE_AVAILABLE = True
except ImportError:
    _YFINANCE_AVAILABLE = False
    yf = None  # type: ignore


class YFinanceUSFeed(DataFeed):
    """美股数据源（yfinance）。

    支持标的格式：
    - AAPL (苹果)
    - MSFT (微软)
    - TSLA (特斯拉)
    - SPY (标普500 ETF)
    - QQQ (纳斯达克100 ETF)

    注意：yfinance 免费接口有速率限制，生产环境建议使用付费数据源。
    """

    def __init__(self) -> None:
        super().__init__()
        if not _YFINANCE_AVAILABLE:
            _logger.warning("yfinance 未安装，美股数据源不可用。请运行: pip install yfinance")

    @property
    def name(self) -> str:
        return "yfinance_us"

    def support(self, symbol: str, exchange: Exchange, interval: Interval) -> bool:
        """仅支持美股交易所 + 日线/小时线。"""
        if not _YFINANCE_AVAILABLE:
            return False
        return exchange in (Exchange.NYSE, Exchange.NASDAQ) and interval in (
            Interval.DAILY, Interval.HOUR,
        )

    async def query_history(self, req: HistoryRequest) -> List[BarData]:
        """查询美股历史数据。"""
        if not _YFINANCE_AVAILABLE:
            _logger.error("yfinance 未安装")
            return []

        if not self.support(req.symbol, req.exchange, req.interval):
            _logger.warning("不支持的标的/交易所/周期: %s %s %s",
                           req.symbol, req.exchange, req.interval)
            return []

        try:
            # yfinance  ticker
            ticker = yf.Ticker(req.symbol)

            # 确定时间范围
            start = req.start or datetime(2020, 1, 1)
            end = req.end or datetime.now()

            # 确定间隔
            interval_map = {
                Interval.DAILY: "1d",
                Interval.HOUR: "1h",
            }
            yf_interval = interval_map.get(req.interval, "1d")

            # 下载数据
            df = ticker.history(start=start, end=end, interval=yf_interval)

            if df.empty:
                _logger.warning("yfinance 返回空数据: %s", req.symbol)
                return []

            # 转换为 BarData
            bars: List[BarData] = []
            for idx, row in df.iterrows():
                bar = BarData(
                    symbol=req.symbol,
                    exchange=req.exchange,
                    datetime=idx.to_pydatetime(),
                    interval=req.interval,
                    volume=float(row.get("Volume", 0)),
                    open_price=float(row.get("Open", 0)),
                    high_price=float(row.get("High", 0)),
                    low_price=float(row.get("Low", 0)),
                    close_price=float(row.get("Close", 0)),
                    gateway_name="yfinance",
                )
                bars.append(bar)

            _logger.info("yfinance 加载 %s 完成: %d 根K线", req.symbol, len(bars))
            return bars

        except Exception as exc:  # noqa: BLE001
            _logger.exception("yfinance 查询失败 %s: %s", req.symbol, exc)
            return []


# 注册到数据源注册表
if _YFINANCE_AVAILABLE:
    from .registry import DATA_FEED_REGISTRY
    DATA_FEED_REGISTRY["yfinance_us"] = YFinanceUSFeed
    _logger.info("美股数据源已注册: yfinance_us")


__all__ = ["YFinanceUSFeed"]
