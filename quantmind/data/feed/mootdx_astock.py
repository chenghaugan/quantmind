"""A 股数据源：mootdx（通达信 TCP 7709，无 token、不封 IP）。

来自你提供的 a-stock-data 仓库思路；akshare 作为兜底源。
延迟导入，未安装不影响 core 测试。
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import List, Optional

from .base import BaseDataFeed, HistoryRequest, resolve_ohlc_columns
from ...core.constant import Exchange, Interval
from ...core.object import BarData

_logger = logging.getLogger("quantmind.data.mootdx_astock")

# A 股代码 -> mootdx 市场号（0=深市 1=沪市），简化映射
_SH_MARKETS = {"60", "68", "11", "113", "110"}  # 沪市前缀


def _market_of(code: str) -> int:
    return 1 if code[:2] in _SH_MARKETS else 0


class MootdxAStockFeed(BaseDataFeed):
    name = "mootdx_astock"

    async def fetch_bar_data(self, req: HistoryRequest) -> List[BarData]:
        # 主源：腾讯（akshare stock_zh_a_daily，GET 不封 IP、本网络下 ~1s）；
        # 回退：mootdx 通达信 → akshare 东财。
        # 延迟导入 + 分层降级，保证任一源失败不影响整链。
        try:
            return await self._fetch_tencent(req)
        except Exception as exc:  # noqa: BLE001
            _logger.warning("腾讯A股源失败，尝试 mootdx: %s", exc)
        try:
            return await self._fetch_mootdx(req)
        except Exception as exc:  # noqa: BLE001
            _logger.warning("mootdx 失败，回退 akshare: %s", exc)
            return await self._fetch_akshare_fallback(req)

    async def _fetch_tencent(self, req: HistoryRequest) -> List[BarData]:
        """腾讯日线（akshare ``stock_zh_a_daily``）：股票代码需带 sh/sz 前缀。"""
        import akshare as ak

        prefix = "sh" if _market_of(req.symbol) == 1 else "sz"
        df = await asyncio.to_thread(
            ak.stock_zh_a_daily,
            symbol=f"{prefix}{req.symbol}",
            adjust="qfq",
        )
        return self._df_to_bars(df, req)

    async def _fetch_mootdx(self, req: HistoryRequest) -> List[BarData]:
        # 通达信源返回不复权价格，与链路其他源（腾讯/东财 adjust="qfq"）口径不一致。
        # 为避免混合口径静默回写缓存与数据库造成假跳变，弃用本源走 qfq fallback。
        # （若未来接入，需用复权因子处理后再返回）
        raise ValueError(
            "mootdx 源返回不复权价格，与全链路 qfq 口径不一致，已禁用（走 qfq fallback）"
        )

    async def _fetch_akshare_fallback(self, req: HistoryRequest) -> List[BarData]:
        import akshare as ak

        df = await asyncio.to_thread(
            ak.stock_zh_a_hist,
            symbol=req.symbol,
            period=self._freq(req.interval),
            adjust="qfq",
        )
        return self._df_to_bars(df, req)

    @staticmethod
    def _freq(interval: Interval) -> str:
        freq = {
            "1d": "daily", "5m": "5", "15m": "15", "30m": "30", "1h": "60",
        }.get(interval.value)
        if freq is None:
            # 未映射周期直接报错让 registry 走 fallback，
            # 避免静默降级成日线后以错误周期标签入库污染存储。
            raise ValueError(
                f"mootdx 数据源不支持周期 {interval.value}，"
                f"支持：1d/5m/15m/30m/1h"
            )
        return freq

    @staticmethod
    def _df_to_bars(df, req: HistoryRequest) -> List[BarData]:
        bars: List[BarData] = []
        if df is None or len(df) == 0:
            return bars
        cols = resolve_ohlc_columns(df)
        date_col = cols.get("date")
        o = cols.get("open")
        h = cols.get("high")
        l = cols.get("low")
        c = cols.get("close")
        v = cols.get("volume")
        to = cols.get("turnover")
        if not (date_col and o and h and l and c):
            # 关键列缺失，无法解析（避免 row[None] 抛错后整源降级到 mock）
            raise ValueError(
                f"A股数据缺少关键列，实际列名={list(df.columns)}"
            )
        for _, row in df.iterrows():
            bars.append(
                BarData(
                    symbol=req.symbol,
                    exchange=req.exchange,
                    datetime=_parse_dt(row[date_col]),
                    interval=req.interval,
                    open_price=float(row[o]),
                    high_price=float(row[h]),
                    low_price=float(row[l]),
                    close_price=float(row[c]),
                    volume=float(row[v]) if v else 0.0,
                    turnover=float(row[to]) if to else 0.0,
                )
            )
        return bars


def _parse_dt(value) -> datetime:
    if isinstance(value, datetime):
        return value
    s = str(value)
    # 先尝试完整时间戳（盘中分钟数据如 "2024-01-02 09:05:00"），
    # 再退日期解析；顺序颠倒会把盘中数据截断成 00:00 静默损毁
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return datetime.strptime(s[:10], "%Y-%m-%d")
