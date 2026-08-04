"""DataService: 行情数据查询（带分页） + 数据质量体检"""
import logging
from datetime import datetime, timedelta
from typing import Optional

from ...core.constant import Exchange, Interval
from ...data.feed.base import HistoryRequest
from ...data import DataManager
from ...data.quality import check_bars
from ..schemas import BarOut


_logger = logging.getLogger("quantmind.api")


class DataService:
    def __init__(self, dm: DataManager):
        self.dm = dm

    async def query_bars(
        self,
        symbol: str,
        exchange: str,
        interval: str,
        start: Optional[str],
        end: Optional[str],
        page: int,
        page_size: int,
    ) -> dict:
        """查询行情数据（支持分页）"""
        try:
            exch = Exchange(exchange.upper())
            interv = Interval(interval)
            req = HistoryRequest(
                symbol=symbol,
                exchange=exch,
                interval=interv,
                start=datetime.fromisoformat(start) if start else None,
                end=datetime.fromisoformat(end) if end else None,
            )
            bars = await self.dm.get_bar_data(req)

            if not bars:
                _logger.warning(f"数据查询返回空结果: {symbol}.{exchange} {interval}")
                return {
                    "data": [],
                    "pagination": {
                        "page": page,
                        "page_size": page_size,
                        "total": 0,
                        "total_pages": 0,
                    },
                }

            # 分页处理
            total = len(bars)
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            page_bars = bars[start_idx:end_idx]

            result = [
                BarOut(
                    symbol=b.symbol,
                    exchange=b.exchange.value,
                    datetime=b.datetime.isoformat(),
                    interval=b.interval.value,
                    open=b.open_price,
                    high=b.high_price,
                    low=b.low_price,
                    close=b.close_price,
                    volume=b.volume,
                )
                for b in page_bars
            ]

            _logger.info(f"数据查询成功: {symbol}.{exchange} {interval}, 共 {total} 条, 返回 {len(result)} 条")

            return {
                "data": result,
                "pagination": {
                    "page": page,
                    "page_size": page_size,
                    "total": total,
                    "total_pages": (total + page_size - 1) // page_size,
                },
            }
        except Exception as e:
            _logger.error(f"数据查询失败: {symbol}.{exchange} - {str(e)}", exc_info=True)
            raise ValueError(f"数据查询失败: {str(e)}") from e

    async def quality_report(
        self,
        symbol: str,
        exchange: str,
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
        freshness_days: Optional[int] = None,
    ) -> dict:
        """数据质量体检：间隙 / 异常尖峰 / 换月跳变 / 新鲜度，并给出 0-100 评分。"""
        try:
            exch = Exchange(exchange.upper())
            interv = Interval(interval)
            req = HistoryRequest(
                symbol=symbol,
                exchange=exch,
                interval=interv,
                start=datetime.fromisoformat(start) if start else None,
                end=datetime.fromisoformat(end) if end else None,
            )
            bars = await self.dm.get_bar_data(req)
        except Exception as e:  # noqa: BLE001
            _logger.error("数据质量检查失败: %s.%s - %s", symbol, exchange, e, exc_info=True)
            raise ValueError(f"数据质量检查失败: {e}") from e

        if not bars:
            return {
                "symbol": f"{symbol}.{exchange.upper()}",
                "total": 0,
                "gaps": 0,
                "outliers": 0,
                "rollover_jumps": 0,
                "last_ts": None,
                "stale": True,
                "issues": ["空数据：该标的在所选区间内没有任何 K 线"],
                "score": 0.0,
            }

        fresh = timedelta(days=freshness_days) if freshness_days else None
        rep = check_bars(bars, interv, freshness=fresh)

        total = max(rep.total, 1)
        score = 100.0
        score -= min(35.0, rep.gaps / total * 100 * 3)
        score -= min(35.0, rep.outliers / total * 100 * 5)
        score -= min(15.0, rep.rollover_jumps * 3.0)
        if rep.stale:
            score -= 20.0
        score = round(max(0.0, score), 1)

        return {
            "symbol": rep.symbol or f"{symbol}.{exchange.upper()}",
            "total": rep.total,
            "gaps": rep.gaps,
            "outliers": rep.outliers,
            "rollover_jumps": rep.rollover_jumps,
            "last_ts": rep.last_ts.isoformat() if rep.last_ts else None,
            "stale": rep.stale,
            "issues": rep.issues or ["未发现明显问题"],
            "score": score,
        }
