"""FactorService: 因子评估（带缓存）"""
import time
import logging
from typing import Dict, Optional

from ...core.constant import Exchange, Interval
from ...data.feed.base import HistoryRequest
from ...data import DataManager
from ...research import FactorRegistry, FactorEvaluator, eval_factor_expression
from ..schemas import FactorRequest, FactorResult

_logger = logging.getLogger("quantmind.api")


class FactorService:
    """因子评估服务，内置 1 小时 TTL 缓存"""

    _CACHE_TTL = 3600  # 秒

    def __init__(self, dm: DataManager):
        self.dm = dm
        self._cache: Dict[str, tuple] = {}  # key -> (timestamp, result)

    def _get_cached(self, key: str) -> Optional[FactorResult]:
        if key in self._cache:
            ts, result = self._cache[key]
            if time.time() - ts < self._CACHE_TTL:
                return result
        return None

    def _set_cached(self, key: str, result: FactorResult) -> None:
        self._cache[key] = (time.time(), result)

    @staticmethod
    def _resolve_factor(name: str, window: int):
        """按名称解析内置因子"""
        reg = FactorRegistry()
        try:
            return reg.get(name)
        except KeyError:
            pass
        if "_" in name:
            kind, _, w = name.rpartition("_")
            try:
                w = int(w)
            except ValueError:
                w = window
            from ...research.technical import build_factor
            return build_factor(kind, w)
        from ...research.technical import build_factor
        return build_factor(name, window)

    async def evaluate(self, req: FactorRequest) -> FactorResult:
        """评估因子，优先查缓存"""
        cache_key = f"{req.symbol}_{req.exchange}_{req.factor}_{req.expression}_{req.window}_{req.forward_periods}"

        cached = self._get_cached(cache_key)
        if cached is not None:
            _logger.info(f"Factor cache hit: {cache_key}")
            return cached

        try:
            exch = Exchange(req.exchange.upper())
            interv = Interval(req.interval)
            bars = await self.dm.get_bar_data(
                HistoryRequest(symbol=req.symbol, exchange=exch, interval=interv)
            )
            if not bars:
                _logger.warning(f"因子评估失败：无数据 {req.symbol}.{req.exchange}")
                return FactorResult(factor_name=req.factor, n_samples=0, error="无可用数据")

            if req.expression:
                from ...research.factors.base import bars_to_df
                df = bars_to_df(bars)
                series = eval_factor_expression(req.expression, df)
                name = req.expression
            else:
                f = self._resolve_factor(req.factor, req.window)
                series = f.compute(bars)
                name = f.meta.name
            series.name = name
            rep = FactorEvaluator().evaluate(series, bars, forward_periods=req.forward_periods)
            result = FactorResult(
                factor_name=rep.factor_name,
                ic_mean=rep.ic_mean,
                ir=rep.ir,
                ic_std=rep.ic_std,
                ic_positive_ratio=rep.ic_positive_ratio,
                ic_decay=rep.ic_decay,
                top_quantile_return=rep.top_quantile_return,
                long_short_return=rep.long_short_return,
                n_samples=rep.n_samples,
            )

            self._set_cached(cache_key, result)
            _logger.info(f"Factor cached: {cache_key}")
            return result
            
        except Exception as e:
            _logger.error(f"因子评估异常 {cache_key}: {str(e)}", exc_info=True)
            return FactorResult(factor_name=req.factor, n_samples=0, error=str(e))
