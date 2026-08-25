"""FactorService: 因子评估（带缓存）——单标的时序 / 多标的（截面）双模式"""
import asyncio
import logging
import time
from typing import Dict, Optional

from ...core.constant import Exchange, Interval
from ...data.feed.base import HistoryRequest
from ...data import DataManager
from ...research import FactorRegistry, FactorEvaluator, eval_factor_expression
from ...research.factors import Panel, list_alpha_cs
from ...research.cross_sectional_backtest import cross_sectional_backtest
from ..schemas import FactorRequest, FactorResult

_logger = logging.getLogger("quantmind.api")


def _sanitize(o):
    import math
    from datetime import datetime
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, (int, str, bool)) or o is None:
        return o
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(x) for x in o]
    if isinstance(o, datetime):
        return o.isoformat()
    return o


class FactorService:
    """因子评估服务，内置 1 小时 TTL 缓存"""

    _CACHE_TTL = 3600  # 秒

    def __init__(self, dm: DataManager):
        self.dm = dm
        self._cache: Dict[str, tuple] = {}  # key -> (timestamp, result)

    @staticmethod
    def cs_factors() -> list:
        """多标（截面）模式可用的截面 Alpha 因子清单。"""
        return list_alpha_cs()


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
        """单标的时序因子评估，优先查缓存（向后兼容 /factor 端点）。"""
        cache_key = f"{req.symbol}_{req.exchange}_{req.factor}_{req.expression}_{req.window}_{req.forward_periods}_{req.start}_{req.end}"

        cached = self._get_cached(cache_key)
        if cached is not None:
            _logger.info(f"Factor cache hit: {cache_key}")
            return cached

        try:
            exch = Exchange((req.exchange or "SHFE").upper())
            interv = Interval(req.interval or "1d")
            # 构造 HistoryRequest，支持可选日期范围
            history_kwargs = {"symbol": req.symbol, "exchange": exch, "interval": interv}
            if req.start:
                from datetime import datetime as _dt
                history_kwargs["start"] = _dt.fromisoformat(req.start)
            if req.end:
                from datetime import datetime as _dt
                history_kwargs["end"] = _dt.fromisoformat(req.end)
            bars = await self.dm.get_bar_data(HistoryRequest(**history_kwargs))
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

    async def evaluate_dict(self, req: FactorRequest) -> dict:
        """统一评估入口：多标（symbols≥2）→ 截面契约 dict；否则单标 → FactorResult 转 dict。"""
        multi = [s.strip() for s in (req.symbols or []) if s and s.strip()]
        if len(multi) >= 2:
            out = await self._evaluate_multi(req, multi)
            if "error" in out:
                return out
            # 补齐 n_symbols/date 等便于前端统一展示
            return out
        # 单标的：走 FactorResult 并补齐完整指标
        rep = await self.evaluate(req)
        d = rep.model_dump() if hasattr(rep, "model_dump") else dict(rep)
        return _sanitize(d)
    async def _evaluate_multi(self, req, symbols) -> dict:
        """多标的截面评估：构造面板 → cross_sectional_backtest → 返回完整契约。

        与旧 CrossSectionService 等效，但并入 FactorService（失败闭合、向后兼容）。
        """
        cache_key = ("multi", str(req.expression), tuple(sorted(symbols)),
                     req.exchange, req.forward_periods, req.n_groups,
                     bool(req.long_short), float(req.cost_rate), req.start, req.end)
        cached = self._get_cached(str(cache_key))
        if cached is not None:
            return _sanitize(cached)
        try:
            default_exch = (req.exchange or "SHFE").upper()
            interv = Interval(req.interval or "1d")
            tasks = []
            for s in symbols:
                sym, exch_str = s, default_exch
                if "." in s:
                    head, _, ex = s.rpartition(".")
                    if head and ex:
                        sym, exch_str = head.strip(), ex.strip().upper()
                hreq = HistoryRequest(
                    symbol=sym, exchange=Exchange(exch_str), interval=interv,
                    start=req.start, end=req.end,
                )
                tasks.append(self.dm.get_bar_data(hreq))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            bars_by_symbol, missing = {}, []
            for sym, res in zip(symbols, results):
                if isinstance(res, Exception) or not res:
                    missing.append(sym)
                    continue
                bars_by_symbol[sym] = res
            if len(bars_by_symbol) < 2:
                return {"error": f"可用标的不足 2 个（缺失: {missing or '无'}）"}
            panel = Panel.from_bars(bars_by_symbol)
            loop = asyncio.get_running_loop()
            rep = await loop.run_in_executor(
                None,
                lambda: cross_sectional_backtest(
                    panel, req.factor,
                    forward_periods=req.forward_periods,
                    n_groups=req.n_groups,
                    long_short=req.long_short,
                    cost_rate=req.cost_rate,
                    expression=req.expression,
                ),
            )
            out = dict(_sanitize(rep))
            out["n_symbols"] = len(panel.symbols)
            dates = list(panel.dates)
            out["date_range"] = [
                dates[0].isoformat() if dates else None,
                dates[-1].isoformat() if dates else None,
            ]
            out["missing"] = missing
            out["symbols"] = list(panel.symbols)
            out = _sanitize(out)
            self._set_cached(str(cache_key), out)
            return out
        except Exception as exc:  # noqa: BLE001 失败闭合
            _logger.error("多标因子评估异常: %s", exc, exc_info=True)
            return {"error": f"多标因子评估失败：{exc}"}
