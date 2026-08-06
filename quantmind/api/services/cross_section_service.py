"""CrossSectionService: 多标的截面因子研究 / 多空组合回测。"""
from __future__ import annotations

import asyncio
import logging
import math
from datetime import datetime
from typing import Any, Dict, List

from ...core.constant import Exchange, Interval
from ...data import DataManager
from ...data.feed.base import HistoryRequest
from ...research.factors import Panel, list_alpha_cs
from ...research.cross_sectional_backtest import cross_sectional_backtest
from ...research.evaluator import FactorEvaluator


_logger = logging.getLogger("quantmind.api")


def _sanitize(o: Any) -> Any:
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(x) for x in o]
    if isinstance(o, datetime):
        return o.isoformat()
    return o


class CrossSectionService:
    def __init__(self, dm: DataManager):
        self.dm = dm

    @staticmethod
    def factors() -> List[str]:
        return list_alpha_cs()

    async def _build_panel(self, symbols, exchange, interval, start, end):
        default_exch = exchange.upper()
        interv = Interval(interval or "1d")
        tasks = []
        for s in symbols:
            sym = s.strip()
            exch_str = default_exch
            if "." in sym:
                head, _, ex = sym.rpartition(".")
                if head and ex:
                    sym, exch_str = head.strip(), ex.strip().upper()
            req = HistoryRequest(
                symbol=sym,
                exchange=Exchange(exch_str),
                interval=interv,
                start=datetime.fromisoformat(start) if start else None,
                end=datetime.fromisoformat(end) if end else None,
            )
            tasks.append(self.dm.get_bar_data(req))
        results = await asyncio.gather(*tasks, return_exceptions=True)

        bars_by_symbol: Dict[str, list] = {}
        missing: List[str] = []
        for sym, res in zip(symbols, results):
            if isinstance(res, Exception) or not res:
                missing.append(sym)
                continue
            bars_by_symbol[sym] = res

        if len(bars_by_symbol) < 2:
            raise ValueError(
                f"可用标的不足 2 个（缺失: {missing or '无'}），截面研究至少需要 2 个标的"
            )
        return Panel.from_bars(bars_by_symbol), missing

    async def run(self, req) -> dict:
        symbols = [s.strip() for s in req.symbols if s and s.strip()]
        if len(symbols) < 2:
            raise ValueError("截面研究至少需要 2 个标的")

        panel, missing = await self._build_panel(
            symbols,
            req.exchange,
            getattr(req, "interval", "1d"),
            getattr(req, "start", None),
            getattr(req, "end", None),
        )

        loop = asyncio.get_running_loop()

        if getattr(req, "backtest", True):
            out = await loop.run_in_executor(
                None,
                lambda: cross_sectional_backtest(
                    panel,
                    req.factor,
                    forward_periods=req.forward_periods,
                    n_groups=req.n_groups,
                    long_short=req.long_short,
                    cost_rate=req.cost_rate,
                ),
            )
        else:
            evaluator = FactorEvaluator()
            reports = await loop.run_in_executor(
                None,
                lambda: evaluator.evaluate_cross_sectional_panel(
                    [req.factor],
                    panel,
                    forward_periods=req.forward_periods,
                    n_groups=req.n_groups,
                ),
            )
            rep = reports.get(req.factor)
            out = {
                "factor": req.factor,
                "n_symbols": len(panel.symbols),
                "n_dates": len(panel.dates),
                "ic_report": rep.to_dict() if rep else None,
                "portfolio": None,
            }

        out["symbols"] = list(panel.symbols)
        out["missing"] = missing
        dates = list(panel.dates)
        out["date_range"] = [
            dates[0].isoformat() if dates else None,
            dates[-1].isoformat() if dates else None,
        ]
        _logger.info(
            "截面研究完成: factor=%s symbols=%d dates=%d", req.factor, len(panel.symbols), len(dates)
        )
        return _sanitize(out)
