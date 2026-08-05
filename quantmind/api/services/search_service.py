"""SearchService: 表达式截面评估 + 因子迭代搜索（CoT）的 API 服务层。

复用 DataManager 构造多标的面板（index=日期, columns=标的），把 P0 的
``evaluate_expression`` 与 P1 的 ``FactorSearcher.cot_search`` 暴露为
可供 FastAPI / CLI 调用的方法。

与现状一致：无 LLM key 或离线时，CoT 回落到确定性变异器，保证流程可跑通。
"""
from __future__ import annotations

import asyncio
import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional

from ...core.constant import Exchange, Interval
from ...data import DataManager
from ...data.feed.base import HistoryRequest
from ...research import (
    evaluate_expression as _eval_expr,
    batch_evaluate_expressions,
    FactorSearcher,
    SearchResult,
    create_algo,
    list_algos,
    dedup_expressions as _dedup_exprs,
    factor_expression_backtest as _expr_backtest,
)
from ...research.factors.alpha_cs import Panel

_logger = logging.getLogger("quantmind.api")


def _sanitize(o: Any) -> Any:
    """把 numpy float/NaN/datetime 规整为 JSON 可序列化值。"""
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


def _flt(x) -> Optional[float]:
    """把可能是 NaN 的 float 规整为 float 或 None（用于 metric 展示）。"""
    try:
        f = float(x)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


class SearchService:
    """因子表达式评估与迭代搜索服务。"""

    def __init__(self, dm: DataManager, provider=None) -> None:
        self.dm = dm
        self.provider = provider  # 可选 LLMProvider；None → CoT 回落 mock 变异器

    # -- 面板构造（复用 CrossSectionService 逻辑） ---------------------------
    async def _build_panel(
        self,
        symbols: List[str],
        exchange: str = "SHFE",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Panel:
        symbols = [s for s in (symbols or []) if s and s.strip()]
        if len(symbols) < 2:
            raise ValueError("表达式截面研究至少需要 2 个标的")
        exch = Exchange(exchange.upper())
        interv = Interval(interval or "1d")
        tasks = [
            self.dm.get_bar_data(
                HistoryRequest(
                    symbol=s,
                    exchange=exch,
                    interval=interv,
                    start=datetime.fromisoformat(start) if start else None,
                    end=datetime.fromisoformat(end) if end else None,
                )
            )
            for s in symbols
        ]
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
                f"可用标的不足 2 个（缺失: {missing or '无'}）："
                "请确认行情数据可用"
            )
        return Panel.from_bars(bars_by_symbol)

    # -- 表达式评估 ----------------------------------------------------------
    async def evaluate_expression(
        self,
        expression: str,
        symbols: List[str],
        exchange: str = "SHFE",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
        forward_periods: int = 1,
        market: str = "",
    ) -> dict:
        """对单个表达式做「面板求值 → 截面 IC 评估」，返回报告 dict。"""
        panel = await self._build_panel(symbols, exchange, interval, start, end)
        loop = asyncio.get_running_loop()
        rep = await loop.run_in_executor(
            None,
            lambda: _eval_expr(expression, panel, forward_periods=forward_periods,
                               market=market, use_cache=False),
        )
        out = rep.to_dict()
        out["n_symbols"] = len(panel.symbols)
        out["n_dates"] = len(panel.dates)
        out["symbols"] = list(panel.symbols)
        dates = list(panel.dates)
        out["date_range"] = [dates[0].isoformat() if dates else None,
                             dates[-1].isoformat() if dates else None]
        return _sanitize(out)

    async def evaluate_expressions_batch(
        self,
        expressions: List[str],
        symbols: List[str],
        exchange: str = "SHFE",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
        forward_periods: int = 1,
        market: str = "",
    ) -> List[dict]:
        """批量评估多个表达式。"""
        panel = await self._build_panel(symbols, exchange, interval, start, end)
        loop = asyncio.get_running_loop()
        reports = await loop.run_in_executor(
            None,
            lambda: batch_evaluate_expressions(
                expressions, panel, forward_periods=forward_periods,
                market=market, use_cache=False,
            ),
        )
        return [_sanitize(r.to_dict()) for r in reports]

    # -- 迭代搜索（co / ea / tot） ---------------------------------------------
    async def search(
        self,
        seed: str,
        symbols: List[str],
        exchange: str = "SHFE",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
        algo: str = "co",
        rounds: int = 6,
        forward_periods: int = 1,
        market: str = "",
        val_symbols: Optional[List[str]] = None,
        val_start: Optional[str] = None,
        val_end: Optional[str] = None,
    ) -> dict:
        """对 seed 表达式做指定算法（co/ea/tot）的迭代搜索，返回 ``SearchResult`` dict。

        可选 ``val_symbols/val_start/val_end`` 提供独立验证期面板做防泄漏评估。
        ``algo`` 未知时按默认回落 ``co``（链式精炼）。
        """
        panel = await self._build_panel(symbols, exchange, interval, start, end)

        val_panel: Optional[Panel] = None
        if val_symbols:
            val_panel = await self._build_panel(
                val_symbols, exchange, interval, val_start, val_end)

        algo_name = algo if algo in list_algos() else "co"
        # 每类算法的迭代参数：co=rounds, ea=generations, tot=depth
        algo_kwargs = {
            "co": {"rounds": rounds},
            "ea": {"generations": rounds},
            "tot": {"depth": rounds},
        }.get(algo_name, {"rounds": rounds})

        searcher = create_algo(algo_name, provider=self.provider, **algo_kwargs)

        async def _run() -> SearchResult:
            return await searcher.run(
                seed, panel, val_panel=val_panel,
                forward_periods=forward_periods, market=market,
            )

        # search 内部是 async（LLM/评估），直接在事件循环中跑（评估经 executor）
        result = await _run()
        out = result.to_dict()
        out["algo"] = algo_name
        out["n_symbols"] = len(panel.symbols)
        out["date_range"] = [panel.dates[0].isoformat() if len(panel.dates) else None,
                             panel.dates[-1].isoformat() if len(panel.dates) else None]
        return _sanitize(out)

    async def cot_search(
        self,
        seed: str,
        symbols: List[str],
        exchange: str = "SHFE",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
        rounds: int = 6,
        forward_periods: int = 1,
        market: str = "",
        val_symbols: Optional[List[str]] = None,
        val_start: Optional[str] = None,
        val_end: Optional[str] = None,
        algo: str = "co",
    ) -> dict:
        """``cot_search`` 为 :meth:`search` 的向后兼容别名（默认 algo=co）。"""
        return await self.search(
            seed, symbols, exchange=exchange, interval=interval, start=start,
            end=end, algo=algo, rounds=rounds, forward_periods=forward_periods,
            market=market, val_symbols=val_symbols, val_start=val_start, val_end=val_end,
        )

    # -- 因子去冗余（相关性聚类） --------------------------------------------
    async def dedup(
        self,
        expressions: List[str],
        symbols: List[str],
        exchange: str = "SHFE",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
        correlation_threshold: float = 0.7,
        min_abs_metric: float = 0.0,
        forward_periods: int = 1,
        market: str = "",
        compute_ic: bool = True,
    ) -> dict:
        """对一批表达式做相关性聚类去冗余，返回每簇代表性因子。"""
        panel = await self._build_panel(symbols, exchange, interval, start, end)
        loop = asyncio.get_running_loop()
        kept = await loop.run_in_executor(
            None,
            lambda: _dedup_exprs(
                expressions, panel, correlation_threshold=correlation_threshold,
                min_abs_metric=min_abs_metric, forward_periods=forward_periods,
                market=market, compute_ic=compute_ic,
            ),
        )
        return {
            "n_input": len([e for e in expressions if e and e.strip()]),
            "n_input_unique": len(expressions),
            "n_kept": len(kept),
            "representatives": [
                {"expression": c["name"], "metric": _flt(c["metric"]),
                 "n_removed": len(c["cluster"]) - 1,
                 "cluster": list(c["cluster"])}
                for c in kept
            ],
            "correlation_threshold": correlation_threshold,
        }

    # -- 表达式 → 截面多空组合回测（研究闭环） ------------------------------
    async def backtest_expression(
        self,
        expression: str,
        symbols: List[str],
        exchange: str = "SHFE",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
        forward_periods: int = 1,
        n_groups: int = 5,
        long_short: bool = True,
        cost_rate: float = 0.0,
    ) -> dict:
        """对挖掘出的 DSL 因子表达式直接做截面多空组合回测。"""
        panel = await self._build_panel(symbols, exchange, interval, start, end)
        loop = asyncio.get_running_loop()
        out = await loop.run_in_executor(
            None,
            lambda: _expr_backtest(
                expression, panel, forward_periods=forward_periods,
                n_groups=n_groups, long_short=long_short, cost_rate=cost_rate,
            ),
        )
        return _sanitize(out)
