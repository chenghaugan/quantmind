"""OptimizeService: 策略参数寻优（网格搜索）。

把 backtest.optimizer.grid_search 暴露为 API 能力，供 Web「参数优化」页面调用。
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List

from ...core.constant import Exchange, Interval
from ...core.contracts import default_size
from ...data import DataManager
from ...data.feed.base import HistoryRequest
from ...backtest.optimizer import grid_search, optuna_optimize
from .backtest_service import _STRATEGY_MAP, _sanitize
from ...strategy.mined import MINED_STRATEGIES


_logger = logging.getLogger("quantmind.api")

# 各策略推荐的搜索空间（Web 端默认填充）
DEFAULT_PARAM_SPACE: Dict[str, Dict[str, List[Any]]] = {
    "dual_ma": {"fast": [3, 5, 10, 15], "slow": [20, 30, 40, 60]},
    # multifactor 仅 size/max_pos（及 specs）真正影响结果；lookback/top_n/threshold 均不生效。
    "multifactor": {"size": [1, 3, 5, 10], "max_pos": [0.5, 1.0, 2.0]},
    "vol_target": {"target_vol": [0.10, 0.15, 0.20], "lookback": [20, 40, 60]},
    "pair": {"window": [20, 40, 60], "entry_z": [1.5, 2.0, 2.5]},
}

# AI 挖掘策略的通用参数空间（按类 parameters 字段生成）
_MINE_SPACE_HINTS: Dict[str, List[Any]] = {
    "threshold": [0.2, 0.4, 0.6],
    "size": [1, 5, 10],
    "max_pos": [0.5, 1.0],
    "trend_window": [40, 60, 90],
    "break_window": [10, 20, 30],
}

METRICS = [
    ("sharpe", "夏普比率"),
    ("total_return", "总收益率"),
    ("max_drawdown", "最大回撤"),
    ("win_rate", "胜率"),
]


class OptimizeService:
    def __init__(self, dm: DataManager, resolver=None):
        self.dm = dm
        # 策略解析器：默认只覆盖内置 + 规范挖掘类；由 app 注入 BacktestService 的
        # _resolve_strategy_class 可一并解析 knowledge.db 已沉淀策略，保证
        # 「页11 下拉能选 = /optimize 一定能跑」的一致性契约。
        self._resolve = resolver or self._default_resolve

    @staticmethod
    def _default_resolve(name: str):
        return _STRATEGY_MAP.get(name) or MINED_STRATEGIES.get(name)

    def param_space_of(self, strategy: str) -> Dict[str, List[Any]]:
        if strategy in DEFAULT_PARAM_SPACE:
            return dict(DEFAULT_PARAM_SPACE[strategy])
        # AI 挖掘/注册策略：按其类 parameters 字段给一组通用搜索空间
        cls = self._resolve(strategy)
        if cls is not None:
            space: Dict[str, List[Any]] = {}
            for p in getattr(cls, "parameters", []):
                if p in _MINE_SPACE_HINTS:
                    space[p] = list(_MINE_SPACE_HINTS[p])
            return space
        return {}

    @staticmethod
    def metrics() -> List[dict]:
        return [{"key": k, "label": v} for k, v in METRICS]

    async def optimize(self, req) -> dict:
        strategy_class = self._resolve(req.strategy)
        if strategy_class is None:
            raise ValueError(
                f"未知策略: {req.strategy}（可选: {list(_STRATEGY_MAP) + list(MINED_STRATEGIES)}）"
            )

        method = getattr(req, "method", "grid") or "grid"
        if method == "optuna":
            # ---------- Optuna 贝叶斯：参数名 -> [low, high, step] ----------
            param_ranges = {
                k: tuple(list(v)[:3] if isinstance(v, (list, tuple)) else (v, v + 1, 1))
                for k, v in (getattr(req, "param_ranges", None) or {}).items()
            }
            param_ranges = {k: v for k, v in param_ranges.items() if v and len(v) >= 2}
            if not param_ranges:
                raise ValueError(
                    "Optuna 需要 param_ranges（参数名 -> [low, high, step]），如 {'fast': [3,15,2], 'slow': [20,60,5]}"
                )
            param_defs = {k: tuple(float(x) if i < 2 else int(x) for i, x in enumerate(v))
                          for k, v in param_ranges.items()}
            n_trials = int(getattr(req, "n_trials", 30) or 30)
            max_combos = getattr(req, "max_combos", 200) or 200
            n_trials = min(n_trials, max_combos)
        else:
            # ---------- 网格搜索 ----------
            param_space = req.param_space or DEFAULT_PARAM_SPACE.get(req.strategy) or {}
            param_space = {k: list(v) for k, v in param_space.items() if v}
            if not param_space:
                raise ValueError("参数空间为空，至少需要一个待搜索参数")

            combos = 1
            for v in param_space.values():
                combos *= len(v)
            max_combos = getattr(req, "max_combos", 200) or 200
            if combos > max_combos:
                raise ValueError(
                    f"参数组合数 {combos} 超过上限 {max_combos}，请缩小搜索空间或调高上限"
                )

        exch = Exchange(req.exchange.upper())
        hist = HistoryRequest(
            symbol=req.symbol,
            exchange=exch,
            interval=Interval(getattr(req, "interval", "1d") or "1d"),
            start=datetime.fromisoformat(req.start) if getattr(req, "start", None) else None,
            end=datetime.fromisoformat(req.end) if getattr(req, "end", None) else None,
        )
        bars = await self.dm.get_bar_data(hist)
        if not bars:
            raise ValueError(f"没有取到 {req.symbol}.{req.exchange} 的行情数据")

        vt_symbol = f"{req.symbol}.{exch.value}"
        data = {vt_symbol: bars}
        sizes = {vt_symbol: default_size(vt_symbol)}

        loop = asyncio.get_running_loop()
        if method == "optuna":
            res = await loop.run_in_executor(
                None,
                lambda: optuna_optimize(
                    strategy_class,
                    data,
                    vt_symbol,
                    param_defs,
                    n_trials=n_trials,
                    metric=req.metric,
                    sizes=sizes,
                ),
            )
            combos = n_trials
        else:
            res = await loop.run_in_executor(
                None,
                lambda: grid_search(
                    strategy_class,
                    data,
                    vt_symbol,
                    param_space,
                    metric=req.metric,
                    sizes=sizes,
                    capital=req.capital,
                ),
            )

        rows = sorted(
            res.results,
            key=lambda r: (r.get("metric") if r.get("metric") is not None else -1e18),
            reverse=True,
        )
        _logger.info(
            "参数寻优完成(%s): %s %s combos=%d best=%s",
            method, req.strategy, vt_symbol, combos, res.best_setting,
        )
        return _sanitize(
            {
                "strategy": req.strategy,
                "vt_symbol": vt_symbol,
                "metric": req.metric,
                "method": method,
                "combos": combos,
                "bars": len(bars),
                "best_setting": res.best_setting,
                "best_metric": res.best_metric,
                "results": rows,
                "param_names": list((param_defs if method == "optuna" else param_space).keys()),
            }
        )
