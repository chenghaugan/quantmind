"""参数优化：网格搜索（内置，无需额外依赖）+ Optuna（可选）。

用于对策略参数寻优，例如双均线的 (fast, slow) 组合。
"""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from ..core.object import BarData
from .engine import BacktestEngine

_logger = logging.getLogger("quantmind.backtest.optimizer")


@dataclass
class OptimizeResult:
    best_setting: Dict[str, Any]
    best_metric: float
    results: List[dict]


def _pick(report, metric: str) -> float:
    if metric == "sharpe":
        return report.sharpe
    if metric == "total_return":
        return report.total_return
    if metric == "calmar":
        return report.calmar
    if metric == "max_drawdown":  # 越小越好，取负
        return -report.max_drawdown
    return report.sharpe


def grid_search(
    strategy_class,
    data: Dict[str, List[BarData]],
    vt_symbol: str,
    param_space: Dict[str, List[Any]],
    metric: str = "sharpe",
    sizes: Optional[Dict[str, float]] = None,
    capital: float = 1_000_000.0,
) -> OptimizeResult:
    """网格搜索最优参数。"""
    keys = list(param_space.keys())
    best_setting: Dict[str, Any] = {}
    best_metric = float("-inf")
    results: List[dict] = []
    for combo in itertools.product(*(param_space[k] for k in keys)):
        setting = dict(zip(keys, combo))
        engine = BacktestEngine(data, capital=capital, sizes=sizes)
        engine.add_strategy(strategy_class, vt_symbol, setting)
        report = engine.run()
        m = _pick(report, metric)
        results.append({"setting": setting, metric: round(m, 4),
                        "sharpe": report.sharpe, "total_return": report.total_return})
        if m > best_metric:
            best_metric = m
            best_setting = setting
    return OptimizeResult(best_setting=best_setting, best_metric=round(best_metric, 4), results=results)


def optuna_optimize(
    strategy_class,
    data: Dict[str, List[BarData]],
    vt_symbol: str,
    param_defs: Dict[str, tuple],  # name -> (low, high, step)
    n_trials: int = 30,
    metric: str = "sharpe",
    sizes: Optional[Dict[str, float]] = None,
) -> OptimizeResult:
    """Optuna 贝叶斯优化（需安装 optuna）。"""
    try:
        import optuna  # type: ignore
    except ImportError:
        _logger.warning("未安装 optuna，回退网格搜索")
        grid = {k: list(range(low, high + 1, step)) for k, (low, high, step) in param_defs.items()}
        return grid_search(strategy_class, data, vt_symbol, grid, metric, sizes)

    def objective(trial):
        setting = {k: trial.suggest_int(k, low, high, step)
                   for k, (low, high, step) in param_defs.items()}
        engine = BacktestEngine(data, sizes=sizes)
        engine.add_strategy(strategy_class, vt_symbol, setting)
        report = engine.run()
        return _pick(report, metric)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return OptimizeResult(
        best_setting=study.best_params,
        best_metric=round(study.best_value, 4),
        results=[{"setting": dict(t), "value": round(t.value, 4)} for t in study.trials],
    )
