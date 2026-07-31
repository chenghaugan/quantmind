"""回测模块。"""
from .engine import BacktestEngine
from .analyzer import PerformanceAnalyzer, PerformanceReport
from .broker import fill_price, commission
from .cost import (
    CostModel, CONTRACT_COST_TABLE, default_cost_table, lookup_cost,
    compute_commission, apply_slippage, compute_margin,
)
from .optimizer import grid_search, optuna_optimize, OptimizeResult
from .diagnostics import (
    limit_day_mask, detect_lookahead, diagnose_overfitting, health_checks, HealthReport,
)
from .walkforward import walk_forward, WalkForwardResult, WalkForwardFold

__all__ = [
    "BacktestEngine",
    "PerformanceAnalyzer",
    "PerformanceReport",
    "fill_price",
    "commission",
    "CostModel",
    "CONTRACT_COST_TABLE",
    "default_cost_table",
    "lookup_cost",
    "compute_commission",
    "apply_slippage",
    "compute_margin",
    "grid_search",
    "optuna_optimize",
    "OptimizeResult",
    "limit_day_mask",
    "detect_lookahead",
    "diagnose_overfitting",
    "health_checks",
    "HealthReport",
    "walk_forward",
    "WalkForwardResult",
    "WalkForwardFold",
]
