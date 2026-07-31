"""回测模块。"""
from .engine import BacktestEngine
from .analyzer import PerformanceAnalyzer, PerformanceReport
from .broker import fill_price, commission
from .optimizer import grid_search, optuna_optimize, OptimizeResult
from .diagnostics import (
    limit_day_mask, detect_lookahead, diagnose_overfitting, health_checks, HealthReport,
)

__all__ = [
    "BacktestEngine",
    "PerformanceAnalyzer",
    "PerformanceReport",
    "fill_price",
    "commission",
    "grid_search",
    "optuna_optimize",
    "OptimizeResult",
    "limit_day_mask",
    "detect_lookahead",
    "diagnose_overfitting",
    "health_checks",
    "HealthReport",
]
