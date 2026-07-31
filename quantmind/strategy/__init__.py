"""策略模块。"""
from .base import CtaTemplate
from .context import StrategyContext, parse_vt_symbol
from .dual_ma import DualMaStrategy
from .multifactor import MultiFactorStrategy
from .allweather import VolTargetStrategy
from .pair import PairTradingStrategy, build_spread_bars
from .runners import run_strategy

__all__ = [
    "CtaTemplate",
    "StrategyContext",
    "parse_vt_symbol",
    "DualMaStrategy",
    "MultiFactorStrategy",
    "VolTargetStrategy",
    "PairTradingStrategy",
    "build_spread_bars",
    "run_strategy",
]
