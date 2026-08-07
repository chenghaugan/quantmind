"""因子子包：技术因子 + Alpha101/191 + 期货席位因子 + 表达式 DSL。"""
from .base import Factor, FactorMeta, bars_to_df, expanding_zscore, rolling_zscore
from .technical import (
    MomentumFactor, MeanReversionFactor, VolatilityFactor, VolumeChangeFactor,
    OpenInterestChangeFactor, TermStructureFactor, build_factor,
)
from .expression import eval_factor_expression, ExpressionError
from .registry import FactorRegistry, build_factor_registry
from .alpha101 import AlphaFactor, build_alpha_factor, list_alpha101
from .alpha191 import Alpha191Factor, build_alpha191_factor, list_alpha191
from .alpha_cs import (
    Panel, compute_alpha_cross_sectional, list_alpha_cs,
)
from .gtja191 import (
    Gtja191Factor, build_gtja191_factor, list_gtja191,
)
from .qlib158 import (
    Qlib158Factor, build_qlib158_factor, list_qlib158,
)
from .academic import (
    AcademicFactor, build_academic_factor, list_academic,
)
from .seat_futures import (
    SeatFactor, compute_seat_factors, make_synthetic_seat_df,
)

__all__ = [
    "Factor", "FactorMeta", "bars_to_df", "expanding_zscore", "rolling_zscore",
    "MomentumFactor", "MeanReversionFactor", "VolatilityFactor", "VolumeChangeFactor",
    "OpenInterestChangeFactor", "TermStructureFactor", "build_factor",
    "eval_factor_expression", "ExpressionError",
    "FactorRegistry", "build_factor_registry",
    "AlphaFactor", "build_alpha_factor", "list_alpha101",
    "Alpha191Factor", "build_alpha191_factor", "list_alpha191",
    "Panel", "compute_alpha_cross_sectional", "list_alpha_cs",
    "Gtja191Factor", "build_gtja191_factor", "list_gtja191",
    "Qlib158Factor", "build_qlib158_factor", "list_qlib158",
    "AcademicFactor", "build_academic_factor", "list_academic",
    "SeatFactor", "compute_seat_factors", "make_synthetic_seat_df",
]
