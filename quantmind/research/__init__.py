"""研究模块：因子、评估、多因子组合。"""
from .factors.base import Factor, FactorMeta, bars_to_df, expanding_zscore, rolling_zscore
from .factors.technical import (
    MomentumFactor,
    MeanReversionFactor,
    VolatilityFactor,
    VolumeChangeFactor,
    OpenInterestChangeFactor,
    TermStructureFactor,
    build_factor,
)
from .factors.registry import FactorRegistry, build_factor_registry
from .factors.expression import eval_factor_expression, ExpressionError
from .factors.alpha101 import AlphaFactor, build_alpha_factor, list_alpha101
from .factors.alpha191 import Alpha191Factor, build_alpha191_factor, list_alpha191
from .factors.seat_futures import SeatFactor, compute_seat_factors, make_synthetic_seat_df
from .evaluator import FactorEvaluator, FactorReport
from .eval import (
    evaluate_expression,
    batch_evaluate_expressions,
    FactorEvalCache,
)
from .factors.panel_expr import panel_eval_expression, list_panel_operators
from .search import FactorSearcher, SearchResult, SearchStep, mutate_expressions
from .target import FactorSpec, MultiFactorModel, build_model_from_specs, icir_weights
from .neutralize import winsorize, cross_sectional_neutralize, orthogonalize_factors

__all__ = [
    "Factor",
    "FactorMeta",
    "bars_to_df",
    "expanding_zscore",
    "rolling_zscore",
    "MomentumFactor",
    "MeanReversionFactor",
    "VolatilityFactor",
    "VolumeChangeFactor",
    "OpenInterestChangeFactor",
    "TermStructureFactor",
    "build_factor",
    "FactorRegistry",
    "build_factor_registry",
    "eval_factor_expression",
    "ExpressionError",
    "AlphaFactor",
    "build_alpha_factor",
    "list_alpha101",
    "Alpha191Factor",
    "build_alpha191_factor",
    "list_alpha191",
    "SeatFactor",
    "compute_seat_factors",
    "make_synthetic_seat_df",
    "FactorEvaluator",
    "FactorReport",
    "evaluate_expression",
    "batch_evaluate_expressions",
    "FactorEvalCache",
    "panel_eval_expression",
    "list_panel_operators",
    "FactorSearcher",
    "SearchResult",
    "SearchStep",
    "mutate_expressions",
    "FactorSpec",
    "MultiFactorModel",
    "build_model_from_specs",
    "icir_weights",
    "winsorize",
    "cross_sectional_neutralize",
    "orthogonalize_factors",
]
