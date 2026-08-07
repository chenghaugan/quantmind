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
from .factors.gtja191 import Gtja191Factor, build_gtja191_factor, list_gtja191
from .factors.qlib158 import Qlib158Factor, build_qlib158_factor, list_qlib158
from .factors.academic import AcademicFactor, build_academic_factor, list_academic
from .factors.seat_futures import SeatFactor, compute_seat_factors, make_synthetic_seat_df
from .evaluator import FactorEvaluator, FactorReport
from .eval import (
    evaluate_expression,
    batch_evaluate_expressions,
    FactorEvalCache,
)
from .factors.panel_expr import panel_eval_expression, list_panel_operators
from .search import (
    FactorSearcher,
    EASearcher,
    ToTSearcher,
    SearchResult,
    SearchStep,
    mutate_expressions,
    BaseAlgo,
    create_algo,
    list_algos,
)
from .judge import (
    judge_signal,
    judge_pairwise,
    judge_ranking,
    judge_scoring,
    score_signal_accuracy,
    score_pairwise_accuracy,
    score_ranking,
    score_scoring,
)
from .split import time_split, regime_labels, PanelSplitter, SplitResult
from .factors.seed_pool import list_seed_pool, DEFAULT_SEED_POOL, FactorPairStore
from .dedup import (
    dedup_expressions,
    dedup_factor_panels,
    factor_correlation_matrix,
    greedy_cluster_dedup,
)
from .cross_sectional_backtest import (
    cross_sectional_backtest,
    factor_expression_backtest,
)
from .pipeline import PipelineConfig, StepReport, run_pipeline
# NOTE: E2EConfig / run_e2e 用惰性导入（PEP 562 __getattr__），
# 避免 ai.factor_gen ↔ research ↔ ai.agent 的加载期循环导入。
from .combine import (
    cs_rank_panel,
    cs_zscore_panel,
    standardize_panel,
    equal_weights,
    icir_weights as combine_icir_weights,
    inverse_variance_weights,
    min_variance_weights,
    combine_factor_panels,
    optimize_weights,
    composite_backtest,
)
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
    "Gtja191Factor",
    "build_gtja191_factor",
    "list_gtja191",
    "Qlib158Factor",
    "build_qlib158_factor",
    "list_qlib158",
    "AcademicFactor",
    "build_academic_factor",
    "list_academic",
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
    "EASearcher",
    "ToTSearcher",
    "SearchResult",
    "SearchStep",
    "mutate_expressions",
    "BaseAlgo",
    "create_algo",
    "list_algos",
    "judge_signal",
    "judge_pairwise",
    "judge_ranking",
    "judge_scoring",
    "score_signal_accuracy",
    "score_pairwise_accuracy",
    "score_ranking",
    "score_scoring",
    "time_split",
    "regime_labels",
    "PanelSplitter",
    "SplitResult",
    "list_seed_pool",
    "DEFAULT_SEED_POOL",
    "FactorPairStore",
    "dedup_expressions",
    "dedup_factor_panels",
    "factor_correlation_matrix",
    "greedy_cluster_dedup",
    "cross_sectional_backtest",
    "factor_expression_backtest",
    "PipelineConfig",
    "StepReport",
    "run_pipeline",
    "E2EConfig",
    "run_e2e",
    "cs_rank_panel",
    "cs_zscore_panel",
    "standardize_panel",
    "equal_weights",
    "combine_icir_weights",
    "inverse_variance_weights",
    "min_variance_weights",
    "combine_factor_panels",
    "optimize_weights",
    "composite_backtest",
    "FactorSpec",
    "MultiFactorModel",
    "build_model_from_specs",
    "icir_weights",
    "winsorize",
    "cross_sectional_neutralize",
    "orthogonalize_factors",
]


def __getattr__(name: str):
    """惰性导出 orchestrator 的 E2EConfig / run_e2e（打破加载期循环导入）。

    保持 ``from quantmind.research import E2EConfig, run_e2e`` 兼容，
    仅在首次访问时才真正导入 orchestrator（及其 ai.agent 依赖）。
    """
    if name in ("E2EConfig", "run_e2e"):
        from .orchestrator import E2EConfig as _E2EConfig
        from .orchestrator import run_e2e as _run_e2e

        return {"E2EConfig": _E2EConfig, "run_e2e": _run_e2e}[name]
    raise AttributeError(f"module 'quantmind.research' has no attribute {name!r}")
