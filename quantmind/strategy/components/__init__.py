"""5 组件模块化算法框架（借鉴 Lean）——组合策略组件。

提供 5 个可组合组件（Universe/Alpha/Portfolio/Risk/Execution）+ 装配用的
``ComposableStrategy``。与现有单体 ``CtaTemplate`` 模板共存。

用法示例：
    from quantmind.strategy.components import (
        ComposableStrategy, MultiFactorAlpha, IdentityPortfolio,
        ExecutionModel, NullRisk,
    )

    # 在 backtest/CLI 中用 setting 传入组件即可装配
"""
from __future__ import annotations

from .base import (
    AlphaSignal,
    UniverseModel,
    AlphaModel,
    PortfolioModel,
    RiskModel,
    ExecutionModel,
)
from .execution import ExecutionModel as _ExecutionModelImpl, TargetExecution
from .risk import NullRisk, RiskGateModel
from .alpha import MultiFactorAlpha, MomentumAlpha, MultiFactorMultiSymbolAlpha
from .universe import AllUniverse, RuleUniverse
from .portfolio import IdentityPortfolio, EqualWeightPortfolio
from .composable import ComposableStrategy

__all__ = [
    "AlphaSignal",
    "UniverseModel",
    "AlphaModel",
    "PortfolioModel",
    "RiskModel",
    "ExecutionModel",
    "TargetExecution",
    "NullRisk",
    "RiskGateModel",
    "MultiFactorAlpha",
    "MomentumAlpha",
    "MultiFactorMultiSymbolAlpha",
    "AllUniverse",
    "RuleUniverse",
    "IdentityPortfolio",
    "EqualWeightPortfolio",
    "ComposableStrategy",
]
