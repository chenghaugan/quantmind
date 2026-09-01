"""5 组件模块化算法框架（借鉴 Lean/QuantConnect）——组件接口定义。

把策略拆解为 5 个可独立替换/复用的组件：

  - ``UniverseModel``：选择可交易标的（Universe Selection）
  - ``AlphaModel``：生成交易信号/目标仓位（Alpha / Signal）
  - ``PortfolioModel``：把多标的信号聚合为组合目标权重（Portfolio Construction）
  - ``RiskModel``：按风控约束调整目标仓位（Risk Management）
  - ``ExecutionModel``：把目标权重转化为实际下单（Execution）

与现有 ``CtaTemplate`` 的关系：``ComposableStrategy``（见 composable.py）是一个
``CtaTemplate`` 子类，用上述 5 个组件装配而成。**现有单体模板策略保持不变**，
新组件模型与旧模板可共存、可混用。

本文件只定义 Protocol 接口与信号数据结构，不含实现。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, runtime_checkable

from ...core.object import BarData


@dataclass
class AlphaSignal:
    """Alpha 组件产出的信号：某标的希望在 ``bar`` 时刻调整到的目标仓位。

    :param vt_symbol: 标的（如 ``rb0.SHFE``）。
    :param direction: 方向：+1 多头 / -1 空头 / 0 平仓。
    :param magnitude: 风险调整后的目标仓位权重（0~1）。
    """
    vt_symbol: str
    direction: int = 0
    magnitude: float = 0.0
    source: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def target(self) -> float:
        """带符号的目标仓位权重（正多 / 负空）。"""
        return self.direction * abs(self.magnitude)


@runtime_checkable
class UniverseModel(Protocol):
    """标的池选择。"""

    def select(self, candidates: List[str], context=None) -> List[str]:
        """从候选标的中选出可交易子集。"""
        ...


@runtime_checkable
class AlphaModel(Protocol):
    """信号生成。"""

    def on_init(self, context) -> None: ...
    def on_bar(self, bar: BarData) -> Optional[float]:
        """对当前 K 线产出目标仓位（带符号）。返回 None 表示本次不变。"""
        ...


@runtime_checkable
class PortfolioModel(Protocol):
    """组合聚合。

    支持两种调用形态（实现可只实现其一，另一个基于它派生）：
      - ``apply(signal, context)``：单标的目标仓位调整（向后兼容单标的流程）。
      - ``apply_all(signals, universe, context)``：把多标的带符号目标仓位聚合为组合目标权重。
    """

    def apply(self, signal: Optional[AlphaSignal], context=None) -> Optional[AlphaSignal]:
        """对单标的目标仓位做组合级调整。默认透传。"""
        ...

    def apply_all(
        self,
        signals: Dict[str, float],
        universe: List[str],
        context=None,
    ) -> Dict[str, float]:
        """把多标的带符号目标仓位聚合为组合目标权重。

        :param signals: ``vt_symbol -> 带符号目标仓位``（Alpha 最新信号）。
        :param universe: 当前选中的标的集合（M5）。
        :returns: ``vt_symbol -> 组合调整后的目标权重``，可在重平衡日逐标的执行。
        """
        ...


@runtime_checkable
class RiskModel(Protocol):
    """风控过滤。"""

    def apply(self, target: Optional[float], bar: BarData, context=None,
              vt_symbol: Optional[str] = None) -> Optional[float]:
        """按风控约束调整目标仓位；返回 None 表示拒绝本次调仓。

        ``vt_symbol``：本次调仓的目标标的（多标的组合下可能与 bar 的标的不同）。
        """
        ...


@runtime_checkable
class ExecutionModel(Protocol):
    """执行。"""

    def set_target(self, vt_symbol: str, target: float) -> Optional[str]:
        """把目标仓位交给上下文执行。返回 order_id。"""
        ...


__all__ = [
    "AlphaSignal",
    "UniverseModel",
    "AlphaModel",
    "PortfolioModel",
    "RiskModel",
    "ExecutionModel",
]
