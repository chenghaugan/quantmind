"""PortfolioModel 实现：多标的信号聚合为组合目标权重（M4）。

  - ``IdentityPortfolio``：单标的透传（默认）；多标的下对选中标的逐项透传。
  - ``EqualWeightPortfolio``：把多标的带符号目标仓位按**相等权重**分配到组合预算，
    每个选中标的获得 ``budget / N`` 的仓位上限，适合演示「多标的组合构建」。

两者都通过 ``apply_all(signals, universe, context)`` 输出 ``{vt: 组合调整后权重}``。
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .base import AlphaSignal, PortfolioModel as PortfolioModelProtocol


class IdentityPortfolio(PortfolioModelProtocol):
    """透传组合：直接返回输入的 Alpha 信号，不做聚合调整。"""

    def apply(self, signal: Optional[AlphaSignal], context=None) -> Optional[AlphaSignal]:
        return signal

    def apply_all(
        self,
        signals: Dict[str, float],
        universe: List[str],
        context=None,
    ) -> Dict[str, float]:
        """多标的下对选中标的逐项透传（Universe 未选中的标的忽略）。"""
        if not universe:
            return dict(signals)
        return {vt: signals.get(vt, 0.0) for vt in universe}


class EqualWeightPortfolio(PortfolioModelProtocol):
    """等权组合：把组合预算按选中标的总数均分，作为每标的目标仓位上限。

    单标的（``N=1``）退化为 ``budget * signal``，与组合仓位语义一致。

    :param budget: 组合总仓位预算（0~1，表示组合最大净敞口/杠杆）。默认 1.0。
    """

    def __init__(self, budget: float = 1.0) -> None:
        self.budget = float(budget)

    def apply(self, signal: Optional[AlphaSignal], context=None) -> Optional[AlphaSignal]:
        # 单标的流程：组合预算即该标的上限
        if signal is None:
            return None
        signal.magnitude = min(abs(signal.magnitude), self.budget)
        return signal

    def apply_all(
        self,
        signals: Dict[str, float],
        universe: List[str],
        context=None,
    ) -> Dict[str, float]:
        if not universe:
            return {}
        n = len(universe)
        one = self.budget / n
        out: Dict[str, float] = {}
        for vt in universe:
            s = signals.get(vt, 0.0)
            out[vt] = s * one
        return out


__all__ = ["IdentityPortfolio", "EqualWeightPortfolio"]
