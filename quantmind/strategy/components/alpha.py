"""AlphaModel 实现：信号/目标仓位生成。

  - ``MultiFactorAlpha``：由多因子规格（FactorSpec）合成复合信号 → 目标仓位。
    把现有 ``MultiFactorStrategy`` 的信号生成部分重构为独立组件。
  - ``MomentumAlpha``：简单双均线动量示例（快速上穿慢速 → 多头）。

两者产出的都是**带符号目标仓位**（正多 / 负空），供 Portfolio/Risk/Execution 使用。
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from ...core.object import BarData
from ...research.target import FactorSpec, MultiFactorModel, build_model_from_specs
from .base import AlphaModel as AlphaModelProtocol


class MultiFactorAlpha(AlphaModelProtocol):
    """多因子合成 → 目标仓位（Alpha 组件）。"""

    parameters = ["specs", "threshold", "size", "max_pos"]

    def __init__(self, specs: Optional[List[FactorSpec]] = None,
                 threshold: float = 0.3, size: int = 1, max_pos: float = 1.0) -> None:
        self.specs = specs or [
            FactorSpec(name="momentum_20", kind="momentum", window=20, weight=1.0),
            FactorSpec(name="reversion_60", kind="mean_reversion", window=60, weight=-0.5),
            FactorSpec(name="vol_20", kind="volatility", window=20, weight=-0.3),
        ]
        self.threshold = threshold
        self.size = size
        self.max_pos = max_pos
        self.model: Optional[MultiFactorModel] = None
        self._target_series: Optional[pd.Series] = None
        self._idx = 0
        self.vt_symbol: str = ""

    def on_init(self, context) -> None:
        """:class:`MultiFactorModel` 预计算复合信号 → 目标序列。"""
        self.vt_symbol = context.vt_symbols[0]
        bars = context.get_history(self.vt_symbol, 9999)
        self.model = build_model_from_specs(self.specs, bars)
        self._target_series = self.model.target_position(
            bars, size=self.size, max_pos=self.max_pos)
        self._idx = 0

    def on_bar(self, bar: BarData) -> Optional[float]:
        """返回当前根的目标仓位；序列耗尽时返回 None。"""
        if self._target_series is None or self._idx >= len(self._target_series):
            return None
        target = float(self._target_series.iloc[self._idx])
        self._idx += 1
        return target


class MomentumAlpha(AlphaModelProtocol):
    """双均线动量 Alpha：快线上穿慢线 → 满仓多头，否则满仓空头。"""

    parameters = ["fast", "slow", "size", "max_pos"]

    def __init__(self, fast: int = 5, slow: int = 20,
                 size: int = 1, max_pos: float = 1.0) -> None:
        self.fast = fast
        self.slow = slow
        self.size = size
        self.max_pos = max_pos
        self._closes: List[float] = []
        self.vt_symbol: str = ""

    def on_init(self, context) -> None:
        self.vt_symbol = context.vt_symbols[0]

    def on_bar(self, bar: BarData) -> Optional[float]:
        self._closes.append(bar.close_price)
        if len(self._closes) < self.slow:
            return None
        last = self._closes[-1]
        fast_ma = sum(self._closes[-self.fast:]) / self.fast
        slow_ma = sum(self._closes[-self.slow:]) / self.slow
        direction = 1 if fast_ma > slow_ma else -1
        return direction * self.max_pos * self.size


class MultiFactorMultiSymbolAlpha(AlphaModelProtocol):
    """多因子 Alpha（多标的）：为每个标的独立预计算目标序列，按 ``bar.vt_symbol`` 查表。

    与 ``MultiFactorAlpha`` 的区别：后者仅服务单一主标的（``context.vt_symbols[0]``），
    本组件对标的池中每个标的分别 ``get_history`` 建模型与目标序列，on_bar 时按当前
    标的返回其目标仓位，供 M4 组合聚合使用。
    """

    parameters = ["specs", "threshold", "size", "max_pos"]

    def __init__(self, specs: Optional[List["FactorSpec"]] = None,
                 threshold: float = 0.3, size: int = 1, max_pos: float = 1.0) -> None:
        self.specs = specs or [
            FactorSpec(name="momentum_20", kind="momentum", window=20, weight=1.0),
            FactorSpec(name="reversion_60", kind="mean_reversion", window=60, weight=-0.5),
            FactorSpec(name="vol_20", kind="volatility", window=20, weight=-0.3),
        ]
        self.threshold = threshold
        self.size = size
        self.max_pos = max_pos
        self._models: Dict[str, "MultiFactorModel"] = {}
        self._series: Dict[str, pd.Series] = {}
        self._idx: Dict[str, int] = {}

    def on_init(self, context) -> None:
        """对标的池中每个标的独立构建多因子模型与目标序列。"""
        for vt in context.vt_symbols:
            bars = context.get_history(vt, 9999)
            self._models[vt] = build_model_from_specs(self.specs, bars)
            self._series[vt] = self._models[vt].target_position(
                bars, size=self.size, max_pos=self.max_pos)
            self._idx[vt] = 0

    def on_bar(self, bar: BarData) -> Optional[float]:
        """返回当前标的的目标仓位；该标的序列耗尽返回 None。"""
        vt = bar.vt_symbol or f"{bar.symbol}.{bar.exchange.value}"
        series = self._series.get(vt)
        if series is None:
            return None
        idx = self._idx.get(vt, 0)
        if idx >= len(series):
            return None
        self._idx[vt] = idx + 1
        return float(series.iloc[idx])


__all__ = ["MultiFactorAlpha", "MomentumAlpha", "MultiFactorMultiSymbolAlpha"]
