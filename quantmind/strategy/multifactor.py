"""多因子组合策略：把多个因子标准化加权合成复合信号，映射为目标仓位。

对应 5 组件框架 Alpha -> Portfolio。演示「多因子组合形成策略」。
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

from ..core.object import BarData
from .base import CtaTemplate
from ..research.target import FactorSpec, MultiFactorModel, build_model_from_specs


class MultiFactorStrategy(CtaTemplate):
    """多因子组合策略。"""

    author = "QuantMind"
    parameters = ["specs", "threshold", "size", "max_pos"]

    def __init__(self, context, setting=None) -> None:
        # 默认：动量(20) + 均值回复(60) + 波动率(20)，权重 1/-0.5/-0.3
        self.specs: List[FactorSpec] = [
            FactorSpec(name="momentum_20", kind="momentum", window=20, weight=1.0),
            FactorSpec(name="reversion_60", kind="mean_reversion", window=60, weight=-0.5),
            FactorSpec(name="vol_20", kind="volatility", window=20, weight=-0.3),
        ]
        self.threshold = 0.3
        self.size = 1
        self.max_pos = 1.0
        self.model: MultiFactorModel = None
        self.target_series: pd.Series = None
        self._idx = 0
        super().__init__(context, setting)

    def on_init(self) -> None:
        # 用历史预计算复合信号（combine 内部为扩张窗口 z-score，无前视）
        vt = self.vt_symbols[0]
        bars = self.context.get_history(vt, 9999)
        self.model = build_model_from_specs(self.specs, bars)
        self.target_series = self.model.target_position(bars, size=self.size, max_pos=self.max_pos)
        self.context.write_log(
            f"多因子模型就绪：{len(self.specs)} 个因子，目标序列长度 {len(self.target_series)}"
        )

    def on_bar(self, bar: BarData) -> None:
        if self.target_series is None or self._idx >= len(self.target_series):
            return
        target = float(self.target_series.iloc[self._idx])
        self._idx += 1
        self.set_target(bar.vt_symbol, target)
        self.pos = target
