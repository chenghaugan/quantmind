"""全天候(All-Weather)风格策略：波动率目标 + 风险平价精神，单标的实现。

桥水全天候的核心思想：用**波动率目标(vol-targeting)**把不同资产的风险归一，再
按风险贡献配置——使组合对单一宏观 régime 不过度暴露。在 QuantMind 单标的 CTA
框架下，这里实现其最关键的「波动率目标 + 趋势过滤」：

  target_t = clip( target_vol / realized_vol_t * sign(momentum_t), -max_pos, +max_pos )

即：用年化波动率倒数缩放仓位（波动越高仓位越低），方向与中期动量一致。
这是 All-Weather「风险平价」思想的单资产体现，可作为多资产风险平价配置的基石组件。
"""
from __future__ import annotations

import math
from typing import List

import numpy as np
import pandas as pd

from ..core.object import BarData
from ..research.factors.base import bars_to_df
from .base import CtaTemplate


class VolTargetStrategy(CtaTemplate):
    """波动率目标 + 动量过滤策略（All-Weather 精神，单标的）。"""

    author = "QuantMind"
    parameters = ["lookback", "target_vol", "momentum_win", "size", "max_pos"]

    def __init__(self, context, setting=None) -> None:
        self.lookback = 20          # 波动率估计窗口（交易日）
        self.target_vol = 0.20      # 目标年化波动率
        self.momentum_win = 60      # 中期动量窗口
        self.size = 1
        self.max_pos = 1.0
        self._target: pd.Series = None
        self._idx = 0
        super().__init__(context, setting)

    def on_init(self) -> None:
        vt = self.vt_symbols[0]
        bars = self.context.get_history(vt, 9999)
        df = bars_to_df(bars)
        if df.empty:
            self._target = pd.Series(dtype=float)
            return
        ret = df["close"].pct_change()
        # 年化已实现波动率（252 交易日）
        vol = ret.rolling(self.lookback, min_periods=5).std() * math.sqrt(252)
        # 中期动量（方向）
        mom = df["close"] / df["close"].shift(self.momentum_win) - 1.0
        # 波动率目标仓位：目标波动 / 实际波动，方向与动量一致
        target = self.target_vol / vol * np.sign(mom)
        target = target.clip(-self.max_pos, self.max_pos).fillna(0.0)
        self._target = target
        self.context.write_log(
            f"波动率目标模型就绪：目标波动 {self.target_vol:.0%}，信号长度 {len(target)}"
        )

    def on_bar(self, bar: BarData) -> None:
        if self._target is None or self._idx >= len(self._target):
            return
        t = float(self._target.iloc[self._idx])
        self._idx += 1
        oid = self.set_target(bar.vt_symbol, t * self.size)
        if oid == "":
            # 风控拒单：不更新 pos，下一根 bar 重试
            return
        self.pos = t * self.size
