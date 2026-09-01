"""双均线策略（最经典的趋势/动量策略，用于验证回测与多模式切换）。

快线上穿慢线 -> 看多（期货多/空，股票仅多）；下穿 -> 看空/空仓。
"""
from __future__ import annotations

from typing import List

from ..core.constant import Direction
from ..core.object import BarData
from ..core.utility import ArrayManager
from .base import CtaTemplate


class DualMaStrategy(CtaTemplate):
    """双均线 CTA 策略。"""

    author = "QuantMind"
    parameters = ["fast", "slow", "size", "max_pos", "fixed_size"]

    def __init__(self, context, setting=None) -> None:
        self.fast = 5
        self.slow = 20
        self.size = 1          # 合约乘数（期货 rb=10 / IF=300，由外部传入）
        self.max_pos = 1.0     # 最大仓位比例（0~1）
        self.fixed_size = 1    # 单次调仓手数（=size*max_pos 的目标净仓比例由 set_target 决定）
        # ArrayManager 缓冲按**应用 settings 后**的窗口惰性构建（与 ChanThirdBuyStrategy 同款修复）：
        # 提前按默认 slow+10 定长会让 setting 调大窗口后 sma 恒返回 0，策略静默变成恒多/恒空
        self.am = None
        self.last_target = 0.0
        super().__init__(context, setting)

    def on_init(self) -> None:
        pass

    def on_bar(self, bar: BarData) -> None:
        if self.am is None:
            self.am = ArrayManager(max(self.fast, self.slow) + 5)
        self.am.update_bar(bar)
        if not self.am.inited:
            return
        fast_ma = self.am.sma(self.fast)
        slow_ma = self.am.sma(self.slow)
        if fast_ma > slow_ma:
            target = self.max_pos
        else:
            target = -self.max_pos
        target_vol = target * self.size
        if target_vol != self.last_target:
            oid = self.set_target(bar.vt_symbol, target_vol)
            if oid == "":
                # 风控拒单：保留 last_target，下一根 bar 重试
                return
            self.last_target = target_vol
            self.pos = target_vol
