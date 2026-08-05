"""ExecutionModel 实现：把目标仓位交给 StrategyContext 执行。

``context.set_target`` 已处理开/平 Offset 判定与 EVENT_SIGNAL 广播，
这里仅做转发，使执行环节可独立替换（例如接入 VWAP/TWAP 算法执行）。
"""
from __future__ import annotations

from typing import Optional

from ...core.object import BarData
from .base import ExecutionModel as ExecutionModelProtocol


class TargetExecution(ExecutionModelProtocol):
    """标准目标仓位执行：委托给 ``StrategyContext.set_target``。"""

    def __init__(self, context=None) -> None:
        # context 在 ComposableStrategy.on_start 时绑定，因为此时才拿到引擎 context
        self.context = context

    def bind(self, context) -> None:
        """绑定运行上下文（回测/模拟/实盘引擎）。"""
        self.context = context

    def set_target(self, vt_symbol: str, target: float) -> Optional[str]:
        if self.context is None:
            return None
        return self.context.set_target(vt_symbol, target)


# 向后兼容别名
ExecutionModel = TargetExecution


__all__ = ["TargetExecution", "ExecutionModel"]
