"""UniverseModel 实现：标的池选择（M5）。

  - ``AllUniverse``：透传所有候选标的（默认无过滤）。
  - ``RuleUniverse``：按规则过滤（最小历史长度 / 最小平均成交量 / 数据完整性）。

``select(candidates, context)`` 的 ``context`` 为带 ``vt_symbols`` 与 ``get_history`` 的
组件上下文（见 ``composable._ComponentContext``），用于逐标的取历史判断规则。
"""
from __future__ import annotations

from typing import List, Optional

from .base import UniverseModel as UniverseModelProtocol


class AllUniverse(UniverseModelProtocol):
    """透传标的池：不过滤，返回全部候选。"""

    def select(self, candidates: List[str], context=None) -> List[str]:
        return list(candidates or [])


class RuleUniverse(UniverseModelProtocol):
    """规则化标的池：过滤候选标的。

    :param min_bars: 最少历史 K 线数（不足以训练/预热因子则剔除）。
    :param min_avg_volume: 最小平均成交量（流动性过滤，0 表示不启用）。
    """

    def __init__(self, min_bars: int = 120, min_avg_volume: float = 0.0) -> None:
        self.min_bars = max(1, int(min_bars))
        self.min_avg_volume = float(min_avg_volume)

    def select(self, candidates: List[str], context=None) -> List[str]:
        if not candidates:
            return []
        selected: List[str] = []
        for vt in candidates:
            bars = context.get_history(vt, self.min_bars) if context is not None else []
            if len(bars) < self.min_bars:
                continue
            if self.min_avg_volume > 0:
                vols = [b.volume for b in bars if b.volume is not None]
                if not vols or (sum(vols) / len(vols)) < self.min_avg_volume:
                    continue
            selected.append(vt)
        return selected


__all__ = ["AllUniverse", "RuleUniverse"]
