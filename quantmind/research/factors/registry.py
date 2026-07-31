"""因子注册表：集中管理与按名检索因子。

参考 vnpy 的因子库思想，但支持动态注册（便于 AI 生成的因子入库）。
"""
from __future__ import annotations

from typing import Dict, List

from .base import Factor
from .technical import (
    MomentumFactor,
    MeanReversionFactor,
    VolatilityFactor,
    VolumeChangeFactor,
    OpenInterestChangeFactor,
    TermStructureFactor,
)
from .alpha101 import build_alpha_factor, list_alpha101
from .alpha191 import build_alpha191_factor, list_alpha191
from .seat_futures import SeatFactor


class FactorRegistry:
    """因子注册表。"""

    def __init__(self) -> None:
        self._factors: Dict[str, Factor] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        for f in (
            MomentumFactor(20),
            MomentumFactor(60),
            MeanReversionFactor(60),
            VolatilityFactor(20),
            VolumeChangeFactor(5),
            OpenInterestChangeFactor(20),
            TermStructureFactor(20),
        ):
            self.register(f)
        # 注册代表性 Alpha101/191 子集（便于 /factors 展示与直接取用；完整集合可用 build_alpha_factor）
        _alpha_subset = [
            "alpha002", "alpha006", "alpha012", "alpha017", "alpha021",
            "alpha026", "alpha033", "alpha038", "alpha054", "alpha075",
            "alpha093", "alpha101",
        ]
        for name in _alpha_subset:
            if name in list_alpha101():
                self.register(build_alpha_factor(name))
        _alpha191_subset = ["alpha191_007", "alpha191_012", "alpha191_042", "alpha191_056", "alpha191_081"]
        for name in _alpha191_subset:
            if name in list_alpha191():
                self.register(build_alpha191_factor(name))
        # 期货席位因子（需席位净持仓数据，compute 时再提供；此处仅登记元信息）
        for name in ("F1_net_position", "F3_net_ratio", "F7_net_zscore", "F8_seat_sentiment"):
            self.register(SeatFactor(name))

    def register(self, factor: Factor) -> None:
        self._factors[factor.meta.name] = factor

    def get(self, name: str) -> Factor:
        f = self._factors.get(name)
        if f is None:
            raise KeyError(f"因子未注册: {name}")
        return f

    def list_factors(self) -> List[dict]:
        return [
            {
                "name": f.meta.name,
                "category": f.meta.category,
                "description": f.meta.description,
                "params": f.params,
            }
            for f in self._factors.values()
        ]


def build_factor_registry() -> FactorRegistry:
    return FactorRegistry()
