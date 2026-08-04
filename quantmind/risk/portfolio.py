"""组合级风控：多策略 / 多标的的暴露、集中度、相关性约束。

与单策略 :class:`~quantmind.risk.engine.RiskEngine` 是**不同粒度**：
- RiskEngine  检查单策略内一笔委托是否合规（手数/保证金/回撤/时段）。
- 本模块检查**整个组合**层面：总敞口、单标的集中度、策略间相关性，
  防止"每个策略单独看没事，合在一起却暴露过度 / 高度同涨同跌"。

口径说明
--------
- 暴露（exposure）以**名义比例**计（相对组合总权益），便于跨策略比较。
- ``PortfolioRiskState`` 只做记账；``PortfolioRiskEngine`` 做判定并复用
  :class:`~quantmind.risk.limits.RiskDecision` / ``RiskCode``。

用法示例
--------
::

    state = PortfolioRiskState()
    eng = PortfolioRiskEngine(PortfolioLimits(max_gross_exposure=0.8))
    eng.attach(state)
    state.set_position("strat_a", "rb0.SHFE", volume=10, value=300_000)
    dec = eng.check_position("strat_a", "rb0.SHFE", value=300_000, total_equity=1_000_000)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .limits import RiskCode, RiskDecision

_logger = logging.getLogger("quantmind.risk.portfolio")


@dataclass
class PortfolioLimits:
    """组合级限额。``None`` 表示不检查该项。"""

    max_gross_exposure: Optional[float] = None    # 总敞口（多头+空头）/ 权益 上限
    max_net_exposure: Optional[float] = None      # 净敞口 / 权益 上限
    max_position_concentration: Optional[float] = None  # 单标的市值 / 权益 上限
    max_strategy_correlation: Optional[float] = None     # 策略收益两两相关性上限

    def to_dict(self) -> dict:
        return {
            "max_gross_exposure": self.max_gross_exposure,
            "max_net_exposure": self.max_net_exposure,
            "max_position_concentration": self.max_position_concentration,
            "max_strategy_correlation": self.max_strategy_correlation,
        }

    @classmethod
    def conservative(cls) -> "PortfolioLimits":
        return cls(
            max_gross_exposure=0.9,
            max_net_exposure=0.5,
            max_position_concentration=0.3,
            max_strategy_correlation=0.8,
        )


@dataclass
class PositionBookEntry:
    """组合中某个标的的一笔头寸记录。"""

    strategy_id: str
    vt_symbol: str
    volume: float = 0.0
    value: float = 0.0             # 名义市值（人民币，正=多头，负=空头）

    def to_dict(self) -> dict:
        return {
            "strategy_id": self.strategy_id,
            "vt_symbol": self.vt_symbol,
            "volume": self.volume,
            "value": self.value,
        }


class PortfolioRiskState:
    """多策略组合的风险记账。"""

    def __init__(self) -> None:
        # strategy_id -> {vt_symbol -> PositionBookEntry}
        self.positions: Dict[str, Dict[str, PositionBookEntry]] = {}
        # strategy_id -> 单策略权益
        self.equities: Dict[str, float] = {}

    def set_position(self, strategy_id: str, vt_symbol: str, volume: float = 0.0,
                     value: float = 0.0) -> None:
        bucket = self.positions.setdefault(strategy_id, {})
        bucket[vt_symbol] = PositionBookEntry(strategy_id, vt_symbol, volume, value)

    def add_value(self, strategy_id: str, vt_symbol: str, value_delta: float) -> None:
        bucket = self.positions.setdefault(strategy_id, {})
        entry = bucket.get(vt_symbol, PositionBookEntry(strategy_id, vt_symbol))
        entry.value += value_delta
        bucket[vt_symbol] = entry

    def set_equity(self, strategy_id: str, equity: float) -> None:
        self.equities[strategy_id] = equity

    def total_equity(self) -> float:
        return sum(self.equities.values()) or 0.0

    # ---- 敞口 ----
    def gross_exposure(self) -> float:
        """总敞口 = Σ|各标的名义市值|。"""
        return sum(abs(e.value) for bucket in self.positions.values() for e in bucket.values())

    def net_exposure(self) -> float:
        """净敞口 = Σ 各标的名义市值（带符号）。"""
        return sum(e.value for bucket in self.positions.values() for e in bucket.values())

    def long_exposure(self) -> float:
        return sum(e.value for bucket in self.positions.values() for e in bucket.values() if e.value > 0)

    def short_exposure(self) -> float:
        return abs(sum(e.value for bucket in self.positions.values() for e in bucket.values() if e.value < 0))

    def position_value(self, vt_symbol: str) -> float:
        """某标的名义市值（跨策略汇总，绝对值求和 = 该标的集中度）。"""
        return sum(abs(e.value) for bucket in self.positions.values()
                   for k, e in bucket.items() if k == vt_symbol)

    def to_dict(self) -> dict:
        equity = self.total_equity()
        return {
            "total_equity": round(equity, 2),
            "gross_exposure": round(self.gross_exposure(), 2),
            "net_exposure": round(self.net_exposure(), 2),
            "long_exposure": round(self.long_exposure(), 2),
            "short_exposure": round(self.short_exposure(), 2),
            "gross_exposure_ratio": round(self.gross_exposure() / equity, 4) if equity else 0.0,
            "net_exposure_ratio": round(self.net_exposure() / equity, 4) if equity else 0.0,
        }


class PortfolioRiskEngine:
    """组合级风控判定。"""

    def __init__(self, policy: Optional[PortfolioLimits] = None, state: Optional[PortfolioRiskState] = None) -> None:
        self.policy = policy or PortfolioLimits()
        # correlation_matrix: strategy_id 索引的收益相关性表（check_strategy_correlation 时传入/更新）
        self.correlation_matrix: Optional[pd.DataFrame] = None

    def attach(self, state: PortfolioRiskState) -> "PortfolioRiskEngine":
        self.state = state
        return self

    def check_position(self, strategy_id: str, vt_symbol: str, value: float,
                       total_equity: Optional[float] = None, existing_value: float = 0.0) -> RiskDecision:
        """判定某策略对某标的的加仓是否导致组合层面超限。

        :param value: 本次拟增加的**名义市值**（带符号）。
        :param existing_value: 该标的当前已有市值（绝对值）。
        :returns: RiskDecision；通过为 ``passed=True``。
        """
        total_equity = total_equity or self.total_equity()
        if total_equity <= 0:
            return RiskDecision.ok(vt_symbol)

        projected_position_value = existing_value + abs(value)
        # 集中度
        if self.policy.max_position_concentration is not None:
            conc = projected_position_value / total_equity
            if conc > self.policy.max_position_concentration:
                return RiskDecision.reject(
                    RiskCode.POSITION_CONCENTRATION,
                    f"{vt_symbol} 集中度 {conc:.1%} 超上限 "
                    f"{self.policy.max_position_concentration:.1%}",
                    vt_symbol,
                )
        return RiskDecision.ok(vt_symbol)

    def check_exposure(self, gross_exposure: float, net_exposure: float,
                       total_equity: Optional[float] = None) -> RiskDecision:
        """基于当前状态校验组合敞口。"""
        total_equity = total_equity or self.total_equity()
        if total_equity <= 0:
            return RiskDecision.ok("")
        if self.policy.max_gross_exposure is not None:
            ratio = gross_exposure / total_equity
            if ratio > self.policy.max_gross_exposure:
                return RiskDecision.reject(
                    RiskCode.EXPOSURE_LIMIT,
                    f"总敞口 {ratio:.0%} 超上限 {self.policy.max_gross_exposure:.0%}",
                    "",
                )
        if self.policy.max_net_exposure is not None:
            ratio = abs(net_exposure) / total_equity
            if ratio > self.policy.max_net_exposure:
                return RiskDecision.reject(
                    RiskCode.EXPOSURE_LIMIT,
                    f"净敞口 {ratio:.0%} 超上限 {self.policy.max_net_exposure:.0%}",
                    "",
                )
        return RiskDecision.ok("")

    def update_correlation_matrix(self, matrix: pd.DataFrame) -> None:
        """更新策略收益相关性矩阵（index/columns 为 strategy_id）。"""
        self.correlation_matrix = matrix

    def check_strategy_correlation(self) -> List[RiskDecision]:
        """检查策略间两两相关性，返回超限的违例列表（空=无违例）。"""
        if not self.policy.max_strategy_correlation or self.correlation_matrix is None:
            return []
        mat = self.correlation_matrix.astype(float)
        violations: List[RiskDecision] = []
        cols = list(mat.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                corr = mat.iloc[i, j]
                if np.isnan(corr):
                    continue
                if corr > self.policy.max_strategy_correlation:
                    violations.append(RiskDecision.reject(
                        RiskCode.STRATEGY_CORRELATION,
                        f"策略 {cols[i]}/{cols[j]} 相关性 {corr:.2f} 超上限 "
                        f"{self.policy.max_strategy_correlation:.2f}",
                        f"{cols[i]}/{cols[j]}",
                    ))
        return violations

    def total_equity(self) -> float:
        return self.state.total_equity() if hasattr(self, "state") and self.state else 0.0

    def to_dict(self) -> dict:
        return {
            "policy": self.policy.to_dict(),
            "state": self.state.to_dict() if hasattr(self, "state") and self.state else {},
            "correlations": self.correlation_matrix.round(3).to_dict()
            if self.correlation_matrix is not None else {},
        }


def compute_strategy_correlation(strategy_returns: Dict[str, pd.Series]) -> pd.DataFrame:
    """由多策略收益序列计算两两相关性表（index/columns 为策略名）。"""
    df = pd.DataFrame({name: s.astype(float) for name, s in strategy_returns.items()})
    return df.corr().fillna(0.0)


__all__ = [
    "PortfolioLimits",
    "PositionBookEntry",
    "PortfolioRiskState",
    "PortfolioRiskEngine",
    "compute_strategy_correlation",
]
