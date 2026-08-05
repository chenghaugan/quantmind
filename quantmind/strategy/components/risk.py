"""RiskModel 实现：复用现有 RiskEngine / RiskLimits 作为组合策略的风控组件。

``RiskGateModel`` 包装 ``quantmind.risk.RiskEngine`` 做**交易前风控闸门**：
对"目标仓位 - 当前持仓"的差值合成一笔委托，走一遍 ``check_order``，
被拒则返回 None（本次调仓被拦截），放行则返回原目标仓位。

``NullRisk`` 为透传（默认无风控）。
"""
from __future__ import annotations

from typing import Optional

from ...core.constant import Direction, Exchange, Offset
from ...core.gateway import OrderRequest
from ...core.object import BarData
from ...risk import RiskEngine, RiskLimits
from .base import RiskModel as RiskModelProtocol


class NullRisk(RiskModelProtocol):
    """透传：不做任何风控过滤。"""

    def apply(self, target: Optional[float], bar: BarData, context=None) -> Optional[float]:
        return target


class RiskGateModel(RiskModelProtocol):
    """基于 RiskEngine 的交易前风控闸门。

    :param engine: 可选；缺省时用 ``RiskLimits()`` 构造。
    :param allow_reduce: True 时减仓/平仓即使拒绝也放行（避免锁死仓位）。
    """

    def __init__(self, engine: Optional[RiskEngine] = None,
                 allow_reduce: bool = True) -> None:
        self.engine = engine or RiskEngine(limits=RiskLimits())
        self.allow_reduce = allow_reduce

    def apply(self, target: Optional[float], bar: BarData, context=None) -> Optional[float]:
        if target is None:
            return None
        vt = bar.vt_symbol or f"{bar.symbol}.{bar.exchange.value}"
        sym, exch = vt.rsplit(".", 1)
        cur_vol = 0.0
        if context is not None:
            pos = context.get_position(vt)
            cur_vol = pos.volume if pos else 0.0
        delta = target - cur_vol
        if abs(delta) < 1e-9:
            return target
        increasing = abs(target) >= abs(cur_vol)
        req = OrderRequest(
            symbol=sym,
            exchange=Exchange(exch),
            direction=Direction.LONG if delta > 0 else Direction.SHORT,
            offset=Offset.OPEN if increasing else Offset.CLOSE,
            volume=abs(delta),
            price=bar.close_price,
        )
        decision = self.engine.check_order(
            req,
            position=context.get_position(vt) if context is not None else None,
            last_price=bar.close_price,
        )
        if decision.passed:
            return target
        # 减仓/平仓拒单：默认放行（避免仓位被锁死）
        if not increasing and self.allow_reduce:
            return target
        return None


__all__ = ["NullRisk", "RiskGateModel"]
