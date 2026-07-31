"""策略生命周期与晋升闸门（对应规划 Lifecycle 状态机）。

状态：IDEA -> RESEARCH -> BACKTEST -> PAPER -> APPROVED -> LIVE
晋升需通过校验门（例如回测夏普、最大回撤、风险参数），避免未经验证的策略直接进入实盘。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

_logger = logging.getLogger("quantmind.paper.promotion")

UTC = timezone.utc


class LifecycleState(Enum):
    IDEA = "IDEA"
    RESEARCH = "RESEARCH"
    BACKTEST = "BACKTEST"
    PAPER = "PAPER"
    APPROVED = "APPROVED"
    LIVE = "LIVE"
    REJECTED = "REJECTED"


_PROMOTION_ORDER = [
    LifecycleState.IDEA,
    LifecycleState.RESEARCH,
    LifecycleState.BACKTEST,
    LifecycleState.PAPER,
    LifecycleState.APPROVED,
    LifecycleState.LIVE,
]


@dataclass
class PromotionGate:
    """晋升校验门配置。"""

    min_sharpe: float = 0.5
    max_drawdown: float = -0.30   # 最大回撤下限（负值）
    min_paper_days: int = 1
    require_risk_review: bool = True


@dataclass
class LifecycleRecord:
    """单策略生命周期记录。"""

    strategy_id: str
    state: LifecycleState = LifecycleState.IDEA
    history: List[Dict] = field(default_factory=list)
    metrics: Dict = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def transition(self, to: LifecycleState, note: str = "") -> None:
        self.history.append({
            "from": self.state.value,
            "to": to.value,
            "at": datetime.now(UTC).isoformat(),
            "note": note,
        })
        self.state = to


class LifecycleManager:
    """生命周期管理器。"""

    def __init__(self, gate: PromotionGate | None = None) -> None:
        self.gate = gate or PromotionGate()
        self.records: Dict[str, LifecycleRecord] = {}

    def get_or_create(self, strategy_id: str) -> LifecycleRecord:
        if strategy_id not in self.records:
            self.records[strategy_id] = LifecycleRecord(strategy_id)
        return self.records[strategy_id]

    def can_promote(self, rec: LifecycleRecord, to: LifecycleState) -> tuple:
        """返回 (是否可晋升, 原因列表)。"""
        reasons: List[str] = []
        cur = rec.state
        if to == LifecycleState.LIVE and cur not in (LifecycleState.APPROVED, LifecycleState.PAPER):
            reasons.append(f"当前状态 {cur.value} 不允许直接进入 LIVE")
        if to == LifecycleState.LIVE:
            m = rec.metrics
            if m.get("sharpe", 0.0) < self.gate.min_sharpe:
                reasons.append(f"夏普 {m.get('sharpe')} < {self.gate.min_sharpe}")
            if m.get("max_drawdown", 0.0) < self.gate.max_drawdown:
                reasons.append(f"最大回撤 {m.get('max_drawdown')} 超过阈值 {self.gate.max_drawdown}")
            if self.gate.require_risk_review and "risk_reviewed" not in rec.notes:
                reasons.append("未完成风控复核")
        return (len(reasons) == 0, reasons)

    def promote(self, strategy_id: str, to: LifecycleState, metrics: Optional[Dict] = None,
                note: str = "") -> tuple:
        rec = self.get_or_create(strategy_id)
        if metrics:
            rec.metrics.update(metrics)
        if note:
            rec.notes.append(note)
        ok, reasons = self.can_promote(rec, to)
        if not ok:
            return False, reasons
        rec.transition(to, note)
        return True, []
