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

    min_sharpe: float = 1.0          # 年化夏普 ≥ 1.0（量化行业基本门槛）
    max_drawdown: float = -0.15      # 最大回撤下限 -15%（更严格的风控）
    min_paper_days: int = 30         # 模拟盘至少跑 30 天
    min_calmar: float = 1.0          # Calmar 比率 = 年化收益 / |最大回撤| ≥ 1.0
    min_win_rate: float = 0.45       # 交易胜率 ≥ 45%
    require_risk_review: bool = True


@dataclass
class LifecycleRecord:
    """单策略生命周期记录。"""

    strategy_id: str
    state: LifecycleState = LifecycleState.IDEA
    history: List[Dict] = field(default_factory=list)
    metrics: Dict = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    #: 可选持久化回调（由 LifecycleManager 注入）；None 时 transition 零开销。
    _persist: Optional = field(default=None, repr=False, compare=False)

    def transition(self, to: LifecycleState, note: str = "") -> None:
        old = self.state.value
        self.history.append({
            "from": old,
            "to": to.value,
            "at": datetime.now(UTC).isoformat(),
            "note": note,
        })
        self.state = to
        if self._persist is not None:
            try:
                self._persist(old, to.value, note)
            except Exception:
                _logger.exception("lifecycle 持久化 transition 失败（不影响内存状态）: %s",
                                  self.strategy_id)


class LifecycleManager:
    """生命周期管理器。

    可选 ``store`` 持久层（提供 get_strategy_lifecycle / update_strategy_state /
    push_strategy_transition / upsert_strategy_lifecycle）。store 为 None 时保持
    纯内存行为，完全向后兼容；store 存在时每次晋升/变更同步落库，重启可恢复。
    所有 store 调用用 try/except 包裹，落库失败不影响内存状态机。
    """

    def __init__(self, gate: PromotionGate | None = None, store: Optional = None) -> None:
        self.gate = gate or PromotionGate()
        self.store = store
        self.records: Dict[str, LifecycleRecord] = {}

    def get_or_create(self, strategy_id: str) -> LifecycleRecord:
        if strategy_id in self.records:
            return self.records[strategy_id]

        rec = self._load_from_db(strategy_id)
        if rec is None:
            rec = LifecycleRecord(strategy_id)
            # 新建时若可持久化则落一个 IDEA 初始行
            if self.store is not None:
                try:
                    self.store.upsert_strategy_lifecycle(strategy_id, state="IDEA")
                except Exception:
                    _logger.exception("lifecycle upsert 失败：%s", strategy_id)
        if self.store is not None:
            rec._persist = self._make_persist(strategy_id)
        self.records[strategy_id] = rec
        return rec

    def _make_persist(self, strategy_id: str):
        """构造绑定到指定策略的持久化回调（供 LifecycleRecord.transition 调用）。"""

        def _cb(from_state: str, to_state: str, note: str) -> None:
            try:
                self.store.push_strategy_transition(
                    strategy_id, from_state, to_state, note)
            except Exception:
                _logger.exception("lifecycle push transition 失败：%s", strategy_id)

        return _cb

    def _load_from_db(self, strategy_id: str) -> Optional[LifecycleRecord]:
        if self.store is None:
            return None
        try:
            data = self.store.get_strategy_lifecycle(strategy_id)
        except Exception:
            _logger.exception("lifecycle get 失败：%s", strategy_id)
            return None
        if data is None:
            return None
        try:
            state = LifecycleState(data["state"])
        except (KeyError, ValueError):
            state = LifecycleState.IDEA
        history = data.get("history") or []
        metrics: Dict = {}
        for k in ("sharpe", "max_drawdown", "composite_fwd_ic"):
            v = data.get(k)
            if v is not None:
                metrics[k] = v
        for k in ("status", "reason"):
            v = data.get(k)
            if v:
                metrics[k] = v
        notes = [h.get("note", "") for h in history if h.get("note")]
        return LifecycleRecord(
            strategy_id, state=state, history=history,
            metrics=metrics, notes=notes,
        )

    def can_promote(self, rec: LifecycleRecord, to: LifecycleState) -> tuple:
        """返回 (是否可晋升, 原因列表)。"""
        reasons: List[str] = []
        cur = rec.state
        if to == LifecycleState.LIVE and cur not in (LifecycleState.APPROVED, LifecycleState.PAPER):
            reasons.append(f"当前状态 {cur.value} 不允许直接进入 LIVE")
        if to == LifecycleState.LIVE:
            m = rec.metrics
            # 1. 夏普门槛
            if m.get("sharpe", 0.0) < self.gate.min_sharpe:
                reasons.append(f"夏普 {m.get('sharpe')} < {self.gate.min_sharpe}")
            # 2. 最大回撤门槛
            if m.get("max_drawdown", 0.0) < self.gate.max_drawdown:
                reasons.append(f"最大回撤 {m.get('max_drawdown')} 超过阈值 {self.gate.max_drawdown}")
            # 3. 模拟盘天数门槛
            paper_days = m.get("paper_days", 0)
            if paper_days < self.gate.min_paper_days:
                reasons.append(f"模拟盘天数 {paper_days} < {self.gate.min_paper_days}")
            # 4. Calmar 比率门槛
            calmar = m.get("calmar")
            if calmar is not None and calmar < self.gate.min_calmar:
                reasons.append(f"Calmar 比率 {calmar} < {self.gate.min_calmar}")
            # 5. 胜率门槛
            win_rate = m.get("win_rate")
            if win_rate is not None and win_rate < self.gate.min_win_rate:
                reasons.append(f"胜率 {win_rate} < {self.gate.min_win_rate}")
            # 6. 风控复核
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
        # transition 会触发 store.push_strategy_transition（若已注入 _persist）
        rec.transition(to, note)
        # 晋升成功后同步状态与真实回测指标落库
        if self.store is not None:
            try:
                self.store.update_strategy_state(
                    strategy_id,
                    state=to.value,
                    sharpe=(metrics or {}).get("sharpe"),
                    max_drawdown=(metrics or {}).get("max_drawdown"),
                    status=(metrics or {}).get("status") or "",
                    reason=(metrics or {}).get("reason", "") if metrics else "",
                    brief=(metrics or {}).get("brief", "") if metrics else "",
                )
            except Exception:
                _logger.exception("lifecycle update_strategy_state 失败：%s", strategy_id)
        return True, []
