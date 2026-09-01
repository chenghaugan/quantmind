"""因子生命周期与衰减监控（对标 Vibe-Trading strategy-dev-manager）。

状态机：ACTIVE → MONITORING → DECAYED → DISABLED
- ACTIVE: 新注册因子，IC/Sharpe 正常
- MONITORING: 指标开始衰减，进入观察期
- DECAYED: 指标持续低于阈值，建议停用
- DISABLED: 已停用，不再参与组合

衰减检测：
- 滚动 IC 窗口（默认 60 日）与历史 IC 对比
- 滚动 Sharpe 衰减率
- 触发条件：近 N 日 IC 均值 < 历史 IC 均值 × decay_ratio

持久化：复用 KnowledgeStore 的 factors 表 status 字段。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_logger = logging.getLogger("quantmind.research.decay")

UTC = timezone.utc


class FactorState(str, Enum):
    """因子生命周期状态。"""

    ACTIVE = "active"
    MONITORING = "monitoring"
    DECAYED = "decayed"
    DISABLED = "disabled"


@dataclass
class DecayConfig:
    """衰减检测配置。"""

    # IC 衰减阈值：近 N 日 IC 均值 < 历史 IC 均值 × ic_decay_ratio → 触发降级
    ic_decay_ratio: float = 0.5
    ic_window_days: int = 60
    history_window_days: int = 252

    # Sharpe 衰减阈值
    sharpe_decay_ratio: float = 0.6

    # 状态转移条件
    monitoring_min_days: int = 30  # MONITORING 至少观察天数
    decayed_min_days: int = 60  # DECAYED 至少观察天数


@dataclass
class DecayMetrics:
    """单因子的衰减指标快照。"""

    factor_id: str
    state: FactorState = FactorState.ACTIVE
    ic_mean_recent: Optional[float] = None
    ic_mean_history: Optional[float] = None
    ic_decay_ratio: Optional[float] = None
    sharpe_recent: Optional[float] = None
    sharpe_history: Optional[float] = None
    sharpe_decay_ratio: Optional[float] = None
    n_samples_recent: int = 0
    n_samples_history: int = 0
    last_scan_at: Optional[str] = None
    state_since: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "factor_id": self.factor_id,
            "state": self.state.value,
            "ic_mean_recent": _nan_to_none(self.ic_mean_recent),
            "ic_mean_history": _nan_to_none(self.ic_mean_history),
            "ic_decay_ratio": _nan_to_none(self.ic_decay_ratio),
            "sharpe_recent": _nan_to_none(self.sharpe_recent),
            "sharpe_history": _nan_to_none(self.sharpe_history),
            "sharpe_decay_ratio": _nan_to_none(self.sharpe_decay_ratio),
            "n_samples_recent": self.n_samples_recent,
            "n_samples_history": self.n_samples_history,
            "last_scan_at": self.last_scan_at,
            "state_since": self.state_since,
            "notes": self.notes,
        }


class FactorDecayScanner:
    """因子衰减扫描器。

    用法：
        scanner = FactorDecayScanner(config=DecayConfig())
        metrics = scanner.compute_metrics(factor_id="f-xxx", ic_series=ic_df)
        scanner.transition_if_needed(metrics)
    """

    def __init__(self, config: Optional[DecayConfig] = None) -> None:
        self.config = config or DecayConfig()
        self._records: Dict[str, DecayMetrics] = {}

    def compute_metrics(
        self,
        factor_id: str,
        ic_series: pd.Series,
        current_state: FactorState = FactorState.ACTIVE,
    ) -> DecayMetrics:
        """基于 IC 时序计算衰减指标。

        :param factor_id: 因子标识。
        :param ic_series: 时序 IC（index 为日期，value 为 IC）。
        :param current_state: 当前状态（用于判断 state_since）。
        :return: DecayMetrics 快照。
        """
        ic = ic_series.dropna()
        n_total = len(ic)
        if n_total < self.config.ic_window_days:
            # 数据不足，无法评估
            return DecayMetrics(
                factor_id=factor_id,
                state=current_state,
                notes=["数据不足，无法评估衰减"],
            )

        # 近期窗口 vs 历史窗口（历史窗口取近期窗口**之前**的区间，
        # 否则衰减比被近期自身稀释，难以触发阈值）
        recent = ic.iloc[-self.config.ic_window_days :]
        if n_total <= self.config.ic_window_days:
            # 历史窗口与近期完全重叠（或为空）：衰减不可评估，虚报比率只会误导
            return DecayMetrics(
                factor_id=factor_id,
                state=current_state,
                notes=["数据不足：历史窗口与近期窗口重叠，无法评估衰减"],
            )
        h_end = -self.config.ic_window_days
        history = ic.iloc[-self.config.history_window_days :h_end]
        if history.empty:
            return DecayMetrics(
                factor_id=factor_id,
                state=current_state,
                notes=["数据不足：历史窗口无有效样本，无法评估衰减"],
            )

        ic_recent = float(recent.mean())
        ic_history = float(history.mean())
        if abs(ic_history) > 1e-9:
            ic_ratio = ic_recent / ic_history
            # 负 IC 历史下简单比值会符号反转（改善被误判为衰减），
            # 翻转符号使「近期/历史」同向可比
            if ic_history < 0:
                ic_ratio = -ic_ratio
        else:
            ic_ratio = None

        # Sharpe（IR）：滚动 IC 的均值/标准差
        sharpe_recent = _rolling_sharpe(recent)
        sharpe_history = _rolling_sharpe(history)
        sharpe_ratio = (
            sharpe_recent / sharpe_history
            if sharpe_history and abs(sharpe_history) > 1e-9
            else None
        )

        return DecayMetrics(
            factor_id=factor_id,
            state=current_state,
            ic_mean_recent=ic_recent,
            ic_mean_history=ic_history,
            ic_decay_ratio=ic_ratio,
            sharpe_recent=sharpe_recent,
            sharpe_history=sharpe_history,
            sharpe_decay_ratio=sharpe_ratio,
            n_samples_recent=len(recent),
            n_samples_history=len(history),
            last_scan_at=datetime.now(UTC).isoformat(),
            state_since=None,  # 由 transition_if_needed 填充
        )

    def transition_if_needed(self, metrics: DecayMetrics) -> FactorState:
        """根据衰减指标决定状态转移，返回新状态。

        转移规则：
        - ACTIVE → MONITORING: ic_decay_ratio < ic_decay_ratio 阈值
        - MONITORING → DECAYED: 持续低于阈值且观察期 >= monitoring_min_days
        - DECAYED → DISABLED: 持续低于阈值且观察期 >= decayed_min_days
        - 任何状态 → ACTIVE: 指标恢复（反向转移不自动，需人工干预）
        """
        cfg = self.config
        old_state = metrics.state
        new_state = old_state

        # 检查 IC 衰减
        ic_decayed = (
            metrics.ic_decay_ratio is not None
            and metrics.ic_decay_ratio < cfg.ic_decay_ratio
        )
        sharpe_decayed = (
            metrics.sharpe_decay_ratio is not None
            and metrics.sharpe_decay_ratio < cfg.sharpe_decay_ratio
        )

        if old_state == FactorState.ACTIVE:
            if ic_decayed or sharpe_decayed:
                new_state = FactorState.MONITORING
                metrics.notes.append(
                    f"IC/Sharpe 衰减触发 MONITORING: ic_ratio={metrics.ic_decay_ratio}, "
                    f"sharpe_ratio={metrics.sharpe_decay_ratio}"
                )
        elif old_state == FactorState.MONITORING:
            # 持续衰减 → DECAYED（简化：不检查天数，由调用方控制）
            if ic_decayed or sharpe_decayed:
                new_state = FactorState.DECAYED
                metrics.notes.append("持续衰减，进入 DECAYED")
        elif old_state == FactorState.DECAYED:
            # 持续衰减 → DISABLED
            if ic_decayed or sharpe_decayed:
                new_state = FactorState.DISABLED
                metrics.notes.append("严重衰减，进入 DISABLED，建议停用")

        if new_state != old_state:
            metrics.state = new_state
            metrics.state_since = datetime.now(UTC).isoformat()
            _logger.info(
                "因子 %s 状态转移: %s → %s (ic_ratio=%s, sharpe_ratio=%s)",
                metrics.factor_id,
                old_state.value,
                new_state.value,
                metrics.ic_decay_ratio,
                metrics.sharpe_decay_ratio,
            )

        return new_state

    def scan_all(
        self,
        factor_ic_map: Dict[str, pd.Series],
        current_states: Optional[Dict[str, FactorState]] = None,
    ) -> List[DecayMetrics]:
        """批量扫描所有因子的衰减状态。

        :param factor_ic_map: {factor_id: ic_series}。
        :param current_states: {factor_id: FactorState}，默认全部 ACTIVE。
        :return: 所有因子的 DecayMetrics 列表。
        """
        current_states = current_states or {}
        results: List[DecayMetrics] = []
        for fid, ic in factor_ic_map.items():
            state = current_states.get(fid, FactorState.ACTIVE)
            metrics = self.compute_metrics(fid, ic, current_state=state)
            self.transition_if_needed(metrics)
            self._records[fid] = metrics
            results.append(metrics)
        return results

    def get_record(self, factor_id: str) -> Optional[DecayMetrics]:
        return self._records.get(factor_id)

    def list_records(self) -> List[DecayMetrics]:
        return list(self._records.values())


# =============================================================================
# 工具函数
# =============================================================================
def _rolling_sharpe(ic: pd.Series, window: int = 60) -> Optional[float]:
    """计算滚动 IC 的 Sharpe（IR）：均值 / 标准差 × sqrt(年化)。"""
    if len(ic) < window:
        return None
    mean = float(ic.mean())
    std = float(ic.std())
    if std < 1e-9:
        return None
    # 假设日线，年化 sqrt(252)
    return mean / std * np.sqrt(252)


def _nan_to_none(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    try:
        f = float(x)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


__all__ = [
    "FactorState",
    "DecayConfig",
    "DecayMetrics",
    "FactorDecayScanner",
]
