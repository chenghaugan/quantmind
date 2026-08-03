"""持仓 / 资金对账（实盘化 P0）。

实盘最危险的状态不是亏损，而是**策略以为自己的持仓 A，账户里实际是 B**。
成因很常见：漏收成交回报、隔夜手工干预、部分成交后程序重启、
夜盘断线重连、交易所强平。此时策略继续按错误持仓下单，会把小问题放大成大事故。

对账规则
--------
  - 每次连接网关成功后、每个交易时段开始前、每日收盘后，各做一次全量对账。
  - 差异超过容差 → **默认触发 SOFT 熔断**（禁开仓），必须人工确认后 ``resume()``。
  - 对账只报告，不自动「修正」——自动改持仓等于把错误状态写死，得不偿失。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from ..core.object import PositionData

_logger = logging.getLogger("quantmind.live.reconcile")
UTC = timezone.utc


@dataclass
class PositionDiff:
    """单个合约的持仓差异。"""

    vt_symbol: str
    local_volume: float
    remote_volume: float
    kind: str  # "MISMATCH" | "MISSING_REMOTE"（本地有远端无） | "MISSING_LOCAL"（远端有本地无）

    @property
    def delta(self) -> float:
        return self.remote_volume - self.local_volume

    def to_dict(self) -> dict:
        return {
            "vt_symbol": self.vt_symbol,
            "local_volume": self.local_volume,
            "remote_volume": self.remote_volume,
            "delta": self.delta,
            "kind": self.kind,
        }


@dataclass
class ReconcileReport:
    """对账报告。"""

    ok: bool = True
    checked: int = 0
    diffs: List[PositionDiff] = field(default_factory=list)
    account_ok: bool = True
    local_equity: Optional[float] = None
    remote_equity: Optional[float] = None
    equity_delta: float = 0.0
    time: datetime = field(default_factory=lambda: datetime.now(UTC))
    note: str = ""

    def summary(self) -> str:
        if self.ok and self.account_ok:
            return f"对账通过：{self.checked} 个合约一致"
        parts = []
        if not self.ok:
            parts.append(f"{len(self.diffs)} 个合约持仓不一致")
        if not self.account_ok:
            parts.append(f"权益差 {self.equity_delta:,.2f} 元")
        return "对账失败：" + "；".join(parts)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok and self.account_ok,
            "position_ok": self.ok,
            "account_ok": self.account_ok,
            "checked": self.checked,
            "diffs": [d.to_dict() for d in self.diffs],
            "local_equity": self.local_equity,
            "remote_equity": self.remote_equity,
            "equity_delta": round(self.equity_delta, 2),
            "time": self.time.isoformat(),
            "summary": self.summary(),
            "note": self.note,
        }


def reconcile_positions(
    local: Dict[str, PositionData],
    remote: Dict[str, PositionData],
    tolerance: float = 1e-6,
) -> ReconcileReport:
    """比对本地推算持仓与网关查询持仓。

    零持仓视为「不存在」，因此本地 0 / 远端无记录不算差异。
    """
    report = ReconcileReport()
    keys = set(local) | set(remote)
    for vt in sorted(keys):
        lv = local[vt].volume if vt in local else 0.0
        rv = remote[vt].volume if vt in remote else 0.0
        if abs(lv) < tolerance and abs(rv) < tolerance:
            continue
        report.checked += 1
        if abs(lv - rv) <= tolerance:
            continue
        if abs(lv) < tolerance:
            kind = "MISSING_LOCAL"
        elif abs(rv) < tolerance:
            kind = "MISSING_REMOTE"
        else:
            kind = "MISMATCH"
        report.diffs.append(PositionDiff(vt, lv, rv, kind))
    report.ok = not report.diffs
    if not report.ok:
        for d in report.diffs:
            _logger.error(
                "[RECONCILE] %s 持仓不一致：本地 %s / 网关 %s（差 %s）",
                d.vt_symbol, d.local_volume, d.remote_volume, d.delta,
            )
    return report


def reconcile_account(
    report: ReconcileReport,
    local_equity: Optional[float],
    remote_equity: Optional[float],
    tolerance: float = 1.0,
    tolerance_ratio: Optional[float] = 0.001,
) -> ReconcileReport:
    """在已有报告上追加资金对账（容差取绝对值与比例中较大者）。"""
    report.local_equity = local_equity
    report.remote_equity = remote_equity
    if local_equity is None or remote_equity is None:
        return report
    delta = remote_equity - local_equity
    report.equity_delta = delta
    tol = tolerance
    if tolerance_ratio is not None and remote_equity:
        tol = max(tol, abs(remote_equity) * tolerance_ratio)
    report.account_ok = abs(delta) <= tol
    if not report.account_ok:
        _logger.error(
            "[RECONCILE] 权益不一致：本地 %.2f / 网关 %.2f（差 %.2f，容差 %.2f）",
            local_equity, remote_equity, delta, tol,
        )
    return report


def reconcile(
    local_positions: Dict[str, PositionData],
    remote_positions: Dict[str, PositionData],
    local_equity: Optional[float] = None,
    remote_equity: Optional[float] = None,
    tolerance: float = 1e-6,
    equity_tolerance: float = 1.0,
    risk_engine=None,
    halt_on_mismatch: bool = True,
) -> ReconcileReport:
    """一站式对账：持仓 + 资金，失败时可自动触发 SOFT 熔断。

    ``risk_engine`` 传入 :class:`~quantmind.risk.engine.RiskEngine` 时，
    对账不通过会调用 ``halt(level="SOFT")``——**只禁开仓，允许平仓**，
    避免在状态不明时继续加仓。
    """
    report = reconcile_positions(local_positions, remote_positions, tolerance)
    reconcile_account(report, local_equity, remote_equity, equity_tolerance)
    failed = not (report.ok and report.account_ok)
    if failed and halt_on_mismatch and risk_engine is not None:
        risk_engine.halt(f"对账失败：{report.summary()}", level="SOFT")
        report.note = "已触发 SOFT 熔断（禁开仓），需人工确认后 resume()"
    return report
