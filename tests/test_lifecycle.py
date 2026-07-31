"""生命周期晋升闸门测试。"""
from __future__ import annotations

import pytest

from quantmind.paper.promotion import LifecycleManager, LifecycleState, PromotionGate


def test_promote_to_live_blocked_without_metrics():
    mgr = LifecycleManager(PromotionGate(min_sharpe=0.5, max_drawdown=-0.30))
    ok, reasons = mgr.promote("s1", LifecycleState.LIVE, metrics={"sharpe": 0.1, "max_drawdown": -0.1})
    assert ok is False
    assert len(reasons) > 0


def test_promote_to_live_allowed_with_good_metrics():
    mgr = LifecycleManager(PromotionGate(min_sharpe=0.5, max_drawdown=-0.30))
    # 真实阶梯：IDEA -> PAPER -> LIVE
    ok1, _ = mgr.promote("s1", LifecycleState.PAPER)
    assert ok1 is True
    ok, reasons = mgr.promote("s1", LifecycleState.LIVE,
                              metrics={"sharpe": 1.2, "max_drawdown": -0.12},
                              note="risk_reviewed")
    assert ok is True, reasons
    assert mgr.get_or_create("s1").state == LifecycleState.LIVE


def test_promote_invalid_transition():
    mgr = LifecycleManager()
    ok, reasons = mgr.promote("s2", LifecycleState.LIVE, metrics={"sharpe": 2.0, "max_drawdown": -0.05},
                              note="risk_reviewed")
    # 从 IDEA 直接到 LIVE 不允许
    assert ok is False
