"""策略生命周期 · 跨切面端到端闭环测试（任务 F4 轨道）。

覆盖「注册 → 回测 → 模拟盘 → 复用检索」全链路，全部离线：
  (a) 注册：upsert_strategy_lifecycle 落行，断言 run_id/idea/state 正确。
  (b) 回测：模拟真实回测 risk_xray，走 judge_strategy 判读并回填 lifecycle
      status —— sharpe≥0.5 → verified，sharpe<0.5 → rejected。
  (c) 模拟盘：metrics 含 sharpe/max_drawdown，走 run_strategy_knowledge_loop
      生成 brief + PAPER 状态 + status verified。
  (d) 复用检索：successful_strategies / failed_strategies / strategy_kb_context
      能检索回该策略的 run_id/idea/brief。

所有用例用 tmp_path 指向的独立 SQLite 文件 + mock provider（provider.name=="mock"
使 judge 走规则兜底），绝不真发网络 / LLM。
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from quantmind.knowledge import KnowledgeStore
from quantmind.paper.promotion import LifecycleManager, LifecycleState
from quantmind.research.knowledge_loop import (
    judge_strategy,
    run_strategy_knowledge_loop,
    strategy_kb_context,
)

#: 晋升门配置（与 API 用同一套默认值）
_STRATEGY_GATE = {"min_sharpe": 0.5, "min_drawdown": -0.30}


def _mock_provider():
    """Mock LLM provider —— name=="mock" 时 judge/brief 走规则兜底，不发网络。"""
    return SimpleNamespace(name="mock", chat=None)


def _backtest_store_lifecycle(store, sid, risk_xray):
    """复刻 API `_persist_backtest_lifecycle`：回填真实指标 + judge 判读落库。

    返回判读 dict；risk_xray 缺 sharpe 时返回 None（安全跳过）。
    """
    if (risk_xray or {}).get("return", {}).get("sharpe") is None:
        return None
    sharpe = risk_xray["return"]["sharpe"]
    mdd = (risk_xray.get("risk") or {}).get("max_drawdown")
    store.update_strategy_state(
        sid, state="BACKTEST", sharpe=sharpe, max_drawdown=mdd, status="", reason="",
    )
    judged = asyncio.run(judge_strategy(
        _mock_provider(),
        {"sharpe": sharpe, "max_drawdown": mdd, "state": "BACKTEST"},
        gate=_STRATEGY_GATE, fallback_rules=True,
    ))
    store.update_strategy_state(
        sid, status=judged.get("status"), reason=judged.get("reason"),
    )
    return judged


def _paper_store_lifecycle(store, sid, metrics, idea=""):
    """复刻 API `_persist_paper_lifecycle`：模拟盘判读 + brief 落库。"""
    loop = asyncio.run(run_strategy_knowledge_loop(
        store, _mock_provider(),
        [{
            "strategy_id": sid, "state": "PAPER",
            "sharpe": metrics.get("sharpe"),
            "max_drawdown": metrics.get("max_drawdown"),
            "status": "paper",
        }],
        idea=idea,
    ))
    judged = (loop.get("judged") or [{}])[0] if loop.get("judged") else {
        "status": "active", "reason": "", "tags": [],
    }
    store.update_strategy_state(
        sid,
        state="PAPER",  # 复刻 paper 端点 promote(...PAPER...) 先置状态
        status=judged.get("status"), reason=judged.get("reason"),
        brief=loop.get("brief") or "",
    )
    return judged, loop


# ---------------------------------------------------------------------------
# (a) 注册
# ---------------------------------------------------------------------------
def test_register_writes_lifecycle_row(tmp_path):
    store = KnowledgeStore(db_path=str(tmp_path / "kb.db"))
    sid = store.upsert_strategy_lifecycle(
        "dual_ma", run_id="r1", idea="缠论3买", state="RESEARCH",
        source="AI生成", code="def algo(): pass", code_safe=True,
    )
    assert sid == "dual_ma"
    rec = store.get_strategy_lifecycle("dual_ma")
    assert rec is not None
    assert rec["run_id"] == "r1"
    assert rec["idea"] == "缠论3买"
    assert rec["state"] == "RESEARCH"
    assert rec["strategy_id"] == "dual_ma"


# ---------------------------------------------------------------------------
# (b) 回测：judge + 判读落库
# ---------------------------------------------------------------------------
def test_backtest_verified_high_sharpe(tmp_path):
    store = KnowledgeStore(db_path=str(tmp_path / "bt.db"))
    store.upsert_strategy_lifecycle("gt", run_id="r-bt-1", idea="趋势", state="RESEARCH")
    risk_xray = {
        "return": {"sharpe": 1.2, "total": 0.35},
        "risk": {"max_drawdown": -0.10},
    }
    judged = _backtest_store_lifecycle(store, "gt", risk_xray)
    assert judged["status"] == "verified"

    rec = store.get_strategy_lifecycle("gt")
    assert rec["state"] == "BACKTEST"
    assert rec["sharpe"] == 1.2
    assert rec["max_drawdown"] == -0.10
    assert rec["status"] == "verified"
    assert rec["reason"]


def test_backtest_rejected_low_sharpe(tmp_path):
    store = KnowledgeStore(db_path=str(tmp_path / "bt2.db"))
    store.upsert_strategy_lifecycle("bd", run_id="r-bt-2", idea="反转", state="RESEARCH")
    risk_xray = {
        "return": {"sharpe": 0.2, "total": 0.05},
        "risk": {"max_drawdown": -0.05},
    }
    judged = _backtest_store_lifecycle(store, "bd", risk_xray)
    assert judged["status"] == "rejected"

    rec = store.get_strategy_lifecycle("bd")
    assert rec["state"] == "BACKTEST"
    assert rec["sharpe"] == 0.2
    assert rec["status"] == "rejected"
    assert "夏普" in rec["reason"]


def test_backtest_skips_without_sharpe(tmp_path):
    store = KnowledgeStore(db_path=str(tmp_path / "bt3.db"))
    store.upsert_strategy_lifecycle("no", run_id="r-bt-3", state="RESEARCH")
    judged = _backtest_store_lifecycle(store, "no", None)
    assert judged is None
    rec = store.get_strategy_lifecycle("no")
    assert rec["state"] == "RESEARCH"  # 未被改动
    assert rec["sharpe"] is None


# ---------------------------------------------------------------------------
# (c) 模拟盘：run_strategy_knowledge_loop → brief + PAPER + verified
# ---------------------------------------------------------------------------
def test_paper_loop_generates_brief_and_paper_state(tmp_path):
    store = KnowledgeStore(db_path=str(tmp_path / "pp.db"))
    sid = store.upsert_strategy_lifecycle("pd", run_id="r-pp-1", idea="缠论3买", state="BACKTEST")

    metrics = {"sharpe": 0.9, "max_drawdown": -0.12, "trade_count": 12}
    judged, loop = _paper_store_lifecycle(store, sid, metrics, idea="缠论3买")
    assert judged["status"] == "verified"
    assert loop.get("brief")

    rec = store.get_strategy_lifecycle(sid)
    assert rec["state"] == "PAPER"
    assert rec["status"] == "verified"
    assert rec["brief"]  # 经验 brief 非空（中文）
    assert rec["brief"].strip()


# ---------------------------------------------------------------------------
# (d) 复用检索：成功/失败策略 + strategy_kb_context
# ---------------------------------------------------------------------------
def test_reuse_retrieval_success_validates_full_loop(tmp_path):
    store = KnowledgeStore(db_path=str(tmp_path / "reuse.db"))
    # 注册
    store.upsert_strategy_lifecycle(
        "good1", run_id="r1", idea="缠论3买", state="RESEARCH", source="AI生成")
    store.upsert_strategy_lifecycle(
        "bad1", run_id="r2", idea="均值回归", state="RESEARCH", source="AI生成")
    # 回测：good1 verified，bad1 rejected
    _backtest_store_lifecycle(store, "good1", {
        "return": {"sharpe": 1.1}, "risk": {"max_drawdown": -0.08},
    })
    _backtest_store_lifecycle(store, "bad1", {
        "return": {"sharpe": 0.2}, "risk": {"max_drawdown": -0.05},
    })
    # 模拟盘：good1 → verified + brief
    _paper_store_lifecycle(store, "good1", {"sharpe": 1.0, "max_drawdown": -0.1}, idea="缠论3买")

    ok = store.successful_strategies(idea="缠论")
    assert any("good1" in (s.get("strategy_id") or "") for s in ok)
    good = next(s for s in ok if s.get("strategy_id") == "good1")
    assert good["run_id"] == "r1"
    assert good["idea"] == "缠论3买"
    assert good["brief"] and good["brief"].strip()
    assert good["status"] == "verified"

    failed = store.failed_strategies()
    assert any("bad1" in (s.get("strategy_id") or "") for s in failed)

    # strategy_kb_context 也能检索回该策略的 run_id/idea
    ctx = asyncio.run(strategy_kb_context(store, idea="缠论"))
    assert any("good1" in s for s in ctx["success"])
    assert not any("bad1" in s for s in ctx["success"])


def test_lifecycle_manager_reuse_with_store(tmp_path):
    """LifecycleManager(store=store) 配合持久化晋升后，仍能检索同一策略。"""
    store = KnowledgeStore(db_path=str(tmp_path / "mgr.db"))
    mgr = LifecycleManager(store=store)
    assert mgr.promote("strat2", LifecycleState.RESEARCH, note="registered")[0] is True
    assert mgr.promote(
        "strat2", LifecycleState.BACKTEST,
        metrics={"sharpe": 1.2, "max_drawdown": -0.12},
        note="backtest ok",
    )[0] is True
    assert mgr.promote(
        "strat2", LifecycleState.PAPER,
        metrics={"sharpe": 1.0, "max_drawdown": -0.2, "status": "paper"},
        note="paper run",
    )[0] is True

    rec = store.get_strategy_lifecycle("strat2")
    assert rec["state"] == "PAPER"
    assert rec["sharpe"] == 1.0
    hits = store.successful_strategies()
    assert any(h["strategy_id"] == "strat2" for h in hits)
