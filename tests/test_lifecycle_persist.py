"""策略生命周期持久化 + 策略级 AI 沉淀闭环 落库端测试。

覆盖：
- store 层：upsert / update_state（部分字段更新不影响其它）/ push_transition（history 追加）
  / get / list / successful / failed（按 idea 检索）/ search(kind='lifecycle')
- promotion：构造带 store 的 LifecycleManager，阶梯晋升后 get_strategy_lifecycle 反映
  state/history；新建 manager 用同 store 后 get_or_create 能恢复（证明重启可恢复）。
所有用例使用 ``tmp_path`` 指向的独立 SQLite 文件，离线无网络。
"""
from __future__ import annotations

import pytest

from quantmind.knowledge import KnowledgeStore
from quantmind.paper.promotion import LifecycleManager, LifecycleState


# ---------------------------------------------------------------------------
# store 层
# ---------------------------------------------------------------------------
def test_upsert_and_get_lifecycle(tmp_path):
    store = KnowledgeStore(db_path=str(tmp_path / "kb.db"))
    sid = store.upsert_strategy_lifecycle(
        "dual_ma", run_id="run-1", idea="螺纹钢动量", state="RESEARCH",
        source="AI生成", code="def algo(): pass", code_safe=True,
        symbols=["rb0.SHFE"], status="active", reason="趋势延续", brief="经验brief",
    )
    assert sid == "dual_ma"

    rec = store.get_strategy_lifecycle("dual_ma")
    assert rec["strategy_id"] == "dual_ma"
    assert rec["run_id"] == "run-1"
    assert rec["idea"] == "螺纹钢动量"
    assert rec["state"] == "RESEARCH"
    assert rec["source"] == "AI生成"
    assert rec["status"] == "active"
    assert rec["reason"] == "趋势延续"
    assert rec["brief"] == "经验brief"
    # history/symbols 解析为 list
    assert rec["history"] == []
    assert rec["symbols"] == ["rb0.SHFE"]
    assert rec["metadata"]["strategy_id"] == "dual_ma"
    assert rec["metadata"]["symbols"] == ["rb0.SHFE"]


def test_get_missing_returns_none(tmp_path):
    store = KnowledgeStore(db_path=str(tmp_path / "kb.db"))
    assert store.get_strategy_lifecycle("nope") is None


def test_update_state_only_changes_given_fields(tmp_path):
    store = KnowledgeStore(db_path=str(tmp_path / "kb.db"))
    store.upsert_strategy_lifecycle(
        "s1", idea="动量", state="RESEARCH", source="AI生成",
        status="active", reason="旧原因", brief="旧brief",
    )
    store.update_strategy_state("s1", state="BACKTEST", sharpe=1.2, max_drawdown=-0.1)

    rec = store.get_strategy_lifecycle("s1")
    assert rec["state"] == "BACKTEST"
    assert rec["sharpe"] == 1.2
    assert rec["max_drawdown"] == -0.1
    # 未传字段保持不变
    assert rec["idea"] == "动量"
    assert rec["source"] == "AI生成"
    assert rec["status"] == "active"
    assert rec["reason"] == "旧原因"
    assert rec["brief"] == "旧brief"


def test_update_state_missing_row_upserts(tmp_path):
    store = KnowledgeStore(db_path=str(tmp_path / "kb.db"))
    store.update_strategy_state("ghost", state="RESEARCH", status="active")
    rec = store.get_strategy_lifecycle("ghost")
    assert rec["state"] == "RESEARCH"
    assert rec["status"] == "active"


def test_push_transition_appends_history(tmp_path):
    store = KnowledgeStore(db_path=str(tmp_path / "kb.db"))
    store.upsert_strategy_lifecycle("s1", state="IDEA")
    store.push_strategy_transition("s1", "IDEA", "RESEARCH", note="start research")
    store.push_strategy_transition("s1", "RESEARCH", "BACKTEST", note="backtest ok")

    rec = store.get_strategy_lifecycle("s1")
    assert rec["state"] == "BACKTEST"
    hist = rec["history"]
    assert len(hist) == 2
    assert hist[0]["from"] == "IDEA"
    assert hist[0]["to"] == "RESEARCH"
    assert hist[0]["note"] == "start research"
    assert hist[1]["from"] == "RESEARCH"
    assert hist[1]["to"] == "BACKTEST"
    assert hist[1]["note"] == "backtest ok"
    assert all("at" in h for h in hist)


def test_push_transition_missing_row_creates(tmp_path):
    store = KnowledgeStore(db_path=str(tmp_path / "kb.db"))
    store.push_strategy_transition("fresh", "IDEA", "RESEARCH", note="初次")
    rec = store.get_strategy_lifecycle("fresh")
    assert rec["state"] == "RESEARCH"
    assert len(rec["history"]) == 1
    assert rec["history"][0]["to"] == "RESEARCH"


def test_list_strategy_lifecycles_filter_and_order(tmp_path):
    store = KnowledgeStore(db_path=str(tmp_path / "kb.db"))
    store.upsert_strategy_lifecycle("s1", state="PAPER", source="AI生成")
    store.upsert_strategy_lifecycle("s2", state="PAPER", source="builtin")
    store.upsert_strategy_lifecycle("s3", state="REJECTED", source="AI生成")
    recs = store.list_strategy_lifecycles()
    # 缺省全部返回，按 updated_at 倒序
    assert {r["state"] for r in recs} == {"PAPER", "REJECTED"}

    paper = store.list_strategy_lifecycles(state="PAPER")
    assert all(r["state"] == "PAPER" for r in paper)

    ai = store.list_strategy_lifecycles(source="AI生成")
    assert all(r["source"] == "AI生成" for r in ai)
    assert {"s1", "s3"} <= {r["strategy_id"] for r in ai}


def test_successful_strategies(tmp_path):
    store = KnowledgeStore(db_path=str(tmp_path / "kb.db"))
    store.update_strategy_state(
        "alpha1", state="PAPER", sharpe=1.5, status="paper", reason="好",
        brief="动量强")
    store.update_strategy_state(
        "alpha2", state="BACKTEST", sharpe=0.9, status="backtested", reason="中")
    store.update_strategy_state(
        "alpha3", state="REJECTED", sharpe=-0.2, status="rejected",
        reason="样本外失效", brief="坏")
    hits = store.successful_strategies()
    assert {h["strategy_id"] for h in hits} == {"alpha1", "alpha2"}
    # 按 sharpe 降序
    sharpes = [h["sharpe"] for h in hits]
    assert sharpes == sorted(sharpes, reverse=True)
    for h in hits:
        assert "strategy_id" in h and "state" in h and "status" in h
        assert "sharpe" in h and "max_drawdown" in h and "reason" in h
        assert "metadata" in h


def test_successful_strategies_idea_filter(tmp_path):
    store = KnowledgeStore(db_path=str(tmp_path / "kb.db"))
    store.update_strategy_state(
        "动量策略", state="PAPER", sharpe=1.2, status="paper", reason="趋势")
    store.update_strategy_state(
        "均值回归", state="PAPER", sharpe=1.0, status="paper", reason="反转")
    hits = store.successful_strategies(idea="动量")
    assert len(hits) >= 1
    assert any("动量" in h["strategy_id"] or "动量" in h["reason"] for h in hits)


def test_failed_strategies(tmp_path):
    store = KnowledgeStore(db_path=str(tmp_path / "kb.db"))
    store.update_strategy_state(
        "bad1", state="REJECTED", sharpe=-0.1, status="rejected",
        reason="样本外失效")
    store.update_strategy_state(
        "good1", state="PAPER", sharpe=1.2, status="paper")
    hits = store.failed_strategies()
    assert {h["strategy_id"] for h in hits} == {"bad1"}
    assert hits[0]["reason"] == "样本外失效"


def test_search_kind_lifecycle(tmp_path):
    store = KnowledgeStore(db_path=str(tmp_path / "kb.db"))
    store.update_strategy_state(
        "双均线", state="PAPER", sharpe=1.2, status="paper", reason="趋势跟踪")
    hits = store.search("双均线", kind="lifecycle")
    assert len(hits) > 0
    assert all(h["kind"] == "lifecycle" for h in hits)
    assert hits[0]["metadata"]["strategy_id"] == "双均线"


# ---------------------------------------------------------------------------
# promotion 持久化闭环
# ---------------------------------------------------------------------------
def _step_mgr(store: KnowledgeStore, sid: str = "p1") -> LifecycleManager:
    mgr = LifecycleManager(store=store)
    # 注册即入 RESEARCH
    assert mgr.promote(sid, LifecycleState.RESEARCH, note="registered")[0] is True
    assert mgr.promote(sid, LifecycleState.BACKTEST,
                       metrics={"sharpe": 1.1, "max_drawdown": -0.12},
                       note="backtest ok")[0] is True
    assert mgr.promote(sid, LifecycleState.PAPER,
                       metrics={"sharpe": 1.0, "max_drawdown": -0.2,
                                "status": "paper"},
                       note="paper run")[0] is True
    return mgr


def test_promotion_persists_state_and_history(tmp_path):
    store = KnowledgeStore(db_path=str(tmp_path / "kb.db"))
    mgr = _step_mgr(store)

    rec = store.get_strategy_lifecycle("p1")
    assert rec["state"] == "PAPER"
    assert rec["sharpe"] == 1.0
    assert rec["max_drawdown"] == -0.2
    assert rec["status"] == "paper"
    # 每次晋升都记录一次 transition
    hist = rec["history"]
    assert len(hist) == 3
    # 初始 get_or_create 也应落了一个 IDEA 行（被 RESEARCH 覆盖前 upsert）
    # transition 序列：IDEA->RESEARCH, RESEARCH->BACKTEST, BACKTEST->PAPER
    assert hist[0]["from"] == "IDEA"
    assert hist[0]["to"] == "RESEARCH"
    assert hist[1]["to"] == "BACKTEST"
    assert hist[2]["to"] == "PAPER"


def test_promotion_restart_recover(tmp_path):
    """新 manager 用同 store 后 get_or_create 能恢复 state/history（重启可恢复）。"""
    store = KnowledgeStore(db_path=str(tmp_path / "kb.db"))
    _step_mgr(store)

    # 模拟重启：全新 manager，共用同一持久层
    mgr2 = LifecycleManager(store=store)
    rec = mgr2.get_or_create("p1")
    assert rec.state == LifecycleState.PAPER
    assert len(rec.history) == 3
    assert rec.history[0]["from"] == "IDEA"
    assert rec.history[0]["to"] == "RESEARCH"
    # metrics 从库恢复
    assert rec.metrics.get("sharpe") == 1.0
    assert rec.metrics.get("status") == "paper"


def test_promotion_no_store_still_works():
    """store 为 None 时保持纯内存，行为与旧版一致。"""
    mgr = LifecycleManager()
    ok, _ = mgr.promote("s1", LifecycleState.RESEARCH, note="r")
    assert ok is True
    assert mgr.get_or_create("s1").state == LifecycleState.RESEARCH


def test_promotion_failed_transition_not_persisted(tmp_path):
    """失败的晋升（被闸门拒绝）不写入状态/历史。"""
    store = KnowledgeStore(db_path=str(tmp_path / "kb.db"))
    mgr = LifecycleManager(store=store)
    mgr.promote("g", LifecycleState.PAPER)
    # 从 RESEARCH 尝试直接 LIVE 但夏普不足 → 被拒
    ok, reasons = mgr.promote(
        "g", LifecycleState.LIVE,
        metrics={"sharpe": 0.1, "max_drawdown": -0.1},
        note="risk_reviewed")
    assert ok is False
    rec = store.get_strategy_lifecycle("g")
    # 未被拒的 transition 不应进入 history，state 保持 PAPER
    assert rec["state"] == "PAPER"
    assert all(h["to"] != "LIVE" for h in rec["history"])
