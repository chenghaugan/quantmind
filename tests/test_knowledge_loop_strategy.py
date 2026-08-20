"""knowledge_loop 策略级 AI 沉淀闭环测试。

覆盖（全部离线，MockProvider + 规则兜底 + stub store）：
    - judge_strategy：sharpe=0.8,state=PAPER → verified；sharpe=0.3 → rejected(夏普)；
      state=RESEARCH 无指标 → active。
    - summarize_strategy_experience：返回 4 字段且 brief 非空中文段落。
    - strategy_kb_context：stub store 1 成功 1 失败 → ctx.success/fail 正确；空 store → 空结构。
    - run_strategy_knowledge_loop：2 条记录 → judged 长度 2、brief 非空。
"""
from __future__ import annotations

import pytest

from quantmind.ai.provider import MockProvider
from quantmind.research.knowledge_loop import (
    format_strategy_kb_context,
    judge_strategy,
    run_strategy_knowledge_loop,
    strategy_kb_context,
    summarize_strategy_experience,
)


@pytest.fixture
def provider() -> MockProvider:
    return MockProvider()


@pytest.mark.asyncio
async def test_judge_strategy_verified(provider):
    """sharpe=0.8 ≥ 0.5，state=PAPER → verified。"""
    strat = {
        "run_id": "strat_001",
        "state": "PAPER",
        "sharpe": 0.8,
        "max_drawdown": -0.15,
        "composite_fwd_ic": 0.03,
        "status": "backtested",
    }
    res = await judge_strategy(provider, strat)
    assert res["status"] == "verified"
    assert isinstance(res["reason"], str) and res["reason"]
    assert isinstance(res["tags"], list) and len(res["tags"]) >= 1
    assert "paper" in res["tags"]


@pytest.mark.asyncio
async def test_judge_strategy_rejected_low_sharpe(provider):
    """sharpe=0.3 < 0.5 → rejected，reason 含「夏普」。"""
    strat = {"run_id": "strat_002", "state": "BACKTEST", "sharpe": 0.3, "max_drawdown": -0.1}
    res = await judge_strategy(provider, strat)
    assert res["status"] == "rejected"
    assert "夏普" in res["reason"]


@pytest.mark.asyncio
async def test_judge_strategy_rejected_drawdown(provider):
    """max_drawdown 超限（回撤过深）→ rejected，reason 含「回撤」。"""
    strat = {"run_id": "strat_003", "state": "BACKTEST", "sharpe": 0.8, "max_drawdown": -0.5}
    res = await judge_strategy(provider, strat)
    assert res["status"] == "rejected"
    assert "回撤" in res["reason"]


@pytest.mark.asyncio
async def test_judge_strategy_active_no_metrics(provider):
    """state=RESEARCH 且无 sharpe → active（未回测）。"""
    strat = {"run_id": "strat_004", "state": "RESEARCH", "status": "ideation"}
    res = await judge_strategy(provider, strat)
    assert res["status"] == "active"
    assert "未回测" in res["reason"] or "尚无" in res["reason"]


@pytest.mark.asyncio
async def test_judge_strategy_live_state(provider):
    """state=LIVE，sharpe 达标 → verified 且 tags 含《live》。"""
    strat = {"run_id": "strat_005", "state": "LIVE", "sharpe": 1.5}
    res = await judge_strategy(provider, strat)
    assert res["status"] == "verified"
    assert "live" in res["tags"]


@pytest.mark.asyncio
async def test_summarize_strategy_experience(provider):
    """返回 4 字段且 brief 为可读中文段落，非空。"""
    strategies = [
        {
            "run_id": "strat_001",
            "state": "PAPER",
            "status": "verified",
            "sharpe": 0.8,
            "reason": "夏普达标，通过晋升门。",
        },
        {
            "run_id": "strat_002",
            "state": "BACKTEST",
            "status": "rejected",
            "sharpe": 0.3,
            "reason": "夏普不达标，被拒。",
        },
    ]
    res = await summarize_strategy_experience(provider, strategies, idea="动量策略")
    for key in ("brief", "effective_patterns", "failure_traps", "next_suggestions"):
        assert key in res
    assert isinstance(res["brief"], str) and res["brief"].strip()
    assert any("\u4e00" <= ch <= "\u9fff" for ch in res["brief"])
    assert "strat_001" in res["effective_patterns"]
    assert any("夏普" in t for t in res["failure_traps"])


@pytest.mark.asyncio
async def test_summarize_strategy_experience_empty(provider):
    """空记录 → brief 仍为非空中文段落（不计策略数）。"""
    res = await summarize_strategy_experience(provider, [], idea="空")
    assert isinstance(res["brief"], str) and res["brief"].strip()


@pytest.mark.asyncio
async def test_strategy_kb_context_nonempty():
    """stub store 返回 1 成功 1 失败 → ctx.success/fail 正确，format 非空。"""
    store = _StubStrategyStore(
        success=[{"strategy_id": "strat_001", "state": "PAPER", "sharpe": 0.8}],
        failed=[{"strategy_id": "strat_002", "state": "BACKTEST", "reason": "夏普不达标"}],
    )
    ctx = await strategy_kb_context(store, idea="动量")
    assert any("strat_001" in s for s in ctx["success"])
    assert any("strat_002" in f for f in ctx["fail"])
    txt = format_strategy_kb_context(ctx)
    assert "历史已验证策略" in txt
    assert "历史被拒策略" in txt


@pytest.mark.asyncio
async def test_strategy_kb_context_empty():
    """空 store → 空结构，format 返回空串。"""
    store = _StubStrategyStore(success=[], failed=[])
    ctx = await strategy_kb_context(store, idea="无")
    assert ctx == {"success": [], "fail": [], "briefs": []}
    assert format_strategy_kb_context(ctx) == ""


@pytest.mark.asyncio
async def test_strategy_kb_context_missing_methods():
    """store 缺方法（getattr 降级）→ 安全返回空结构，不抛异常。"""
    store = object()
    ctx = await strategy_kb_context(store, idea="无")
    assert ctx == {"success": [], "fail": [], "briefs": []}


@pytest.mark.asyncio
async def test_run_strategy_knowledge_loop(provider):
    """2 条策略记录 → judged 长度 2、brief 非空。"""
    records = [
        {"run_id": "strat_001", "state": "PAPER", "sharpe": 0.8, "max_drawdown": -0.15},
        {"run_id": "strat_002", "state": "BACKTEST", "sharpe": 0.3, "max_drawdown": -0.1},
    ]
    store = _StubStrategyStore(success=[], failed=[])
    res = await run_strategy_knowledge_loop(store, provider, records, idea="动量")
    assert len(res["judged"]) == 2
    for j in res["judged"]:
        assert "status" in j and "reason" in j and "tags" in j
    assert isinstance(res["brief"], str) and res["brief"].strip()
    assert set(["effective_patterns", "failure_traps", "next_suggestions"]).issubset(res.keys())


@pytest.mark.asyncio
async def test_run_strategy_knowledge_loop_empty(provider):
    """无策略记录 → judged=[]、brief=「本次无策略记录」。"""
    store = _StubStrategyStore(success=[], failed=[])
    res = await run_strategy_knowledge_loop(store, provider, [], idea="无")
    assert res["judged"] == []
    assert res["brief"] == "本次无策略记录"


class _StubStrategyStore:
    """策略级知识库占位实现（store.successful_strategies/failed_strategies 尚由另一任务加入）。

    测试内用 stub 而非真正动 store.py，本模块可独立离线测试。
    """

    def __init__(self, success=None, failed=None):
        self._success = success or []
        self._failed = failed or []

    def successful_strategies(self, idea="", statuses=("verified", "paper", "backtested"),
                              top_k=20):
        return self._success[:top_k]

    def failed_strategies(self, idea="", statuses=("rejected",), top_k=20):
        return self._failed[:top_k]

    def list_items(self, kind=None, limit=50):
        return []
