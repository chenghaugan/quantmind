"""knowledge_loop（AI 持续学习闭环）测试。

覆盖（全部离线，MockProvider + 规则兜底）：
    - judge_trial：verified 样本（test_ic>0, sharpe>0, train 同正）与 rejected 样本。
    - summarize_experience：返回四字段且 brief 非空中文段落。
    - kb_search_context：空库返回空结构，format 返回空串。
    - run_knowledge_loop：最小 run_report（2 个 step）→ trials 长度 2 且 brief 非空。
"""
from __future__ import annotations

import pytest

from quantmind.ai.provider import MockProvider
from quantmind.research.knowledge_loop import (
    format_kb_context,
    judge_trial,
    kb_search_context,
    run_knowledge_loop,
    summarize_experience,
)


@pytest.fixture
def provider() -> MockProvider:
    return MockProvider()


@pytest.mark.asyncio
async def test_judge_trial_verified(provider):
    """OOS 稳定正 IC 且 train 同正 → verified。"""
    trial = {
        "expression": "delta(close, 20)",
        "train_ic": 0.05,
        "val_ic": 0.04,
        "test_ic": 0.06,
        "test_sharpe": 1.8,
        "test_mdd": -0.12,
        "seed": 42,
    }
    res = await judge_trial(provider, trial, idea="动量")
    assert res["status"] == "verified"
    assert isinstance(res["reason"], str) and res["reason"]
    assert isinstance(res["tags"], list) and len(res["tags"]) >= 1
    assert "momentum" in res["tags"]


@pytest.mark.asyncio
async def test_judge_trial_rejected(provider):
    """test_ic 非正 → rejected（OOS 失效）。"""
    trial = {
        "expression": "-delta(close, 5)",
        "train_ic": 0.03,
        "val_ic": 0.0,
        "test_ic": -0.02,
        "test_sharpe": -0.5,
        "test_mdd": -0.3,
        "seed": 7,
    }
    res = await judge_trial(provider, trial, idea="反转")
    assert res["status"] == "rejected"
    assert "OOS" in res["reason"] or "无OOS" in res["reason"]
    assert "mean_reversion" in res["tags"]


@pytest.mark.asyncio
async def test_judge_trial_active(provider):
    """无 test 数据 → rejected（无OOS）。"""
    trial = {
        "expression": "std(close, 10)",
        "train_ic": 0.04,
        "val_ic": 0.02,
        "seed": 1,
    }
    res = await judge_trial(provider, trial)
    assert res["status"] == "rejected"
    assert "无OOS" in res["reason"] or "OOS" in res["reason"]


@pytest.mark.asyncio
async def test_summarize_experience(provider):
    """返回四字段且 brief 为可读中文段落，非空。"""
    trials = [
        {
            "expression": "delta(close, 20)",
            "status": "verified",
            "reason": "样本外稳定正IC且夏普为正。",
            "tags": ["momentum", "rank"],
            "test_ic": 0.06,
        },
        {
            "expression": "mean(close,1)-close",
            "status": "rejected",
            "reason": "OOS失效：期限结构在3个月窗口失效。",
            "tags": ["term_structure", "overfit"],
            "test_ic": -0.01,
        },
    ]
    run = {"idea": "期货期限结构", "representative_count": 2}
    res = await summarize_experience(provider, run, trials, idea="期货期限结构")
    for key in ("brief", "effective_themes", "failure_traps", "next_suggestions"):
        assert key in res
    assert isinstance(res["brief"], str) and res["brief"].strip()
    # brief 是中文段落，非空且非空字段
    assert any("\u4e00" <= ch <= "\u9fff" for ch in res["brief"])
    assert "momentum" in res["effective_themes"]
    assert any("OOS失效" in t or "期限结构" in t for t in res["failure_traps"])


def test_format_kb_context_empty():
    """空库 → format 返回空串（不污染 prompt）。"""
    assert format_kb_context({"success": [], "fail": [], "briefs": []}) == ""


@pytest.mark.asyncio
async def test_kb_search_context_empty():
    """空库（stub store 返回空）→ 返回空结构。"""
    store = _StubStore(success=[], failed=[])
    ctx = await kb_search_context(store, idea="动量")
    assert ctx == {"success": [], "fail": [], "briefs": []}
    assert format_kb_context(ctx) == ""


@pytest.mark.asyncio
async def test_kb_search_context_nonempty():
    """非空库 → 成功表达式 / 失败清单被提取。"""
    store = _StubStore(
        success=[{"expression": "delta(close,20)", "test_ic": 0.06, "status": "verified"}],
        failed=[{"expression": "mean(close,1)-close", "test_ic": -0.01, "status": "rejected"}],
    )
    ctx = await kb_search_context(store, idea="期限")
    assert "delta(close,20)" in ctx["success"]
    assert any("mean(close,1)-close" in f for f in ctx["fail"])
    txt = format_kb_context(ctx)
    assert "Historical verified factor patterns" in txt
    assert "Historical failure pitfalls" in txt


@pytest.mark.asyncio
async def test_run_knowledge_loop(provider):
    """最小 run_report（2 个 step）→ trials 长度 2 且 brief 非空。"""
    run_report = {
        "summary": {"representative_count": 2, "n_verified_hypotheses": 1},
        "steps": [
            {
                "expression": "delta(close, 20)",
                "train_ic": 0.05,
                "val_ic": 0.04,
                "test_ic": 0.06,
                "test_sharpe": 1.8,
                "test_mdd": -0.1,
                "removed_redundant": False,
            },
            {
                "expression": "-delta(close, 5)",
                "train_ic": 0.03,
                "val_ic": 0.0,
                "test_ic": -0.02,
                "test_sharpe": -0.5,
                "test_mdd": -0.2,
                "removed_redundant": False,
            },
        ],
        "composite": {},
        "evidence": {"hypotheses": [{"statement": "h1", "status": "verified"}]},
    }
    store = _StubStore(success=[], failed=[])
    res = await run_knowledge_loop(store, provider, run_report, idea="动量")
    assert len(res["trials"]) == 2
    for t in res["trials"]:
        assert "status" in t and "reason" in t and "tags" in t
    assert isinstance(res["brief"], str) and res["brief"].strip()
    assert set(["effective_themes", "failure_traps", "next_suggestions"]).issubset(res.keys())


@pytest.mark.asyncio
async def test_run_knowledge_loop_no_steps(provider):
    """无有效 steps → trials=[] 且 brief 为一句提示。"""
    store = _StubStore(success=[], failed=[])
    run_report = {"summary": {}, "steps": [], "composite": {}, "evidence": {}}
    res = await run_knowledge_loop(store, provider, run_report, idea="无试验")
    assert res["trials"] == []
    assert res["brief"] == "本次未产生可比对因子试验"


class _StubStore:
    """知识库占位实现（store.successful_factors/failed_factors 尚由另一任务加入）。

    测试内用 stub 而非真正动 store.py，保证本模块可独立测试。
    """

    def __init__(self, success=None, failed=None):
        self._success = success or []
        self._failed = failed or []

    def successful_factors(self, idea="", statuses=("verified", "passed"), top_k=20):
        return self._success[:top_k]

    def failed_factors(self, idea="", statuses=("rejected", "redundant"), top_k=30):
        return self._failed[:top_k]

    def list_items(self, kind=None, limit=50):
        return []
