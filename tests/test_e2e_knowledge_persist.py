"""e2e 知识库持久化 + 知识库上下文注入（C1/C2）测试。

覆盖：
    - SearchService._ingest_report：把 run+trials+brief 落库 e2e_runs / factor_trials，
      返回 run_id / judged_trials / brief；KnowledgeStore 可查询到。
    - build_chain_prompt(kb_block=...)：注入历史知识文本；不带时行为不变。
    - build_kb_block：空上下文返回空串，非空返回注入文本。

全部离线：MockProvider + 临时 SQLite KnowledgeStore(db_path)。
"""
from __future__ import annotations

import pytest

from quantmind.ai.provider import MockProvider
from quantmind.api.services.search_service import SearchService
from quantmind.knowledge import KnowledgeStore
from quantmind.research.knowledge_loop import format_kb_context
from quantmind.research.search.prompts import build_chain_prompt, build_kb_block


@pytest.fixture
def provider() -> MockProvider:
    return MockProvider()


def _minimal_run_report() -> dict:
    """构造一份最小 run_report，供 run_knowledge_loop 判读 + 落库。"""
    return {
        "idea": "螺纹钢动量",
        "summary": {
            "representative_count": 2,
            "n_verified_hypotheses": 1,
            "mean_test_ic": 0.06,
        },
        "steps": [
            {
                "expression": "delta(close, 20)",
                "train_ic": 0.05,
                "val_ic": 0.04,
                "test_ic": 0.06,
                "test_sharpe": 1.8,
                "test_mdd": -0.1,
            },
            {
                "expression": "mean(close,1)-close",
                "train_ic": 0.01,
                "test_ic": -0.02,
                "test_sharpe": -0.4,
                "test_mdd": -0.2,
            },
        ],
        "composite": {"scheme": "icir", "ic_mean": 0.055, "sharpe": 1.1},
        "evidence": {
            "hypotheses": [{"statement": "h1", "status": "verified"}],
            "factors": [{"name": "f1", "expression": "delta(close,20)"}],
            "verified_exprs": ["delta(close, 20)"],
        },
        "strategy": {"code": "class S:\n    pass", "code_safe": True},
        "pipeline": {
            "config": {"algo": "co", "rounds": 3},
            "composite": {"scheme": "icir", "ic_mean": 0.055, "sharpe": 1.1},
            "steps": [
                {"expression": "delta(close, 20)"},
                {"expression": "mean(close,1)-close"},
            ],
        },
        "knowledge": {},
    }


@pytest.mark.asyncio
async def test_ingest_report_persists_run_trials_brief(tmp_path, provider):
    """_ingest_report → e2e_runs 1 行、factor_trials ≥ 代表数、brief 非空。"""
    store = KnowledgeStore(db_path=str(tmp_path / "kb.db"))
    svc = SearchService(dm=None, provider=provider)

    out = await svc._ingest_report(
        _minimal_run_report(), idea="螺纹钢动量", symbols=["rb0.SHFE"],
        asset_class="期货", market="SHFE", store=store,
    )

    # 返回结构含 run_id / judged_trials / brief
    assert isinstance(out["run_id"], str) and out["run_id"]
    assert isinstance(out["judged_trials"], list) and len(out["judged_trials"]) == 2
    for t in out["judged_trials"]:
        for k in ("expression", "status", "reason", "tags"):
            assert k in t
    assert isinstance(out["brief"], str) and out["brief"].strip()
    assert "effective_themes" in out and "failure_traps" in out

    # e2e_runs 恰 1 行，brief 已回填
    runs = store.list_runs()
    assert len(runs) == 1
    assert runs[0]["run_id"] == out["run_id"]
    assert isinstance(out["brief"], str) and out["brief"].strip()

    # 本轮代表数 ≥ 2
    trials = store.trials_for_run(out["run_id"])
    assert len(trials) >= 2

    # finish 后 brief 非空（run metadata 含 brief，或 finish 不覆盖则 from_fallback）
    run = store.get_run(out["run_id"])
    meta = run["metadata"] or {}
    assert meta.get("n_representative") == 2
    assert meta.get("status") == "done"


@pytest.mark.asyncio
async def test_ingest_report_failure_does_not_raise(tmp_path, provider):
    """无 steps 的极端 report：run_knowledge_loop 仍返回兜底，_ingest_report 不抛异常。"""
    store = KnowledgeStore(db_path=str(tmp_path / "kb.db"))
    svc = SearchService(dm=None, provider=provider)
    report = {"summary": {}, "steps": [], "composite": {}, "evidence": {},
              "knowledge": {}}
    out = await svc._ingest_report(report, idea="空", symbols=[], store=store)
    assert "run_id" in out and "judged_trials" in out and "brief" in out


def test_build_kb_block_empty():
    """空上下文 → build_kb_block 返回空串（不污染 prompt）。"""
    assert build_kb_block(None) == ""
    assert build_kb_block({}) == ""
    assert build_kb_block({"success": [], "fail": [], "briefs": []}) == ""


def test_build_kb_block_nonempty():
    """非空上下文 → build_kb_block 返回包含历史知识与避坑的文本。"""
    ctx = {
        "success": ["delta(close,20)"],
        "fail": ["mean(close,1)-close (test_ic=-0.0100)"],
        "briefs": ["近期验证因子: delta(close,20)"],
    }
    block = build_kb_block(ctx)
    assert isinstance(block, str) and block
    assert "Historical verified factor patterns" in block
    assert "Historical failure pitfalls" in block
    assert "delta(close,20)" in block
    assert "mean(close,1)-close" in block


def test_build_chain_prompt_with_kb_block():
    """build_chain_prompt(kb_block=...) → 注入文本出现在 seed 之前。"""
    kb = build_kb_block({
        "success": ["delta(close,20)"],
        "fail": ["mean(close,1)-close"],
        "briefs": [],
    })
    prompt = build_chain_prompt(
        seed="rank(close,20)", history=[], best_expression="rank(close,20)",
        best_rank_ic=0.03, instruction="", kb_block=kb,
    )
    assert "Historical verified factor patterns" in prompt
    assert "delta(close,20)" in prompt
    assert "避免重复失败模式" in prompt or "避免重复" in prompt
    # 知识块插在 seed 之前
    assert prompt.index("delta(close,20)") < prompt.index("Seed factor")


def test_build_chain_prompt_without_kb_block_unchanged():
    """不带 kb_block → 行为与历史一致（无注入文本）。"""
    prompt = build_chain_prompt(
        seed="rank(close,20)", history=[], best_expression="rank(close,20)",
        best_rank_ic=0.03,
    )
    assert "Historical verified factor patterns" not in prompt
    assert "避免重复失败模式" not in prompt
    assert prompt.startswith("Seed factor")
