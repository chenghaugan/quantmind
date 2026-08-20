"""知识库（KnowledgeStore）测试：落库 / 检索 / 列表 / 空库 / 方法论 / 领域知识增强。

所有用例使用 ``tmp_path`` 指向的独立 SQLite 文件，避免污染默认库。
"""
from __future__ import annotations

import asyncio

import pytest

from quantmind.knowledge import KnowledgeStore
from quantmind.knowledge.seeds import ensure_seed_data
from quantmind.ai.provider import MockProvider
from quantmind.ai.knowledge_enrichment import KnowledgeBrief, enrich_idea
from quantmind.ai.factor_gen import generate_factors


def _seed(store: KnowledgeStore) -> None:
    store.ingest_factor(
        name="动量20", expression="delta(close,20)", idea="螺纹钢动量",
        ic=0.05, ir=0.3, status="verified",
        symbols=["S0", "S1"], asset_class="期货", market="SHFE",
    )
    store.ingest_factor(
        name="波动率", expression="std(close,20)", idea="低波动",
        ic=0.02, ir=0.1, status="pending",
        symbols=["S0"], asset_class="股票", market="CN",
    )
    store.ingest_strategy(
        code="def algo():\n    return []", code_safe=True,
        idea="螺纹钢动量", composite_scheme="icir", composite_sharpe=0.8,
        symbols=["S0", "S1"],
    )
    store.ingest_research_log(
        idea="螺纹钢动量研究",
        hypotheses=[{"id": "h1", "statement": "动量有效", "status": "VERIFIED"}],
        evidence={"verified_exprs": ["delta(close,20)"]},
    )


def test_ingest_all_kinds(tmp_path):
    """三类对象（factor/strategy/research_log）各自返回非空 kb_id。"""
    store = KnowledgeStore(db_path=str(tmp_path / "kb.db"))
    fid = store.ingest_factor(
        name="动量", expression="delta(close,5)", idea="测试", symbols=["S0"])
    sid = store.ingest_strategy(
        code="def algo():\n    return []", code_safe=True, idea="测试")
    rid = store.ingest_research_log(idea="测试")

    assert isinstance(fid, str) and fid
    assert isinstance(sid, str) and sid
    assert isinstance(rid, str) and rid
    assert len({fid, sid, rid}) == 3


def test_search_returns_hits(tmp_path):
    """search 返回 list，每条含 kb_id/kind/text/score，score 为数值。"""
    store = KnowledgeStore(db_path=str(tmp_path / "kb.db"))
    _seed(store)
    hits = store.search("动量", top_k=5)
    assert isinstance(hits, list)
    assert len(hits) > 0
    for hit in hits:
        assert "kb_id" in hit
        assert "kind" in hit
        assert "text" in hit
        assert "score" in hit
        assert isinstance(hit["score"], (int, float))


def test_search_kind_filter(tmp_path):
    """kind 过滤后只返回该种类的结果。"""
    store = KnowledgeStore(db_path=str(tmp_path / "kb.db"))
    _seed(store)
    hits = store.search("动量", kind="factor")
    assert isinstance(hits, list)
    assert len(hits) > 0
    assert all(h["kind"] == "factor" for h in hits)


def test_list_items(tmp_path):
    """list_items 返回非空 list，每条含 kb_id/kind/text。"""
    store = KnowledgeStore(db_path=str(tmp_path / "kb.db"))
    _seed(store)
    items = store.list_items()
    assert isinstance(items, list)
    assert len(items) > 0
    for item in items:
        assert "kb_id" in item
        assert "kind" in item
        assert "text" in item


def test_empty_db(tmp_path):
    """全新空库：search 与 list_items 均返回空 list。"""
    store = KnowledgeStore(db_path=str(tmp_path / "empty.db"))
    assert store.search("xx") == []
    assert store.list_items() == []


# ---------------------------------------------------------------------------
# 方法论（methodology）扩类测试
# ---------------------------------------------------------------------------
def test_ingest_methodology(tmp_path):
    """methodology 类型可落库，返回非空 kb_id。"""
    store = KnowledgeStore(db_path=str(tmp_path / "kb.db"))
    mid = store.ingest_methodology(
        title="缠论第三类买点", concept="回抽不破中枢上沿ZG",
        summary="中枢形成后的次级别买点", content="第一/二/三类买点",
        source="seed", tags=["技术分析", "缠论"],
    )
    assert isinstance(mid, str) and mid


def test_search_methodology_kind(tmp_path):
    """kind='methodology' 过滤只返回方法论，且字段齐全。"""
    store = KnowledgeStore(db_path=str(tmp_path / "kb.db"))
    store.ingest_methodology(title="缠论第三类买点", concept="回抽不破中枢",
                             summary="次级别买点", source="seed",
                             tags=["缠论"])
    store.ingest_factor(name="动量20", expression="delta(close,20)", idea="动量")
    hits = store.search("缠论", kind="methodology")
    assert len(hits) > 0
    assert all(h["kind"] == "methodology" for h in hits)
    meta = hits[0]["metadata"]
    assert meta["title"] == "缠论第三类买点"
    assert meta["tags"] == ["缠论"]


def test_list_methodology_unique_kind(tmp_path):
    """list_items(kind='methodology') 只列出方法论条目。"""
    store = KnowledgeStore(db_path=str(tmp_path / "kb.db"))
    store.ingest_methodology(title="海龟交易法则", concept="唐奇安突破",
                             source="seed")
    ensure_seed_data(store)  # 幂等：不重复写入同名 title
    items = store.list_items(kind="methodology")
    assert items, "应有方法论条目"
    assert all(i["kind"] == "methodology" for i in items)
    titles = [i["metadata"]["title"] for i in items]
    assert titles.count("海龟交易法则") == 1  # 幂等：同名 title 不重复写


def test_seed_data_idempotent(tmp_path):
    """ensure_seed_data 幂等：二次调用不重复写入。"""
    store = KnowledgeStore(db_path=str(tmp_path / "kb.db"))
    first = ensure_seed_data(store)
    second = ensure_seed_data(store)
    assert first >= 5                     # 内置至少 5-6 条
    assert second == 0                    # 已存在同名 title → 不重复写
    items = store.list_items(kind="methodology")
    assert len(items) == first


def test_methodology_does_not_break_legacy(tmp_path):
    """扩类后，原有三类行为完全不变（回归门槛）。"""
    store = KnowledgeStore(db_path=str(tmp_path / "kb.db"))
    _seed(store)
    store.ingest_methodology(title="布林带", concept="均值回归", source="seed")
    # 原三类不受影响
    assert all(h["kind"] == "factor" for h in store.search("动量", kind="factor"))
    items = store.list_items()
    kinds = {i["kind"] for i in items}
    assert {"factor", "strategy", "research_log"} <= kinds


# ---------------------------------------------------------------------------
# 领域知识增强：enrich_idea 降级与提炼测试（MockProvider 离线）
# ---------------------------------------------------------------------------
class _BriefProvider(MockProvider):
    """返回固定 KnowledgeBrief JSON 的 stub provider（模拟真实 LLM）。"""

    def __init__(self, brief_json: str) -> None:
        self.brief_json = brief_json

    async def chat(self, system: str, user: str) -> str:
        return self.brief_json


_CANON_BRIEF = (
    '{"concept":"缠论第三类买点",'
    '"definition":"回抽不重新进入中枢区间（不跌破中枢上沿ZG）",'
    '"buy_signal_rules":["回抽不破ZG则看多"],'
    '"candidate_factors":[{"kind":"momentum","reason":"趋势延续"}]}'
)


def test_enrich_idea_online_parses(tmp_path):
    """在线（真实验结果）：能解析出 LLM 的 KnowledgeBrief 字段。"""
    store = KnowledgeStore(db_path=str(tmp_path / "kb.db"))
    store.ingest_methodology(title="缠论", concept="中枢", summary="三类买点",
                             content="回抽不破ZG", source="seed")
    brief = asyncio.run(enrich_idea(
        _BriefProvider(_CANON_BRIEF), "缠论第三类买点", kb=store, web=False))
    assert brief.concept == "缠论第三类买点"
    assert "回抽" in brief.definition
    assert brief.buy_signal_rules == ["回抽不破ZG则看多"]
    assert brief.candidate_factors[0]["kind"] == "momentum"
    assert "seed" in brief.sources       # 溯源含库内 source
    assert brief.kb_hits                 # kb_hits 填充库内命中文本


def test_enrich_idea_offline_degrades(tmp_path):
    """离线（无库内命中 + 无网络）：降级为启发式，不崩。"""
    store = KnowledgeStore(db_path=str(tmp_path / "kb.db"))  # 空库
    brief = asyncio.run(enrich_idea(
        MockProvider(), "缠论第三类买点", kb=store, web=False))
    # 启发式降级：candidate_factors 至少含 momentum / mean_reversion 两个方向
    kinds = {c["kind"] for c in brief.candidate_factors}
    assert {"momentum", "mean_reversion"} <= kinds
    assert brief.concept                        # concept 回退为 idea
    assert len(brief.kb_hits) == 0              # 空库无命中


def test_enrich_idea_with_kb_hits_degrades_def(tmp_path):
    """离线但有库内命中：definition 回退为命中首条 summary。"""
    store = KnowledgeStore(db_path=str(tmp_path / "kb.db"))
    store.ingest_methodology(title="海龟交易法则", concept="唐奇安突破",
                             summary="唐奇安通道突破入场", source="seed")
    brief = asyncio.run(enrich_idea(
        MockProvider(), "海龟交易法则", kb=store, web=False))
    assert "唐奇安" in brief.definition        # 命中首条 summary 作为 definition
    assert brief.sources == ["seed"]


# ---------------------------------------------------------------------------
# generate_factors 带/不带 knowledge（MockProvider 下确定性）
# ---------------------------------------------------------------------------
def test_generate_factors_without_knowledge():
    """无 knowledge：保持原行为（回归门槛）。"""
    fs = asyncio.run(generate_factors(MockProvider(), "螺纹钢动量"))
    assert fs and len(fs) > 0
    assert all(f.kind in
               {"momentum", "mean_reversion", "volatility", "volume_change",
                "open_interest_change", "term_structure"} for f in fs)


def test_generate_factors_with_knowledge():
    """带 knowledge：注入背景后仍产出合法 FactorSpec（确定性）。"""
    kb = KnowledgeBrief(
        concept="缠论3买", definition="回抽不破中枢上沿ZG",
        buy_signal_rules=["回抽不破ZG则看多"],
        candidate_factors=[{"kind": "momentum", "reason": "趋势延续"}],
    )
    fs = asyncio.run(generate_factors(MockProvider(), "缠论第三类买点", knowledge=kb))
    assert fs and len(fs) > 0
    for f in fs:
        assert f.kind in {"momentum", "mean_reversion", "volatility",
                          "volume_change", "open_interest_change",
                          "term_structure"}
        assert f.window > 0


def test_e2e_run_roundtrip_and_trials(tmp_path, monkeypatch):
    """e2e 运行历史：run 统计 + 因子试验明细（含丰富指标）可读回。"""
    from quantmind.knowledge import KnowledgeStore

    k = KnowledgeStore(str(tmp_path / "kb.db"))
    rid = "run-20240101-abc"
    k.start_e2e_run(rid, "螺纹钢动量", "期货", "", ["rb0"], "", "", "co", 3, 2)
    k.ingest_factor_trial(rid, "delta(close,10)", "co", "s", 0.05, 0.03, 0.02,
                          1.2, 0.05, -0.02, True, "verified", "强", [])
    k.ingest_factor_trial(rid, "mean(close,20)-close", "co", "s", -0.02, 0.0, -0.01,
                          -0.5, -0.02, -0.01, False, "rejected", "弱", ["e1"])
    k.finish_e2e_run(rid, n_representative=2, n_verified_hypotheses=1,
                     composite_scheme="icir", composite_fwd_ic=0.04,
                     composite_sharpe=1.1, brief="经验", status="done")

    runs = k.list_runs()
    assert len(runs) == 1
    m = runs[0]["metadata"]
    assert m["composite_fwd_ic"] == 0.04 and m["composite_sharpe"] == 1.1
    assert m["composite_scheme"] == "icir" and m["brief"] == "经验"
    assert m["n_representative"] == 2 and m["n_verified_hypotheses"] == 1

    tr = k.trials_for_run(rid)
    assert len(tr) == 2
    rep = [t for t in tr if t["metadata"]["is_representative"]]
    assert len(rep) == 1 and rep[0]["metadata"]["test_sharpe"] == 1.2
    # get_run 一致
    assert k.get_run(rid)["metadata"]["composite_fwd_ic"] == 0.04
    assert k.get_run("不存在") is None
