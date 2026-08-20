"""方法论知识层：三态 enrich_idea + meta 回填 + 忠实因子 kind 的回归测试。

覆盖「前置知识库 → 联网 → 不懂就回问用户」的通用机制：
  - methodology.meta 机器可读字段（implementable/kind/operator）落库与回填
  - enrich_idea 三态：可实现(ok) / 需澄清(needs_input) / 启发式降级
  - 已注册可实现方法论强制可实现的优先级
  - chan_third_buy 因子 kind 的表达式映射与求值
  - 无真实资料时不编造（can_implement=False）
"""
import asyncio
import os
import tempfile

import pytest

from quantmind.ai.knowledge_enrichment import enrich_idea
from quantmind.ai.provider import LLMProvider
from quantmind.knowledge import KnowledgeStore
from quantmind.knowledge.seeds import ensure_seed_data


class FakeProvider(LLMProvider):
    """可控 provider：返回给定 JSON 或抛错。"""

    def __init__(self, text="", exc=None):
        self._text = text
        self._exc = exc

    async def chat(self, *a, **k):
        if self._exc is not None:
            raise self._exc
        return self._text


def _kb():
    d = tempfile.mkdtemp()
    return KnowledgeStore(os.path.join(d, "kb.db"))


async def _enrich(provider, idea, kb, web=False):
    return await enrich_idea(provider, idea, kb=kb, web=web)


# -- meta 落库 / 回填 --------------------------------------------------------
def test_ingest_methodology_with_meta_and_backfill():
    kb = _kb()
    kid = kb.ingest_methodology("黄金分割线", "概念", "摘要", "内容", "user",
                                ["黄金"], meta={"implementable": True, "kind": "mean_reversion"})
    h = kb.search("黄金分割线", top_k=1, kind="methodology")
    assert h[0]["metadata"]["meta"]["kind"] == "mean_reversion"

    # 回填：旧种子无 meta → update_methodology_meta 补齐
    kb.ingest_methodology("缠论第三类买点", "c", "s", "k", "seed", ["缠论"])
    assert kb.update_methodology_meta("缠论第三类买点", {"implementable": True, "kind": "chan_third_buy"})
    h2 = kb.search("缠论第三类买点", top_k=1, kind="methodology")
    assert h2[0]["metadata"]["meta"]["kind"] == "chan_third_buy"
    # 无匹配 title 返回 False
    assert kb.update_methodology_meta("不存在的方法论", {"implementable": True}) is False


def test_ensure_seed_data_backfills_meta_on_existing_row():
    kb = _kb()
    # 预置旧版缠论 seed（无 meta），模拟历史库
    kb.ingest_methodology("缠论第三类买点", "c", "s", "k", "seed", ["缠论"])
    n = ensure_seed_data(kb)
    h = kb.search("缠论第三类买点", top_k=1, kind="methodology")
    assert (h[0]["metadata"] or {}).get("meta", {}).get("kind") == "chan_third_buy"
    # 新 seed（黄金分割线）也入库且带 meta
    g = kb.search("黄金分割线", top_k=1, kind="methodology")
    assert g and (g[0]["metadata"] or {}).get("meta", {}).get("implementable") is True


# -- 三态 enrich_idea ---------------------------------------------------------
def test_enrich_needs_input_when_no_source_and_llm_cannot_implement():
    kb = _kb()
    prov = FakeProvider(
        '{"concept":"某冷门法","definition":"","buy_signal_rules":[],'
        '"candidate_factors":[],"can_implement":false,'
        '"missing":["缺定义","缺计算规则"]}'
    )
    brief = asyncio.run(_enrich(prov, "某冷门法", kb))
    assert brief.can_implement is False
    assert brief.missing                                            # 列出缺什么
    assert brief.to_dict()["can_implement"] is False


def test_enrich_needs_input_when_absolutely_no_source():
    kb = _kb()
    prov = FakeProvider('{"concept":"x","definition":"虚构造","buy_signal_rules":[],"candidate_factors":[]}')
    brief = asyncio.run(_enrich(prov, "不可名状的绝密策略", kb))      # 空库 + 无网络
    assert brief.can_implement is False


def test_enrich_ok_when_registered_implementable_seed():
    kb = _kb()
    kb.ingest_methodology("缠论第三类买点", "概念", "摘要", "内容", "seed",
                          ["缠论"], meta={"implementable": True, "kind": "chan_third_buy"})
    # 即使 LLM 误报 can_implement=false，已注册的可实现方法论仍被强制标记为可实现
    prov = FakeProvider('{"concept":"缠论第三类买点","definition":"d","candidate_factors":[],'
                        '"can_implement":false,"missing":["x"]}')
    brief = asyncio.run(_enrich(prov, "缠论第三类买点", kb))
    assert brief.can_implement is True
    assert brief.missing == []


def test_enrich_heuristic_offline_with_no_source_says_cannot_implement():
    kb = _kb()
    prov = FakeProvider(exc=RuntimeError("offline"))
    brief = asyncio.run(_enrich(prov, "离线且无资料的冷门法", kb))
    assert brief.can_implement is False            # 离线 + 无库内种子 → 回问用户，而非编造
    assert brief.missing


def test_enrich_heuristic_with_seed_can_implement():
    kb = _kb()
    kb.ingest_methodology("威科夫量价分析", "概念", "威科夫四阶段", "内容", "seed", ["威科夫"])
    brief = asyncio.run(_enrich(FakeProvider(exc=RuntimeError("offline")), "威科夫", kb))
    assert brief.can_implement is True             # 有库内种子 → 按种子实现（不触发回问）


# -- 忠实因子 kind -----------------------------------------------------------
def test_expr_map_chan_third_buy_maps():
    from quantmind.ai.expr_map import factor_spec_to_expression
    from quantmind.research.target import FactorSpec
    spec = FactorSpec(name="ctb", kind="chan_third_buy", window=12, weight=1.0)
    expr = factor_spec_to_expression(spec)
    assert "delta" in expr and "ts_max" in expr and "12" in expr


# -- agent 护栏：can_implement=False → 短路，不编造因子 -----------------------
def test_generic_learning_loop_for_arbitrary_idea():
    """通用性证明：一个与缠论/黄金完全无关的全新 idea，不靠任何硬编码也能跑通。

    第一次：库内/网络无资料 → needs_input（回问用户）；
    用户补充信息入库(source=user) → 下次直接可理解，无需任何预设 kind。
    """
    idea = "某另类定增折价拥挤度新术语"     # 与缠论/黄金无关的任意 idea

    # 1) 无资料 → 需澄清
    prov = FakeProvider('{"concept":"x","candidate_factors":[],'
                        '"can_implement":false,"missing":["缺定义"]}')
    brief = asyncio.run(_enrich(prov, idea, _kb()))
    assert brief.can_implement is False

    # 2) 用户补充 → 入库为方法论
    kb = _kb()
    kb.ingest_methodology(idea, "核心定义", "摘要", "详细内容与计算规则",
                          "user", ["方法论"])
    # 3) 同 idea 下次查询 → 可理解（基于库内资料，无 meta.implementable 强制）
    prov2 = FakeProvider('{"concept":"x","definition":"定义",'
                         '"buy_signal_rules":["规则"],'
                         '"candidate_factors":[{"kind":"momentum","reason":"r"}],'
                         '"can_implement":true,"missing":[]}')
    b2 = asyncio.run(_enrich(prov2, idea, kb))
    assert b2.can_implement is True


def test_agent_guard_shortcircuits_without_fabricating(monkeypatch):
    from quantmind.ai import agent as agent_mod
    from quantmind.ai.agent import AutoResearchAgent
    from quantmind.ai.knowledge_enrichment import KnowledgeBrief

    async def fake_enrich(provider, idea, **k):
        return KnowledgeBrief(concept=idea, definition="", can_implement=False, missing=["缺定义", "缺规则"])

    monkeypatch.setattr(agent_mod, "enrich_idea", fake_enrich)
    ag = AutoResearchAgent(provider=FakeProvider("{}"))
    out = asyncio.run(ag.research_with_evidence("某冷门法", panel=None))
    assert out.needs_input == ["缺定义", "缺规则"]
    assert out.factors == []          # 未编造因子
    assert out.to_dict()["needs_input"]
