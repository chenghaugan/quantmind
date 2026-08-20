"""领域知识增强层：在 idea → 因子之前，把「知识库方法论 + 网络补充」提炼为 KnowledgeBrief。

对应「领域知识获取层」三步：
  1. :func:`enrich_idea` 先从知识库 ``kind="methodology"`` 检索命中资料；
  2. 命中不足（< 2 条）时调用 :func:`quantmind.knowledge.web_source.gather_web` 补充网络资料；
  3. 交给 LLM 把 (idea + 库内命中 + 网络资料) 提炼成严格 JSON 的
     :class:`KnowledgeBrief`（含方向性因子建议），并注入后续因子生成提示词。

解析失败 / 无有效输出时降级为**确定性启发式**（concept=idea、definition=命中首条 summary），
保证离线、可测试、绝不崩。
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from ..knowledge import KnowledgeStore
from ..knowledge import web_source
from .provider import LLMProvider
from .prompts import KNOWLEDGE_RICH_SYSTEM, knowledge_prompt

_logger = logging.getLogger("quantmind.ai.enrichment")

__all__ = ["KnowledgeBrief", "enrich_idea"]

#: LLM 必须严格产出的 JSON 字段白名单（防御虚构/额外字段）。
_BRIEF_KEYS = ("concept", "definition", "buy_signal_rules", "candidate_factors")
#: can_implement / missing 单独解析（不参与 has_content 判定，缺失也回推默认）。
_IMPLEMENT_KEYS = ("can_implement", "missing")


@dataclass
class KnowledgeBrief:
    """领域知识增强后的可因子化摘要。

    :param concept: 核心概念一句话（如「缠论第三类买点」）。
    :param definition: 精确定义（供因子生成引用）。
    :param buy_signal_rules: 可判定的买入/做多信号规则列表。
    :param candidate_factors: 方向性因子建议（含 kind 与理由）。
    :param sources: 溯源：库内 source 去重 + 网络 url。
    :param kb_hits: 库内命中的方法论原文文本。
    :param can_implement: 能否基于现有资料给出**忠实、可计算**的因子实现。
        False 表示结论不可信，应提示用户补充信息而非编造。
    :param missing: 无法实现时缺什么（回问用户时的提示项）。
    """

    concept: str
    definition: str
    buy_signal_rules: List[str] = field(default_factory=list)
    candidate_factors: List[dict] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    kb_hits: List[str] = field(default_factory=list)
    can_implement: bool = True
    missing: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """JSON 友好的 dict 表示（供序列化 / 入库 / 契约）。"""
        return {
            "concept": self.concept,
            "definition": self.definition,
            "buy_signal_rules": list(self.buy_signal_rules),
            "candidate_factors": list(self.candidate_factors),
            "sources": list(self.sources),
            "kb_hits": list(self.kb_hits),
            "can_implement": self.can_implement,
            "missing": list(self.missing),
        }


async def enrich_idea(
    provider: LLMProvider,
    idea: str,
    kb: Optional[KnowledgeStore] = None,
    web: bool = True,
    top_k: int = 4,
) -> KnowledgeBrief:
    """为想法注入领域知识：检索 + 网络补充 + LLM 提炼为 :class:`KnowledgeBrief`。

    Args:
        provider: LLM Provider（可离线 Mock）。
        idea: 投资想法。
        kb: 知识库；None → 用默认 ``KnowledgeStore()``。
        web: 是否在库内命中不足时联网补充。
        top_k: 库内方法论检索的 top_k。
    """
    kb = kb or KnowledgeStore()

    # 1) 库内方法论检索
    kb_hits: List[dict] = []
    try:
        kb_hits = kb.search(idea, top_k=top_k, kind="methodology") or []
    except Exception as exc:  # noqa: BLE001
        _logger.debug("知识库方法论检索失败: %s", exc)
        kb_hits = []

    # 2) 命中不足时联网补充
    web_hits: List[dict] = []
    if web and len(kb_hits) < 2:
        try:
            web_hits = await web_source.gather_web(idea) or []
        except Exception as exc:  # noqa: BLE001
            _logger.debug("网络资料获取失败: %s", exc)
            web_hits = []

    # 已注册且带算子/kind 的方法论：可忠实实现，优先级最高（LLM 仍可细化，但标记可实现）。
    implementable_meta = False
    if kb_hits:
        m = (kb_hits[0].get("metadata") or {}).get("meta") or {}
        implementable_meta = bool(m.get("implementable")) and bool(
            m.get("kind") or m.get("operator")
        )

    # 3) 交给 LLM 提炼；失败则降级启发式
    try:
        resp = await provider.chat(
            KNOWLEDGE_RICH_SYSTEM,
            knowledge_prompt(idea, kb_hits, web_hits),
        )
        brief = _parse_brief(resp, idea, kb_hits, web_hits)
        if brief is not None:
            return _finalize(brief, implementable_meta)
    except Exception as exc:  # noqa: BLE001
        _logger.warning("领域知识提炼失败，降级启发式: %s", exc)

    return _finalize(_heuristic_brief(idea, kb_hits, web_hits), implementable_meta)


def _finalize(brief: KnowledgeBrief, implementable_meta: bool) -> KnowledgeBrief:
    """已注册可实现的方法论强制标记可实现（不受 LLM/降级影响）。"""
    if implementable_meta:
        brief.can_implement = True
        brief.missing = []
    return brief


def _no_real_source(kb_hits: List[dict], web_hits: List[dict]) -> bool:
    """无可用真实资料（库内无命中，且网络补充无真实 url——启发式占位无 url）。"""
    if kb_hits:
        return False
    return not any(str((w or {}).get("url") or "").strip() for w in web_hits)


# ---------------------------------------------------------------------------
# LLM 结果解析 / 降级
# ---------------------------------------------------------------------------
def _parse_brief(resp: str, idea: str, kb_hits: List[dict],
                 web_hits: List[dict]) -> Optional[KnowledgeBrief]:
    """把 LLM 输出解析为 KnowledgeBrief；结构不符返回 None（触发降级）。

    判定「有效」：原始 dict 至少含有 definition / buy_signal_rules /
    candidate_factors 三者之一（即 LLM 真的依据资料给出了可因子化输出）；
    否则视为无效并触发启发式降级。
    """
    try:
        data = json.loads(resp)
    except (TypeError, ValueError):
        # 容忍被 ```json ... ``` 包裹的回复
        cleaned = _strip_code_fence(resp or "")
        try:
            data = json.loads(cleaned)
        except (TypeError, ValueError):
            return None
    if not isinstance(data, dict):
        return None
    has_content = (
        str(data.get("definition") or "").strip()
        or bool(data.get("buy_signal_rules"))
        or bool(data.get("candidate_factors"))
    )
    if not has_content:
        return None
    concept = str(data.get("concept") or idea or "").strip()
    definition = str(data.get("definition") or "").strip()
    buy_rules = _as_str_list(data.get("buy_signal_rules"))
    candidates = _as_factor_list(data.get("candidate_factors"))
    # 防御：LLM 只允许基于给定资料，不得虚构 concept/definition
    if not concept:
        return None
    # can_implement / missing：无真实资料→False（回问用户）；有资料→信任 LLM，否则默认 True。
    no_real = _no_real_source(kb_hits, web_hits)
    raw_ci = data.get("can_implement")
    if no_real:
        can_implement = False
        missing = _as_str_list(data.get("missing")) or [
            "该想法在知识库与网络均无可靠资料，无法确认其定义与量化实现方式，请补充交易方法论说明。"
        ]
    elif raw_ci is False:
        can_implement = False
        missing = _as_str_list(data.get("missing")) or [
            "现有资料不足以实现，请补充关键定义/规则/示例。"
        ]
    else:
        can_implement = True
        missing = _as_str_list(data.get("missing"))
    return KnowledgeBrief(
        concept=concept,
        definition=definition,
        buy_signal_rules=buy_rules,
        candidate_factors=candidates,
        sources=_collect_sources(kb_hits, web_hits),
        kb_hits=[str(h.get("text") or "") for h in kb_hits if h.get("text")],
        can_implement=can_implement,
        missing=missing,
    )


def _heuristic_brief(idea: str, kb_hits: List[dict], web_hits: List[dict]) -> KnowledgeBrief:
    """确定性降级：concept=idea，definition=命中首条 summary，否则空串。"""
    definition = ""
    if kb_hits:
        meta = (kb_hits[0].get("metadata") or {})
        definition = str(meta.get("summary") or meta.get("concept") or "") or ""
    # 有库内方法论可按其实现；无任何真实资料 → 回问用户。
    can_implement = bool(kb_hits)
    missing = []
    if not can_implement:
        missing = ["库内与网络均无该方法论资料，无法忠实实现，请补充其定义与量化实现要点。"]
    return KnowledgeBrief(
        concept=idea or "",
        definition=definition,
        candidate_factors=[
            {"kind": "momentum", "reason": "降级启发式：默认方向性候选"},
            {"kind": "mean_reversion", "reason": "降级启发式：默认方向性候选"},
        ],
        sources=_collect_sources(kb_hits, web_hits),
        kb_hits=[str(h.get("text") or "") for h in kb_hits if h.get("text")],
        can_implement=can_implement,
        missing=missing,
    )


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _collect_sources(kb_hits: List[dict], web_hits: List[dict]) -> List[str]:
    """溯源：库内 source 字段去重 + 网络 url（过滤空值）。"""
    seen: List[str] = []
    for h in kb_hits:
        src = str((h.get("metadata") or {}).get("source") or "").strip()
        if src and src not in seen:
            seen.append(src)
    for w in web_hits:
        url = str((w or {}).get("url") or "").strip()
        if url and url not in seen:
            seen.append(url)
    return seen


def _as_str_list(raw) -> List[str]:
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if isinstance(raw, str) and raw.strip():
        return [raw.strip()]
    return []


def _as_factor_list(raw) -> List[dict]:
    """把 candidate_factors 规整为 ``[{"kind","reason"}]``；kind 仅保留合法词。"""
    valid = {"momentum", "mean_reversion", "volatility", "volume_change",
             "open_interest_change", "term_structure", "chan_third_buy"}
    out: List[dict] = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind") or "").strip().lower()
            if kind not in valid:
                kind = "momentum"
            out.append({
                "kind": kind,
                "reason": str(item.get("reason") or "").strip(),
            })
    return out


def _strip_code_fence(s: str) -> str:
    """去掉 ```json ... ``` 包裹标记（容错 LLM 常见输出）。"""
    return s.strip().lstrip("`").rstrip("`").strip()
