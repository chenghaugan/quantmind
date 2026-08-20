"""因子生成：把想法转化为可入库的因子规格（FactorSpec 列表）。

供研究阶段自动生成候选因子，并可直接进入 ``MultiFactorModel`` 回测。

在生成时会调用领域知识增强层产出的 :class:`~quantmind.ai.knowledge_enrichment.KnowledgeBrief`
（若有），把其精确定义 / 买入规则 / 方向性候选注入提示词，避免 LLM 凭训练记忆
把「缠论3买」等专精概念定义错；无 knowledge 时保持原行为（回归门槛）。
"""
from __future__ import annotations

import json
import logging
from typing import List, Optional

from ..research.target import FactorSpec
from .provider import LLMProvider
from .prompts import FACTOR_SYSTEM, factor_prompt, factor_prompt_knowledge

_logger = logging.getLogger("quantmind.ai.factor_gen")

_VALID_KINDS = {
    "momentum", "mean_reversion", "volatility",
    "volume_change", "open_interest_change", "term_structure",
    "chan_third_buy",
}


async def generate_factors(
    provider: LLMProvider,
    idea: str,
    knowledge: Optional["object"] = None,
) -> List[FactorSpec]:
    """生成因子规格列表。

    Args:
        provider: LLM Provider。
        idea: 投资想法。
        knowledge: 可选领域知识增强结果。可为 :class:`KnowledgeBrief` 实例或其
            ``to_dict()`` dict；非 None 时会把 definition / buy_signal_rules /
            candidate_factors 注入提示词。None 或无有效字段时保持原行为。
    """
    system = FACTOR_SYSTEM
    prompt = factor_prompt(idea)
    brief_dict = _brief_to_dict(knowledge)
    if brief_dict is not None:
        prompt = factor_prompt_knowledge(idea, brief_dict)
        system = (
            FACTOR_SYSTEM
            + " 已提供领域知识背景，请严格依据背景中的定义与方向性建议设计因子，"
              "不得虚构超出该背景的结构。"
        )
    try:
        resp = await provider.chat(system, prompt)
        data = json.loads(resp)
        specs: List[FactorSpec] = []
        for item in data.get("factors", []):
            kind = item.get("kind", "momentum")
            if kind not in _VALID_KINDS:
                kind = "momentum"
            specs.append(FactorSpec(
                name=item.get("name", f"{kind}_{item.get('window', 20)}"),
                kind=kind,
                window=int(item.get("window", 20)),
                weight=float(item.get("weight", 1.0)),
            ))
        if specs:
            return specs
    except (json.JSONDecodeError, Exception) as exc:  # noqa: BLE001
        _logger.warning("因子生成解析失败，使用默认: %s", exc)
    return [
        FactorSpec(name="momentum_20", kind="momentum", window=20, weight=1.0),
        FactorSpec(name="mean_reversion_60", kind="mean_reversion", window=60, weight=-0.5),
    ]


def _brief_to_dict(knowledge) -> Optional[dict]:
    """把知识增强结果规整为 dict；无有效字段返回 None（保持原行为）。"""
    if knowledge is None:
        return None
    if hasattr(knowledge, "to_dict"):
        d = knowledge.to_dict()
    elif isinstance(knowledge, dict):
        d = dict(knowledge)
    else:
        d = None
    if not isinstance(d, dict):
        return None
    if not (d.get("definition") or d.get("buy_signal_rules")
            or d.get("candidate_factors")):
        return None
    return d
