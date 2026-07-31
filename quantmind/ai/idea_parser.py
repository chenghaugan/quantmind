"""自然语言想法解析（idea -> 研究规格）。

流程：构造提示词 -> 调用 LLM Provider -> 解析 JSON -> 产出 ``ResearchSpec``。
Mock Provider 下同样返回结构一致的结果，保证离线可跑。
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from .provider import LLMProvider, MockProvider
from .prompts import RESEARCH_SYSTEM, research_prompt

_logger = logging.getLogger("quantmind.ai.idea_parser")


@dataclass
class ResearchSpec:
    """研究规格。"""

    idea: str
    asset_class: str = "期货"
    hypothesis: str = ""
    suggested_factors: List[str] = field(default_factory=list)
    risk_notes: List[str] = field(default_factory=list)
    raw: str = ""

    def to_dict(self) -> dict:
        return {
            "idea": self.idea,
            "asset_class": self.asset_class,
            "hypothesis": self.hypothesis,
            "suggested_factors": self.suggested_factors,
            "risk_notes": self.risk_notes,
        }


def _heuristic_fallback(idea: str) -> ResearchSpec:
    asset = "期货"
    if any(k in idea for k in ["期权", "option"]):
        asset = "期权"
    elif any(k in idea for k in ["港股", "hk", "腾讯", "美团"]):
        asset = "港股"
    elif any(k in idea for k in ["a股", "股票", "茅台"]):
        asset = "A股"
    return ResearchSpec(
        idea=idea,
        asset_class=asset,
        hypothesis=f"基于「{idea}」，相关因子具有稳定的预测能力",
        suggested_factors=["momentum_20", "mean_reversion_60", "volatility_20"],
        risk_notes=["前视偏差校验", "样本外验证", "流动性与滑点", "极端行情止损"],
    )


async def parse_idea(provider: LLMProvider, idea: str, asset_class: str = "") -> ResearchSpec:
    """把自然语言想法解析为 ``ResearchSpec``。"""
    try:
        resp = await provider.chat(RESEARCH_SYSTEM, research_prompt(idea, asset_class))
        data = json.loads(resp)
        return ResearchSpec(
            idea=idea,
            asset_class=data.get("asset_class", "期货"),
            hypothesis=data.get("hypothesis", ""),
            suggested_factors=data.get("suggested_factors", []),
            risk_notes=data.get("risk_notes", []),
            raw=resp,
        )
    except (json.JSONDecodeError, Exception) as exc:  # noqa: BLE001
        _logger.warning("研究规格解析失败，使用启发式兜底: %s", exc)
        return _heuristic_fallback(idea)
