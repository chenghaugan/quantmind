"""因子生成：把想法转化为可入库的因子规格（FactorSpec 列表）。

供研究阶段自动生成候选因子，并可直接进入 ``MultiFactorModel`` 回测。
"""
from __future__ import annotations

import json
import logging
from typing import List

from ..research.target import FactorSpec
from .provider import LLMProvider
from .prompts import FACTOR_SYSTEM, factor_prompt

_logger = logging.getLogger("quantmind.ai.factor_gen")

_VALID_KINDS = {
    "momentum", "mean_reversion", "volatility",
    "volume_change", "open_interest_change", "term_structure",
}


async def generate_factors(provider: LLMProvider, idea: str) -> List[FactorSpec]:
    """生成因子规格列表。"""
    try:
        resp = await provider.chat(FACTOR_SYSTEM, factor_prompt(idea))
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
