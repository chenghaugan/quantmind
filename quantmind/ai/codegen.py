"""策略代码生成：把研究规格转化为可运行的策略类源码。

生成代码经 ``sandbox`` 的 AST 安全校验后才允许导入执行（防 LLM 幻觉/恶意代码）。
"""
from __future__ import annotations

import logging
from typing import List

from ..research.target import FactorSpec
from .provider import LLMProvider
from .prompts import CODE_SYSTEM, code_prompt

_logger = logging.getLogger("quantmind.ai.codegen")

_TEMPLATE = '''from quantmind.strategy.multifactor import MultiFactorStrategy
from quantmind.research.target import FactorSpec


class GeneratedStrategy(MultiFactorStrategy):
    """由 AI 生成（idea={idea!r}）。"""

    def __init__(self, context, setting=None):
        self.specs = [
{specs}
        ]
        self.threshold = {threshold}
        self.size = {size}
        self.max_pos = {max_pos}
        super().__init__(context, setting)
'''


def _spec_block(specs: List[FactorSpec]) -> str:
    lines = []
    for s in specs:
        lines.append(
            f"            FactorSpec(name={s.name!r}, kind={s.kind!r}, "
            f"window={s.window}, weight={s.weight!r}),"
        )
    return "\n".join(lines)


async def generate_strategy_code(
    provider: LLMProvider,
    idea: str,
    specs: List[FactorSpec],
    threshold: float = 0.3,
    size: int = 1,
    max_pos: float = 1.0,
) -> str:
    """生成策略源码字符串（异步，因 Provider.chat 为协程）。

    优先调用 LLM（若存在真实实现）；否则用本地模板（确定性、可安全执行）。
    """
    # 真实 Provider 可在此分支调用 provider.chat；Mock 与兜底统一用模板
    try:
        resp = await provider.chat(CODE_SYSTEM, code_prompt(idea))
        if resp.strip().startswith("from ") or "class " in resp:
            return resp
    except Exception as exc:  # noqa: BLE001
        _logger.warning("代码生成调用失败，使用模板: %s", exc)
    return _TEMPLATE.format(
        idea=idea, specs=_spec_block(specs),
        threshold=threshold, size=size, max_pos=max_pos,
    )
