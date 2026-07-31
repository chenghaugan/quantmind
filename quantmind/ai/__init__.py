"""AI 研究模块：想法解析、因子生成、策略代码生成、安全沙箱。

对外提供 ``ResearchAgent``：一个把「自然语言 idea -> 研究规格 -> 因子 -> 策略代码」
串起来的高层智能体，所有步骤可离线（MockProvider）运行。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

from .provider import LLMProvider, MockProvider, build_provider
from .idea_parser import ResearchSpec, parse_idea
from .factor_gen import generate_factors
from .codegen import generate_strategy_code
from .sandbox import validate_code, compile_strategy, SandboxViolation
from ..research.target import FactorSpec

_logger = logging.getLogger("quantmind.ai")


@dataclass
class ResearchOutput:
    """AI 研究的完整产出。"""

    spec: ResearchSpec
    factors: List[FactorSpec] = field(default_factory=list)
    code: str = ""
    code_safe: bool = False
    code_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "spec": self.spec.to_dict(),
            "factors": [
                {"name": f.name, "kind": f.kind, "window": f.window, "weight": f.weight}
                for f in self.factors
            ],
            "code_safe": self.code_safe,
            "code_errors": self.code_errors,
            "code": self.code[:500],
        }


class ResearchAgent:
    """研究智能体：idea -> 研究规格 -> 因子 -> 策略代码（沙箱校验）。"""

    def __init__(self, provider: LLMProvider | None = None) -> None:
        self.provider = provider or MockProvider()

    async def research(self, idea: str, asset_class: str = "") -> ResearchOutput:
        spec = await parse_idea(self.provider, idea, asset_class)
        factors = await generate_factors(self.provider, idea)
        code = await generate_strategy_code(self.provider, idea, factors)
        ok, errors = validate_code(code)
        return ResearchOutput(spec=spec, factors=factors, code=code, code_safe=ok, code_errors=errors)


__all__ = [
    "LLMProvider",
    "MockProvider",
    "build_provider",
    "ResearchSpec",
    "parse_idea",
    "generate_factors",
    "generate_strategy_code",
    "validate_code",
    "compile_strategy",
    "SandboxViolation",
    "FactorSpec",
    "ResearchAgent",
    "ResearchOutput",
]
