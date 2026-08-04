"""自动研究智能体：多步假设循环 + 研究日志 + 假设跟踪 + 策略解释/事实表。

对应框架文档 Phase 4「自动化研究 agent / 持续学习 / 策略解释与文档」的落地。
在既有三步 ``ResearchAgent``（idea → 规格 → 因子 → 代码）之上，新增：

- **多步假设循环**：维护 ``Hypothesis`` 列表，区分 proposed / verified / rejected，
  并记录每条假设的证据（evidence）。
- **研究日志**：逐动作记录 ``ResearchLogEntry``，支持事后审计与研究者回放。
- **策略解释**：``generate_explanation`` 用自然语言综合研究结论。
- **事实表**：``generate_fact_sheet`` 产出合规/审查用结构化事实表。
- **前视护栏**：生成代码后额外跑 ``lookahead_warnings``，纳入 ``code_safe``。

全部可离线（MockProvider）运行，确定性、可测试。
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from .provider import LLMProvider, MockProvider
from .idea_parser import ResearchSpec, parse_idea
from .factor_gen import generate_factors
from .codegen import generate_strategy_code
from .sandbox import validate_code
from .safety import lookahead_warnings
from ..research.target import FactorSpec

_logger = logging.getLogger("quantmind.ai.agent")


class HypothesisStatus(str, Enum):
    PROPOSED = "proposed"
    VERIFIED = "verified"
    REJECTED = "rejected"


@dataclass
class Hypothesis:
    """单条研究假设。"""

    id: str
    statement: str
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    evidence: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "statement": self.statement,
            "status": self.status.value,
            "evidence": self.evidence,
        }


@dataclass
class ResearchLogEntry:
    """研究日志中的一步。"""

    step: str
    action: str
    input: str = ""
    output: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "action": self.action,
            "input": self.input,
            "output": self.output,
            "timestamp": round(self.timestamp, 3),
        }


@dataclass
class AutoResearchOutput:
    """自动研究智能体的完整产出。"""

    spec: ResearchSpec
    factors: List[FactorSpec] = field(default_factory=list)
    code: str = ""
    code_safe: bool = False
    code_errors: List[str] = field(default_factory=list)
    lookahead_warnings: List[str] = field(default_factory=list)
    hypotheses: List[Hypothesis] = field(default_factory=list)
    log: List[ResearchLogEntry] = field(default_factory=list)
    explanation: str = ""
    fact_sheet: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "spec": self.spec.to_dict(),
            "factors": [
                {"name": f.name, "kind": f.kind, "window": f.window, "weight": f.weight}
                for f in self.factors
            ],
            "code_safe": self.code_safe,
            "code_errors": self.code_errors,
            "lookahead_warnings": self.lookahead_warnings,
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "log": [l.to_dict() for l in self.log],
            "explanation": self.explanation,
            "fact_sheet": self.fact_sheet,
            "code": self.code[:500],
        }


class AutoResearchAgent:
    """自动多步研究智能体。

    :param provider: LLM Provider；默认 Mock。
    :param max_steps: 循环最多执行步数。
    """

    def __init__(self, provider: LLMProvider | None = None, max_steps: int = 3) -> None:
        self.provider = provider or MockProvider()
        self.max_steps = max_steps

    def _log(self, out: AutoResearchOutput, step: str, action: str,
             input: str = "", output: str = "") -> None:
        out.log.append(ResearchLogEntry(step=step, action=action, input=input, output=output))

    async def research(self, idea: str, asset_class: str = "") -> AutoResearchOutput:
        out = AutoResearchOutput(
            spec=ResearchSpec(idea=idea, asset_class=asset_class),
        )

        # 步骤 0：想法解析
        spec = await parse_idea(self.provider, idea, asset_class)
        out.spec = spec
        self._log(out, 0, "parse_idea", input=idea,
                  output=spec.to_dict().__repr__())
        _logger.info("[Agent] 已解析想法 → 资产=%s 假设=%s",
                     spec.asset_class, spec.hypothesis)

        # 建立初始假设
        h0 = Hypothesis(
            id="H0",
            statement=spec.hypothesis or f"基于「{idea}」的因子具有稳定预测能力",
            status=HypothesisStatus.PROPOSED,
            evidence="由想法解析产出",
        )
        out.hypotheses.append(h0)

        # 步骤 1：因子生成（对每个建议因子/生成因子尝试评估）
        factors = await generate_factors(self.provider, idea)
        out.factors = factors
        self._log(out, 1, "generate_factors", input=idea,
                  output=f"[{', '.join(f.name for f in factors)}]")

        # 对生成的每条因子赋予假设验证状态（离线启发式：有权重且窗口合理视为 verified）
        for i, f in enumerate(factors):
            if f.window > 0 and abs(f.weight) > 0:
                h = Hypothesis(
                    id=f"H{i+1}",
                    statement=f"因子 {f.name}(kind={f.kind}, window={f.window}) 具有预测能力",
                    status=HypothesisStatus.VERIFIED,
                    evidence="由因子生成与规格匹配启发式验证（离线）",
                )
            else:
                h = Hypothesis(
                    id=f"H{i+1}",
                    statement=f"因子 {f.name} 参数不完整，需人工复核",
                    status=HypothesisStatus.REJECTED,
                    evidence="权重或窗口无效",
                )
            out.hypotheses.append(h)

        # 步骤 2：策略代码生成 + 沙箱校验 + 前视扫描
        code = await generate_strategy_code(self.provider, idea, factors)
        out.code = code
        self._log(out, 2, "generate_strategy_code", input=idea, output=f"code_len={len(code)}")

        ok_sandbox, errors = validate_code(code)
        lookahead = lookahead_warnings(code)
        out.code_errors = list(errors)
        out.lookahead_warnings = lookahead
        out.code_safe = bool(ok_sandbox) and not lookahead
        self._log(out, 3, "validate_strategy", input=code,
                  output=f"sandbox={ok_sandbox}, lookahead={len(lookahead)}")

        # 步骤 3：生成解释与事实表
        out.explanation = generate_explanation(out)
        out.fact_sheet = generate_fact_sheet(out)
        self._log(out, 4, "generate_explanation_factsheet", input="",
                  output=f"explanation_len={len(out.explanation)}")

        return out


def generate_explanation(out: AutoResearchOutput) -> str:
    """用自然语言综合研究结论，生成人类可读的策略解释。"""
    if not out.spec:
        return ""
    factors_txt = ", ".join(f.name for f in out.factors) if out.factors else "（无候选因子）"
    verified = [h for h in out.hypotheses if h.status == HypothesisStatus.VERIFIED]
    rejected = [h for h in out.hypotheses if h.status == HypothesisStatus.REJECTED]
    lines = [
        f"投资想法：{out.spec.idea}",
        f"资产类别：{out.spec.asset_class or '未指定'}",
        f"研究假设：{out.spec.hypothesis}",
        f"候选因子：{factors_txt}",
        f"假设验证：{len(verified)} 条通过、{len(rejected)} 条待人工复核。",
        f"策略代码：{'已通过沙箱与前视校验，可安全回测' if out.code_safe else '存在风险（' + '; '.join(out.code_errors + out.lookahead_warnings) + '）'}",
        f"风险要点：{'；'.join(out.spec.risk_notes) if out.spec.risk_notes else '（无）'}",
    ]
    return "\n".join(lines)


def generate_fact_sheet(out: AutoResearchOutput, metrics: Optional[dict] = None) -> dict:
    """产出合规/审查用结构化事实表。

    :param metrics: 可选回测/绩效指标（如 sharpe/max_drawdown），并入事实表。
    """
    return {
        "strategy_name": f"ai_generated_{out.spec.asset_class or 'generic'}",
        "idea": out.spec.idea,
        "hypothesis": out.spec.hypothesis,
        "asset_class": out.spec.asset_class,
        "factors": [
            {"name": f.name, "kind": f.kind, "window": f.window, "weight": f.weight}
            for f in out.factors
        ],
        "code_safe": out.code_safe,
        "validation_notes": {
            "sandbox": bool(not out.code_errors),
            "lookahead": list(out.lookahead_warnings),
        },
        "risk_notes": list(out.spec.risk_notes),
        "metrics": metrics or {},
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


__all__ = [
    "Hypothesis",
    "HypothesisStatus",
    "ResearchLogEntry",
    "AutoResearchOutput",
    "AutoResearchAgent",
    "generate_explanation",
    "generate_fact_sheet",
]
