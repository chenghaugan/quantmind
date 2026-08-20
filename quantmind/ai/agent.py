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
from .knowledge_enrichment import KnowledgeBrief, enrich_idea
from .codegen import generate_strategy_code
from .sandbox import validate_code
from .safety import lookahead_warnings
from .expr_map import factor_spec_to_expression
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
    knowledge: Optional[dict] = None  # 领域知识增强层产出的 KnowledgeBrief.to_dict() 或 None
    #: 方法论知识层判定「无法忠实实现」时非空：列出需要用户补充的信息（非错误，需澄清）。
    needs_input: List[str] = field(default_factory=list)

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
            "knowledge": self.knowledge,
            "needs_input": list(self.needs_input),
            "code": self.code[:500],
        }


class AutoResearchAgent:
    """自动多步研究智能体。

    :param provider: LLM Provider；默认 Mock。
    :param max_steps: 循环最多执行步数。
    :param use_knowledge: 是否启用领域知识增强层（idea → 因子 前先检索/提炼知识）。
    :param web_fallback: 库内方法论命中不足时是否联网补充资料。
    """

    def __init__(self, provider: LLMProvider | None = None, max_steps: int = 3,
                 use_knowledge: bool = True, web_fallback: bool = True) -> None:
        self.provider = provider or MockProvider()
        self.max_steps = max_steps
        self.use_knowledge = use_knowledge
        self.web_fallback = web_fallback

    def _log(self, out: AutoResearchOutput, step: str, action: str,
             input: str = "", output: str = "") -> None:
        out.log.append(ResearchLogEntry(step=step, action=action, input=input, output=output))

    async def _enrich(self, out: AutoResearchOutput, idea: str) -> Optional[dict]:
        """领域知识增强阶段：检索方法论 + 提炼 KnowledgeBrief，存入 out.knowledge。"""
        if not self.use_knowledge:
            out.knowledge = None
            return None
        brief: KnowledgeBrief = await enrich_idea(
            self.provider, idea, web=self.web_fallback,
        )
        out.knowledge = brief.to_dict()
        self._log(out, -1, "enrich_idea", input=idea,
                  output=f"concept={brief.concept}, sources={len(brief.sources)}, "
                         f"factors={len(brief.candidate_factors)}, "
                         f"can_implement={brief.can_implement}")
        return brief

    async def _guard(self, out: AutoResearchOutput, brief: Optional[KnowledgeBrief]) -> bool:
        """方法论知识层护栏：无法忠实实现时，短路返回并回问用户，不编造因子。"""
        if brief is None or brief.can_implement:
            return False
        out.needs_input = list(brief.missing) or [
            "该方法论无法从现有资料忠实实现，请补充其定义与量化实现要点。"
        ]
        return True

    async def research(self, idea: str, asset_class: str = "") -> AutoResearchOutput:
        out = AutoResearchOutput(
            spec=ResearchSpec(idea=idea, asset_class=asset_class),
        )

        # 领域知识增强阶段：idea → 因子 前先检索/提炼领域知识（可选）
        brief = await self._enrich(out, idea)
        if await self._guard(out, brief):
            return out

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
        factors = await generate_factors(self.provider, idea, knowledge=brief)
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

    async def research_with_evidence(
        self,
        idea: str,
        panel,
        asset_class: str = "",
        verify_threshold: float = 0.02,
        forward_periods: int = 1,
        market: str = "",
        run_search: bool = False,
        max_rounds: int = 2,
        use_cache: bool = False,
        use_knowledge: Optional[bool] = None,
        web_fallback: Optional[bool] = None,
    ) -> AutoResearchOutput:
        """闭环研究：用**真实面板截面 IC 证据**验证因子假设（而非离线启发式）。

        复用 :meth:`research` 的步骤 0-2（想法解析、因子生成、代码沙箱校验），但把
        「每条因子假设的验证」改为：将因子映射为面板 DSL 表达式 → 用 P0 统一评估入口
        ``evaluate_expression`` 求真实截面 IC → 依据阈值标记 VERIFIED / REJECTED，
        并把真实指标写入 evidence 与事实表。

        可选 ``run_search=True``：以 IC 绝对值最高的因子为 seed，调用 P1 ``FactorSearcher``
        链式精炼搜索改进表达式。

        :param idea: 投资想法。
        :param panel: ``Panel`` 或 ``{symbol: List[BarData]}``（评估/搜索用的面板）。
        :param asset_class: 资产类别，透传想法解析。
        :param verify_threshold: |IC| 通过阈值（默认 0.02），用于 VERIFIED 判定。
        :param forward_periods: 前向收益周期，透传评估。
        :param market: 市场标识，透传评估/缓存键。
        :param run_search: 是否额外运行 CoT 迭代搜索改进。
        :param max_rounds: CoT 搜索轮数（``run_search=True`` 时生效）。
        :param use_cache: 评估是否启用 SQLite 持久缓存。
        :param use_knowledge: 是否启用领域知识增强；None → 用实例默认
            （``self.use_knowledge``）。
        :param web_fallback: 库内方法论命中不足时是否联网补充；None → 用实例默认。
        :return: ``AutoResearchOutput``（hypotheses 携带真实 IC 证据，fact_sheet 带 metrics）。
        """
        # 延迟导入，避免 research.search.cot -> ai.agent 的循环依赖
        from ..research import evaluate_expression
        from ..research.search.cot import FactorSearcher

        if use_knowledge is not None:
            self.use_knowledge = use_knowledge
        if web_fallback is not None:
            self.web_fallback = web_fallback

        out = AutoResearchOutput(
            spec=ResearchSpec(idea=idea, asset_class=asset_class),
        )

        # 领域知识增强阶段（可选，复用 _enrich）
        brief = await self._enrich(out, idea)
        if await self._guard(out, brief):
            return out

        # 步骤 0-2 与 research() 保持一致
        spec = await parse_idea(self.provider, idea, asset_class)
        out.spec = spec
        self._log(out, 0, "parse_idea", input=idea, output=spec.to_dict().__repr__())

        h0 = Hypothesis(
            id="H0",
            statement=spec.hypothesis or f"基于「{idea}」的因子具有稳定预测能力",
            status=HypothesisStatus.PROPOSED,
            evidence="由想法解析产出",
        )
        out.hypotheses.append(h0)

        factors = await generate_factors(self.provider, idea, knowledge=brief)
        out.factors = factors
        self._log(out, 1, "generate_factors", input=idea,
                  output=f"[{', '.join(f.name for f in factors)}]")

        # 闭环：真实 IC 验证每条因子假设
        metrics_summary: Dict[str, dict] = {}
        best_seed: Optional[str] = None
        best_abs_ic = 0.0
        for i, f in enumerate(factors):
            expr = factor_spec_to_expression(f)
            f.expression = expr  # 写回表达式便于下游复核/复评
            h = Hypothesis(
                id=f"H{i+1}",
                statement=f"因子 {f.name}(kind={f.kind}, window={f.window}) 具有预测能力",
                status=HypothesisStatus.PROPOSED,
                evidence="待面板 IC 评估",
            )
            try:
                rep = evaluate_expression(
                    expr, panel, forward_periods=forward_periods,
                    market=market, use_cache=use_cache,
                    factor_name=f.name,
                )
                metrics = _report_metrics(rep)
                metrics_summary[f.name] = metrics
                ic_mean = metrics.get("ic_mean")
                if metrics.get("n_samples", 0) > 0 and ic_mean is not None:
                    h.evidence = (f"面板IC评估：ic_mean={ic_mean:.4f}, "
                                  f"ir={metrics.get('ir')}, "
                                  f"ic_positive_ratio={metrics.get('ic_positive_ratio')}, "
                                  f"n_samples={metrics['n_samples']}")
                    if abs(ic_mean) >= verify_threshold:
                        h.status = HypothesisStatus.VERIFIED
                    else:
                        h.status = HypothesisStatus.REJECTED
                else:
                    h.status = HypothesisStatus.REJECTED
                    h.evidence = "面板 IC 评估无有效样本（n_samples=0 或 ic_mean 缺失）"
            except Exception as exc:  # noqa: BLE001
                _logger.warning("因子 %s 评估失败: %s", f.name, exc)
                h.status = HypothesisStatus.REJECTED
                h.evidence = f"评估异常：{exc}"
                self._log(out, 1, "evaluate_factor", input=expr, output=f"error={exc}")
            else:
                self._log(out, 1, "evaluate_factor", input=expr,
                          output=f"ic_mean={metrics.get('ic_mean')}, "
                                 f"status={h.status.value}")
                if metrics.get("ic_mean") is not None and abs(metrics["ic_mean"]) > best_abs_ic:
                    best_abs_ic = abs(metrics["ic_mean"])
                    best_seed = expr
            out.hypotheses.append(h)

        # 可选：CoT 迭代搜索改进
        if run_search and best_seed:
            try:
                self._log(out, 1, "cot_search", input=best_seed,
                          output=f"rounds={max_rounds}")
                searcher = FactorSearcher(provider=self.provider, rounds=max_rounds)
                res = await searcher.cot_search(
                    best_seed, panel, forward_periods=forward_periods,
                    market=market, instruction=spec.hypothesis or "",
                )
                best_expr = res.best_expression
                best_ic = res.best_ic if res.best_ic == res.best_ic else float("nan")
                h_search = Hypothesis(
                    id=f"H{COT_H_PREFIX}",
                    statement=f"CoT 搜索改进表达式「{best_expr}」预测能力",
                    status=(HypothesisStatus.VERIFIED if best_ic == best_ic
                            and abs(best_ic) >= verify_threshold else HypothesisStatus.REJECTED),
                    evidence=(f"CoT 搜索 best_ic={best_ic:.4f}, "
                              f"steps={len(res.steps)}" if best_ic == best_ic
                              else "CoT 搜索无有效 IC"),
                )
                out.hypotheses.append(h_search)
                metrics_summary["_cot_search_best"] = {
                    "expression": best_expr,
                    "ic": round(float(best_ic), 6) if best_ic == best_ic else None,
                }
                self._log(out, 1, "cot_search", input=best_seed,
                          output=f"best={best_expr}, best_ic={best_ic}")
            except Exception as exc:  # noqa: BLE001
                _logger.warning("CoT 搜索失败: %s", exc)
                self._log(out, 1, "cot_search", input=best_seed, output=f"error={exc}")

        # 步骤 2：策略代码生成 + 沙箱校验 + 前视扫描（与 research 一致，但跳过重复假设标记）
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

        # 步骤 3：解释 + 事实表（并入真实评估 metrics）
        out.explanation = generate_explanation(out)
        out.fact_sheet = generate_fact_sheet(out, metrics=metrics_summary)
        self._log(out, 4, "generate_explanation_factsheet", input="",
                  output=f"explanation_len={len(out.explanation)}")

        return out


def _report_metrics(rep) -> dict:
    """把 ``FactorReport`` 规整为 JSON 友好的指标 dict（NaN → None）。"""
    d = rep.to_dict() if hasattr(rep, "to_dict") else {}
    out = {}
    for k, v in d.items():
        if isinstance(v, float):
            out[k] = v if v == v else None  # NaN → None
        else:
            out[k] = v
    return out


# CoT 搜索追加假设的 id 延续序号由 agent 实例维护；此处用固定前缀+索引占位，
# 实际 id 在运行时据此生成。
COT_H_PREFIX = "cot"


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
    "KnowledgeBrief",
    "enrich_idea",
]
