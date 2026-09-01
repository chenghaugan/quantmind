"""端到端流水线**编排器**：把两条既有能力链打通成一条可一键触发的完整链路。

实际系统里两条能力链此前互相割裂：
  - **A 线（AI 研究 Agent）**：``quantmind.ai.agent.AutoResearchAgent`` ——
    Idea → 因子生成 → **真实面板 IC 证据验证**（Hypothesis VERIFIED/REJECTED）→ 策略代码生成 → AST 沙箱。
  - **B 线（因子挖掘流水线）**：``quantmind.research.pipeline.run_pipeline`` ——
    多 seed → co/ea/tot 迭代搜索 → 去冗余 → 防泄漏切分 → 逐代表 OOS 多空回测 → 复合 alpha。

本编排器做的仅是「打针」：
  1. 用 A 线产出的 **VERIFIED 因子表达式 + 最高 IC 因子** 作为 B 线流水线的搜索种子
     （A 线 IC 证据 回灌 B 线挖掘 的断链闭合）。
  2. 跑 B 线挖掘 → 去冗余 → OOS 多空回测 → 复合 alpha。
  3. 把复合 alpha 对应的代表因子 / A 线证据因子 交给 ``generate_strategy_code``
     生成**可部署策略代码**（并过 AST 沙箱校验）。
  4. 产出统一契约 dict，供 REST 端点 / 前端页 / 知识库沉淀消费。

设计要点：
  - **纯内存面板操作**，与 ``run_pipeline`` 同级，不依赖 DataManager/网络；
    真实数据由调用方先用 ``Panel.from_bars`` 构造好。
  - **可插拔 provider**：``provider=None`` → Mock（离线可跑/可测）。
  - **不重写核心**：summary/steps/composite/strategy 全部来自既有模块，仅做组装与契约化。
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# 注意：这里**不**顶层导入 ``..ai.*``（agent/codegen/sandbox/safety/expr_map）。
# 那些模块依赖 ``..research.target``，若在 research 包 __init__ 初始化期间被顶层加载，
# 会触发 ``research -> orchestrator -> ai.agent -> factor_gen -> research.target`` 循环导入。
# 因此 ai.* 统一在 run_e2e 内部延迟导入（函数级），research/__init__ 因此可安全急切导出。
from .factors.alpha_cs import Panel
from .pipeline import PipelineConfig, StepReport, run_pipeline

_logger = logging.getLogger("quantmind.research.orchestrator")

__all__ = ["E2EConfig", "run_e2e"]

DEFAULT_IDEA = "螺纹钢期货动量与期限结构因子组合策略"


@dataclass
class E2EConfig:
    """端到端编排器配置。

    复用 :class:`quantmind.research.pipeline.PipelineConfig` 的全部挖掘参数，
    额外叠加 AI 证据研究阶段与策略代码生成阶段的参数。
    """

    idea: str = DEFAULT_IDEA
    asset_class: str = "期货"
    # -- AI 证据研究阶段（A 线）--
    verify_threshold: float = 0.02        # |IC| 通过阈值（Hypothesis VERIFIED 依据）
    run_search: bool = False              # 是否在证据阶段额外跑 CoT 搜索改进
    max_rounds: int = 2                   # 证据阶段 CoT 轮数
    # -- 领域知识增强层 --
    use_knowledge: bool = True            # 是否在 idea→因子 前注入领域知识
    web_fallback: bool = True             # 库内方法论命中不足时是否联网补充
    # -- 因子挖掘阶段（B 线，透传 PipelineConfig）--
    seeds: Optional[List[str]] = None     # 用户提供的额外种子；None → 用证据阶段产出的种子
    algo: str = "co"                      # co | ea | tot
    rounds: int = 3                       # 每 seed 迭代深度
    forward_periods: int = 1
    market: str = ""
    train_frac: float = 0.6
    val_frac: float = 0.2
    dedup_threshold: float = 0.7
    min_abs_ic: float = 0.03
    run_composite: bool = True
    composite_scheme: str = "icir"        # equal | icir | inv_var | min_var
    composite_standardize: str = "zscore"
    n_groups: int = 5
    long_short: bool = True
    cost_rate: float = 0.0
    # 因子侧宽松净成本/换手闸门（透传 PipelineConfig）
    net_gate: bool = False
    max_turnover: float = 0.0
    min_net_sharpe: float = 0.0
    max_candidates: int = 8
    persist_pairs: bool = False           # 编排器默认不额外落库，由知识库负责沉淀
    # 持续学习闭环：历史知识库上下文（success/fail/briefs）注入挖掘搜索 LLM prompt
    knowledge_context: Optional[dict] = None
    # -- 策略代码生成阶段 --
    code_threshold: float = 0.3
    code_size: int = 1
    code_max_pos: float = 1.0


def _hypothesis_repr(status: HypothesisStatus) -> str:
    return status.value


def run_e2e(
    panel: Panel,
    config: Optional[E2EConfig] = None,
    provider: Optional[LLMProvider] = None,
    val_panel: Optional[Panel] = None,
    test_panel: Optional[Panel] = None,
    knowledge_context: Optional[dict] = None,
    progress: Optional[dict] = None,
) -> Dict[str, object]:
    """运行端到端编排：AI 证据研究 → 因子挖掘 → OOS 复合 alpha → 策略代码生成。

    Args:
        panel: 面板（训练/全期）。若同时传 val/test 用,可先自行切分；否则内部自动切分。
        config: 编排配置。
        provider: LLM provider；None → Mock（离线可跑/可测）。
        val_panel: 可选验证期面板（透传给挖掘阶段的 val 报告）。
        test_panel: 可选测试期面板（透传给挖掘阶段的 OOS 回测）。
        knowledge_context: 可选历史知识库上下文（``kb_search_context`` 输出）。
            None → 不注入，搜索保持原行为；非空 → 注入挖掘阶段 co/ea/tot 的 LLM prompt。

    Returns:
        统一契约 dict::

            {
              "idea", "client_ready",
              "evidence": { "hypotheses": [...], "factors": [...],
                            "verified_exprs": [...], "fact_sheet": {...} },
              "pipeline": { "config": {...}, "summary": {...}, "steps": [...],
                            "composite": {...} },
              "strategy": { "code", "code_safe", "code_errors", "lookahead" },
              "knowledge": { concept, definition, buy_signal_rules,
                             candidate_factors, sources, kb_hits }  # 或空 dict
            }
    """
    # 函数级延迟导入 ai.*（见模块顶部注释，避免加载期循环导入）
    from ..ai.provider import MockProvider
    from ..ai.agent import AutoResearchAgent, HypothesisStatus
    from ..ai.codegen import generate_strategy_code
    from ..ai.sandbox import compile_strategy, validate_code
    from ..ai.safety import lookahead_warnings
    from ..ai.expr_map import factor_spec_to_expression

    config = config or E2EConfig()
    provider = provider or MockProvider()
    if panel is None or panel.close.empty:
        raise ValueError("训练期面板为空")

    def _stage(msg: str, cur: int, tot: int) -> None:
        """向任务框架透出阶段进度（借鉴 LLM 策略挖掘的 progress 模式）。"""
        if progress is None:
            return
        try:
            progress.clear()
            progress.update({"current": cur, "total": tot, "message": msg})
        except Exception:  # noqa: BLE001
            pass

    # =====================================================================
    # 阶段 1：AI 证据研究（A 线）——真实面板 IC 验证，回灌种子
    # =====================================================================
    _stage("AI 证据研究中…", 1, 4)
    agent = AutoResearchAgent(provider=provider,
                              use_knowledge=config.use_knowledge,
                              web_fallback=config.web_fallback)
    evidence = asyncio.run(agent.research_with_evidence(
        config.idea,
        panel,
        asset_class=config.asset_class,
        verify_threshold=config.verify_threshold,
        forward_periods=config.forward_periods,
        market=config.market,
        run_search=config.run_search,
        max_rounds=config.max_rounds,
        use_cache=False,
    ))

    # 方法论知识层护栏：无法忠实实现时提前返回（请用户补充信息），跳过挖掘与代码生成。
    if getattr(evidence, "needs_input", None):
        return {
            "idea": config.idea,
            "client_ready": True,
            "needs_input": {"missing": list(evidence.needs_input)},
            "evidence": {
                "hypotheses": [
                    {"id": h.id, "statement": h.statement,
                     "status": _hypothesis_repr(h.status), "evidence": h.evidence}
                    for h in evidence.hypotheses
                ],
                "factors": [],
                "verified_exprs": [],
                "fact_sheet": dict(evidence.fact_sheet or {}),
            },
            "pipeline": None,
            "strategy": {"code": "", "code_safe": False, "code_errors": [],
                          "lookahead": []},
            "knowledge": evidence.knowledge or {},
        }

    # 提取 VERIFIED 因子表达式（含证据阶段写回的 spec.expression）
    verified_exprs: List[str] = []
    verified_factors = []
    # hypotheses[0] 是 H0（Idea 级假设），后续依次对应每个因子
    for h, f in zip(evidence.hypotheses[1:], evidence.factors):
        if h.status == HypothesisStatus.VERIFIED:
            expr = f.expression or factor_spec_to_expression(f)
            if expr and expr not in verified_exprs:
                verified_exprs.append(expr)
                verified_factors.append(f)
    # 兜底：无 VERIFIED 也用全部生成因子（保证 B 线至少可跑）
    if not verified_exprs:
        for f in evidence.factors:
            expr = f.expression or factor_spec_to_expression(f)
            if expr and expr not in verified_exprs:
                verified_exprs.append(expr)
    # 合并用户显式种子（证据表达式在前，作为优先挖掘起点）→ 去重保序
    user_seeds = [s for s in (config.seeds or []) if s and s.strip()]
    seeds = list(dict.fromkeys([*verified_exprs, *user_seeds]))
    if not seeds:
        raise ValueError("无可用种子：AI 证据研究未产出有效因子，且未提供用户种子")

    # =====================================================================
    # 阶段 2-3：因子挖掘 + OOS 复合 alpha（B 线，复用 run_pipeline）
    # =====================================================================
    pipe_cfg = PipelineConfig(
        seeds=seeds,
        algo=config.algo if config.algo in ("co", "ea", "tot") else "co",
        rounds=config.rounds,
        forward_periods=config.forward_periods,
        market=config.market,
        train_frac=config.train_frac,
        val_frac=config.val_frac,
        dedup_threshold=config.dedup_threshold,
        min_abs_ic=config.min_abs_ic,
        run_composite=config.run_composite,
        composite_scheme=config.composite_scheme,
        composite_standardize=config.composite_standardize,
        n_groups=config.n_groups,
        long_short=config.long_short,
        cost_rate=config.cost_rate,
        net_gate=config.net_gate,
        max_turnover=config.max_turnover,
        min_net_sharpe=config.min_net_sharpe,
        max_candidates=config.max_candidates,
        persist_pairs=config.persist_pairs,
        knowledge_context=(
            knowledge_context if knowledge_context is not None else config.knowledge_context
        ),
    )
    _stage("因子挖掘与 OOS 复合中…", 2, 4)
    pipeline_report = run_pipeline(
        panel,
        config=pipe_cfg,
        provider=provider,
        val_panel=val_panel,
        test_panel=test_panel,
    )
    # 去掉不可 JSON 序列化的内存对象（若有）
    composite = pipeline_report.get("composite")
    if isinstance(composite, dict):
        composite.pop("composite", None)

    # =====================================================================
    # 阶段 4：策略代码生成（复用 A 线 codegen + 沙箱）
    # =====================================================================
    # 优先用复合 alpha 的代表因子，否则回退 A 线证据因子（保证 specs 足够表达 idea）
    code_specs = verified_factors or evidence.factors
    if not code_specs:
        code_specs = _specs_from_representatives(pipeline_report.get("steps") or [])
    _stage("策略代码生成中…", 3, 4)
    code = asyncio.run(generate_strategy_code(
        provider,
        config.idea,
        code_specs,
        threshold=config.code_threshold,
        size=config.code_size,
        max_pos=config.code_max_pos,
    ))
    # 沙箱对齐（借鉴 LLM 策略挖掘）：不仅查 AST，还要求代码可实例化为 CTA 策略类，
    # 确保产出的代码可直接注册进策略引擎参与真实回测。
    ok_sandbox, _compile_err, errors = compile_strategy(code, require_base="CtaTemplate")
    lookahead = lookahead_warnings(code)

    # =====================================================================
    # 汇总契约
    # =====================================================================
    return {
        "idea": config.idea,
        "client_ready": True,
        "evidence": {
            "hypotheses": [
                {
                    "id": h.id,
                    "statement": h.statement,
                    "status": _hypothesis_repr(h.status),
                    "evidence": h.evidence,
                }
                for h in evidence.hypotheses
            ],
            "factors": [
                {
                    "name": f.name, "kind": f.kind, "window": f.window,
                    "weight": f.weight, "expression": f.expression or "",
                }
                for f in evidence.factors
            ],
            "verified_exprs": verified_exprs,
            "fact_sheet": {
                k: v for k, v in (evidence.fact_sheet or {}).items()
            },
        },
        "pipeline": {
            "config": pipeline_report.get("config", {}),
            "summary": pipeline_report.get("summary", {}),
            "steps": pipeline_report.get("steps", []),
            "composite": composite,
        },
        "strategy": {
            "code": code,
            "code_safe": bool(ok_sandbox) and not lookahead,
            "code_errors": list(errors),
            "lookahead": list(lookahead),
        },
        "knowledge": evidence.knowledge or {},
    }


def _specs_from_representatives(steps: List[Dict[str, object]]) -> list:
    """从流水线代表因子 steps 反构 FactorSpec（当无 A 线因子时兜底）。

    面板 DSL 表达式无法可靠反解 kind/window，这里统一归为 ``momentum``，
    权重按 OOS Sharpe 的正负取 ±1，至少让生成的策略类可编译可跑（真实权重
    应与挖掘复合结果一致，正式接入时由调用方把代表因子映射为 FactorSpec）。
    """
    from ..research.target import FactorSpec

    specs: List[FactorSpec] = []
    for s in (steps or [])[:8]:
        expr = str(s.get("expression", "")).strip()
        if not expr:
            continue
        sh = s.get("test_sharpe")
        w = -1.0 if (isinstance(sh, (int, float)) and sh == sh and sh < 0) else 1.0
        specs.append(FactorSpec(
            name="rep_" + str(len(specs) + 1),
            kind="momentum",
            window=20,
            weight=float(w),
            expression=expr,
        ))
    return specs
