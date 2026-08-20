"""AI 持续学习闭环（research/knowledge_loop.py）。

三大职责：
  1. :func:`judge_trial`      —— 对单个挖掘出的因子试验做 AI 质量判读
                             （verified / active / rejected + 一句话原因 + 模式标签）。
  2. :func:`summarize_experience` —— 对整次 e2e 运行做 AI 经验归纳 brief。
  3. :func:`kb_search_context` / :func:`format_kb_context` —— 从知识库读取「成功因子模式 +
                             失败避坑清单」，构造供下次挖掘 LLM 实时参考的上下文文本。

设计哲学与 :mod:`quantmind.research.judge` 一致：**LLM 优先 + 规则兜底**。
离线 / 无 key / provider 为 Mock 或 LLM 调用失败时，全部回落内置规则，
保证模块无 key 无网络也能跑通、可测试。本模块只负责 AI 判读与经验归纳，
不负责落库（落库由调用方 search_service 负责，属于后续任务）。

公共 API（全部 async）：
    - judge_trial(provider, trial, idea="", fallback_rules=True) -> dict
    - summarize_experience(provider, run, trials, idea="") -> dict
    - kb_search_context(store, idea="", max_success=8, max_fail=8) -> dict
 辅助（同步）：
    - format_kb_context(ctx) -> str
 集成入口：
    - run_knowledge_loop(store, provider, run_report, idea="") -> dict

策略级 AI 沉淀闭环（并行于因子级，同样 LLM 优先 + 规则兜底）：
    - judge_strategy(provider, strat, gate=None, fallback_rules=True) -> dict
    - summarize_strategy_experience(provider, strategies, idea="") -> dict
    - strategy_kb_context(store, idea="", max_success=6, max_fail=6) -> dict
 辅助（同步）：
    - format_strategy_kb_context(ctx) -> str
 集成入口：
    - run_strategy_knowledge_loop(store, provider, strategy_records, idea="") -> dict
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from typing import Any, Dict, List, Optional

from ..ai.provider import LLMProvider

_logger = logging.getLogger("quantmind.research.knowledge_loop")

__all__ = [
    "judge_trial",
    "summarize_experience",
    "kb_search_context",
    "format_kb_context",
    "run_knowledge_loop",
    "judge_strategy",
    "summarize_strategy_experience",
    "strategy_kb_context",
    "format_strategy_kb_context",
    "run_strategy_knowledge_loop",
]


# =============================================================================
# 提示词（风格对齐 judge.py 的 _SIGNAL_SYSTEM / _SCORING_SYSTEM）
# =============================================================================
_JUDGE_SYSTEM = (
    "You are an expert quantitative researcher judging whether a mined alpha factor "
    "is worth keeping. A factor should be 'verified' only if it shows a stable "
    "positive out-of-sample (OOS) information coefficient (IC) and a positive "
    "long-short Sharpe. If it has no OOS data, or its OOS IC / Sharpe is not positive, "
    "it must be 'rejected'. Anything in between (candidate proposed but not proven) "
    "is 'active'. "
    "Respond with ONLY a JSON object: "
    '{"status": "verified"|"active"|"rejected", "reason": "<one short sentence in Chinese>", '
    '"tags": ["<mode-tag>", ...]}. Tags are 3-6 short English labels like '
    '"momentum", "mean_reversion", "term_structure", "volatility", "price_volume", '
    '"rank", "overfit", "low_IC", "unstable".'
)

_BRIEF_SYSTEM = (
    "You are an expert quantitative research team lead writing a concise lessons-"
    "learned brief after a factor-mining run. Based on the trials below, distill: "
    "effective_themes (which factor patterns actually worked), failure_traps (which "
    "patterns failed and why, e.g. overfitting or term-structure failing in a window), "
    "and next_suggestions (what to try next). Respond with ONLY a JSON object: "
    '{"brief": "<readable Chinese paragraph>", "effective_themes": ["..."], '
    '"failure_traps": ["..."], "next_suggestions": ["..."]}.'
)


# =============================================================================
# 内部工具
# =============================================================================
def _num(value: Any) -> Optional[float]:
    """安全地把任意值转成 float；NaN / None / 非法返回 None。"""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f


def _is_mock_provider(provider) -> bool:
    """判定 provider 是否为离线 Mock（此时不发网络，直接规则兜底）。"""
    if provider is None:
        return True
    return getattr(provider, "name", "base") == "mock"


async def _chat_json(provider: LLMProvider, system: str, user: str) -> Optional[dict]:
    """调用 provider 并尝试解析 JSON dict；失败返回 None（不抛异常）。"""
    try:
        text = await provider.chat(system, user)
    except Exception as exc:  # noqa: BLE001
        _logger.warning("knowledge_loop LLM 调用失败: %s", exc)
        return None
    if not text:
        return None
    blob = re.search(r"\{.*\}", text, re.S)
    if not blob:
        _logger.debug("knowledge_loop LLM 回复无 JSON 块: %r", text[:80])
        return None
    try:
        data = json.loads(blob.group(0))
    except json.JSONDecodeError:
        _logger.debug("knowledge_loop LLM JSON 解析失败: %r", text[:80])
        return None
    return data if isinstance(data, dict) else None


# ---- 规则兜底：judge ---------------------------------------------------------
def _rule_tags(expression: str) -> List[str]:
    """根据表达式子串打模式标签（标签规则兜底）。"""
    tags: List[str] = []
    expr = expression or ""
    if "delta" in expr.lower():
        tags.append("momentum")
    if "-delta" in expr.lower():
        tags.append("mean_reversion")
    if "std" in expr.lower() or "std" in expr:
        tags.append("volatility")
    if "corr" in expr.lower() or "corr" in expr:
        tags.append("price_volume")
    if "rank" in expr.lower() or "rank" in expr:
        tags.append("rank")
    if "mean(close,1)-close" in expr.lower():
        tags.append("term_structure")
    if not tags:
        tags.append("generic")
    return tags[:6]


def _ic_positive_ratio(ic_series: list) -> Optional[float]:
    """从 ic_series 计算 IC 正值比例（需至少 20 个有效值）。"""
    valid = [x for x in (ic_series or []) if x is not None and x == x]
    if len(valid) < 20:
        return None
    return sum(1 for x in valid if x > 0) / len(valid)


def _ic_half_life(ic_series: list) -> Optional[float]:
    """从 ic_series 估计信号衰减半衰期（需至少 10 个有效值）。"""
    valid = [x for x in (ic_series or []) if x is not None and x == x]
    if len(valid) < 10:
        return None
    n = len(valid)
    half = n // 2
    if half == 0:
        return None
    first_half = sum(valid[:half]) / half
    second_half = sum(valid[half:]) / (n - half)
    if first_half <= 0 or second_half <= 0:
        return None
    if second_half >= first_half:
        return float("inf")
    ratio = first_half / second_half
    try:
        return half / math.log2(ratio)
    except (ValueError, ZeroDivisionError):
        return None


def _rule_judge(trial: dict) -> dict:
    """规则判读：严格版——OOS IC≥0.03 且 Sharpe≥0.5 且 train 同向 → verified。

    新增检查：IC 正比例≥0.52、IC 衰减半衰期≥2、年化换手<30。
    """
    expression = str(trial.get("expression") or "")
    train_ic = _num(trial.get("train_ic"))
    test_ic = _num(trial.get("test_ic"))
    test_sharpe = _num(trial.get("test_sharpe"))
    test_mdd = _num(trial.get("test_mdd"))
    turnover = _num(trial.get("turnover_annual"))
    tags = _rule_tags(expression)

    # 从 ic_series 提取 IC 正比例和半衰期（供质量检查）
    ic_series = trial.get("ic_series") or []
    ic_pos_ratio = _ic_positive_ratio(ic_series)
    half_life = _ic_half_life(ic_series)

    # 1. 无 OOS 数据 → rejected
    if test_ic is None:
        return {"status": "rejected",
                "reason": "无OOS：缺少 test_ic 数据，无法证明样本外有效性，判定为失败候选。",
                "tags": tags}

    # 2. test_ic 低于最低信号强度 → rejected
    if test_ic < 0.02:
        return {"status": "rejected",
                "reason": f"OOS信号过弱：test_ic={test_ic:.4f}<0.02，低于噪声门槛，判定为失败候选。",
                "tags": tags}

    # 3. 多空亏钱 → rejected
    if test_sharpe is not None and test_sharpe < 0:
        return {"status": "rejected",
                "reason": f"多空亏损：test_sharpe={test_sharpe:.3f}<0，样本外多空组合亏钱，判定为失败候选。",
                "tags": tags}

    # 4. IC 正比例检查（需足够数据）
    if ic_pos_ratio is not None and ic_pos_ratio < 0.52:
        return {"status": "rejected",
                "reason": f"IC不稳定：IC正值比例={ic_pos_ratio:.2%}<52%，因子预测力不可靠，判定为失败候选。",
                "tags": tags}

    # 5. 半衰期检查（信号衰减太快）
    if half_life is not None and half_life != float("inf") and half_life < 2:
        return {"status": "rejected",
                "reason": f"信号衰减过快：半衰期={half_life:.1f}<2期，无法支撑实际交易，判定为失败候选。",
                "tags": tags}

    # 6. 换手率检查
    if turnover is not None and turnover > 30:
        return {"status": "rejected",
                "reason": f"换手过高：年化换手={turnover:.1f}>30，交易成本将吃掉alpha，判定为失败候选。",
                "tags": tags}

    # 7. verified：IC + Sharpe + train 全部达标
    if test_ic >= 0.03 and test_sharpe is not None and test_sharpe >= 0.5:
        if train_ic is not None and train_ic > 0:
            reason = (
                f"样本外稳定正IC(test_ic={test_ic:.4f}≥0.03)且多空夏普达标"
                f"(sharpe={test_sharpe:.3f}≥0.5)，train期同向为正，判定为可靠因子。"
            )
            return {"status": "verified", "reason": reason, "tags": tags}
        else:
            reason = (
                f"OOS达标(test_ic={test_ic:.4f}, sharpe={test_sharpe:.3f})"
                f"但train_ic={f'{train_ic:.4f}' if train_ic is not None else 'n/a'}非正，"
                f"训练期信号方向不一致，需进一步验证。"
            )
            return {"status": "active", "reason": reason, "tags": tags}

    # 8. active：IC 或 Sharpe 至少有一个存在但未达标
    if test_ic is not None or test_sharpe is not None:
        parts = []
        if test_ic is not None and test_ic < 0.03:
            parts.append(f"test_ic={test_ic:.4f}<0.03")
        if test_sharpe is not None and test_sharpe < 0.5:
            parts.append(f"sharpe={test_sharpe:.3f}<0.5")
        if test_sharpe is None:
            parts.append("多空夏普缺失")
        reason = f"OOS部分达标但未达门槛({', '.join(parts)})，继续研究。"
        return {"status": "active", "reason": reason, "tags": tags}

    # 9. 兆底 rejected
    return {"status": "rejected",
            "reason": "OOS指标均缺失，无法证明样本外有效性，判定为失败候选。",
            "tags": tags}


# ---- 规则兜底：brief --------------------------------------------------------
def _rule_brief(run: dict, trials: List[dict], idea: str) -> dict:
    """规则归纳：从 trials 提取 tags/reason 拼成 experience brief。"""
    eff_tags: List[str] = []
    fail_traps: List[str] = []
    suggestions: List[str] = []
    verified = [t for t in trials if t.get("status") == "verified"]
    rejected = [t for t in trials if (t.get("status") or "").lower() in ("rejected", "redundant")]

    for t in verified:
        for tag in t.get("tags") or []:
            if tag and tag not in eff_tags:
                eff_tags.append(tag)
    for t in rejected:
        reason = (t.get("reason") or "").strip()
        if reason and reason not in fail_traps:
            fail_traps.append(reason)

    if verified:
        suggestions.append(
            "在本次成功主题（" + "/".join(eff_tags[:5]) + "）基础上继续变异，保持稳定样本外IC。"
        )
    if rejected:
        suggestions.append("对本次失败候选的因子族降低权重，优先尝试未被证伪的主题。")
    if not suggestions:
        suggestions.append("扩大搜索空间，尝试不同窗口与截面算子组合。")

    title = idea or run.get("idea") or "未命名的挖掘任务"
    n_rep = run.get("representative_count") or run.get("n_verified_hypotheses") or len(verified)
    brief = (
        f"本次针对「{title}」的因子挖掘：共判读 {len(trials)} 个代表性因子试验，"
        f"其中 {len(verified)} 个验证通过、{len(rejected)} 个被拒。"
    )
    if eff_tags:
        brief += f"有效的模式主题包括：{('/'.join(eff_tags[:6]))}。"
    else:
        brief += "尚未形成稳定的有效模式主题。"
    if fail_traps:
        brief += "主要失败陷阱：" + "；".join(fail_traps[:3]) + "。"
    else:
        brief += "本次未发现明显失败陷阱。"
    brief += " 建议下次挖掘聚焦已验证主题并规避被拒路径。"

    return {
        "brief": brief,
        "effective_themes": eff_tags,
        "failure_traps": fail_traps,
        "next_suggestions": suggestions,
    }


# =============================================================================
# 1. judge_trial —— 单因子试验判读
# =============================================================================
async def judge_trial(
    provider: LLMProvider,
    trial: dict,
    idea: str = "",
    fallback_rules: bool = True,
) -> dict:
    """判读一个因子试验（成功/失败 + 原因 + 模式标签）。

    Args:
        provider: LLM Provider（Mock 或真实；Mock 时直接走规则）。
        trial: 含 ``expression/train_ic/val_ic/test_ic/test_sharpe/test_mdd/seed``
            的试验 dict（字段可能缺省）。
        idea: 研究主题（辅助 LLM 判读）。
        fallback_rules: 为 ``True`` 时直接走规则兜底（不调 LLM）。

    Returns:
        ``{"status": "verified"|"active"|"rejected", "reason": str, "tags": [str]}``
    """
    use_llm = (not fallback_rules) and (not _is_mock_provider(provider))
    if use_llm:
        user = _build_judge_user(trial, idea)
        data = await _chat_json(provider, _JUDGE_SYSTEM, user)
        status = (data or {}).get("status")
        if status in ("verified", "active", "rejected"):
            reason = str((data or {}).get("reason") or "").strip()
            tags = (data or {}).get("tags") or []
            if not isinstance(tags, list):
                tags = []
            tags = [str(t) for t in tags if isinstance(t, str) and t][:6]
            if not reason:
                reason = "LLM 给出判定，未附带原因。" if status != "rejected" else "LLM 判定为失败候选。"
            return {
                "status": status,
                "reason": reason,
                "tags": tags or _rule_tags(str(trial.get("expression") or "")),
            }
        _logger.debug("judge_trial LLM 结果不可用，回落规则。")

    rule = _rule_judge(trial)
    rule["status"] = rule["status"] if rule["status"] in ("verified", "rejected") else "active"
    return rule


def _build_judge_user(trial: dict, idea: str) -> str:
    """构造 judge 的用户消息（各 IC/夏普指标 + 表达式）。"""
    lines = []
    if idea:
        lines.append(f"Research idea: {idea}")
    lines.append(f"Factor expression: {trial.get('expression')}")
    for key in ("train_ic", "val_ic", "test_ic", "test_sharpe", "test_mdd"):
        val = _num(trial.get(key))
        lines.append(f"{key}={f'{val:.4f}' if val is not None else 'n/a'}")
    if "seed" in trial:
        lines.append(f"seed={trial.get('seed')}")
    lines.append("Decide status and provide tags.")
    return "\n".join(lines)


# =============================================================================
# 2. summarize_experience —— 整次运行经验归纳 brief
# =============================================================================
async def summarize_experience(
    provider: LLMProvider,
    run: dict,
    trials: List[dict],
    idea: str = "",
) -> dict:
    """对该次 run 的全部因子试验作经验归纳，返回一份 AI brief。

    Args:
        provider: LLM Provider。
        run: 该次运行的摘要 dict（含 ``idea`` 等）。
        trials: 全部 factor_trials（含 ``status/reason/tags/test_ic``）。
        idea: 研究主题（优先于 run.idea）。

    Returns:
        ``{"brief": str, "effective_themes": [str], "failure_traps": [str],
          "next_suggestions": [str]}`` —— brief 始终是可读中文段落。
    """
    use_llm = not _is_mock_provider(provider)
    if use_llm:
        user = _build_brief_user(run, trials, idea)
        data = await _chat_json(provider, _BRIEF_SYSTEM, user)
        if data is not None and data.get("brief"):
            brief = str(data.get("brief") or "").strip()
            if brief:
                return {
                    "brief": brief,
                    "effective_themes": [str(x) for x in (data.get("effective_themes") or [])],
                    "failure_traps": [str(x) for x in (data.get("failure_traps") or [])],
                    "next_suggestions": [str(x) for x in (data.get("next_suggestions") or [])],
                }
        _logger.debug("summarize_experience LLM 结果不可用，回落规则。")

    return _rule_brief(run, trials, idea)


def _build_brief_user(run: dict, trials: List[dict], idea: str) -> str:
    """构造 brief 的用户消息：run.idea + 逐条(expression,status,reason,tags,test_ic)。"""
    lines = []
    the_idea = idea or run.get("idea") or ""
    if the_idea:
        lines.append(f"Research idea: {the_idea}")
    lines.append("Trials:")
    for i, t in enumerate(trials):
        ic = _num(t.get("test_ic"))
        lines.append(
            f"  [{i}] expr={t.get('expression')} status={t.get('status')} "
            f"reason={t.get('reason')} tags={','.join(t.get('tags') or [])} "
            f"test_ic={f'{ic:.4f}' if ic is not None else 'n/a'}"
        )
    lines.append("Write a lessons-learned brief in JSON.")
    return "\n".join(lines)


# =============================================================================
# 3. kb_search_context —— 成功模式 + 失败清单上下文
# =============================================================================
async def kb_search_context(
    store,
    idea: str = "",
    max_success: int = 8,
    max_fail: int = 8,
) -> dict:
    """从知识库读取成功/失败因子与近期 brief，构造供下次挖掘 LLM 参考的上下文。

    Args:
        store: KnowledgeStore（须提供 ``successful_factors`` / ``failed_factors``）。
        idea: 检索主题过滤。
        max_success / max_fail: 各取多少条。

    Returns:
        ``{"success": [str expressions], "fail": [str], "briefs": [str recent brief]}``
    """
    try:
        success = await _maybe_await(store.successful_factors(
            idea=idea, statuses=("verified", "passed"), top_k=max_success))
    except Exception as exc:  # noqa: BLE001
        _logger.warning("kb successful_factors 不可用（方法可能尚未实现）: %s", exc)
        success = []

    try:
        failed = await _maybe_await(store.failed_factors(
            idea=idea, statuses=("rejected", "redundant"), top_k=max_fail))
    except Exception as exc:  # noqa: BLE001
        _logger.warning("kb failed_factors 不可用（方法可能尚未实现）: %s", exc)
        failed = []

    success_exprs: List[str] = []
    for item in success or []:
        expr = (item.get("expression") or "") if isinstance(item, dict) else ""
        if expr and expr not in success_exprs:
            success_exprs.append(expr)

    fail_items: List[str] = []
    for item in failed or []:
        if not isinstance(item, dict):
            continue
        expr = item.get("expression") or ""
        ic = _num(item.get("test_ic"))
        seg = f"{expr} (test_ic={ic:.4f})" if ic is not None else expr
        if seg and seg not in fail_items:
            fail_items.append(seg)

    # 近期 brief：兜底用最近研究日志的 evidence/idea 近似（尽力而为，可缺省）
    briefs: List[str] = await _load_recent_briefs(store, idea, max_fail)

    return {
        "success": success_exprs,
        "fail": fail_items,
        "briefs": briefs,
    }


async def _maybe_await(obj) -> Any:
    """兼容同步/异步 store 方法。"""
    if asyncio.iscoroutine(obj):
        return await obj
    return obj


async def _load_recent_briefs(store, idea: str, limit: int) -> List[str]:
    """尽力读取近期研究日志作为 brief 上下文；不可用则返回空列表。"""
    briefs: List[str] = []
    try:
        logs_method = getattr(store, "list_items", None)
        if not logs_method:
            return briefs
        logs = await _maybe_await(logs_method(kind="research_log", limit=limit))
        for log in logs or []:
            meta = (log or {}).get("metadata") or {}
            ev = meta.get("evidence") or {}
            if isinstance(ev, dict) and ev.get("verified_exprs"):
                briefs.append("近期验证因子: " + ", ".join(str(x) for x in ev["verified_exprs"]))
    except Exception as exc:  # noqa: BLE001
        _logger.debug("读取近期 brief 失败: %s", exc)
    return briefs[:limit]


def format_kb_context(ctx: dict) -> str:
    """把 ``kb_search_context`` 结果拼成可注入搜索 prompt 的结构化文本。

    库为空时返回空串（不污染 prompt）。
    """
    success = ctx.get("success") or []
    fail = ctx.get("fail") or []
    briefs = ctx.get("briefs") or []
    if not success and not fail and not briefs:
        return ""

    lines: List[str] = []
    lines.append("## Historical verified factor patterns:")
    if success:
        for s in success:
            lines.append(f"- {s}")
    else:
        lines.append("- (none)")

    lines.append("## Historical failure pitfalls:")
    if fail:
        for f in fail:
            lines.append(f"- {f}")
    else:
        lines.append("- (none)")

    if briefs:
        lines.append("## Past experience brief:")
        for b in briefs:
            lines.append(f"- {b}")

    return "\n".join(lines)


# =============================================================================
# 4. run_knowledge_loop —— 集成入口
# =============================================================================
async def run_knowledge_loop(
    store,
    provider: LLMProvider,
    run_report: dict,
    idea: str = "",
) -> dict:
    """e2e 运行后集成入口：并行判读代表因子 + 生成经验 brief。

    Args:
        store: KnowledgeStore（本入口不使用，保留签名以供后续落库流程一致）。
        provider: LLM Provider。
        run_report: 一次 e2e 的运行报告，含 ``summary/steps/composite/evidence``。
        idea: 研究主题。

    Returns:
        ``{"trials": [ {**step, "status", "reason", "tags"} ], "brief": str,
          "effective_themes": [], "failure_traps": [], "next_suggestions": []}``

    注意：本入口只做 AI 判读与归纳，不负责落库（落库由调用方 search_service 处理）。
    """
    steps = run_report.get("steps") or (run_report.get("pipeline") or {}).get("steps") or []
    valid_steps = [s for s in steps if isinstance(s, dict) and s.get("expression")]

    judged: List[dict] = []
    if valid_steps:
        results = await asyncio.gather(
            *(judge_trial(provider, s, idea=idea, fallback_rules=True) for s in valid_steps)
        )
        for step, res in zip(valid_steps, results):
            judged.append({**step, **res})

    if not judged:
        brief_text = "本次未产生可比对因子试验"
        _brief = {
            "brief": brief_text,
            "effective_themes": [],
            "failure_traps": [],
            "next_suggestions": [],
        }
    else:
        _brief = await summarize_experience(provider, run_report, judged, idea=idea)

    return {
        "trials": judged,
        "brief": _brief.get("brief", ""),
        "effective_themes": _brief.get("effective_themes", []),
        "failure_traps": _brief.get("failure_traps", []),
        "next_suggestions": _brief.get("next_suggestions", []),
    }


# =============================================================================
# 策略级 AI 沉淀闭环（strategy-level knowledge loop）
# -----------------------------------------------------------------------------
# 与因子级闭环（judge_trial 等）并列：对**整条策略**做晋升门判读、经验归纳、
# 以及从历史成功/失败策略中构造供下次挖掘复用的上下文。同样遵循
# 「LLM 优先 + 规则兜底」，离线 / mock / 无 key / LLM 失败时全部回落内置规则。
# 只负责 AI 判读与归纳，不负责落库。
# =============================================================================

# ---- 策略判读提示词 -----------------------------------------------------------
_STRATEGY_JUDGE_SYSTEM = (
    "You are an expert quantitative portfolio manager judging whether a whole "
    "strategy deserves promotion through the research gate. A strategy should be "
    "'verified' only if it clears the promotion threshold: Sharpe >= min_sharpe, "
    "max drawdown not exceeding the allowed limit, and state reached "
    "BACKTEST/PAPER or above. If its Sharpe is below min_sharpe, or its max "
    "drawdown exceeds the allowed limit, it must be 'rejected'. If it is only an "
    "IDEA/RESEARCH without backtest metrics yet, it is 'active'. Respond with ONLY "
    'a JSON object: {"status": "verified"|"active"|"rejected", "reason": '
    '"<one short sentence in Chinese>", "tags": ["<tag>", ...]}. Tags are 2-6 short '
    'English labels like "paper", "live", "backtested", "high_sharpe", '
    '"low_sharpe", "no_composite".'
)

_STRATEGY_BRIEF_SYSTEM = (
    "You are an expert quantitative research team lead writing a concise lessons-"
    "learned brief after a strategy-level e2e run. Based on the strategy records "
    "below, distill: effective_patterns (which verified strategies / reusable high-"
    "order templates actually worked), failure_traps (which strategies were rejected "
    "and why), and next_suggestions (what to try next). Respond with ONLY a JSON "
    'object: {"brief": "<readable Chinese paragraph>", "effective_patterns": '
    '["..."], "failure_traps": ["..."], "next_suggestions": ["..."]}.'
)


# ---- 规则兜底：strategy judge -------------------------------------------------
def _rule_strategy_tags(strat: dict) -> List[str]:
    """根据 state 与 sharpe/composite 打策略标签（tags 规则兜底）。"""
    tags: List[str] = []
    state = str(strat.get("state") or "").upper()
    state_tag = {
        "PAPER": "paper",
        "LIVE": "live",
        "BACKTEST": "backtested",
    }.get(state)
    if state_tag and state_tag not in tags:
        tags.append(state_tag)

    sharpe = _num(strat.get("sharpe"))
    if sharpe is not None:
        if sharpe >= 1.5:
            tags.append("high_sharpe")
        elif sharpe < 1.0:
            tags.append("low_sharpe")
    if strat.get("composite_fwd_ic") is not None or strat.get("composite_fwd_ic") != "":
        tags.append("composite")
    if not tags:
        tags.append("generic")
    return tags[:6]


def _rule_strategy_judge(strat: dict, gate: dict) -> dict:
    """策略判读规则：严格版——夏普≥1.0、回撤≤-15%、Calmar≥1.0、胜率≥45%、模拟盘≥30天。"""
    state = str(strat.get("state") or "").upper()
    sharpe = _num(strat.get("sharpe"))
    mdd = _num(strat.get("max_drawdown"))
    calmar = _num(strat.get("calmar"))
    win_rate = _num(strat.get("win_rate"))
    paper_days = _num(strat.get("paper_days"))
    min_sharpe = _num(gate.get("min_sharpe", 1.0))
    min_drawdown = _num(gate.get("min_drawdown", -0.15))
    min_calmar = _num(gate.get("min_calmar", 1.0))
    min_win_rate = _num(gate.get("min_win_rate", 0.45))
    min_paper_days = _num(gate.get("min_paper_days", 30))
    tags = _rule_strategy_tags(strat)

    # state 处 IDEA/RESEARCH，尚无回测指标 → active
    if state in ("IDEA", "RESEARCH") and sharpe is None:
        reason = (
            f"策略处于{state}阶段，尚无回测指标(sharpe缺失)，未达晋升门槛，标记为继续研究。"
        )
        return {"status": "active", "reason": reason, "tags": tags}

    # 1. 回撤超限 → rejected
    if mdd is not None and min_drawdown is not None and mdd < min_drawdown:
        reason = (
            f"回撤超限：max_drawdown={mdd:.3f} 低于回撤上限 {min_drawdown:.3f}，"
            "不满足晋升门要求，判定为失败策略。"
        )
        return {"status": "rejected", "reason": reason, "tags": tags}

    # 2. 夏普不达标 → rejected
    if sharpe is not None and min_sharpe is not None and sharpe < min_sharpe:
        reason = (
            f"夏普不达标：sharpe={sharpe:.3f} < 门槛 {min_sharpe:.3f}，"
            "未达到晋升模拟盘的最低要求，判定为失败策略。"
        )
        return {"status": "rejected", "reason": reason, "tags": tags}

    # 3. Calmar 比率不达标 → rejected
    if calmar is not None and min_calmar is not None and calmar < min_calmar:
        reason = (
            f"Calmar不达标：calmar={calmar:.3f} < 门槛 {min_calmar:.3f}，"
            "收益/回撤比不足，判定为失败策略。"
        )
        return {"status": "rejected", "reason": reason, "tags": tags}

    # 4. 胜率不达标 → rejected
    if win_rate is not None and min_win_rate is not None and win_rate < min_win_rate:
        reason = (
            f"胜率不达标：win_rate={win_rate:.3f} < 门槛 {min_win_rate:.3f}，"
            "交易胜率过低，判定为失败策略。"
        )
        return {"status": "rejected", "reason": reason, "tags": tags}

    # 5. 模拟盘天数不足 → rejected
    if paper_days is not None and min_paper_days is not None and paper_days < min_paper_days:
        reason = (
            f"模拟盘天数不足：paper_days={paper_days:.0f} < 门槛 {min_paper_days:.0f}，"
            "统计样本不足，判定为失败策略。"
        )
        return {"status": "rejected", "reason": reason, "tags": tags}

    # sharpe 有且达标 + 所有门槛通过 + state 达 BACKTEST/PAPER 及以上 → verified
    if (
        sharpe is not None
        and (min_sharpe is None or sharpe >= min_sharpe)
        and (mdd is None or min_drawdown is None or mdd >= min_drawdown)
        and (calmar is None or min_calmar is None or calmar >= min_calmar)
        and (win_rate is None or min_win_rate is None or win_rate >= min_win_rate)
        and (paper_days is None or min_paper_days is None or paper_days >= min_paper_days)
        and state in ("BACKTEST", "PAPER", "LIVE")
    ):
        reason = (
            f"策略通过晋升门：sharpe={sharpe:.3f}（≥{min_sharpe:.3f}）"
            + (f"，max_drawdown={mdd:.3f} 未超限" if mdd is not None else "，无回撤记录")
            + (f"，calmar={calmar:.3f}（≥{min_calmar:.3f}）" if calmar is not None else "")
            + (f"，win_rate={win_rate:.3f}（≥{min_win_rate:.3f}）" if win_rate is not None else "")
            + (f"，paper_days={paper_days:.0f}（≥{min_paper_days:.0f}）" if paper_days is not None else "")
            + f"，状态达{state}阶段，可晋升/认可。"
        )
        return {"status": "verified", "reason": reason, "tags": tags}

    # 其余 → active
    reason = "策略尚未满足所有晋升条件，标记为继续研究。"
    return {"status": "active", "reason": reason, "tags": tags}


def _build_strategy_judge_user(strat: dict, gate: dict) -> str:
    """构造策略判读的用户消息（strategy_id/state/sharpe/mdd/calmar/win_rate/paper_days + 门槛）。"""
    lines = []
    sid = strat.get("run_id") or strat.get("strategy_id")
    lines.append(f"strategy_id: {sid}")
    lines.append(f"state: {strat.get('state')}")
    for key in ("sharpe", "max_drawdown", "calmar", "win_rate", "paper_days", "composite_fwd_ic"):
        val = _num(strat.get(key))
        lines.append(f"{key}={f'{val:.4f}' if val is not None else 'n/a'}")
    lines.append(f"gate min_sharpe={gate.get('min_sharpe', 1.0)} "
                 f"min_drawdown={gate.get('min_drawdown', -0.15)} "
                 f"min_calmar={gate.get('min_calmar', 1.0)} "
                 f"min_win_rate={gate.get('min_win_rate', 0.45)} "
                 f"min_paper_days={gate.get('min_paper_days', 30)}")
    lines.append("Decide whether this strategy passes the promotion gate.")
    return "\n".join(lines)


async def judge_strategy(
    provider: LLMProvider,
    strat: dict,
    gate: dict = None,
    fallback_rules: bool = True,
) -> dict:
    """判读一条策略是否通过晋升门（verified / active / rejected + 原因 + 标签）。

    Args:
        provider: LLM Provider（Mock 或真实；Mock 时直接走规则）。
        strat: 含 ``sharpe/max_drawdown/composite_fwd_ic/state/status/run_id(strategy_id)``
            的策略 dict（字段可缺省）。
        gate: 晋升门配置，含 ``min_sharpe``（默认 0.5）与 ``min_drawdown``（默认 -0.30）。
        fallback_rules: 为 ``True`` 时直接走规则兜底（不调 LLM）。

    Returns:
        ``{"status": "verified"|"active"|"rejected", "reason": str, "tags": [str]}``
    """
    gate = dict(gate or {})
    gate.setdefault("min_sharpe", 1.0)
    gate.setdefault("min_drawdown", -0.15)
    gate.setdefault("min_calmar", 1.0)
    gate.setdefault("min_win_rate", 0.45)
    gate.setdefault("min_paper_days", 30)

    use_llm = (not fallback_rules) and (not _is_mock_provider(provider))
    if use_llm:
        user = _build_strategy_judge_user(strat, gate)
        data = await _chat_json(provider, _STRATEGY_JUDGE_SYSTEM, user)
        status = (data or {}).get("status")
        if status in ("verified", "active", "rejected"):
            reason = str((data or {}).get("reason") or "").strip()
            tags = (data or {}).get("tags") or []
            if not isinstance(tags, list):
                tags = []
            tags = [str(t) for t in tags if isinstance(t, str) and t][:6]
            if not reason:
                reason = "LLM 给出策略判定，未附带原因。"
            return {"status": status, "reason": reason, "tags": tags or _rule_strategy_tags(strat)}
        _logger.debug("judge_strategy LLM 结果不可用，回落规则。")

    rule = _rule_strategy_judge(strat, gate)
    rule["status"] = rule["status"] if rule["status"] in ("verified", "active", "rejected") else "active"
    return rule


# ---- 规则兜底：strategy brief -------------------------------------------------
def _rule_strategy_brief(strategies: List[dict], idea: str) -> dict:
    """规则归纳：从 verified/rejected 策略提取 pattern / traps 拼成策略级 brief。"""
    verified = [s for s in strategies if (s.get("status") or "").lower() == "verified"]
    rejected = [s for s in strategies if (s.get("status") or "").lower() == "rejected"]

    effective: List[str] = []
    for s in verified:
        sid = s.get("run_id") or s.get("strategy_id") or str(s.get("name") or "unknown")
        if sid and sid not in effective:
            effective.append(sid)

    fail_traps: List[str] = []
    for s in rejected:
        reason = (s.get("reason") or "").strip()
        if reason and reason not in fail_traps:
            fail_traps.append(reason)

    suggestions: List[str] = []
    if effective:
        suggestions.append("在已验证策略（" + "/".join(effective[:5]) + "）基础上继续演化，保持夏普达标。")
    if fail_traps:
        suggestions.append("规避被拒策略的失败路径，优先尝试未被证伪的策略思路。")
    if not suggestions:
        suggestions.append("扩大策略搜索空间，尝试不同因子与风控组合。")

    title = idea or "本次挖掘任务"
    brief = (
        f"本次针对「{title}」的端到端共产出 {len(strategies)} 条策略，"
        f"其中 {len(verified)} 条回测达标（夏普≥0.5）可晋升模拟盘，"
        f"{len(rejected)} 条因夏普不足或回撤超限被拒。"
    )
    if effective:
        brief += "可复用的成功模板包括：" + ("/".join(effective[:5])) + "。"
    else:
        brief += "尚未形成稳定的成功策略模板。"
    if fail_traps:
        brief += "主要失败陷阱：" + "；".join(fail_traps[:3]) + "。"
    else:
        brief += "本次未发现明显失败陷阱。"
    brief += " 建议下次挖掘聚焦已验证模板并规避被拒路径。"

    return {
        "brief": brief,
        "effective_patterns": effective,
        "failure_traps": fail_traps,
        "next_suggestions": suggestions,
    }


def _build_strategy_brief_user(strategies: List[dict], idea: str) -> str:
    """构造策略 brief 的用户消息：idea + 逐条(strategy_id,state,status,sharpe,reason)。"""
    lines = []
    if idea:
        lines.append(f"Research idea: {idea}")
    lines.append("Strategies:")
    for i, s in enumerate(strategies):
        sharpe = _num(s.get("sharpe"))
        sid = s.get("run_id") or s.get("strategy_id") or "unknown"
        lines.append(
            f"  [{i}] strategy_id={sid} state={s.get('state')} status={s.get('status')} "
            f"sharpe={f'{sharpe:.3f}' if sharpe is not None else 'n/a'} "
            f"reason={s.get('reason')}"
        )
    lines.append("Write a lessons-learned strategy brief in JSON.")
    return "\n".join(lines)


async def summarize_strategy_experience(
    provider: LLMProvider,
    strategies: List[dict],
    idea: str = "",
) -> dict:
    """对一批策略作经验归纳，返回策略级 AI brief。

    Args:
        provider: LLM Provider。
        strategies: 全部策略记录（每个含 ``strategy_id/state/status/sharpe/reason``）。
        idea: 研究主题。

    Returns:
        ``{"brief": str, "effective_patterns": [str], "failure_traps": [str],
          "next_suggestions": [str]}`` —— brief 始终是可读中文段落。
    """
    use_llm = not _is_mock_provider(provider)
    if use_llm:
        user = _build_strategy_brief_user(strategies, idea)
        data = await _chat_json(provider, _STRATEGY_BRIEF_SYSTEM, user)
        if data is not None and data.get("brief"):
            brief = str(data.get("brief") or "").strip()
            if brief:
                return {
                    "brief": brief,
                    "effective_patterns": [str(x) for x in (data.get("effective_patterns") or [])],
                    "failure_traps": [str(x) for x in (data.get("failure_traps") or [])],
                    "next_suggestions": [str(x) for x in (data.get("next_suggestions") or [])],
                }
        _logger.debug("summarize_strategy_experience LLM 结果不可用，回落规则。")

    return _rule_strategy_brief(strategies, idea)


# =============================================================================
# 策略级上下文：历史成功 + 失败(被拒)策略 + 经验 brief
# =============================================================================
async def strategy_kb_context(
    store,
    idea: str = "",
    max_success: int = 6,
    max_fail: int = 6,
) -> dict:
    """从知识库读取历史成功/失败策略，构造策略级复用上下文。

    Args:
        store: KnowledgeStore（须提供 ``successful_strategies`` / ``failed_strategies``）。
        idea: 检索主题过滤。
        max_success / max_fail: 各取多少条。

    Returns:
        ``{"success": [str strategy_id 或 (id|state|sharpe)], "fail": [str],
          "briefs": [str]}`` —— store 缺方法/异常时安全返回空结构。
    """
    success: List[dict] = []
    failed: List[dict] = []

    success_method = getattr(store, "successful_strategies", None)
    if success_method:
        try:
            success = await _maybe_await(success_method(
                idea=idea, statuses=("verified", "paper", "backtested"), top_k=max_success))
        except Exception as exc:  # noqa: BLE001
            _logger.warning("strategy successful_strategies 不可用: %s", exc)
            success = []

    fail_method = getattr(store, "failed_strategies", None)
    if fail_method:
        try:
            failed = await _maybe_await(fail_method(
                idea=idea, statuses=("rejected",), top_k=max_fail))
        except Exception as exc:  # noqa: BLE001
            _logger.warning("strategy failed_strategies 不可用: %s", exc)
            failed = []

    success_items: List[str] = []
    for item in (success or []):
        if not isinstance(item, dict):
            continue
        sid = item.get("strategy_id") or item.get("run_id") or ""
        state = item.get("state") or ""
        sharpe = _num(item.get("sharpe"))
        if sid:
            seg = (f"{sid} | state={state}"
                   + (f" | sharpe={sharpe:.2f}" if sharpe is not None else ""))
            if seg not in success_items:
                success_items.append(seg)

    fail_items: List[str] = []
    for item in (failed or []):
        if not isinstance(item, dict):
            continue
        sid = item.get("strategy_id") or item.get("run_id") or ""
        reason = (item.get("reason") or "").strip()
        seg = (f"{sid} ({reason})" if reason else sid) if sid else reason
        if seg and seg not in fail_items:
            fail_items.append(seg)

    briefs: List[str] = await _load_recent_briefs(store, idea, max_fail)

    return {
        "success": success_items,
        "fail": fail_items,
        "briefs": briefs,
    }


def format_strategy_kb_context(ctx: dict) -> str:
    """把 ``strategy_kb_context`` 结果拼成可注入挖掘 prompt 的结构化文本。

    库为空时返回空串（不污染 prompt）。
    """
    success = ctx.get("success") or []
    fail = ctx.get("fail") or []
    briefs = ctx.get("briefs") or []
    if not success and not fail and not briefs:
        return ""

    lines: List[str] = []
    lines.append("## 历史已验证策略(可复用的高阶模板):")
    if success:
        for s in success:
            lines.append(f"- {s}")
    else:
        lines.append("- (none)")

    lines.append("## 历史被拒策略(避免重蹈):")
    if fail:
        for f in fail:
            lines.append(f"- {f}")
    else:
        lines.append("- (none)")

    if briefs:
        lines.append("## 历史经验brief:")
        for b in briefs:
            lines.append(f"- {b}")

    return "\n".join(lines)


# =============================================================================
# 策略级集成入口
# =============================================================================
async def run_strategy_knowledge_loop(
    store,
    provider: LLMProvider,
    strategy_records: List[dict],
    idea: str = "",
) -> dict:
    """策略级 e2e 集成入口：并行判读每条策略 + 归纳策略级 brief。

    Args:
        store: KnowledgeStore（本入口不使用，保留签名供后续流程一致）。
        provider: LLM Provider。
        strategy_records: 一批策略记录（每条含 ``sharpe/max_drawdown/state/status/run_id``）。
        idea: 研究主题。

    Returns:
        ``{"judged": [ {**record, "status", "reason", "tags"} ], "brief": str,
          "effective_patterns": [], "failure_traps": [], "next_suggestions": []}``

    注意：只做 AI 判读与归纳，不负责落库（落库由调用方处理）。
    """
    judged: List[dict] = []
    if strategy_records:
        results = await asyncio.gather(
            *(judge_strategy(provider, s, fallback_rules=True) for s in strategy_records)
        )
        for record, res in zip(strategy_records, results):
            judged.append({**record, **res})

    if not judged:
        _brief = {
            "brief": "本次无策略记录",
            "effective_patterns": [],
            "failure_traps": [],
            "next_suggestions": [],
        }
    else:
        _brief = await summarize_strategy_experience(provider, judged, idea=idea)

    return {
        "judged": judged,
        "brief": _brief.get("brief", ""),
        "effective_patterns": _brief.get("effective_patterns", []),
        "failure_traps": _brief.get("failure_traps", []),
        "next_suggestions": _brief.get("next_suggestions", []),
    }
