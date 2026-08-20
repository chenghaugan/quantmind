"""AI 提示词模板（研究 / 因子生成 / 代码生成 / 知识增强）。"""
from __future__ import annotations

RESEARCH_SYSTEM = (
    "你是一名量化研究员。把用户的投资想法（idea）解析为结构化研究规格，"
    "严格只返回 JSON：{asset_class, hypothesis, suggested_factors:[...], risk_notes:[...]}。"
)

FACTOR_SYSTEM = (
    "你是因子工程师。把想法转化为因子定义列表，只返回 JSON："
    "{factors:[{name, kind, window, weight}]}，kind 取值："
    "momentum/mean_reversion/volatility/volume_change/open_interest_change/term_structure。"
)

CODE_SYSTEM = (
    "你是量化开发。根据用户想法生成继承 MultiFactorStrategy 的 Python 策略类，"
    "只返回 Python 代码，且只能 import quantmind 下的模块。"
)

KNOWLEDGE_RICH_SYSTEM = (
    "你是量化研究员，负责把「投资想法 + 学习资料」提炼为可因子化的领域知识摘要。"
    "只依据下方给定资料（库内方法论 + 网络补充 + 想法本身）提炼，**不得虚构任何概念/定义/规则**。"
    "严格只返回 JSON："
    "{concept, definition, buy_signal_rules:[...], candidate_factors:[{kind, reason}], "
    "can_implement:boolean, missing:[...]}。"
    "其中 candidate_factors 的 kind 仅取："
    "momentum/mean_reversion/volatility/volume_change/open_interest_change/term_structure。"
    "若**无法仅凭给定资料给出可计算、可信的实现**（资料不足/定义不明），"
    "则 can_implement 置 false，并在 missing 逐一列出还缺什么（定义/计算规则/示例）；"
    "绝不虚构实现。有充分资料时 can_implement 置 true，missing 为空数组。"
)


def research_prompt(idea: str, asset_class: str = "") -> str:
    ac = f"（资产类别：{asset_class}）" if asset_class else ""
    return f"请研究以下投资想法{ac}：{idea}"


def factor_prompt(idea: str) -> str:
    return f"请为以下想法设计因子：{idea}"


def factor_prompt_knowledge(idea: str, brief: dict) -> str:
    """带领域知识背景的因子生成提示词：注入 KnowledgeBrief 的精确定义/规则/候选。"""
    defn = (brief.get("definition") or "").strip()
    rules = brief.get("buy_signal_rules") or []
    candidates = brief.get("candidate_factors") or []
    parts = [f"请为以下想法设计因子：{idea}"]
    if defn:
        parts.append(f"\n概念定义：{defn}")
    if rules:
        parts.append("\n买入/做多信号规则：\n- " + "\n- ".join(str(r) for r in rules))
    if candidates:
        cand_txt = "\n- ".join(
            f"{c.get('kind')}: {c.get('reason')}" for c in candidates
        )
        parts.append(f"\n方向性因子建议：\n- {cand_txt}")
    parts.append("\n请优先把这些方向性建议转化为具体因子（kind 取值同上）。")
    return "\n".join(parts)


def knowledge_prompt(idea: str, kb_hits: list, web_hits: list) -> str:
    """知识增强提示词：把想法 + 库内命中 + 网络资料交给 LLM 提炼为严格 JSON。"""
    lines = [f"投资想法：{idea}", "\n【库内方法论资料】"]
    if kb_hits:
        for h in kb_hits:
            txt = str(h.get("text") or (h.get("metadata") or {}).get("summary") or "")
            if txt:
                lines.append(f"- {txt}")
    else:
        lines.append("（无库内命中）")
    lines.append("\n【网络补充资料】")
    if web_hits:
        for w in web_hits:
            title = str((w or {}).get("title") or "")
            snippet = str((w or {}).get("snippet") or "")
            lines.append(f"- {title}: {snippet}")
    else:
        lines.append("（无网络资料）")
    lines.append(
        "\n请严格只返回 JSON（不得虚构资料之外的任何概念）："
        "{concept, definition, buy_signal_rules:[...], candidate_factors:[{kind, reason}], "
        "can_implement:boolean, missing:[...]}。"
        "若资料不足以实现则 can_implement 置 false 并在 missing 说明缺什么；足以实现则 true 且 missing 为空。"
    )
    return "\n".join(lines)


def code_prompt(idea: str) -> str:
    return f"请生成交易策略代码：{idea}"
