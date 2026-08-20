"""CoT（链式精炼）因子搜索的提示词与文本解析（对标 AlphaBench ``searcher/algo/cot.py``）。

真实 LLM 场景下，把「链历史 + 各因子 IC 指标」呈现给模型，请求其基于此前
结果变异/精炼出一个改进的因子表达式，返回 JSON ``{"expression": "..."}``。
离线/无模型时由调用方回落到 :func:`quantmind.research.search.base.mutate_expressions`。
"""
from __future__ import annotations

import json
import logging
import re
from typing import List, Optional

_logger = logging.getLogger("quantmind.research.search.prompts")

# 允许的变量与示例，辅助 LLM 生成合法表达式
_VAR_HINT = "close, open, high, low, volume, amount"

SEARCH_SYSTEM = (
    "You are an expert quantitative researcher performing iterative alpha factor "
    "mining. You are given a chain of candidate factor expressions with their "
    "evaluation metrics (Rank IC = Spearman correlation between the factor and "
    "next-period cross-sectional stock returns). Your goal is to propose ONE new, "
    "improved factor expression that keeps the predictive signal while improving "
    "stability and reducing redundancy.\n"
    "Use only these variables: $close, $open, $high, $low, $volume, $amount.\n"
    "Use these operators (QLib style): Mean(x,n), Std(x,n), Sum(x,n), Rank(x) "
    "(cross-sectional rank), TsRank(x,n), Min(x,n), Max(x,n), Delay(x,n), Delta(x,n), "
    "Corr(a,b,n), Cov(a,b,n), TsZscore(x,n), Sign(x), Abs(x), Log(x), Power(x,a).\n"
    "Return ONLY a JSON object with a single field \"expression\" containing a valid "
    "factor expression string."
)


def build_kb_block(knowledge_context: dict) -> str:
    """把知识库上下文（历史已验证因子模式 + 失败避坑 + 历史 brief）拼成可注入搜索 prompt 的文本。

    复用 :func:`quantmind.research.knowledge_loop.format_kb_context` 的格式化逻辑；
    上下文为空时返回空串（不污染 prompt）。用 try/except 包裹 import，防循环依赖。
    """
    if not knowledge_context:
        return ""
    try:
        from ..knowledge_loop import format_kb_context
        body = format_kb_context(knowledge_context)
    except Exception as exc:  # noqa: BLE001
        _logger.debug("knowledge_loop.format_kb_context 不可用，跳过知识库注入: %s", exc)
        body = ""
    if not body:
        return ""
    return ("Below is historical knowledge distilled from previous mining runs. "
            "Reference it to avoid repeating known failure patterns and to reuse "
            "verified factor patterns.\n" + body)


def build_chain_prompt(
    seed: str,
    history: List[dict],
    best_expression: str,
    best_rank_ic: Optional[float],
    instruction: str = "",
    kb_block: str = "",
) -> str:
    """构造 CoT 用户消息：把链历史与当前最优呈现给模型。

    Args:
        seed: 初始因子表达式。
        history: 已评估的候选（dict: expression/rank_ic/improved）。
        best_expression: 当前最优表达式。
        best_rank_ic: 当前最优 Rank IC（可为 None）。
        instruction: 附加变体指示（如"更侧重波动率"）。
        kb_block: 可选的历史知识库上下文文本（由 :func:`build_kb_block` 生成）。
            非空时插在 seed 之前，引导模型参考历史经验、避免重复失败模式。

    Returns:
        User 消息字符串。
    """
    lines: List[str] = []
    if kb_block:
        lines.append(kb_block)
        lines.append(
            "参考以上历史知识，避免重复失败模式，并尽量复用已验证的因子模式。"
        )
    lines += [
        f"Seed factor: {seed}",
        "Iterative search history:",
    ]
    if not history:
        lines.append("  (no prior candidates yet)")
    for i, h in enumerate(history):
        ric = h.get("rank_ic")
        ric_s = f"{float(ric):.4f}" if ric is not None and ric == ric else "n/a"
        mark = " <- current best" if h.get("is_best") else ""
        lines.append(f"  Round {i + 1}: {h.get('expression')}  RankIC={ric_s}{mark}")
    lines.append(f"Current best factor: {best_expression}  RankIC="
                 f"{(f'{float(best_rank_ic):.4f}' if best_rank_ic is not None and best_rank_ic == best_rank_ic else 'n/a')}")
    if instruction:
        lines.append(f"Variation direction: {instruction}")
    lines.append(
        "Propose a single new factor expression likely to outperform the current "
        "best. Return JSON: {\"expression\": \"...\"}"
    )
    return "\n".join(lines)


def parse_expression_response(text: str, fallback: Optional[str] = None) -> Optional[str]:
    """从 LLM 回复中解析出单个因子表达式。

    依次尝试：
      1. 整段是 JSON 且含 ``expression`` 字段；
      2. 用正则提取 JSON 块中的 expression；
      3. 直接取首行/去引号的裸表达式。

    Returns:
        表达式字符串；无法解析返回 ``fallback``。
    """
    text = (text or "").strip()
    if not text:
        return fallback

    def _clean(expr: str) -> str:
        expr = expr.strip()
        if expr.startswith("```"):
            # 去掉代码围栏
            expr = re.sub(r"^```[a-zA-Z]*\n?", "", expr)
            expr = re.sub(r"\n?```$", "", expr)
        return expr.strip().strip("\"'`").strip()

    # 1) 整段 JSON
    try:
        data = json.loads(_clean(text))
        if isinstance(data, dict) and data.get("expression"):
            return _clean(str(data["expression"]))
    except (json.JSONDecodeError, ValueError):
        pass

    # 2) 提取 JSON 块
    m = re.search(r"\{[^{}]*\"expression\"\s*:\s*(\"[^\"]+\")\s*[^{}]*\}", text)
    if m:
        try:
            return _clean(json.loads(m.group(1)))
        except (json.JSONDecodeError, ValueError):
            pass

    # 3) 裸表达式：去掉可能的前缀/引号，取第一行
    for line in text.splitlines():
        line = _clean(line)
        if line and "=" not in line.split(":", 1)[0] and not line.startswith("{"):
            # 期望形如 Mean($close, 20) 或 mean(close,20)
            if re.search(r"[\w\$]+\s*\(", line):
                return line
    if "expression" in text:
        m2 = re.search(r"\"expression\"\s*:\s*(\"[^\"]*\")", text)
        if m2:
            try:
                return _clean(json.loads(m2.group(1)))
            except (json.JSONDecodeError, ValueError):
                pass
    return fallback
