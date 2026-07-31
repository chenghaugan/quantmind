"""AI 提示词模板（研究 / 因子生成 / 代码生成）。"""
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


def research_prompt(idea: str, asset_class: str = "") -> str:
    ac = f"（资产类别：{asset_class}）" if asset_class else ""
    return f"请研究以下投资想法{ac}：{idea}"


def factor_prompt(idea: str) -> str:
    return f"请为以下想法设计因子：{idea}"


def code_prompt(idea: str) -> str:
    return f"请生成交易策略代码：{idea}"
