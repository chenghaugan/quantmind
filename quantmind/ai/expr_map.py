"""因子规格 → 面板 DSL 表达式映射。

供 AI 研究智能体（:mod:`quantmind.ai.agent`）将生成的 ``FactorSpec`` 映射为
P0 面板表达式求值器（:func:`quantmind.research.panel_eval_expression`）可求值的
DSL 表达式，从而用真实面板截面 IC 验证因子假设。

面板 DSL 变量仅支持 ``close / open / high / low / volume / amount``，
算子见 :func:`quantmind.research.list_panel_operators`。
"""
from __future__ import annotations

from typing import Optional

from ..research.target import FactorSpec

_DEFAULT_WINDOW = 20

# kind → 表达式模板（{w} 为窗口占位）
_KIND_TEMPLATES: dict[str, str] = {
    "momentum": "delta(close, {w})",
    "mean_reversion": "-delta(close, {w})",
    "volatility": "std(close, {w})",
    "volume_change": "delta(volume, {w})",
    # 面板无 open_interest 变量，近似用成交量变化表达持仓变化信号
    "open_interest_change": "delta(volume, {w})",
    "term_structure": "mean(close, {w}) - close",
}

_KNOWN_KINDS = set(_KIND_TEMPLATES)


def factor_spec_to_expression(spec: FactorSpec) -> str:
    """把 :class:`FactorSpec` 映射为合法的面板 DSL 表达式字符串。

    优先使用 ``spec.expression``（若已提供非空）；否则按 ``spec.kind`` 的模板
    生成，窗口取 ``spec.window``（非法/非正时回退默认 20）。

    :param spec: 因子规格（kind/window 至少一个需有效）。
    :return: 可被 :func:`panel_eval_expression` 求值的表达式字符串。
    """
    if spec.expression and spec.expression.strip():
        return spec.expression.strip()

    kind = spec.kind or "momentum"
    if kind not in _KNOWN_KINDS:
        kind = "momentum"

    window = spec.window if isinstance(spec.window, int) and spec.window > 0 else _DEFAULT_WINDOW
    return _KIND_TEMPLATES[kind].format(w=window)
