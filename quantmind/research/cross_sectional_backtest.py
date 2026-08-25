"""截面因子 -> 多空组合回测（研究与回测的桥接）。

把因子（硬编码截面 Alpha 或**任意 DSL 表达式**）在面板上算出的严格截面因子，
直接转成每日横截面排名驱动的多空组合，并用「次根」前向收益做样本外回测，得到
可比较的权益曲线与绩效指标；同时附上同一因子的截面 IC 报告。这样因子研究（IC）
与组合表现（Sharpe/回撤）在同一条流水线上闭环——**挖掘出的因子**（``co/ea/tot``
搜索产出的表达式）可直接回测，避免「IC 高但组合做不出来」的脱节。

无前视：第 t 日信号只用 t 日及之前的数据（含 close[t]），组合收益用
close[t]→close[t+fp] 的前向收益，符合严谨回测约定。
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .factors.alpha_cs import Panel, compute_alpha_cross_sectional, _ALPHA_CS_FUNCS
from .factors.panel_expr import panel_eval_expression
from .evaluator import FactorEvaluator
from ..backtest.analyzer import PerformanceAnalyzer


def _factor_scores(panel: Panel, factor_name: str, expression: str | None) -> pd.DataFrame:
    """按给定因子名或 DSL 表达式计算因子面板（date×symbol）。"""
    if expression:
        # 兼容别名：面板语法用 delay()，单标的时序语法常用 ref()，此处归一化（失败闭合）。
        import re as _re
        _expr = _re.sub(r"\bref\(([^()]*)\)", r"delay(\1)", expression)
        return panel_eval_expression(_expr, panel)
    if factor_name not in _ALPHA_CS_FUNCS:
        raise KeyError(f"未知截面 Alpha 因子: {factor_name}")
    return compute_alpha_cross_sectional([factor_name], panel)[factor_name]


def _run_portfolio(panel, scores, forward_periods, n_groups, long_short, cost_rate):
    """得分面板 -> 每日横截面多空组合 -> 权益曲线 + 绩效。"""
    n_symbols = len(panel.symbols)
    n_groups = max(2, min(n_groups, n_symbols))
    close = panel.close
    fwd = close.pct_change(forward_periods).shift(-forward_periods)

    curve: list = []
    port_ret: list = []
    equity = 1.0
    for d in scores.index:
        s = scores.loc[d]
        r = fwd.loc[d]
        valid = s.notna() & r.notna()
        if valid.sum() < n_groups:
            curve.append({"date": d, "equity": equity})
            continue
        sv = s[valid]
        rv = r[valid]
        try:
            groups = pd.qcut(sv.rank(method="first"), n_groups, labels=False)
        except ValueError:
            groups = pd.cut(sv.rank(method="first"), n_groups, labels=False)
        long_mask = groups == n_groups - 1
        short_mask = groups == 0
        long_ret = float(rv[long_mask].mean())
        short_ret = float(rv[short_mask].mean())
        day_ret = (long_ret - short_ret) if long_short else long_ret
        day_ret = day_ret - cost_rate
        equity *= (1.0 + day_ret)
        curve.append({"date": d, "equity": equity})
        port_ret.append(day_ret)

    analyzer = PerformanceAnalyzer()
    perf = analyzer.analyze(curve, [])
    return curve, port_ret, perf


def factor_expression_backtest(
    expression: str,
    panel: Panel,
    forward_periods: int = 1,
    n_groups: int = 5,
    long_short: bool = True,
    cost_rate: float = 0.0,
) -> Dict:
    """对**任意 DSL 因子表达式**直接做截面多空组合回测（挖掘 → 回测闭环）。

    :param expression: 面板 DSL 因子表达式（如 ``delta(close,20)``、
        ``ts_zscore(close,30) - rank(close,20)``）。
    其余参数含义同 :func:`cross_sectional_backtest`。
    :returns: {factor, expression, n_symbols, n_dates, ic_report, portfolio{...}}
    """
    if panel.close.empty:
        raise ValueError("面板为空，无法回测")
    scores = _factor_scores(panel, "", expression)
    curve, port_ret, perf = _run_portfolio(
        panel, scores, forward_periods, n_groups, long_short, cost_rate)

    e = FactorEvaluator()
    ic_rep = e.evaluate_factor_panel(scores, panel, forward_periods=forward_periods,
                                     n_groups=n_groups, factor_name=expression)
    return {
        "factor": expression,
        "expression": expression,
        "n_symbols": len(panel.symbols),
        "n_dates": len(curve),
        "ic_report": ic_rep.to_dict(),
        "portfolio": {**perf.to_dict(), "daily_returns": [float(x) for x in port_ret]},
    }


def cross_sectional_backtest(
    panel: Panel,
    factor_name: str,
    forward_periods: int = 1,
    n_groups: int = 5,
    long_short: bool = True,
    cost_rate: float = 0.0,
    expression: str | None = None,
) -> Dict:
    """把面板截面因子转成每日横截面多空组合并回测。

    :param panel: 多标的面板（index=日期，columns=标的）。
    :param factor_name: 截面因子名（alpha002..alpha101 / alpha191_*）；若同时传入
        ``expression``，则以 ``expression`` 为准（DSL 表达式优先）。
    :param forward_periods: 持仓周期（根），收益用 close[t]→close[t+fp]。
    :param n_groups: 分组数；头组做多、尾组做空（等权）。
    :param long_short: True=多空组合，False=仅多头组合。
    :param cost_rate: 每期双边成本近似（直接扣减组合收益），如 0.001。
    :param expression: 可选任意 DSL 因子表达式（优先于 factor_name）。
    :returns: {factor, n_symbols, n_dates, ic_report, portfolio{...}}
    """
    if panel.close.empty:
        raise ValueError("面板为空，无法回测")
    label = expression or factor_name
    scores = _factor_scores(panel, factor_name, expression)
    curve, port_ret, perf = _run_portfolio(
        panel, scores, forward_periods, n_groups, long_short, cost_rate)

    evaluator = FactorEvaluator()
    if expression:
        ic_rep = evaluator.evaluate_factor_panel(
            scores, panel, forward_periods=forward_periods,
            n_groups=n_groups, factor_name=expression)
    else:
        ic_reports = evaluator.evaluate_cross_sectional_panel(
            [factor_name], panel, forward_periods=forward_periods, n_groups=n_groups)
        ic_rep = ic_reports.get(factor_name)

    return {
        "factor": label,
        "expression": expression,
        "n_symbols": len(panel.symbols),
        "n_dates": len(curve),
        "ic_report": ic_rep.to_dict() if ic_rep else None,
        "portfolio": {
            **perf.to_dict(),
            "daily_returns": [float(x) for x in port_ret],
        },
    }

