"""截面因子 -> 多空组合回测（研究与回测的桥接）。

把 :mod:`quantmind.research.factors.alpha_cs` 在面板上算出的**严格截面因子**，直接转成
每日横截面排名驱动的多空组合，并用「次根」前向收益做样本外回测，得到可比较的权益曲线与
绩效指标；同时附上同一因子的截面 IC 报告。这样因子研究（IC）与组合表现（Sharpe/回撤）
在同一条流水线上闭环，避免「IC 高但组合做不出来」的脱节。

无前视：第 t 日信号只用 t 日及之前的数据（含 close[t]），组合收益用 close[t]→close[t+fp]
的前向收益，符合严谨回测约定。
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..core.object import BarData
from .factors.alpha_cs import Panel, compute_alpha_cross_sectional, _ALPHA_CS_FUNCS
from .evaluator import FactorEvaluator
from ..backtest.analyzer import PerformanceAnalyzer


def cross_sectional_backtest(
    panel: Panel,
    factor_name: str,
    forward_periods: int = 1,
    n_groups: int = 5,
    long_short: bool = True,
    cost_rate: float = 0.0,
) -> Dict:
    """把面板截面因子转成每日横截面多空组合并回测。

    :param panel: 多标的面板（index=日期，columns=标的）。
    :param factor_name: 截面因子名（alpha002..alpha101 / alpha191_*）。
    :param forward_periods: 持仓周期（根），收益用 close[t]→close[t+fp]。
    :param n_groups: 分组数；头组做多、尾组做空（等权）。
    :param long_short: True=多空组合，False=仅多头组合。
    :param cost_rate: 每期双边成本近似（直接扣减组合收益），如 0.001。
    :returns: {factor, n_symbols, n_dates, ic_report, portfolio{...}}
    """
    if factor_name not in _ALPHA_CS_FUNCS:
        raise KeyError(f"未知截面 Alpha 因子: {factor_name}")
    if panel.close.empty:
        raise ValueError("面板为空，无法回测")

    # 分组数自适应：不能超过可用标的数，且多空至少需 2 组
    n_symbols = len(panel.symbols)
    n_groups = max(2, min(n_groups, n_symbols))

    scores = compute_alpha_cross_sectional([factor_name], panel)[factor_name]  # date×symbol
    close = panel.close
    fwd = close.pct_change(forward_periods).shift(-forward_periods)  # t 行 = t->t+fp 收益

    dates = list(scores.index)
    equity = 1.0
    curve: List[dict] = []
    port_ret: List[float] = []
    for d in dates:
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

    evaluator = FactorEvaluator()
    ic_reports = evaluator.evaluate_cross_sectional_panel(
        [factor_name], panel, forward_periods=forward_periods, n_groups=n_groups
    )
    ic_rep = ic_reports.get(factor_name)

    return {
        "factor": factor_name,
        "n_symbols": len(panel.symbols),
        "n_dates": len(curve),
        "ic_report": ic_rep.to_dict() if ic_rep else None,
        "portfolio": {
            **perf.to_dict(),
            "daily_returns": [float(x) for x in port_ret],
        },
    }
