"""Walk-forward（滚动窗口）样本外验证。

把一段历史切成多个「训练窗 + 测试窗」折，每折在**测试窗**上跑回测（训练窗仅作指标
预热 burn-in，我们的策略无待拟合参数，故等价于滚动样本外验证）。逐折收集绩效，并用
:func:`diagnose_overfitting` 把「全样本(in-sample)」与「各折样本外均值」对比，给出过拟合
预警——这是严谨回测的标配，避免单段历史上的偶然好成绩被误认为稳定 alpha。

参考 quantskills 的 walk-forward / 过拟合检测方法论做**原生实现**。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from ..core.object import BarData
from .engine import BacktestEngine
from .analyzer import PerformanceReport
from .diagnostics import diagnose_overfitting


@dataclass
class WalkForwardFold:
    """单折结果。"""

    index: int
    start: datetime
    end: datetime
    report: PerformanceReport

    def to_dict(self) -> dict:
        d = self.report.to_dict()
        d.update({
            "fold": self.index,
            "start": self.start.isoformat() if isinstance(self.start, datetime) else str(self.start),
            "end": self.end.isoformat() if isinstance(self.end, datetime) else str(self.end),
        })
        return d


@dataclass
class WalkForwardResult:
    """Walk-forward 总结果。"""

    folds: List[WalkForwardFold]
    aggregate: dict
    overfit_suspected: bool
    detail: dict

    def to_dict(self) -> dict:
        return {
            "folds": [f.to_dict() for f in self.folds],
            "aggregate": self.aggregate,
            "overfit_suspected": self.overfit_suspected,
            "detail": self.detail,
        }


def walk_forward(
    bars: List[BarData],
    strategy_class,
    setting: Optional[dict],
    vt_symbol: str,
    train_window: int = 250,
    test_window: int = 60,
    step: Optional[int] = None,
    sizes: Optional[Dict[str, float]] = None,
    capital: float = 1_000_000.0,
    cost=None,
) -> WalkForwardResult:
    """滚动窗口 walk-forward 验证。

    :param bars: 完整历史 K 线（按时间升序）。
    :param strategy_class: 策略类（如 MultiFactorStrategy）。
    :param setting: 策略参数字典。
    :param vt_symbol: 标的（如 "rb0.SHFE"）。
    :param train_window: 训练/预热窗口长度（根）。注意：当前实现每折仅回放
        测试窗口，训练窗不参与引擎回放；策略自身的 warmup（如均线窗口）
        会在每折开头产生空跑期，折绩效会被相应低估。
    :param test_window: 每折测试窗口长度（根）。
    :param step: 滚动步长（默认 = test_window，即不重叠切分）。
    :param cost: 成本模型（None=旧式单一费率；或 CostModel / dict / True）。
    :returns: 各折绩效 + 聚合指标 + 过拟合预警。
    """
    step = step or test_window
    n = len(bars)
    if n < train_window + test_window:
        raise ValueError(f"样本不足：需要至少 {train_window + test_window} 根，仅有 {n}")

    folds: List[WalkForwardFold] = []
    start_idx = 0
    i = 0
    while start_idx + train_window + test_window <= n:
        test_slice = bars[start_idx + train_window: start_idx + train_window + test_window]
        if not test_slice:
            break
        eng = BacktestEngine({vt_symbol: test_slice}, capital=capital,
                             sizes=sizes, cost_table=cost)
        eng.add_strategy(strategy_class, vt_symbol, setting)
        report = eng.run()
        folds.append(WalkForwardFold(i, test_slice[0].datetime, test_slice[-1].datetime, report))
        start_idx += step
        i += 1

    if not folds:
        raise ValueError("未生成任何折（请检查 train_window/test_window/step 与样本长度）")

    sharpes = [f.report.sharpe for f in folds]
    returns = [f.report.total_return for f in folds]
    sharpes_f = [s for s in sharpes if np.isfinite(s)]
    returns_f = [r for r in returns if np.isfinite(r)]
    mean_sharpe = float(np.mean(sharpes_f)) if sharpes_f else 0.0
    mean_return = float(np.mean(returns_f)) if returns_f else 0.0
    std_return = float(np.std(returns_f)) if len(returns_f) > 1 else 0.0
    positive_rate = float(np.mean([1 if r > 0 else 0 for r in returns_f])) if returns_f else 0.0

    # 过拟合判定：全样本(in-sample) vs 各折样本外均值(out-of-sample)
    full = BacktestEngine({vt_symbol: bars}, capital=capital, sizes=sizes, cost_table=cost)
    full.add_strategy(strategy_class, vt_symbol, setting)
    full_report = full.run()
    oos_report = PerformanceReport(sharpe=mean_sharpe, total_return=mean_return)
    diag = diagnose_overfitting(full_report, oos_report)

    return WalkForwardResult(
        folds=folds,
        aggregate={
            "n_folds": len(folds),
            "train_window": train_window,
            "test_window": test_window,
            "step": step,
            "mean_sharpe": round(mean_sharpe, 3),
            "mean_total_return": round(mean_return, 4),
            "std_total_return": round(std_return, 4),
            "positive_rate": round(positive_rate, 3),
        },
        overfit_suspected=bool(diag["overfit_suspected"]),
        detail=diag,
    )
