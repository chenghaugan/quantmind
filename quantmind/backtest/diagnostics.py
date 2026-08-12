"""回测严谨性诊断：涨跌停剔除、前视(未来函数)检测、过拟合检测、健康度自检。

这些检查参考 quantskills 的方法论（中性化/严谨回测/健康度）做**原生实现**，
用于提升 QuantMind 回测的可信度，不改变因子/策略本身的逻辑。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.constant import Direction
from ..core.object import BarData
from .analyzer import PerformanceReport
from ..research.evaluator import FactorReport, _spearman


def limit_price_range(
    bars: List[BarData],
    idx: int,
    limit_pct: float = 0.10,
) -> Tuple[Optional[float], Optional[float]]:
    """计算第 *idx* 根 K 线的涨跌停价格区间。

    基于前一根 K 线收盘价 × (1 ± limit_pct)，精度 0.01（A 股最小变动价位）。
    首根 K 线（无前收）或前收 ≤ 0 时返回 ``(None, None)``。

    :return: ``(limit_down_price, limit_up_price)``
    """
    if idx <= 0 or idx >= len(bars):
        return None, None
    prev_close = bars[idx - 1].close_price
    if prev_close <= 0:
        return None, None
    limit_up = round(prev_close * (1 + limit_pct), 2)
    limit_down = round(prev_close * (1 - limit_pct), 2)
    return limit_down, limit_up


def limit_day_mask(bars: List[BarData], limit_pct: Optional[float] = 0.10, eps: float = 1e-6) -> List[Optional[str]]:
    """识别每日是否涨跌停（需 ``limit_pct`` 非 None）。

    以「(今收 - 昨收)/昨收」判断：>= limit_pct 视为涨停('up')，<= -limit_pct 视为跌停('down')。
    首根无昨收 -> None；``limit_pct=None`` 全部返回 None（不做限制）。
    """
    if limit_pct is None:
        return [None] * len(bars)
    out: List[Optional[str]] = []
    prev_close = None
    for b in bars:
        if prev_close is None or prev_close == 0:
            out.append(None)
        else:
            chg = (b.close_price - prev_close) / prev_close
            if chg >= limit_pct - eps:
                out.append("up")
            elif chg <= -limit_pct + eps:
                out.append("down")
            else:
                out.append(None)
        prev_close = b.close_price
    return out


def detect_lookahead(
    factor_values: pd.Series,
    returns: pd.Series,
    fwd_returns: pd.Series,
) -> Dict[str, object]:
    """前视(未来函数)检测：比较因子与「同期收益」和「未来收益」的相关。

    若因子与**同期收益**的相关显著高于与**未来收益**的相关（如 >2 倍且同期显著），
    说明因子可能泄露了未来信息（用到了 t 时刻尚未可得的收益）。
    """
    f = factor_values.reset_index(drop=True).astype(float)
    r = returns.reset_index(drop=True).astype(float)
    fr = fwd_returns.reset_index(drop=True).astype(float)
    contemp_ic = _spearman(f, r)
    fwd_ic = _spearman(f, fr)
    suspected = (
        np.isfinite(contemp_ic) and abs(contemp_ic) > 0.05
        and np.isfinite(fwd_ic) and abs(contemp_ic) > 2.0 * abs(fwd_ic)
    )
    return {
        "contemporaneous_ic": contemp_ic,
        "forward_ic": fwd_ic,
        "lookahead_suspected": bool(suspected),
    }


def diagnose_overfitting(
    train: PerformanceReport,
    test: PerformanceReport,
    sharpe_drop: float = 0.5,
) -> Dict[str, object]:
    """过拟合检测：比较样本内/样本外绩效衰减。

    若样本外 Sharpe < sharpe_drop × 样本内 Sharpe，或样本内盈利而样本外亏损，
    判定为疑似过拟合。
    """
    tr_sharpe = train.sharpe if np.isfinite(train.sharpe) else 0.0
    te_sharpe = test.sharpe if np.isfinite(test.sharpe) else 0.0
    tr_ret = train.total_return if np.isfinite(train.total_return) else 0.0
    te_ret = test.total_return if np.isfinite(test.total_return) else 0.0

    sharpe_degraded = (tr_sharpe > 0) and (te_sharpe < sharpe_drop * tr_sharpe)
    sign_flip = (tr_ret > 0) and (te_ret <= 0)
    overfit = bool(sharpe_degraded or sign_flip)
    return {
        "train_sharpe": tr_sharpe,
        "test_sharpe": te_sharpe,
        "train_return": tr_ret,
        "test_return": te_ret,
        "overfit_suspected": overfit,
    }


@dataclass
class HealthReport:
    """健康度自检结果。"""

    checks: Dict[str, Dict[str, object]] = field(default_factory=dict)
    passed: bool = True

    def to_dict(self) -> dict:
        return {"passed": self.passed, "checks": self.checks}


def health_checks(
    factor_values: Optional[pd.Series] = None,
    bars: Optional[List[BarData]] = None,
    report: Optional[FactorReport] = None,
    perf: Optional[PerformanceReport] = None,
    min_samples: int = 250,
    min_abs_ic: float = 0.02,
    min_abs_ir: float = 0.5,
    max_turnover_annual: float = 60.0,
    max_drawdown: float = 0.5,
    min_coverage: float = 0.8,
) -> HealthReport:
    """因子/回测健康度自检（5 项）。

    1. 样本量充足  2. IC 显著  3. 换手率合理  4. 回撤可控  5. 数据覆盖率高。
    任一项不通过则 ``passed=False``（参考项缺失则标记 skipped，不判失败）。
    """
    checks: Dict[str, Dict[str, object]] = {}

    def add(name, ok, detail, skipped=False):
        checks[name] = {"pass": bool(ok) if not skipped else None, "detail": detail, "skipped": skipped}

    # 1. 样本量
    n = len(factor_values) if factor_values is not None else (len(bars) if bars else 0)
    add("sample_size", n >= min_samples, f"n={n} (>= {min_samples})")

    # 2. IC 显著
    if report is not None:
        ic = report.ic_mean if pd.notna(report.ic_mean) else float("nan")
        ir = report.ir if pd.notna(report.ir) else float("nan")
        ok = pd.notna(ic) and abs(ic) >= min_abs_ic and pd.notna(ir) and abs(ir) >= min_abs_ir
        add("ic_significant", ok, f"|IC|={abs(ic):.4f}, |IR|={abs(ir):.3f}")
    else:
        add("ic_significant", True, "无因子报告，跳过", skipped=True)

    # 3. 换手率合理
    if report is not None and pd.notna(report.turnover_annual):
        to = report.turnover_annual
        add("turnover_reasonable", to <= max_turnover_annual, f"年化换手={to:.1f} (<= {max_turnover_annual})")
    else:
        add("turnover_reasonable", True, "无换手率，跳过", skipped=True)

    # 4. 回撤可控
    if perf is not None and pd.notna(perf.max_drawdown):
        mdd = abs(perf.max_drawdown)
        add("drawdown_controlled", mdd <= max_drawdown, f"最大回撤={perf.max_drawdown:.3f} (>= -{max_drawdown})")
    else:
        add("drawdown_controlled", True, "无绩效报告，跳过", skipped=True)

    # 5. 数据覆盖率
    if factor_values is not None:
        cov = float(factor_values.notna().mean())
        add("data_coverage", cov >= min_coverage, f"覆盖率={cov:.3f} (>= {min_coverage})")
    else:
        add("data_coverage", True, "无因子序列，跳过", skipped=True)

    passed = all(v["pass"] for v in checks.values() if v["pass"] is not None)
    return HealthReport(checks=checks, passed=passed)
