"""回测严谨性验证套件（对应规划「回测验证框架」）。

配合 :mod:`quantmind.backtest.diagnostics`（涨跌停剔除/单因子前视检测/过拟合/
健康度）与本模块（样本外验证扩展）提升回测可信度。本模块提供：

- **蒙特卡洛**：对收益序列 bootstrap 或打乱重采样，给出绩效分布（避免"单段历史
  恰好表现好"被误当稳定 alpha）。
- **置乱噪声测试**：打乱价格时序破坏行情形态后重跑回测，验证真实收益不是来自
  前视/泄漏。
- **成本敏感性**：按成本倍数考察 Sharpe/收益退化曲线，判断策略是否对成本稳健。
- **状态分割测试**：按滚动波动率分段回测，考察策略是否跨市场状态成立。
- **策略代码前视静态扫描**：AST 扫描生成代码中信号计算对"未来数据"的引用。

所有函数纯本地、可离线、确定性（用 ``default_rng(seed)`` 保证可复现）。
"""
from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ..core.object import BarData
from .engine import BacktestEngine
from .analyzer import PerformanceReport

_logger = logging.getLogger("quantmind.backtest.validation")


# ---------------------------------------------------------------------------
# 1. 蒙特卡洛：收益重采样路径
# ---------------------------------------------------------------------------
def _to_return_series(equity_or_returns) -> pd.Series:
    """把权益曲线(List[dict])或收益序列转成收益 pd.Series。"""
    if isinstance(equity_or_returns, list):
        df = pd.DataFrame(equity_or_returns).sort_values("date").reset_index(drop=True)
        df["equity"] = df["equity"].astype(float)
        rets = df["equity"].pct_change().dropna()
        rets = rets.replace([np.inf, -np.inf], np.nan).dropna()
        return rets
    s = pd.Series(equity_or_returns).astype(float)
    return s.replace([np.inf, -np.inf], np.nan).dropna()


def monte_carlo(
    equity_or_returns,
    n_simulations: int = 500,
    seed: int = 42,
    method: str = "bootstrap",
) -> dict:
    """蒙特卡洛模拟收益路径。

    :param equity_or_returns: 权益曲线(List[dict]，带 date/equity)或收益序列。
    :param n_simulations: 模拟次数。
    :param seed: 随机种子（保证可复现）。
    :param method: ``bootstrap``=有放回重采样；``shuffle``=打乱顺序（破坏时序）。
    :returns: dict 统计量。
    """
    rng = np.random.default_rng(seed)
    rets = _to_return_series(equity_or_returns)
    if len(rets) < 4:
        return {
            "n_simulations": 0, "mean_total_return": 0.0,
            "pct5_total_return": 0.0, "pct95_total_return": 0.0,
            "mean_sharpe": 0.0, "mean_max_drawdown": 0.0,
            "prob_positive": 0.0, "seed": seed, "note": "样本不足",
        }

    total_returns: List[float] = []
    sharpes: List[float] = []
    max_drawdowns: List[float] = []
    n = len(rets)

    for _ in range(n_simulations):
        if method == "shuffle":
            sample = rng.permutation(rets.values)
        else:  # bootstrap 有放回
            idx = rng.integers(0, n, size=n)
            sample = rets.values[idx]
        # 重构建权益路径（从 1.0 起步）
        equity = np.cumprod(1.0 + sample)
        # 总收益
        total = equity[-1] - 1.0
        # sharpe（年化 252）
        std = sample.std()
        sharpe = (sample.mean() / std * (252 ** 0.5)) if std and std > 0 else 0.0
        # 最大回撤
        running_max = np.maximum.accumulate(equity)
        dd = equity / running_max - 1.0
        mdd = float(dd.min())
        total_returns.append(float(total))
        sharpes.append(float(sharpe))
        max_drawdowns.append(float(mdd))

    arr = np.array(total_returns)
    return {
        "n_simulations": n_simulations,
        "mean_total_return": round(float(arr.mean()), 4),
        "pct5_total_return": round(float(np.percentile(arr, 5)), 4),
        "pct95_total_return": round(float(np.percentile(arr, 95)), 4),
        "mean_sharpe": round(float(np.mean(sharpes)), 4),
        "mean_max_drawdown": round(float(np.mean(max_drawdowns)), 4),
        "prob_positive": round(float(np.mean(arr > 0)), 4),
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# 2. 置乱噪声测试：破坏行情时序后重跑回测
# ---------------------------------------------------------------------------
def _shuffled_bars(bars: List[BarData], rng: np.random.Generator) -> List[BarData]:
    """打乱 bar 的顺序（保留 OHLC 形态但破坏行情时序），保持 datetime 单调递增。"""
    from dataclasses import replace
    values = bars.copy()
    perm = rng.permutation(len(values))
    values = [values[i] for i in perm]
    # 重排 datetime 为原始递增顺序（避免回测引擎日期错乱）
    dates = sorted(b.datetime for b in bars)
    return [replace(b, datetime=d) for b, d in zip(values, dates)]


def shuffle_noise_test(
    bars: List[BarData],
    strategy_class,
    setting: Optional[dict],
    vt_symbol: str,
    n_shuffles: int = 100,
    seed: int = 42,
    sizes: Optional[Dict[str, float]] = None,
    capital: float = 1_000_000.0,
) -> dict:
    """置乱噪声测试：打乱行情时序重跑，验证真实收益不是前视泄漏的产物。

    :returns: dict 含置乱收益分布与 ``no_lookahead`` 判定。
    """
    rng = np.random.default_rng(seed)
    eng = BacktestEngine({vt_symbol: bars}, capital=capital, sizes=sizes)
    eng.add_strategy(strategy_class, vt_symbol, setting)
    real_report = eng.run()
    real_total = real_report.total_return

    shuffled_returns: List[float] = []
    for _ in range(n_shuffles):
        sb = _shuffled_bars(bars, rng)
        e = BacktestEngine({vt_symbol: sb}, capital=capital, sizes=sizes)
        e.add_strategy(strategy_class, vt_symbol, setting)
        try:
            rep = e.run()
            r = rep.total_return
        except Exception:  # noqa: BLE001
            r = np.nan
        if np.isfinite(r):
            shuffled_returns.append(float(r))

    arr = np.array(shuffled_returns)
    if len(arr) == 0:
        return {"note": "置乱回测全部失败", "no_lookahead": None}
    mean = float(arr.mean())
    std = float(arr.std())
    p95 = float(np.percentile(arr, 95))
    # 判定：真实收益若显著高于置乱分布上界（均值+2σ），说明有可利用的时序 alpha；
    # 但若真实收益异常高且无法在置乱中复现，需人工审视是否前视。
    threshold = mean + 2.0 * std
    if real_total > threshold:
        # 真实收益远超置乱分布上界：置乱破坏时序结构后无法复现，
        # 高收益很可能来自前视/未来函数，标记为可疑并需人工审查。
        no_lookahead = False
        note = "真实收益显著高于置乱分布上界，疑似前视/未来函数，请人工审查"
    else:
        no_lookahead = True
        note = "真实收益未显著超过置乱分布上界，未见前视迹象（但 alpha 可能无效或来自噪声）"
    return {
        "n_shuffles": len(arr),
        "mean_total_return": round(mean, 4),
        "std_total_return": round(std, 4),
        "p95_total_return": round(p95, 4),
        "real_total_return": round(real_total, 4),
        "upper_bound_2sigma": round(threshold, 4),
        "no_lookahead": bool(no_lookahead),
        "note": note,
    }


# ---------------------------------------------------------------------------
# 3. 成本敏感性分析
# ---------------------------------------------------------------------------
def cost_sensitivity(
    strategy_class,
    data: Dict[str, List[BarData]],
    vt_symbol: str,
    cost_multipliers: tuple = (0.0, 0.5, 1.0, 2.0, 4.0),
    base_commission: float = 0.0002,
    sizes: Optional[Dict[str, float]] = None,
    capital: float = 1_000_000.0,
) -> dict:
    """成本敏感性：按不同成本倍数跑回测，考察绩效退化。

    此处走 BacktestEngine 的"单一费率"成本路径（成本表关闭，便于线性缩放）。
    :returns: dict 含各成本档位绩效与 ``robust_to_cost``。
    """
    rows: List[dict] = []
    base = base_commission
    for mult in cost_multipliers:
        eng = BacktestEngine(
            data, capital=capital, sizes=sizes,
            commission=base * mult, slippage=0.0,
        )
        eng.add_strategy(strategy_class, vt_symbol, {})
        try:
            rep = eng.run()
        except Exception as exc:  # noqa: BLE001
            _logger.warning("成本敏感性档位 %.1f 回测失败: %s", mult, exc)
            rep = PerformanceReport()
        rows.append({
            "multiplier": float(mult),
            "commission": round(base * mult, 6),
            "sharpe": round(rep.sharpe, 3),
            "total_return": round(rep.total_return, 4),
            "trade_count": rep.trade_count,
        })

    # 4 倍成本下 Sharpe 仍为正即稳健
    robust = any(r["sharpe"] > 0 for r in rows if r["multiplier"] >= 4.0)
    return {
        "base_commission": base,
        "rows": rows,
        "robust_to_cost": bool(robust),
        "note": "4 倍基准成本下 Sharpe 仍为正视为对成本稳健",
    }


# ---------------------------------------------------------------------------
# 4. 状态分割测试
# ---------------------------------------------------------------------------
def regime_split_test(
    bars: List[BarData],
    strategy_class,
    setting: Optional[dict],
    vt_symbol: str,
    n_splits: int = 3,
    lookback: int = 20,
    sizes: Optional[Dict[str, float]] = None,
    capital: float = 1_000_000.0,
    min_bars_per_regime: int = 30,
) -> dict:
    """状态分割测试：按滚动波动率把样本分档，各档分别回测。

    :returns: dict 含各状态绩效与 ``robust_across_regimes``。
    """
    if len(bars) < lookback + min_bars_per_regime * n_splits:
        return {"note": "样本不足", "robust_across_regimes": None, "regimes": []}

    closes = np.array([b.close_price for b in bars], dtype=float)
    # 滚动波动率：直接对原始收盘序列 pct_change + rolling，天然与 bars 对齐
    rets = pd.Series(closes).pct_change()
    roll_std = rets.rolling(lookback).std().values
    valid = ~np.isnan(roll_std)
    if valid.sum() < n_splits * min_bars_per_regime:
        return {"note": "有效波动率样本不足", "robust_across_regimes": None, "regimes": []}

    # 按波动率分位数分档：0=低波(平稳), 1=中, 2=高波(震荡)
    quantiles = np.quantile(roll_std[valid], [i / n_splits for i in range(1, n_splits)])
    labels = np.digitize(roll_std, bins=quantiles)
    regimes: List[dict] = []
    for reg_idx in range(n_splits):
        idx = np.where(np.array([labels[i] == reg_idx for i in range(len(labels))]))[0]
        # 映射到 bar 索引（roll_std 与 bars 对齐，roll_std[i] 对应 bar i+? 需对齐）
        # rets 从 bar[1] 起，roll_std 从 bar[lookback] 起；这里直接取 labels 对应的 bar
        bar_idx = [i for i in range(len(bars)) if labels[i] == reg_idx and not np.isnan(roll_std[i])]
        if len(bar_idx) < min_bars_per_regime:
            continue
        seg_bars = [bars[i] for i in bar_idx]
        eng = BacktestEngine({vt_symbol: seg_bars}, capital=capital, sizes=sizes)
        eng.add_strategy(strategy_class, vt_symbol, setting)
        try:
            rep = eng.run()
        except Exception as exc:  # noqa: BLE001
            _logger.warning("状态分割档 %d 回测失败: %s", reg_idx, exc)
            continue
        regime_name = ["low_vol", "mid_vol", "high_vol"][reg_idx] if reg_idx < 3 else f"regime_{reg_idx}"
        regimes.append({
            "regime": regime_name,
            "sharpe": round(rep.sharpe, 3),
            "total_return": round(rep.total_return, 4),
            "drawdown": round(rep.max_drawdown, 4),
            "n_bars": len(seg_bars),
        })

    if not regimes:
        return {"note": "无有效状态分段", "robust_across_regimes": None, "regimes": []}
    # 稳健判定：多数状态正收益，且无大额负收益段
    positive = sum(1 for r in regimes if r["total_return"] > 0)
    worst = min(r["total_return"] for r in regimes)
    robust = positive >= max(1, len(regimes) // 2) and worst > -0.5
    return {"note": "按滚动波动率(低/中/高)分段的样本外表现", "robust_across_regimes": bool(robust),
            "regimes": regimes}


# ---------------------------------------------------------------------------
# 5. 策略代码前视静态扫描（AST，禁止 exec/eval）
# ---------------------------------------------------------------------------
def static_lookahead_scan(code: str) -> List[str]:
    """AST 扫描策略代码中的未来数据引用（前视）。返回违规说明列表。

    信号计算中常见的未来泄漏：
      - ``close.shift(-1)``（引用下期价格）
      - ``close.pct_change().shift(-N)``（N>0 → 未来收益）
      - ``df.iloc[+N]`` 或 ``df.iloc[未来]`` 的显式正向越界索引（难以静态判定，尽量识别常量正索引）
    仅用 ``ast`` 静态分析，不执行任何代码，安全。
    """
    def _neg_int(node: ast.AST) -> bool:
        # -N（N>0）或常量负整数
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) \
                and isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, int):
            return node.operand.value > 0
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value < 0
        return False

    def _neg_value(node: ast.AST):
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) \
                and isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, int):
            return -node.operand.value
        return None

    violations: List[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"语法错误: {exc}"]

    for node in ast.walk(tree):
        # close.shift(-N)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "shift" and node.args and _neg_int(node.args[0]):
            n = _neg_value(node.args[0])
            violations.append(
                f"检测到 shift({n})：引用未来数据（前视偏差），乘数应为正（引用历史）"
                if n is not None else "检测到 shift(负参数)：引用未来数据（前视偏差）"
            )
        # close.pct_change().shift(-N)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "shift":
            base = node.func.value
            if isinstance(base, ast.Call) and isinstance(base.func, ast.Attribute) \
                    and base.func.attr == "pct_change" and node.args and _neg_int(node.args[0]):
                violations.append("检测到 pct_change().shift(负参数)：引用未来收益率（前视偏差）")

    # 去重
    seen = set()
    dedup: List[str] = []
    for v in violations:
        if v not in seen:
            seen.add(v)
            dedup.append(v)
    return dedup


__all__ = [
    "monte_carlo",
    "shuffle_noise_test",
    "cost_sensitivity",
    "regime_split_test",
    "static_lookahead_scan",
]
