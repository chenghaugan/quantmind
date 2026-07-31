"""WorldQuant 风格时序原语（单资产近似版）。

WorldQuant Alpha 公式中大量使用**截面(cross-sectional) rank / correlation**，
但 QuantMind 的回测/评估多在**单标的时序**语境下运行。为在单标的下仍可计算并评估，
本模块把 ``rank`` 近似为「滚动窗口内的分位排名」（即最后一根在窗口内的百分位），
把 ``correlation/covariance`` 实现为滚动窗口时序相关。`scale` 用滚动窗口内的
绝对值之和归一。这样得出的因子在单标的情况下是**有意义的时序代理**，但在多标的
横截面上会弱于原版——生产环境若需严格截面，应在多标的 DataFrame 上替换 ``rank`` 实现。

公式来源：WorldQuant Alpha101/Alpha191 公开文献（仅重实现数学公式，未复制任何仓库代码）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 滚动 rank 默认窗口（约 1 个交易年的交易日）
RANK_WINDOW = 250


def _rank(s: pd.Series, win: int = RANK_WINDOW) -> pd.Series:
    """分位排名：最后一根在滚动窗口内的百分位（0~1）。近似截面 rank。"""
    s = s.astype(float)
    return s.rolling(win, min_periods=5).apply(lambda x: (x[-1] < x).mean(), raw=True)


def _delay(s: pd.Series, d: int) -> pd.Series:
    return s.shift(d)


def _delta(s: pd.Series, d: int) -> pd.Series:
    return s - s.shift(d)


def _corr(a: pd.Series, b: pd.Series, d: int) -> pd.Series:
    return a.rolling(d, min_periods=max(2, d // 2)).corr(b)


def _cov(a: pd.Series, b: pd.Series, d: int) -> pd.Series:
    return a.rolling(d, min_periods=max(2, d // 2)).cov(b)


def _ts_min(s: pd.Series, d: int) -> pd.Series:
    return s.rolling(d, min_periods=1).min()


def _ts_max(s: pd.Series, d: int) -> pd.Series:
    return s.rolling(d, min_periods=1).max()


def _ts_arg_max(s: pd.Series, d: int) -> pd.Series:
    """距窗口内最大值的周期数（d-1 - argmax）。"""

    def f(x: np.ndarray) -> float:
        return d - 1 - int(np.argmax(x))

    return s.rolling(d, min_periods=1).apply(f, raw=True)


def _ts_rank(s: pd.Series, d: int) -> pd.Series:
    """时序排名：最后一根在窗口内的百分位（同 _rank 但窗口=d）。"""
    return s.rolling(d, min_periods=1).apply(lambda x: (x[-1] < x).mean(), raw=True)


def _signed_power(s: pd.Series, a: float) -> pd.Series:
    return np.sign(s) * (np.abs(s) ** a)


def _scale(s: pd.Series, a: float = 1.0, win: int = RANK_WINDOW) -> pd.Series:
    """按滚动窗口内绝对值之和缩放，使 sum|缩放后| ≈ a。"""
    denom = s.abs().rolling(win, min_periods=5).sum().replace(0, np.nan)
    return s * a / denom


def _decay_linear(s: pd.Series, d: int) -> pd.Series:
    """线性衰减加权均值（权重随时间递增）。"""
    w = np.arange(1, d + 1, dtype=float)
    w = w / w.sum()

    def f(x: np.ndarray) -> float:
        n = len(x)
        return float(np.dot(x, w[-n:]))

    return s.rolling(d, min_periods=1).apply(f, raw=True)


def _slope(y: pd.Series, d: int) -> pd.Series:
    """滚动窗口内 y 对 [0..d-1] 的 OLS 斜率（Alpha191 回归类因子用）。

    注意：滚动窗口在预热期传入的数组长度可能小于 ``d``，故 ``x`` 需在函数内按
    实际长度构造。
    """

    def f(ys: np.ndarray) -> float:
        n = len(ys)
        if n < 2:
            return 0.0
        x = np.arange(n, dtype=float)
        xc = x - x.mean()
        denom = float((xc ** 2).sum())
        if denom == 0:
            return 0.0
        ys_c = ys - ys.mean()
        return float((xc * ys_c).sum() / denom)

    return y.rolling(d, min_periods=max(2, d // 2)).apply(f, raw=True)


def _sma(s: pd.Series, d: int) -> pd.Series:
    return s.rolling(d, min_periods=1).mean()


def _std(s: pd.Series, d: int) -> pd.Series:
    return s.rolling(d, min_periods=1).std()


def _sum(s: pd.Series, d: int) -> pd.Series:
    return s.rolling(d, min_periods=1).sum()
