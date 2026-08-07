"""WorldQuant 风格时序原语（单资产近似版）。

WorldQuant Alpha 公式中大量使用**截面(cross-sectional) rank / correlation**，
但 QuantMind 的回测/评估多在**单标的时序**语境下运行。为在单标的下仍可计算并评估，
本模块把 ``rank`` 近似为「滚动窗口内的分位排名」（即最后一根在窗口内的百分位），
把 ``correlation/covariance`` 实现为滚动窗口时序相关。`scale` 用滚动窗口内的
绝对值之和归一。这样得出的因子在单标的情况下是**有意义的时序代理**，但在多标的
横截面上会弱于原版——生产环境若需严格截面，应在多标的 DataFrame 上替换 ``rank`` 实现。

公式来源：WorldQuant Alpha101/Alpha191 公开文献（仅重实现数学公式，未复制任何仓库代码）。

本模块同时提供两种 ``rank``：
  - :func:`_rank` —— 单标的滚动窗口内分位（时序近似），用于 ``cli factor`` 单标的评估；
  - :func:`_rank_cs` —— **严格截面 rank**（每行跨列百分位），用于多标的面板因子计算。
所有时间序列表原语（``_delta``/``_ts_*``/``_corr``/``_cov``/``_slope``/``_decay_linear``
等）均天然支持面板 DataFrame（按列逐标的计算），可与 ``_rank_cs`` 组合得到可信截面因子。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 滚动 rank 默认窗口（约 1 个交易年的交易日）
RANK_WINDOW = 250


def _rank(s: pd.Series, win: int = RANK_WINDOW) -> pd.Series:
    """分位排名：最后一根在滚动窗口内的百分位（0~1）。近似截面 rank（单标的场景）。"""
    s = s.astype(float)
    return s.rolling(win, min_periods=5).apply(lambda x: (x[-1] < x).mean(), raw=True)


def _rank_cs(df: pd.DataFrame, pct: bool = True) -> pd.DataFrame:
    """严格**截面** rank：对每个时间截面（每行）跨标的（列）做百分位排名（0~1）。

    这是 WorldQuant Alpha 公式中 ``rank`` 的本意——在同一交易日对所有标的研究其相对
    位置，而非单标的的时间序列分位。输入为面板 DataFrame（index=日期，columns=标的），
    输出同型 DataFrame；NaN 不参与排名（pandas 默认）。

    单标的下若误用（仅 1 列），结果退化为常量 0.5，因此截面计算要求面板含 ≥2 个标的。
    """
    return df.rank(axis=1, method="average", pct=pct)


def _delay(s: pd.Series, d: int) -> pd.Series:
    return s.shift(d)


def _delta(s: pd.Series, d: int) -> pd.Series:
    return s - s.shift(d)


def _corr(a: pd.Series, b: pd.Series, d: int) -> pd.Series:
    with np.errstate(divide="ignore", invalid="ignore"):
        return a.rolling(d, min_periods=max(2, d // 2)).corr(b)


def _cov(a: pd.Series, b: pd.Series, d: int) -> pd.Series:
    with np.errstate(divide="ignore", invalid="ignore"):
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


def _ts_arg_min(s: pd.Series, d: int) -> pd.Series:
    """距窗口内最小值的周期数（d-1 - argmin）。"""
    def f(x: np.ndarray) -> float:
        return d - 1 - int(np.argmin(x))
    return s.rolling(d, min_periods=1).apply(f, raw=True)


def _ts_product(s: pd.Series, d: int) -> pd.Series:
    """滚动窗口内乘积（Alpha191 Product 用）。"""
    return s.rolling(d, min_periods=1).apply(lambda x: float(np.prod(x)), raw=True)


def _ts_zscore(s: pd.Series, d: int) -> pd.Series:
    """滚动 z-score（均值/标准差标准化）。"""
    m = s.rolling(d, min_periods=1).mean()
    sd = s.rolling(d, min_periods=1).std().replace(0, np.nan)
    return (s - m) / sd


def _ts_median(s: pd.Series, d: int) -> pd.Series:
    """滚动窗口中位数（Alpha191 Med 用）。"""
    return s.rolling(d, min_periods=1).median()


def _reg_beta(y: pd.Series, x: pd.Series, d: int) -> pd.Series:
    """滚动窗口 OLS 斜率（Alpha191 RegBeta）：以窗口内 x 对 y 回归的 beta。

    用「协方差/方差」等价实现（beta = Cov(y,x)/Var(x)），只依赖逐列的滚动
    运算（``_cov``/``_std``），避免 ``DataFrame.rolling().apply`` 在 pandas 3.0
    下逐列应用、无法拿到多列对齐窗口的缺陷。
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = x.rolling(d, min_periods=max(2, d // 2)).var()
        cov = _cov(y, x, d)
        return cov / denom.replace(0, np.nan)


def _reg_resi(y: pd.Series, x: pd.Series, d: int) -> pd.Series:
    """滚动窗口 OLS 残差（Alpha191 RegResi）：以窗口内 x 对 y 回归的残差序列。

    残差 = y - (alpha + beta·x)，其中 alpha = mean(y) - beta·mean(x)，beta 用
    协方差/方差等价实现（同 :func:`_reg_beta`），仅用逐列滚动运算。
    """
    minp = max(2, d // 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        var_x = x.rolling(d, min_periods=minp).var()
        beta = _cov(y, x, d) / var_x.replace(0, np.nan)
        xmean = x.rolling(d, min_periods=minp).mean()
        ymean = y.rolling(d, min_periods=minp).mean()
        alpha = ymean - beta * xmean
        return y - (alpha + beta * x)


# ----------------------------- 金额/均价辅助（turnover=成交额） -----------------------------
def _vwap(df: pd.DataFrame) -> pd.Series:
    """成交量加权均价：成交额 / 成交量；成交额缺失或为零时退化为 (H+L+C)/3 典型价。"""
    amt = df["turnover"].replace(0, np.nan) if "turnover" in df else np.nan
    v = df["volume"].replace(0, np.nan)
    out = (amt / v)
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    return out.fillna(typical)


def _adv(df: pd.DataFrame, d: int) -> pd.Series:
    """平均日成交额（滚动 d 日均值），用于 Alpha 公式中 advN。"""
    if "turnover" in df:
        return df["turnover"].rolling(d, min_periods=1).mean()
    return (df["close"] * df["volume"]).rolling(d, min_periods=1).mean()
