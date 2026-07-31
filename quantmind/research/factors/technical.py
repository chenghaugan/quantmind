"""技术类因子库：动量、均值回复、波动率、成交量变化、期限结构等。

命名约定：``<factor>_<window>``。所有因子基于历史已知数据，返回与输入等长的序列。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd

from ...core.object import BarData
from .base import Factor, FactorMeta, bars_to_df


class MomentumFactor(Factor):
    """动量因子：过去 n 根 K 线的累计收益率（close_t / close_{t-n} - 1）。

    最经典的截面/时序动量。n=20 约一个月（日线）。
    """

    def __init__(self, window: int = 20) -> None:
        self.window = window
        self.meta = FactorMeta(name=f"momentum_{window}", category="momentum",
                               description=f"过去{window}根K线累计收益率")
        self.params = {"window": window}

    def compute(self, bars: List[BarData]) -> pd.Series:
        df = bars_to_df(bars)
        if df.empty:
            return pd.Series(dtype=float)
        return (df["close"] / df["close"].shift(self.window) - 1.0).fillna(0.0)


class MeanReversionFactor(Factor):
    """均值回复因子：(close - N日均值) / N日标准差 的负值（偏离越大越看多回复）。"""

    def __init__(self, window: int = 60) -> None:
        self.window = window
        self.meta = FactorMeta(name=f"mean_reversion_{window}", category="reversion",
                               description=f"相对{window}日均值的 z-score 取反")
        self.params = {"window": window}

    def compute(self, bars: List[BarData]) -> pd.Series:
        df = bars_to_df(bars)
        if df.empty:
            return pd.Series(dtype=float)
        ma = df["close"].rolling(self.window, min_periods=20).mean()
        sd = df["close"].rolling(self.window, min_periods=20).std()
        z = (df["close"] - ma) / (sd.replace(0, pd.NA))
        return (-z).fillna(0.0)


class VolatilityFactor(Factor):
    """波动率因子：过去 n 根收益率的滚动标准差（去量纲）。"""

    def __init__(self, window: int = 20) -> None:
        self.window = window
        self.meta = FactorMeta(name=f"volatility_{window}", category="volatility",
                               description=f"过去{window}根收益率滚动波动率")
        self.params = {"window": window}

    def compute(self, bars: List[BarData]) -> pd.Series:
        df = bars_to_df(bars)
        if df.empty:
            return pd.Series(dtype=float)
        ret = df["close"].pct_change()
        return ret.rolling(self.window, min_periods=5).std().fillna(0.0)


class VolumeChangeFactor(Factor):
    """成交量变化因子：当日成交量 / N日均量 - 1。"""

    def __init__(self, window: int = 5) -> None:
        self.window = window
        self.meta = FactorMeta(name=f"volume_change_{window}", category="volume",
                               description=f"成交量相对{window}日均量变化")
        self.params = {"window": window}

    def compute(self, bars: List[BarData]) -> pd.Series:
        df = bars_to_df(bars)
        if df.empty:
            return pd.Series(dtype=float)
        avg = df["volume"].rolling(self.window, min_periods=2).mean()
        return (df["volume"] / avg - 1.0).fillna(0.0)


class OpenInterestChangeFactor(Factor):
    """持仓量变化因子：持仓量相对 N 日均值的变化（期货专用，反映资金流向）。"""

    def __init__(self, window: int = 20) -> None:
        self.window = window
        self.meta = FactorMeta(name=f"open_interest_change_{window}", category="futures",
                               description=f"持仓量相对{window}日均量变化")
        self.params = {"window": window}

    def compute(self, bars: List[BarData]) -> pd.Series:
        df = bars_to_df(bars)
        if df.empty or df["open_interest"].abs().sum() == 0:
            return pd.Series([0.0] * len(df)) if not df.empty else pd.Series(dtype=float)
        avg = df["open_interest"].rolling(self.window, min_periods=2).mean()
        return (df["open_interest"] / avg - 1.0).fillna(0.0)


class TermStructureFactor(Factor):
    """期限结构因子：用主力连续(df close)与指数连续的偏离近似（此处以近远月价差代理）。

    简化实现：用 ``close`` 的 N 日变化率减去其 2N 日变化率，近似近月强于远月为正。
    """

    def __init__(self, window: int = 20) -> None:
        self.window = window
        self.meta = FactorMeta(name=f"term_structure_{window}", category="futures",
                               description=f"近远月价差代理（{window}日）")
        self.params = {"window": window}

    def compute(self, bars: List[BarData]) -> pd.Series:
        df = bars_to_df(bars)
        if df.empty:
            return pd.Series(dtype=float)
        near = df["close"].pct_change(self.window)
        far = df["close"].pct_change(self.window * 2)
        return (near - far).fillna(0.0)


# 便于反射式构造
_FACTOR_CLASSES = {
    "momentum": MomentumFactor,
    "mean_reversion": MeanReversionFactor,
    "volatility": VolatilityFactor,
    "volume_change": VolumeChangeFactor,
    "open_interest_change": OpenInterestChangeFactor,
    "term_structure": TermStructureFactor,
}


def build_factor(kind: str, window: int = 20) -> Factor:
    """按名称构造因子（供表达式/AI 生成使用）。"""
    cls = _FACTOR_CLASSES.get(kind)
    if cls is None:
        raise KeyError(f"未知因子类型: {kind}")
    return cls(window=window)
