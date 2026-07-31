"""Alpha191 因子族（代表性子集，pandas 重实现）。

Alpha191 大量使用「对指数/自身做滚动回归斜率、滚动相关」等结构。这里选取可仅由
单标的 OHLCV 计算的、以滚动回归斜率(`_slope`)与滚动相关为核心的代表性公式。
完整的 191 个可按同样模式扩展。公式来源：公开 Alpha191 文献（仅重实现公式）。
"""
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from ...core.object import BarData
from .base import Factor, FactorMeta, bars_to_df
from .wq import _rank, _delay, _delta, _corr, _ts_min, _ts_max, _slope, _std, _sma


def _ret(df):
    return df["close"].pct_change()


# ----------------------------- Alpha191 公式(代表) -----------------------------
def a191_007(df):  # 收盘价的滚动回归斜率（趋势强度）
    return _slope(df["close"], 10)


def a191_012(df):  # 收盘与昨收的滚动相关（均值回复度）
    return _rank(_corr(df["close"], df["close"].shift(1), 10))


def a191_019(df):  # -rank(Δclose5) * rank(volume)
    return -_rank(_delta(df["close"], 5)) * _rank(df["volume"])


def a191_042(df):  # -rank(Δclose1) * rank(volume)
    return -_rank(_delta(df["close"], 1)) * _rank(df["volume"])


def a191_056(df):  # rank(corr(high, low, 5))（波幅协同）
    return _rank(_corr(df["high"], df["low"], 5))


def a191_065(df):  # rank(slope(volume, 10))（量能趋势）
    return _rank(_slope(df["volume"], 10))


def a191_081(df):  # -rank(open - close)（日内强弱）
    return -_rank(df["open"] - df["close"])


def a191_009(df):  # -rank(Δclose1)
    return -_rank(_delta(df["close"], 1))


def a191_038(df):  # corr(close, open, 10)
    return _corr(df["close"], df["open"], 10)


def a191_099(df):  # rank(TsMax(close,10)-close) 反转信号
    return _rank(_ts_max(df["close"], 10) - df["close"])


_ALPHA191_FUNCS: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "alpha191_007": a191_007, "alpha191_012": a191_012, "alpha191_019": a191_019,
    "alpha191_042": a191_042, "alpha191_056": a191_056, "alpha191_065": a191_065,
    "alpha191_081": a191_081, "alpha191_009": a191_009, "alpha191_038": a191_038,
    "alpha191_099": a191_099,
}


class Alpha191Factor(Factor):
    """Alpha191 因子（单标的滚动回归/相关类）。"""

    def __init__(self, name: str) -> None:
        if name not in _ALPHA191_FUNCS:
            raise KeyError(f"未知 Alpha191 因子: {name}")
        self._name = name
        self.meta = FactorMeta(name=name, category="alpha191",
                               description=f"WorldQuant {name} 因子(单标的近似)")
        self.params = {"name": name}

    def compute(self, bars: List[BarData]) -> pd.Series:
        df = bars_to_df(bars)
        if df.empty:
            return pd.Series(dtype=float)
        return _ALPHA191_FUNCS[self._name](df).fillna(0.0)


def list_alpha191() -> List[str]:
    return sorted(_ALPHA191_FUNCS.keys())


def build_alpha191_factor(name: str) -> Alpha191Factor:
    return Alpha191Factor(name)
