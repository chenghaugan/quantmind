"""Alpha101 因子族（代表性子集，pandas 重实现，非拷贝原仓库代码）。

仅选取公式清晰、可由日线 OHLCV 稳定计算的代表性 Alpha。所有因子返回与输入等长的
``pd.Series``，依赖 :mod:`quantmind.research.factors.wq` 的时序原语（单标的滚动近似）。
完整 101 个公式可直接按同样模式扩展。
"""
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from ...core.object import BarData
from .base import Factor, FactorMeta, bars_to_df
from .wq import (
    _rank, _delay, _delta, _corr, _cov, _ts_min, _ts_max, _ts_arg_max,
    _signed_power, _scale, _decay_linear, _slope, _sma, _std, _sum,
)


def _close(df):
    return df["close"].astype(float)


def _ret(df):
    return df["close"].pct_change()


# ----------------------------- Alpha 公式 -----------------------------
def a002(df):  # -corr(rank(Δlog vol 2), rank((close-open)/open), 6)
    logv = np.log(df["volume"].clip(lower=1))
    return -_corr(_rank(_delta(logv, 2)), _rank((df["close"] - df["open"]) / df["open"]), 6)


def a003(df):  # -corr(rank(open), rank(volume), 10)
    return -_corr(_rank(df["open"]), _rank(df["volume"]), 10)


def a006(df):  # -corr(open, volume, 10)
    return -_corr(df["open"], df["volume"], 10)


def a012(df):  # sign(Δvol 1) * (-Δclose 1)
    return np.sign(_delta(df["volume"], 1)) * (-_delta(df["close"], 1))


def a013(df):  # -rank(cov(rank(close), rank(volume), 5))
    return -_rank(_cov(_rank(df["close"]), _rank(df["volume"]), 5))


def a014(df):  # -rank(Δret 3) * corr(open, volume, 10)
    return -_rank(_delta(_ret(df), 3)) * _corr(df["open"], df["volume"], 10)


def a015(df):  # -rank(sum(rank(corr(rank(high-low), rank(volume),5)),2))
    c = _rank(_corr(_rank(df["high"] - df["low"]), _rank(df["volume"]), 5))
    return -_rank(_sum(c, 2))


def a016(df):  # -rank(cov(rank(high), rank(volume), 5))
    return -_rank(_cov(_rank(df["high"]), _rank(df["volume"]), 5))


def a017(df):  # rank(corr(rank(high), rank(volume), 5))
    return _rank(_corr(_rank(df["high"]), _rank(df["volume"]), 5))


def a018(df):  # -rank(std(|close-open|,5)+(close-open) + corr(close,open,10))
    inner = _std((df["close"] - df["open"]).abs(), 5) + (df["close"] - df["open"]) + _corr(df["close"], df["open"], 10)
    return -_rank(inner)


def a019(df):  # -sign((close-delay(close,7))+Δclose7) * (1+rank(sum(ret,250)))
    inner = (df["close"] - df["close"].shift(7)) + _delta(df["close"], 7)
    return -np.sign(inner) * (1 + _rank(_sum(_ret(df), 250)))


def a020(df):  # -rank(open-delay(close,1)) * rank(open-close) * rank(open-delay(close,1))
    a = _rank(df["open"] - df["close"].shift(1))
    b = _rank(df["open"] - df["close"])
    return -a * b * a


def a021(df):  # -rank(TsMax(close,5)-close) / rank(TsMax(close,5)-TsMin(close,5))
    tmax = _ts_max(df["close"], 5)
    tmin = _ts_min(df["close"], 5)
    num = -_rank(tmax - df["close"])
    den = _rank(tmax - tmin).replace(0, np.nan)
    return (num / den).fillna(0.0)


def a022(df):  # -rank(corr(close, volume, 10)) * rank(close)
    return -_rank(_corr(df["close"], df["volume"], 10)) * _rank(df["close"])


def a024(df):  # -rank(TsMax(Δclose1,5)) * rank(corr(close, volume, 20))
    return -_rank(_ts_max(_delta(df["close"], 1), 5)) * _rank(_corr(df["close"], df["volume"], 20))


def a026(df):  # -TsMax(rank(corr(volume, sma(close,2),5)),5)
    c = _rank(_corr(df["volume"], _sma(df["close"], 2), 5))
    return -_ts_max(c, 5)


def a033(df):  # rank(-(1 - open/close))
    return _rank(-(1 - df["open"] / df["close"]))


def a037(df):  # -rank(open-close) * rank(open-delay(close,1))
    return -_rank(df["open"] - df["close"]) * _rank(df["open"] - df["close"].shift(1))


def a038(df):  # -rank(delay(close,1)-close) * rank(open-close) * rank(delay(close,1)-delay(close,2))
    return (-_rank(df["close"].shift(1) - df["close"]) * _rank(df["open"] - df["close"])
            * _rank(df["close"].shift(1) - df["close"].shift(2)))


def a040(df):  # -rank(std(high,10)) * corr(high, low, 10)
    return -_rank(_std(df["high"], 10)) * _corr(df["high"], df["low"], 10)


def a049(df):  # -rank(delay(close,1)-close) * rank(corr(close, volume, 5))
    return -_rank(df["close"].shift(1) - df["close"]) * _rank(_corr(df["close"], df["volume"], 5))


def a051(df):  # -rank(corr(rank(high), rank(volume), 3))
    return -_rank(_corr(_rank(df["high"]), _rank(df["volume"]), 3))


def a054(df):  # -rank(delay(close,1)-close) * rank(delay(close,1)-delay(close,2))
    return -_rank(df["close"].shift(1) - df["close"]) * _rank(df["close"].shift(1) - df["close"].shift(2))


def a060(df):  # -rank(Δclose1)*rank(Δclose1 from -2)*rank(Δclose1 from -3)
    return (-_rank(_delta(df["close"], 1)) * _rank(df["close"].shift(1) - df["close"].shift(2))
            * _rank(df["close"].shift(2) - df["close"].shift(3)))


def a062(df):  # -rank(corr(volume, sma(close,5),5)) * rank(std(close,5))
    return -_rank(_corr(df["volume"], _sma(df["close"], 5), 5)) * _rank(_std(df["close"], 5))


def a071(df):  # -rank(Δclose1) * rank(std(close,20))
    return -_rank(_delta(df["close"], 1)) * _rank(_std(df["close"], 20))


def a075(df):  # rank(corr(close, volume, 10))
    return _rank(_corr(df["close"], df["volume"], 10))


def a083(df):  # -rank(corr(close, volume, 20)) * rank(Δclose1)
    return -_rank(_corr(df["close"], df["volume"], 20)) * _rank(_delta(df["close"], 1))


def a093(df):  # rank(delay(close,1)-close)
    return _rank(df["close"].shift(1) - df["close"])


def a099(df):  # rank(corr(close, open, 10))
    return _rank(_corr(df["close"], df["open"], 10))


def a101(df):  # -rank(cov(rank(close), rank(volume), 10))
    return -_rank(_cov(_rank(df["close"]), _rank(df["volume"]), 10))


_ALPHA_FUNCS: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "alpha002": a002, "alpha003": a003, "alpha006": a006, "alpha012": a012,
    "alpha013": a013, "alpha014": a014, "alpha015": a015, "alpha016": a016,
    "alpha017": a017, "alpha018": a018, "alpha019": a019, "alpha020": a020,
    "alpha021": a021, "alpha022": a022, "alpha024": a024, "alpha026": a026,
    "alpha033": a033, "alpha037": a037, "alpha038": a038, "alpha040": a040,
    "alpha049": a049, "alpha051": a051, "alpha054": a054, "alpha060": a060,
    "alpha062": a062, "alpha071": a071, "alpha075": a075, "alpha083": a083,
    "alpha093": a093, "alpha099": a099, "alpha101": a101,
}


class AlphaFactor(Factor):
    """WorldQuant Alpha 因子（单标的滚动近似版）。"""

    def __init__(self, name: str) -> None:
        if name not in _ALPHA_FUNCS:
            raise KeyError(f"未知 Alpha 因子: {name}")
        self._name = name
        self.meta = FactorMeta(name=name, category="alpha101",
                               description=f"WorldQuant {name} 因子(单标的近似)")
        self.params = {"name": name}

    def compute(self, bars: List[BarData]) -> pd.Series:
        df = bars_to_df(bars)
        if df.empty:
            return pd.Series(dtype=float)
        return _ALPHA_FUNCS[self._name](df).fillna(0.0)


def list_alpha101() -> List[str]:
    return sorted(_ALPHA_FUNCS.keys())


def build_alpha_factor(name: str) -> AlphaFactor:
    return AlphaFactor(name)
