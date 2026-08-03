"""Alpha101/191 因子的**严格截面(cross-sectional)** 实现（面板版）。

与 :mod:`quantmind.research.factors.alpha101` / ``alpha191``（单标的滚动近似）不同，本模块
在**多标的面板**（index=日期，columns=标的）上计算：

  - 所有时间序列表原语（``_delta``/``_ts_*``/``_corr``/``_cov``/``_slope``/``_decay_linear``
    等）按列逐标的计算；
  - 所有 ``rank`` 调用一律走 :func:`wq._rank_cs` —— 在**每个交易日横截面上对所有标的**
    做百分位排名。这正是 WorldQuant Alpha 公式里 ``rank`` 的本意，单标的滚动近似会高估/
    低估其 IC，截面版才能正确评估。

公式来源：WorldQuant Alpha101/Alpha191 公开文献（仅重实现数学公式，未复制任何仓库代码）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from ...core.object import BarData
from .base import bars_to_df
from .wq import (
    _rank_cs, _delay, _delta, _corr, _cov, _ts_min, _ts_max, _ts_arg_max, _ts_rank,
    _signed_power, _scale, _decay_linear, _slope, _sma, _std, _sum,
)


@dataclass
class Panel:
    """多标的面板数据。各字段均为 DataFrame（index=日期，columns=标的），已对齐。"""

    close: pd.DataFrame
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    volume: pd.DataFrame
    amount: pd.DataFrame = field(default_factory=pd.DataFrame)

    @classmethod
    def from_bars(cls, bars_by_symbol: Dict[str, List[BarData]]) -> "Panel":
        """由 ``{symbol: List[BarData]}`` 构建对齐面板（取所有标的日期交集）。"""
        fields = ["open", "high", "low", "close", "volume", "turnover"]
        per_sym: Dict[str, pd.DataFrame] = {}
        for sym, bars in bars_by_symbol.items():
            df = bars_to_df(bars)
            if df.empty:
                continue
            per_sym[sym] = df.set_index("datetime")[fields]
        if not per_sym:
            empty = pd.DataFrame()
            return cls(empty, empty, empty, empty, empty, empty)
        # 以 close 的共同日期为对齐基准
        close = pd.DataFrame({s: d["close"] for s, d in per_sym.items()}).dropna(how="all")
        idx = close.index
        cols = close.columns
        open_ = pd.DataFrame({s: per_sym[s]["open"] for s in cols}).reindex(idx)
        high = pd.DataFrame({s: per_sym[s]["high"] for s in cols}).reindex(idx)
        low = pd.DataFrame({s: per_sym[s]["low"] for s in cols}).reindex(idx)
        volume = pd.DataFrame({s: per_sym[s]["volume"] for s in cols}).reindex(idx)
        amount = pd.DataFrame({s: per_sym[s]["turnover"] for s in cols}).reindex(idx)
        return cls(close=close, open=open_, high=high, low=low, volume=volume, amount=amount)

    @property
    def symbols(self) -> List[str]:
        return list(self.close.columns)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index


# ----------------------------- 面板辅助 -----------------------------
def _ret(P: Panel) -> pd.DataFrame:
    return P.close.pct_change()


def _vwap(P: Panel) -> pd.DataFrame:
    """面板成交量加权均价：amount / volume；amount 缺失/为空时退化为 (H+L+C)/3。"""
    typical = (P.high + P.low + P.close) / 3.0
    if P.amount is None or P.amount.empty:
        return typical
    v = P.volume.replace(0, np.nan)
    out = P.amount / v
    return out.fillna(typical)


def _adv(P: Panel, n: int) -> pd.DataFrame:
    """面板平均日成交额（滚动 n 日均值）；amount 缺失/为空时退化为 volume 近似。"""
    if P.amount is None or P.amount.empty:
        return P.volume
    return P.amount.rolling(n, min_periods=1).mean()


# ----------------------------- Alpha101 面板公式 -----------------------------
def a002(P):  # -corr(rank(Δlog vol 2), rank((close-open)/open), 6)
    logv = np.log(P.volume.clip(lower=1))
    return -_corr(_rank_cs(_delta(logv, 2)), _rank_cs((P.close - P.open) / P.open), 6)


def a003(P):  # -corr(rank(open), rank(volume), 10)
    return -_corr(_rank_cs(P.open), _rank_cs(P.volume), 10)


def a006(P):  # -corr(open, volume, 10)
    return -_corr(P.open, P.volume, 10)


def a012(P):  # sign(Δvol 1) * (-Δclose 1)
    return np.sign(_delta(P.volume, 1)) * (-_delta(P.close, 1))


def a013(P):  # -rank(cov(rank(close), rank(volume), 5))
    return -_rank_cs(_cov(_rank_cs(P.close), _rank_cs(P.volume), 5))


def a014(P):  # -rank(Δret 3) * corr(open, volume, 10)
    return -_rank_cs(_delta(_ret(P), 3)) * _corr(P.open, P.volume, 10)


def a015(P):  # -rank(sum(rank(corr(rank(high-low), rank(volume),5)),2))
    c = _rank_cs(_corr(_rank_cs(P.high - P.low), _rank_cs(P.volume), 5))
    return -_rank_cs(_sum(c, 2))


def a016(P):  # -rank(cov(rank(high), rank(volume), 5))
    return -_rank_cs(_cov(_rank_cs(P.high), _rank_cs(P.volume), 5))


def a017(P):  # rank(corr(rank(high), rank(volume), 5))
    return _rank_cs(_corr(_rank_cs(P.high), _rank_cs(P.volume), 5))


def a018(P):  # -rank(std(|close-open|,5)+(close-open) + corr(close,open,10))
    inner = (_std((P.close - P.open).abs(), 5) + (P.close - P.open)
             + _corr(P.close, P.open, 10))
    return -_rank_cs(inner)


def a019(P):  # -sign((close-delay(close,7))+Δclose7) * (1+rank(sum(ret,250)))
    inner = (P.close - P.close.shift(7)) + _delta(P.close, 7)
    return -np.sign(inner) * (1 + _rank_cs(_sum(_ret(P), 250)))


def a020(P):  # -rank(open-delay(close,1)) * rank(open-close) * rank(open-delay(close,1))
    a = _rank_cs(P.open - P.close.shift(1))
    b = _rank_cs(P.open - P.close)
    return -a * b * a


def a021(P):  # -rank(TsMax(close,5)-close) / rank(TsMax(close,5)-TsMin(close,5))
    tmax = _ts_max(P.close, 5)
    tmin = _ts_min(P.close, 5)
    num = -_rank_cs(tmax - P.close)
    den = _rank_cs(tmax - tmin).replace(0, np.nan)
    return (num / den).fillna(0.0)


def a022(P):  # -rank(corr(close, volume, 10)) * rank(close)
    return -_rank_cs(_corr(P.close, P.volume, 10)) * _rank_cs(P.close)


def a024(P):  # -rank(TsMax(Δclose1,5)) * rank(corr(close, volume, 20))
    return -_rank_cs(_ts_max(_delta(P.close, 1), 5)) * _rank_cs(_corr(P.close, P.volume, 20))


def a026(P):  # -TsMax(rank(corr(volume, sma(close,2),5)),5)
    c = _rank_cs(_corr(P.volume, _sma(P.close, 2), 5))
    return -_ts_max(c, 5)


def a033(P):  # rank(-(1 - open/close))
    return _rank_cs(-(1 - P.open / P.close))


def a037(P):  # -rank(open-close) * rank(open-delay(close,1))
    return -_rank_cs(P.open - P.close) * _rank_cs(P.open - P.close.shift(1))


def a038(P):  # -rank(delay(close,1)-close) * rank(open-close) * rank(delay(close,1)-delay(close,2))
    return (-_rank_cs(P.close.shift(1) - P.close) * _rank_cs(P.open - P.close)
            * _rank_cs(P.close.shift(1) - P.close.shift(2)))


def a040(P):  # -rank(std(high,10)) * corr(high, low, 10)
    return -_rank_cs(_std(P.high, 10)) * _corr(P.high, P.low, 10)


def a049(P):  # -rank(delay(close,1)-close) * rank(corr(close, volume, 5))
    return -_rank_cs(P.close.shift(1) - P.close) * _rank_cs(_corr(P.close, P.volume, 5))


def a051(P):  # -rank(corr(rank(high), rank(volume), 3))
    return -_rank_cs(_corr(_rank_cs(P.high), _rank_cs(P.volume), 3))


def a054(P):  # -rank(delay(close,1)-close) * rank(delay(close,1)-delay(close,2))
    return -_rank_cs(P.close.shift(1) - P.close) * _rank_cs(P.close.shift(1) - P.close.shift(2))


def a060(P):  # -rank(Δclose1)*rank(Δclose1 from -2)*rank(Δclose1 from -3)
    return (-_rank_cs(_delta(P.close, 1)) * _rank_cs(P.close.shift(1) - P.close.shift(2))
            * _rank_cs(P.close.shift(2) - P.close.shift(3)))


def a062(P):  # -rank(corr(volume, sma(close,5),5)) * rank(std(close,5))
    return -_rank_cs(_corr(P.volume, _sma(P.close, 5), 5)) * _rank_cs(_std(P.close, 5))


def a071(P):  # -rank(Δclose1) * rank(std(close,20))
    return -_rank_cs(_delta(P.close, 1)) * _rank_cs(_std(P.close, 20))


def a075(P):  # rank(corr(close, volume, 10))
    return _rank_cs(_corr(P.close, P.volume, 10))


def a083(P):  # -rank(corr(close, volume, 20)) * rank(Δclose1)
    return -_rank_cs(_corr(P.close, P.volume, 20)) * _rank_cs(_delta(P.close, 1))


def a093(P):  # rank(delay(close,1)-close)
    return _rank_cs(P.close.shift(1) - P.close)


def a099(P):  # rank(corr(close, open, 10))
    return _rank_cs(_corr(P.close, P.open, 10))


def a101(P):  # -rank(cov(rank(close), rank(volume), 10))
    return -_rank_cs(_cov(_rank_cs(P.close), _rank_cs(P.volume), 10))


# ----------------------------- 补充高价值经典 Alpha（截面面板版，显式真实公式） -----------------------------
def a001(P):  # rank(Ts_ArgMax(SignedPower(((ret<0)?std(ret,20):close), 2), 5)) - 0.5
    ret = P.close.pct_change()
    std_ret = _std(ret, 20)
    base = pd.DataFrame(np.where(ret < 0, std_ret.values, P.close.values),
                        index=P.close.index, columns=P.close.columns)
    sp = np.sign(base) * (np.abs(base) ** 2.0)
    return _rank_cs(_ts_arg_max(sp, 5)) - 0.5


def a004(P):  # -Ts_Rank(rank(low), 9)
    return -_ts_rank(_rank_cs(P.low), 9)


def a005(P):  # rank(open - sma(vwap,10)) * (-1 * abs(rank(close - vwap)))
    vwap = _vwap(P)
    r1 = _rank_cs(P.open - _sma(vwap, 10))
    r2 = _rank_cs(P.close - vwap).abs()
    return r1 * (-1.0 * r2)


def a007(P):  # (rank(max(open-close,0))+rank(max(low-close,0))+rank(min(open-close,0))+rank(min(high-close,0)))^2
    oc = P.open - P.close
    lc = P.low - P.close
    hc = P.high - P.close
    r = (_rank_cs(oc.clip(lower=0)) + _rank_cs(lc.clip(lower=0))
         + _rank_cs(oc.clip(upper=0)) + _rank_cs(hc.clip(upper=0)))
    return r * r


def a008(P):  # -rank((sum(open,5)*sum(ret,5)) - delay(., 10))
    prod = _sum(P.open, 5) * _sum(_ret(P), 5)
    return -_rank_cs(prod - prod.shift(10))


def a011(P):  # (rank(TsMax(vwap-close,3)) + rank(TsMin(vwap-close,3))) * rank(Δvolume 3)
    vwap = _vwap(P)
    diff = vwap - P.close
    return (_rank_cs(_ts_max(diff, 3)) + _rank_cs(_ts_min(diff, 3))) * _rank_cs(_delta(P.volume, 3))


def a028(P):  # scale((((close-low)-(high-close))/(high-low)))  价格位置
    hl = (P.high - P.low).replace(0, np.nan)
    inner = ((P.close - P.low) - (P.high - P.close)) / hl
    inner = inner.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return _scale(inner)


def a053(P):  # -delta(价格位置, 9)
    hl = (P.high - P.low).replace(0, np.nan)
    inner = ((P.close - P.low) - (P.high - P.close)) / hl
    inner = inner.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return -_delta(inner, 9)


def a055(P):  # -corr(rank(价格位置), rank(volume), 5)
    hl = (P.high - P.low).replace(0, np.nan)
    inner = ((P.close - P.low) - (P.high - P.close)) / hl
    inner = inner.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return -_corr(_rank_cs(inner), _rank_cs(P.volume), 5)


# ----------------------------- Alpha191 面板公式 -----------------------------
def a191_007(P):  # 收盘价的滚动回归斜率（趋势强度）
    return _slope(P.close, 10)


def a191_012(P):  # 收盘与昨收的滚动相关（均值回复度）
    return _rank_cs(_corr(P.close, P.close.shift(1), 10))


def a191_019(P):  # -rank(Δclose5) * rank(volume)
    return -_rank_cs(_delta(P.close, 5)) * _rank_cs(P.volume)


def a191_042(P):  # -rank(Δclose1) * rank(volume)
    return -_rank_cs(_delta(P.close, 1)) * _rank_cs(P.volume)


def a191_056(P):  # rank(corr(high, low, 5))（波幅协同）
    return _rank_cs(_corr(P.high, P.low, 5))


def a191_065(P):  # rank(slope(volume, 10))（量能趋势）
    return _rank_cs(_slope(P.volume, 10))


def a191_081(P):  # -rank(open - close)（日内强弱）
    return -_rank_cs(P.open - P.close)


def a191_009(P):  # -rank(Δclose1)
    return -_rank_cs(_delta(P.close, 1))


def a191_038(P):  # corr(close, open, 10)
    return _corr(P.close, P.open, 10)


def a191_099(P):  # rank(TsMax(close,10)-close) 反转信号
    return _rank_cs(_ts_max(P.close, 10) - P.close)


_ALPHA_CS_FUNCS: Dict[str, Callable[[Panel], pd.DataFrame]] = {
    "alpha001": a001, "alpha002": a002, "alpha003": a003, "alpha004": a004,
    "alpha005": a005, "alpha006": a006, "alpha007": a007, "alpha008": a008,
    "alpha011": a011, "alpha012": a012, "alpha013": a013, "alpha014": a014,
    "alpha015": a015, "alpha016": a016, "alpha017": a017, "alpha018": a018,
    "alpha019": a019, "alpha020": a020, "alpha021": a021, "alpha022": a022,
    "alpha024": a024, "alpha026": a026, "alpha028": a028, "alpha033": a033,
    "alpha037": a037, "alpha038": a038, "alpha040": a040, "alpha049": a049,
    "alpha051": a051, "alpha053": a053, "alpha054": a054, "alpha055": a055,
    "alpha060": a060, "alpha062": a062, "alpha071": a071, "alpha075": a075,
    "alpha083": a083, "alpha093": a093, "alpha099": a099, "alpha101": a101,
    "alpha191_007": a191_007, "alpha191_012": a191_012, "alpha191_019": a191_019,
    "alpha191_042": a191_042, "alpha191_056": a191_056, "alpha191_065": a191_065,
    "alpha191_081": a191_081, "alpha191_009": a191_009, "alpha191_038": a191_038,
    "alpha191_099": a191_099,
}


def list_alpha_cs() -> List[str]:
    return sorted(_ALPHA_CS_FUNCS.keys())


def compute_alpha_cross_sectional(
    names: List[str], panel: Panel
) -> Dict[str, pd.DataFrame]:
    """在面板上计算一组 Alpha 因子（严格截面 rank）。

    返回 ``{name: DataFrame(date×symbol)}``，缺失值以 0.0 填充（便于下游分组/排序）。
    """
    out: Dict[str, pd.DataFrame] = {}
    for name in names:
        if name not in _ALPHA_CS_FUNCS:
            raise KeyError(f"未知截面 Alpha 因子: {name}")
        out[name] = _ALPHA_CS_FUNCS[name](panel).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out
