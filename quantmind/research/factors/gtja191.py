"""GTJA191 因子族（国泰君安短周期价量因子，代表性子集）。

来源：国泰君安证券研报《基于短周期价量特征的多因子选股体系》所提出的 191
个短周期价量因子（常称 GTJA191 / Alpha191，面向 A 股市场）。本模块**仅重实现
其公开数学公式**，未复制任何外部仓库代码；rank/corr/回归等全部用它对应的
QuantMind ``wq.py`` 原语等价实现，不依赖 numba/scipy 及任何外部第三方源。

注释约定：每个 ``g_*`` 函数按原文献公式逐项映射——
  - ``RANK``        → ``_rank``（单标的滚动时序分位近似，等价 WorldQuant rank）
  - ``TSRANK(x,n)`` → ``_ts_rank(x, n)``（窗口内分位）
  - ``CORR/COV``    → ``_corr/_cov``（滚动相关/协方差）
  - ``REGBETA/REGRESI`` → ``_reg_beta/_reg_resi``（滚动 OLS）
  - ``DELTA/DELAY`` → ``_delta/_delay``
  - ``SUM/STD/MAX/MIN/MEAN`` → ``_sum/_std/_ts_max/_ts_min/_sma``
  - ``WMA`` → ``_decay_linear``（线性加权）
  - ``VWAP`` → ``_vwap``（成交额/成交量，缺失退化为典型价）
所有函数输入为 OHLCV DataFrame（含 open/high/low/close/volume，可选 turnover）。

适用性：A 股日线；单标的时序近似。完整的 191 个可按同样「公式 → wq.py 原语」
模式继续扩展，这里选取 25 个最具代表性、公式清晰、日线可稳定计算的经典因子。
"""
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from ...core.object import BarData
from .base import Factor, FactorMeta, bars_to_df
from .wq import (
    _corr,
    _decay_linear,
    _delay,
    _delta,
    _rank,
    _slope,
    _sma,
    _std,
    _sum,
    _ts_max,
    _ts_min,
    _ts_rank,
    _vwap,
)


def _ret(df: pd.DataFrame) -> pd.Series:
    """日收益率 RET = CLOSE / DELAY(CLOSE,1) - 1。"""
    return df["close"].pct_change()


def _wma(s: pd.Series, d: int) -> pd.Series:
    """WMA(s, d)：线性加权移动平均（等价 _decay_linear 的递增权重形式）。"""
    return _decay_linear(s, d)


# ----------------------------- GTJA191 公式（代表性子集） -----------------------------
def g_001(df: pd.DataFrame) -> pd.Series:
    """α001 = -1*CORR(RANK(DELTA(LOG(VOLUME),1)), RANK((CLOSE-OPEN)/OPEN), 6)

    量能变化与日内涨跌幅的相关（取负），捕捉量价背离。
    """
    vol = df["volume"].replace(0, np.nan)
    delta_log_vol = _delta(np.log(vol), 1)
    intraday = (df["close"] - df["open"]) / df["open"].replace(0, np.nan)
    return -1.0 * _corr(_rank(delta_log_vol), _rank(intraday), 6)


def g_002(df: pd.DataFrame) -> pd.Series:
    """α002 = -1*DELTA((((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)), 1)

    收盘在日内区间位置的变化（取负），日内强弱反转。
    """
    denom = (df["high"] - df["low"]).replace(0, np.nan)
    pos = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / denom
    return -1.0 * _delta(pos, 1)


def g_003(df: pd.DataFrame) -> pd.Series:
    """α003 = SUM(CLOSE==DELAY(CLOSE,1)?0 : CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1))
            :MAX(HIGH,DELAY(CLOSE,1))), 6)

    条件式动量反转：按收盘相对前收的位置选择止损/止盈参考计算偏离，滚动 6 期求和。
    """
    c = df["close"]
    pc = _delay(c, 1)
    inner = np.where(c > pc, np.minimum(df["low"], pc), np.maximum(df["high"], pc))
    term = pd.Series(np.where(c == pc, 0.0, c - inner), index=df.index)
    return _sum(term, 6)


def g_006(df: pd.DataFrame) -> pd.Series:
    """α006 = RANK(SIGN(DELTA((OPEN*0.85+HIGH*0.15), 4)))*-1

    开盘主导加权价的 4 期变化符号（取负），开盘动量反转。
    """
    weighted = df["open"] * 0.85 + df["high"] * 0.15
    return -1.0 * _rank(np.sign(_delta(weighted, 4)))


def g_007(df: pd.DataFrame) -> pd.Series:
    """α007 = (RANK(MAX(VWAP-CLOSE,3)) + RANK(MIN(VWAP-CLOSE,3))) * RANK(DELTA(VOLUME,3))

    VWAP 相对收盘的平均偏离强度与量能变化的乘积。
    """
    vwap = _vwap(df)
    vc = vwap - df["close"]
    return (_rank(_ts_max(vc, 3)) + _rank(_ts_min(vc, 3))) * _rank(_delta(df["volume"], 3))


def g_012(df: pd.DataFrame) -> pd.Series:
    """α012 = RANK(OPEN - MEAN(VWAP,10)) * (-1*RANK(ABS(CLOSE - VWAP)))

    开盘相对 10 日均价的吸引力 × 收盘相对 VWAP 偏离度（取负）。
    """
    vwap = _vwap(df)
    open_diff = df["open"] - _sma(vwap, 10)
    abs_diff = (df["close"] - vwap).abs()
    return _rank(open_diff) * (-1.0 * _rank(abs_diff))


def g_013(df: pd.DataFrame) -> pd.Series:
    """α013 = ((HIGH*LOW)^0.5) - VWAP

    高低几何均值与 VWAP 的差值，日内压力位相对量价均价的溢价。
    """
    return np.sqrt(df["high"] * df["low"]) - _vwap(df)


def g_014(df: pd.DataFrame) -> pd.Series:
    """α014 = CLOSE - DELAY(CLOSE, 5)

    5 日收盘动量（区间收益之差）。
    """
    return df["close"] - _delay(df["close"], 5)


def g_015(df: pd.DataFrame) -> pd.Series:
    """α015 = OPEN / DELAY(CLOSE,1) - 1

    隔夜跳空幅度（开盘相对昨收的收益率）。
    """
    return df["open"] / _delay(df["close"], 1).replace(0, np.nan) - 1.0


def g_016(df: pd.DataFrame) -> pd.Series:
    """α016 = (-1*TSMAX(RANK(CORR(VOLUME, VWAP, 5)), 5))

    成交量与 VWAP 短期相关的峰值（取负），量价趋势反转。
    """
    vwap = _vwap(df)
    return -1.0 * _ts_max(_rank(_corr(df["volume"], vwap, 5)), 5)


def g_019(df: pd.DataFrame) -> pd.Series:
    """α019 = CLOSE<DELAY(CLOSE,5)?(CLOSE-DELAY(CLOSE,5))/DELAY(CLOSE,5)
            :(CLOSE==DELAY(CLOSE,5)?0:(CLOSE-DELAY(CLOSE,5))/CLOSE)

    对称条件 5 日收益：下跌时以昨收为分母、上涨时以现价为分母，非对称动量。
    """
    c = df["close"]
    pc = _delay(c, 5)
    diff = c - pc
    term1 = diff / pc.replace(0, np.nan)
    term3 = diff / c.replace(0, np.nan)
    out = pd.Series(np.where(c < pc, term1, np.where(c == pc, 0.0, term3)), index=df.index)
    return out


def g_020(df: pd.DataFrame) -> pd.Series:
    """α020 = (CLOSE - DELAY(CLOSE,6)) / DELAY(CLOSE,6) * 100

    6 日收益率（百分比），短周期动量。
    """
    pc = _delay(df["close"], 6).replace(0, np.nan)
    return (df["close"] - pc) / pc * 100.0


def g_021(df: pd.DataFrame) -> pd.Series:
    """α021 = REGBETA(MEAN(CLOSE,6), SEQUENCE(6))

    6 日均价随时间（等距序号）的滚动回归斜率，趋势强度。等价地用 ``_slope``
    （REGBETA(y, SEQUENCE(n)) ≡ 对等距时间序的 OLS 斜率）。
    """
    return _slope(_sma(df["close"], 6), 6)


def g_026(df: pd.DataFrame) -> pd.Series:
    """α026 = (SUM(CLOSE,7)/7 - CLOSE) + CORR(VWAP, DELAY(CLOSE,5), 230)

    7 日均价相对现价的偏离 + 长周期(230)量价相关，均值回复动量。
    """
    vwap = _vwap(df)
    return (_sma(df["close"], 7) - df["close"]) + _corr(vwap, _delay(df["close"], 5), 230)


def g_027(df: pd.DataFrame) -> pd.Series:
    """α027 = WMA((CLOSE-DELAY(CLOSE,3))/DELAY(CLOSE,3)*100 + (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100, 12)

    双周期动量（3/6 日收益之和）的 12 期线性加权平滑。
    """
    c = df["close"]
    r3 = (c - _delay(c, 3)) / _delay(c, 3).replace(0, np.nan) * 100.0
    r6 = (c - _delay(c, 6)) / _delay(c, 6).replace(0, np.nan) * 100.0
    return _wma(r3 + r6, 12)


def g_033(df: pd.DataFrame) -> pd.Series:
    """α033 = ((-1*TSMIN(LOW,5) + DELAY(TSMIN(LOW,5),5)) * RANK((SUM(RET,240)-SUM(RET,20))/220))
            * TSRANK(VOLUME,5)

    低点回升幅度 × 长/短动量差 rank × 量能时序 rank 的三重组合。
    """
    low5 = _ts_min(df["low"], 5)
    ret = _ret(df)
    mom = (_sum(ret, 240) - _sum(ret, 20)) / 220.0
    return ((-1.0 * low5 + _delay(low5, 5)) * _rank(mom)) * _ts_rank(df["volume"], 5)


def g_037(df: pd.DataFrame) -> pd.Series:
    """α037 = -1*RANK((SUM(OPEN,5)*SUM(RET,5)) - DELAY(SUM(OPEN,5)*SUM(RET,5),10))

    5 期开盘和 × 5 期收益和的动量的 10 期变化（取负反转）。
    """
    p = _sum(df["open"], 5) * _sum(_ret(df), 5)
    return -1.0 * _rank(p - _delay(p, 10))


def g_038(df: pd.DataFrame) -> pd.Series:
    """α038 = ((SUM(HIGH,20)/20) < HIGH) ? -1*DELTA(HIGH,2) : 0

    当日最高价突破 20 日均高时给出 2 期高价变化的负值，摸高反转。
    """
    cond = _sma(df["high"], 20) < df["high"]
    out = -1.0 * _delta(df["high"], 2)
    return pd.Series(np.where(cond, out, 0.0), index=df.index)


def g_045(df: pd.DataFrame) -> pd.Series:
    """α045 = RANK(DELTA((CLOSE*0.6+OPEN*0.4),1)) * RANK(CORR(VWAP, MEAN(VOLUME,150), 15))

    加权收盘动量 rank × 量价长期相关 rank。
    """
    vwap = _vwap(df)
    weighted = df["close"] * 0.6 + df["open"] * 0.4
    return _rank(_delta(weighted, 1)) * _rank(_corr(vwap, _sma(df["volume"], 150), 15))


def g_054(df: pd.DataFrame) -> pd.Series:
    """α054 = -1*RANK((STD(ABS(CLOSE-OPEN)) + (CLOSE-OPEN)) + CORR(CLOSE,OPEN,10))

    波动(价差标准差) + 日内动量 + 开收相关的综合 rank（取负）。
    """
    abs_diff = (df["close"] - df["open"]).abs()
    price_diff = df["close"] - df["open"]
    total = (_std(abs_diff, 10) + price_diff) + _corr(df["close"], df["open"], 10)
    return -1.0 * _rank(total)


def g_060(df: pd.DataFrame) -> pd.Series:
    """α060 = SUM(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW) * VOLUME, 20)

    收盘在日内区间的相对位置（-1~1）乘以成交量后 20 期求和，量价位置指标。
    """
    denom = (df["high"] - df["low"]).replace(0, np.nan)
    pos = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / denom
    return _sum(pos * df["volume"], 20)


def g_079(df: pd.DataFrame) -> pd.Series:
    """α079 = SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1) / SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1) * 100

    RSI 类相对强弱指标（12 期平滑）。
    """
    diff = df["close"] - _delay(df["close"], 1)
    up = np.maximum(diff, 0.0)
    dn = diff.abs()
    denom = _sma(dn, 12).replace(0, np.nan)
    return _sma(up, 12) / denom * 100.0


def g_088(df: pd.DataFrame) -> pd.Series:
    """α088 = (CLOSE - DELAY(CLOSE,20)) / DELAY(CLOSE,20) * 100

    20 日收益率（百分比），中短期动量。
    """
    pc = _delay(df["close"], 20).replace(0, np.nan)
    return (df["close"] - pc) / pc * 100.0


def g_096(df: pd.DataFrame) -> pd.Series:
    """α096 = SMA(SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100, 3, 1), 3, 1)

    KDJ 未成熟随机值（RSV）的双重 3 期平滑，随机摆动。
    """
    ll = _ts_min(df["low"], 9)
    hh = _ts_max(df["high"], 9)
    denom = (hh - ll).replace(0, np.nan)
    rsv = (df["close"] - ll) / denom * 100.0
    return _sma(_sma(rsv, 3), 3)


def g_102(df: pd.DataFrame) -> pd.Series:
    """α102 = SMA(MAX(VOLUME-DELAY(VOLUME,1),0),6,1) / SMA(ABS(VOLUME-DELAY(VOLUME,1)),6,1) * 100

    成交量 RSI（量能相对强弱）。
    """
    diff = df["volume"] - _delay(df["volume"], 1)
    up = np.maximum(diff, 0.0)
    denom = _sma(diff.abs(), 6).replace(0, np.nan)
    return _sma(up, 6) / denom * 100.0


_GTJA191_FUNCS: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "gtja191_001": g_001,
    "gtja191_002": g_002,
    "gtja191_003": g_003,
    "gtja191_006": g_006,
    "gtja191_007": g_007,
    "gtja191_012": g_012,
    "gtja191_013": g_013,
    "gtja191_014": g_014,
    "gtja191_015": g_015,
    "gtja191_016": g_016,
    "gtja191_019": g_019,
    "gtja191_020": g_020,
    "gtja191_021": g_021,
    "gtja191_026": g_026,
    "gtja191_027": g_027,
    "gtja191_033": g_033,
    "gtja191_037": g_037,
    "gtja191_038": g_038,
    "gtja191_045": g_045,
    "gtja191_054": g_054,
    "gtja191_060": g_060,
    "gtja191_079": g_079,
    "gtja191_088": g_088,
    "gtja191_096": g_096,
    "gtja191_102": g_102,
}


class Gtja191Factor(Factor):
    """GTJA191 短周期价量因子（单标的滚动 rank/corr/回归类）。"""

    def __init__(self, name: str) -> None:
        if name not in _GTJA191_FUNCS:
            raise KeyError(f"未知 GTJA191 因子: {name}")
        self._name = name
        self.meta = FactorMeta(
            name=name,
            category="gtja191",
            description=f"GTJA 短周期价量因子 {name}（A股日线，公式重实现）",
        )
        self.params = {"name": name}

    def compute(self, bars: List[BarData]) -> pd.Series:
        df = bars_to_df(bars)
        if df.empty:
            return pd.Series(dtype=float)
        return _GTJA191_FUNCS[self._name](df).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def list_gtja191() -> List[str]:
    """返回全部 GTJA191 因子名（gtja191_xxx）。"""
    return sorted(_GTJA191_FUNCS.keys())


def build_gtja191_factor(name: str) -> Gtja191Factor:
    """按因子名构造 Gtja191Factor 实例。"""
    return Gtja191Factor(name)
