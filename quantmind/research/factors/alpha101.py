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
    _ts_arg_min, _ts_rank, _ts_median, _signed_power, _scale, _decay_linear,
    _slope, _sma, _std, _sum, _vwap, _adv,
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


# ----------------------------- 新增经典 Alpha（忠实官方公式） -----------------------------
def a009(df):
    # (rank(open - Ts_Min(high,2)) > rank(open - Ts_Min(low,2))) ? (high - close) : (close - low)
    # 依据开盘价相对近2日高低点的位置，判断当日为阳线取 high-close、阴线取 close-low。
    cond = _rank(df["open"] - _ts_min(df["high"], 2)) > _rank(df["open"] - _ts_min(df["low"], 2))
    out = np.where(cond, df["high"] - df["close"], df["close"] - df["low"])
    return pd.Series(out, index=df.index)


def a010(df):
    # rank(max(((ret < 0) ? std(ret,20) : close)^2, 5))
    # 过去5日收益率与收盘价平方的最大值做时序分位排名。
    ret = _ret(df)
    base = np.where(ret < 0, _std(ret, 20).values, df["close"].values)
    sp = pd.Series(base, index=df.index) ** 2
    return _rank(_ts_max(sp, 5))


def a023(df):
    # ((sum(high,20)/20) < close) ? (-1 * ts_rank(abs(delta(close,7)),60)) : -1) * rank(corr(volume, close,10))
    # 若收盘价高于20日最高均价，则给出更消极的涨幅震荡排名，否则取 -1。
    cond = (_sum(df["high"], 20) / 20) < df["close"]
    chosen = np.where(cond, (-1.0 * _ts_rank(_delta(df["close"], 7).abs(), 60)).values, -1.0)
    chosen = pd.Series(chosen, index=df.index)
    return chosen * _rank(_corr(df["volume"], df["close"], 10))


def a027(df):
    # (0.5 < rank(sum(corr(rank(volume), rank(vwap),6),2)/2)) ? -1 : 1) * rank(corr(rank(close), rank(median(volume,3)),5))
    # 量价相关的横截面强弱 → 量价相关性与三日量中位数的相关。
    vwap = _vwap(df)
    term = _rank(_sum(_corr(_rank(df["volume"]), _rank(vwap), 6), 2) / 2.0)
    sign = pd.Series(np.where(0.5 < term, -1.0, 1.0), index=df.index)
    med = _ts_median(df["volume"], 3)
    return sign * _rank(_corr(_rank(df["close"]), _rank(med), 5))


def a029(df):
    # min(rank(rank(scale(log(sum(rank(corr(rank(volume), rank(vwap),6)),2))))),5)
    #     + ts_rank(delay(-1*rank(rank(scale(log(sum(rank(corr(rank(volume), rank(vwap),6)),2))))),6),4)
    # 量价相关的对数占比复合排名，叠加其滞后6日的反向时序排名。
    vwap = _vwap(df)
    inner = _rank(_corr(_rank(df["volume"]), _rank(vwap), 6))
    x = _sum(_rank(inner), 2)
    y = _rank(_rank(_scale(np.log(x.clip(lower=1e-12)))))
    return np.minimum(_rank(y), 5.0) + _ts_rank(_delay(-1.0 * y, 6), 4)


def a031(df):
    # rank(rank(rank(decay_linear(-1*rank(rank(delta(close,10))),10)))) + rank(-1*delta(close,3)) + sign(scale(corr(adv20,low,12)))
    # 十日动量衰减 + 三日反转变动 + 成交额-低价相关的方向。
    d1 = _rank(_rank(_delta(df["close"], 10)))
    term1 = _rank(_rank(_rank(_decay_linear(-1.0 * d1, 10))))
    term2 = _rank(-1.0 * _delta(df["close"], 3))
    term3 = np.sign(_scale(_corr(_adv(df, 20), df["low"], 12)))
    return term1 + term2 + term3


def a032(df):
    # scale((sum(close,7)/7 - close)) + 20*scale(corr(vwap, delay(close,5),230))
    # 7日收盘均值偏离 + 均价相对5日前收盘的长程相关放大。
    vwap = _vwap(df)
    return _scale((_sum(df["close"], 7) / 7) - df["close"]) + 20.0 * _scale(
        _corr(vwap, df["close"].shift(5), 230)
    )


def a034(df):
    # rank((1-rank(std(ret,2)/std(ret,5))) + (1-rank(delta(close,1))))
    # 短期波动率占比反转 + 单日涨跌反转的复合排名。
    ret = _ret(df)
    ratio = _std(ret, 2) / _std(ret, 5).replace(0, np.nan)
    inner = (1 - _rank(ratio.replace([np.inf, -np.inf], np.nan))) + (1 - _rank(_delta(df["close"], 1)))
    return _rank(inner.replace([np.inf, -np.inf], np.nan).fillna(0.0))


def a035(df):
    # ts_rank(volume/adv20, 60) * (1 - ts_rank(close/high - 1, 20)) * -1
    # 成交量相对20日均量触顶、同时收盘价相对高点回落时给出反向信号。
    r1 = _ts_rank(df["volume"] / _adv(df, 20), 60)
    r2 = _ts_rank((df["close"] / df["high"]) - 1.0, 20)
    return r1 * (1 - r2) * -1.0


def a041(df):  # rank(vwap-close) / rank(vwap+close)  均价偏离强度
    vwap = _vwap(df)
    num = _rank(vwap - df["close"])
    den = _rank(vwap + df["close"]).replace(0, np.nan)
    return (num / den).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def a042(df):  # -1 * rank(std(high,10)) * corr(high,volume,10)  高价波动与量相关
    return -1.0 * _rank(_std(df["high"], 10)) * _corr(df["high"], df["volume"], 10)


def a043(df):
    # ((close - high) < 0) ? (-1 * rank(corr(close,volume,10))) : (close - high)
    # 当日收盘小于最高价时取量价相关排名，否则取收盘对高位的偏离。
    cond = (df["close"] - df["high"]) < 0
    chosen = np.where(
        cond,
        (-1.0 * _rank(_corr(df["close"], df["volume"], 10))).values,
        (df["close"] - df["high"]).values,
    )
    return pd.Series(chosen, index=df.index)


def a044(df):  # -1 * rank(ts_rank(close,30)) * corr(close,volume,10)  价量相关
    return -1.0 * _rank(_ts_rank(df["close"], 30)) * _corr(df["close"], df["volume"], 10)


def a045(df):
    # -1 * rank(sum(delay(close,5),20)/20) * corr(close,volume,2) * rank(corr(sum(close,5), sum(close,20),2))
    # 5日滞后均价的排名、短期量价相关与长短周期收盘相关三者复合。
    c = df["close"]
    t = -1.0 * _rank(_sum(c.shift(5), 20) / 20) * _corr(c, df["volume"], 2)
    return t * _rank(_corr(_sum(c, 5), _sum(c, 20), 2))


def a050(df):  # -1 * ts_max(rank(corr(rank(volume), rank(vwap),5)),5)  量价相关冲高反转
    vwap = _vwap(df)
    return -1.0 * _ts_max(_rank(_corr(_rank(df["volume"]), _rank(vwap), 5)), 5)


def a052(df):
    # ((-tsmin(low,9)+tsmax(high,9)) < 0) ? 1 : (((sum(close,9)/9) - close) / (sum(close,9)/9)) < 0) ? -1 : 1
    # 近9日高低幅为负时看多；否则按收盘相对9日均价的偏离方向取 ±1。
    c9 = _sum(df["close"], 9) / 9
    inner = (c9 - df["close"]) / c9.replace(0, np.nan)
    cond1 = (-_ts_min(df["low"], 9) + _ts_max(df["high"], 9)) < 0
    out = np.where(cond1, 1.0, np.where(inner.fillna(0.0) < 0, -1.0, 1.0))
    return pd.Series(out, index=df.index)


def a056(df):
    # rank(ts_rank(1/close,20)) * (1-rank(volume/adv20)) * (1 - rank(sum(close,5)/5 - close) / rank(vwap))
    # 倒数收盘时序排名、量能收缩与5日均价偏离三者复合。
    vwap = _vwap(df)
    c = df["close"]
    t1 = _rank(_ts_rank(1.0 / c, 20))
    t2 = 1.0 - _rank(df["volume"] / _adv(df, 20))
    den = _rank(vwap).replace(0, np.nan)
    t3 = (1.0 - (_rank(_sum(c, 5) / 5 - c) / den)).replace([np.inf, -np.inf], np.nan)
    return (t1 * t2 * t3).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def a066(df):
    # (rank(decay_linear(delta(vwap,3),7)) + ts_rank(decay_linear((low*0.9+low*0.1)-vwap,5),6)) * -1
    # 均价三日动量的衰减排名 + 低点相对均价偏离的时序排名，整体反向。
    vwap = _vwap(df)
    t1 = _rank(_decay_linear(_delta(vwap, 3), 7))
    inner = ((df["low"] * 0.9) + (df["low"] * 0.1)) - vwap
    t2 = _ts_rank(_decay_linear(inner, 5), 6)
    return (t1 + t2) * -1.0


def a078(df):
    # rank(corr(sum(low*0.35+vwap*0.65,20), sum(mean(volume,40),20),7))
    # 加权低价均价20日总和与40日均量20日总和的相关。
    vwap = _vwap(df)
    a = _sum((df["low"] * 0.35) + (vwap * 0.65), 20)
    b = _sum(_sma(df["volume"], 40), 20)
    return _rank(_corr(a, b, 7))


def a095(df):  # rank(open - ts_arg_min(close,30))  开盘价相对30日最低价位置
    return _rank(df["open"] - _ts_arg_min(df["close"], 30))


# ----------------------------- 补充高价值经典 Alpha（显式真实公式） -----------------------------
def a001(df):  # rank(Ts_ArgMax(SignedPower(((ret<0)?std(ret,20):close), 2), 5)) - 0.5
    ret = df["close"].pct_change()
    std_ret = _std(ret, 20)
    base = np.where(ret < 0, std_ret.values, df["close"].values)
    sp = _signed_power(pd.Series(base, index=df.index), 2.0)
    return _rank(_ts_arg_max(sp, 5)) - 0.5


def a004(df):  # -Ts_Rank(rank(low), 9)
    return -_ts_rank(_rank(df["low"]), 9)


def a005(df):  # rank(open - sma(vwap,10)) * (-1 * abs(rank(close - vwap)))
    vwap = _vwap(df)
    r1 = _rank(df["open"] - _sma(vwap, 10))
    r2 = _rank(df["close"] - vwap).abs()
    return r1 * (-1.0 * r2)


def a007(df):  # (rank(max(open-close,0))+rank(max(low-close,0))+rank(min(open-close,0))+rank(min(high-close,0)))^2
    oc = df["open"] - df["close"]
    lc = df["low"] - df["close"]
    hc = df["high"] - df["close"]
    r = (_rank(oc.clip(lower=0)) + _rank(lc.clip(lower=0))
         + _rank(oc.clip(upper=0)) + _rank(hc.clip(upper=0)))
    return r * r


def a008(df):  # -rank((sum(open,5)*sum(ret,5)) - delay(., 10))
    prod = _sum(df["open"], 5) * _sum(_ret(df), 5)
    return -_rank(prod - prod.shift(10))


def a011(df):  # (rank(TsMax(vwap-close,3)) + rank(TsMin(vwap-close,3))) * rank(Δvolume 3)
    vwap = _vwap(df)
    diff = vwap - df["close"]
    return (_rank(_ts_max(diff, 3)) + _rank(_ts_min(diff, 3))) * _rank(_delta(df["volume"], 3))


def a028(df):  # scale((((close-low)-(high-close))/(high-low)))  价格位置
    hl = (df["high"] - df["low"]).replace(0, np.nan)
    inner = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / hl
    inner = inner.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return _scale(inner)


def a053(df):  # -delta(价格位置, 9)
    hl = (df["high"] - df["low"]).replace(0, np.nan)
    inner = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / hl
    inner = inner.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return -_delta(inner, 9)


def a055(df):  # -corr(rank(价格位置), rank(volume), 5)
    hl = (df["high"] - df["low"]).replace(0, np.nan)
    inner = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / hl
    inner = inner.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return -_corr(_rank(inner), _rank(df["volume"]), 5)


_ALPHA_FUNCS: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "alpha001": a001, "alpha002": a002, "alpha003": a003, "alpha004": a004,
    "alpha005": a005, "alpha006": a006, "alpha007": a007, "alpha008": a008,
    "alpha009": a009, "alpha010": a010, "alpha011": a011, "alpha012": a012,
    "alpha013": a013, "alpha014": a014, "alpha015": a015, "alpha016": a016,
    "alpha017": a017, "alpha018": a018, "alpha019": a019, "alpha020": a020,
    "alpha021": a021, "alpha022": a022, "alpha023": a023, "alpha024": a024,
    "alpha026": a026, "alpha027": a027, "alpha028": a028, "alpha029": a029,
    "alpha031": a031, "alpha032": a032, "alpha033": a033, "alpha034": a034,
    "alpha035": a035, "alpha037": a037, "alpha038": a038, "alpha040": a040,
    "alpha041": a041, "alpha042": a042, "alpha043": a043, "alpha044": a044,
    "alpha045": a045, "alpha049": a049, "alpha050": a050, "alpha051": a051,
    "alpha052": a052, "alpha053": a053, "alpha054": a054, "alpha055": a055,
    "alpha056": a056, "alpha060": a060, "alpha062": a062, "alpha066": a066,
    "alpha071": a071, "alpha075": a075, "alpha078": a078, "alpha083": a083,
    "alpha093": a093, "alpha095": a095, "alpha099": a099, "alpha101": a101,
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
        return _ALPHA_FUNCS[self._name](df).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def list_alpha101() -> List[str]:
    return sorted(_ALPHA_FUNCS.keys())


def build_alpha_factor(name: str) -> AlphaFactor:
    return AlphaFactor(name)
