"""Qlib158 因子族：常用量价技术指标（pandas/numpy 重实现）。

微软 qlib 内置了一组常用的量价技术指标（RSI / KDJ / MACD / BOLL / ATR / CCI / OBV /
MOM / ROC / WR / BIAS / 量比 等）。本模块借鉴这些指标的**标准公开定义**（标准技术分析法），
用 pandas + numpy + ``wq.py`` 时序原语在单标的 OHLCV DataFrame 上重新实现，不依赖
qlib 运行时，也不引入任何新的第三方依赖。

所有因子函数签名统一为 ``f(df: pd.DataFrame) -> pd.Series``，字段约定：
  - ``open / high / low / close / volume``：必填；
  - ``turnover``：成交额（可选）。涉及「换手率/量比/成交额」的因子优先使用
    ``df["turnover"]``，缺失时对换手率退化为 ``close * volume`` 代理。

每个因子均输出与输入等长的序列，缺失处为 NaN；``compute`` 层会把 inf 替换为 NaN 并
对剩余 NaN 填充 0（与 alpha191 一致）。
"""
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from ...core.object import BarData
from .base import Factor, FactorMeta, bars_to_df
from .wq import _ts_min, _ts_max, _ts_arg_max, _ts_arg_min, _ts_rank, _std, _sma, _sum, _scale

# 常用滚动均值（简单 / 指数），供内部复用
def _ema(s: pd.Series, span: int) -> pd.Series:
    """指数移动平均（EWM，span 对应平滑系数 alpha=2/(span+1)，与多数软件一致）。"""
    return s.ewm(span=span, adjust=False, min_periods=1).mean()


# ----------------------------- 指标实现（每个函数一条标准公式） -----------------------------

def qlib_rsi_14(df):  # 相对强弱指标 RSI(14)：Wilder 平滑，RSI=100*S/(S+L)
    c = df["close"]
    delta = c.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    # Wilder 平滑 = 前一值 + (当前值-前一值)/N，等价于 alpha=1/N 的 EWM
    au = up.ewm(alpha=1.0 / 14, adjust=False, min_periods=1).mean()
    ad = down.ewm(alpha=1.0 / 14, adjust=False, min_periods=1).mean()
    rs = au / ad.replace(0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(100.0).where(ad != 0, 100.0)


def qlib_kdj_k(df):  # KDJ 之 K 值(9,3,3)：K = 2/3*K_prev + 1/3*RSV
    n, m1 = 9, 3
    ll = _ts_min(df["low"], n)
    hh = _ts_max(df["high"], n)
    rng = (hh - ll).replace(0, np.nan)
    rsv = (df["close"] - ll) / rng * 100.0
    k = rsv.fillna(50.0).ewm(alpha=1.0 / m1, adjust=False, min_periods=1).mean()
    return k


def qlib_kdj_d(df):  # KDJ 之 D 值：D = 2/3*D_prev + 1/3*K
    k = qlib_kdj_k(df)
    return k.ewm(alpha=1.0 / 3, adjust=False, min_periods=1).mean()


def qlib_kdj_j(df):  # KDJ 之 J 值：J = 3*K - 2*D
    k = qlib_kdj_k(df)
    d = qlib_kdj_d(df)
    return 3.0 * k - 2.0 * d


def qlib_macd_dif(df):  # MACD 之 DIF(12,26,9)：DIF = EMA12 - EMA26
    c = df["close"]
    return _ema(c, 12) - _ema(c, 26)


def qlib_macd_dea(df):  # MACD 之 DEA(9)：DEA = EMA9(DIF)
    return _ema(qlib_macd_dif(df), 9)


def qlib_macd_hist(df):  # MACD 之 柱(MACD)：2*(DIF - DEA)
    diff = qlib_macd_dif(df)
    dea = qlib_macd_dea(df)
    return 2.0 * (diff - dea)


def qlib_boll_up(df):  # 布林带上轨(20,2)：MID + 2*STD(20,close)
    c = df["close"]
    mid = _sma(c, 20)
    return mid + 2.0 * _std(c, 20)


def qlib_boll_mid(df):  # 布林带中轨(20)：SMA(20)
    return _sma(df["close"], 20)


def qlib_boll_low(df):  # 布林带下轨(20,2)：MID - 2*STD(20,close)
    c = df["close"]
    mid = _sma(c, 20)
    return mid - 2.0 * _std(c, 20)


def qlib_atr_14(df):  # 平均真实波幅 ATR(14)：Wilder 平滑 TrueRange
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat(
        [h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / 14, adjust=False, min_periods=1).mean()


def qlib_cci_20(df):  # 顺势指标 CCI(20)：CCI=(C-TP)/(0.015*MD)
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    ma = tp.rolling(20).mean()
    md = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - ma) / (0.015 * md.replace(0, np.nan))


def qlib_obv(df):  # 能量潮 OBV：累计 (当前量-上一日量) 符号 * 当前量
    c = df["close"]
    v = df["volume"]
    sign = np.sign(c.diff().fillna(0.0))
    return (sign * v).cumsum()


def qlib_mom_10(df):  # 动量 MOM(10)：close - close_10
    return df["close"] - df["close"].shift(10)


def qlib_roc_12(df):  # 变动率 ROC(12)：100*(close-close_12)/close_12
    c = df["close"]
    prev = c.shift(12)
    return 100.0 * (c - prev) / prev.replace(0, np.nan)


def qlib_wr_14(df):  # 威廉指标 WR(14)：100*(HH-close)/(HH-LL)
    n = 14
    hh = _ts_max(df["high"], n)
    ll = _ts_min(df["low"], n)
    return 100.0 * (hh - df["close"]) / (hh - ll).replace(0, np.nan)


def qlib_bias_10(df):  # 乖离率 BIAS(10)：100*(close-MA10)/MA10
    c = df["close"]
    ma = _sma(c, 10)
    return 100.0 * (c - ma) / ma.replace(0, np.nan)


def qlib_turnover_ratio(df):  # 换手率代理：成交额/滚动均价市值；无 turnover 时退化为量/流通股
    # 无流通股本字段，退化为 close*volume 代理（量额比）
    amount = df["turnover"] if "turnover" in df else df["close"] * df["volume"]
    amt_ma = _sma(amount, 5).replace(0, np.nan)
    return amount / amt_ma


def qlib_volatility_20(df):  # 波动率(20)：20 日对数收益标准差（年化尺度近似）
    logret = np.log(df["close"] / df["close"].shift(1))
    return logret.rolling(20).std() * np.sqrt(252)


def qlib_ma_20(df):  # 简单均线 MA(20)
    return _sma(df["close"], 20)


def qlib_ma_60(df):  # 简单均线 MA(60)
    return _sma(df["close"], 60)


def qlib_volume_ratio(df):  # 量比(5)：当日成交量/过去 5 日均量（滚动）
    v = df["volume"]
    avg5 = _sma(v, 6).shift(1).replace(0, np.nan)  # 不含当日，取前 5 日均值
    ret = (v - avg5) / avg5
    return ret


def qlib_wvad(df):  # 威廉变异离散量 WVAD：sum((close-open)/(high-low)*volume) 滚动 24
    c, o, h, l = df["close"], df["open"], df["high"], df["low"]
    rng = (h - l).replace(0, np.nan)
    return _sum((c - o) / rng * df["volume"], 24)


def qlib_emv(df):  # 简易波动指标 EMV(14)：[(H+L)/2 - 昨(H+L)/2] / [(V/100)/(H-L)]
    h, l, v = df["high"], df["low"], df["volume"]
    mid = (h + l) / 2.0
    rng = (h - l).replace(0, np.nan)
    box = (v / 100.0) / rng
    div = mid.diff() / box.replace(0, np.nan)
    return div.rolling(14).mean()


def qlib_vr(df):  # 成交量比率 VR(26)：26 日 (上涨日量)/(下跌日量)*100
    c = df["close"]
    up = c.diff()
    up_amt = up.where(up > 0, 0.0).mul(df["volume"])
    dn_amt = (-up).where(up < 0, 0.0).mul(df["volume"])
    avs = _sum(up_amt, 26)
    avs_dn = _sum(dn_amt, 26)
    return 100.0 * avs / avs_dn.replace(0, np.nan)


def qlib_atr_norm(df):  # 归一化 ATR(14)：ATR/收盘
    return qlib_atr_14(df) / df["close"].replace(0, np.nan)


def qlib_stoch_k(df):  # 随机指标 K(9,3) 快速线（同 KDJ K）
    return qlib_kdj_k(df)


_QLIB_FUNCS: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "qlib_rsi_14": qlib_rsi_14,
    "qlib_kdj_k": qlib_kdj_k,
    "qlib_kdj_d": qlib_kdj_d,
    "qlib_kdj_j": qlib_kdj_j,
    "qlib_macd_dif": qlib_macd_dif,
    "qlib_macd_dea": qlib_macd_dea,
    "qlib_macd_hist": qlib_macd_hist,
    "qlib_boll_up": qlib_boll_up,
    "qlib_boll_mid": qlib_boll_mid,
    "qlib_boll_low": qlib_boll_low,
    "qlib_atr_14": qlib_atr_14,
    "qlib_cci_20": qlib_cci_20,
    "qlib_obv": qlib_obv,
    "qlib_mom_10": qlib_mom_10,
    "qlib_roc_12": qlib_roc_12,
    "qlib_wr_14": qlib_wr_14,
    "qlib_bias_10": qlib_bias_10,
    "qlib_turnover_ratio": qlib_turnover_ratio,
    "qlib_volatility_20": qlib_volatility_20,
    "qlib_ma_20": qlib_ma_20,
}


class Qlib158Factor(Factor):
    """Qlib158 常用量价技术指标因子（pandas 重实现，借鉴微软 qlib 常用指标）。"""

    def __init__(self, name: str) -> None:
        if name not in _QLIB_FUNCS:
            raise KeyError(f"未知 qlib158 因子: {name}")
        self._name = name
        self.meta = FactorMeta(
            name=name,
            category="qlib158",
            description=f"qlib 常用量价技术指标 {name}（标准公式，pandas 重实现）",
        )
        self.params = {"name": name}

    def compute(self, bars: List[BarData]) -> pd.Series:
        df = bars_to_df(bars)
        if df.empty:
            return pd.Series(dtype=float)
        return (
            _QLIB_FUNCS[self._name](df)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )


def list_qlib158() -> List[str]:
    return sorted(_QLIB_FUNCS.keys())


def build_qlib158_factor(name: str) -> Qlib158Factor:
    return Qlib158Factor(name)
