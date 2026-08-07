"""学术风格因子族（价格代理版）。

本模块实现一批经典学术文献中的因子，但以**诚实的价格代理**形式给出。原因：这些因子
（Fama-French 五因子、Carhart 动量、Frazzini-Pedersen Betting-Against-Beta、
Ang 特质波动率等）本需基本面/横截面数据，而 QuantMind 在**单标的时序**语境运行，
故参考 Vibe-Trading 的做法，仅用单标的价格/成交额（OHLCV）近似，并在注释中**如实标注
代理性质**，避免误导为真实横截面因子。

学术来源（仅借鉴其数学定义，未复制源码）：
  - Fama & French (1993, 2015)：市场/规模/价值/盈利/投资五因子。
  - Carhart (1997)：12-1 动量。
  - Frazzini & Pedersen (2014)：Betting Against Beta（BAB）。
  - Ang, Hodrick, Xing & Zhang (2006)：特质波动率。
  - Jegadeesh (1990) / Lehmann (1990)：短期反转。
  - Amihud (2002)：非流动性（流动性）因子。

**前视规避**：全部因子仅使用滚动/扩张窗口内**当时已知**的历史数据（rolling/expanding
窗口右端点即为当前 bar，不借用未来信息），且 `compute` 返回与 K 线等长的序列（不足窗口的
预热期填 NaN，最终 `fillna(0.0)` 由包装类完成）。
"""
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from ...core.object import BarData
from .base import Factor, FactorMeta, bars_to_df
from .wq import (
    _cov, _ts_min, _ts_max, _slope, _std, _adv,
)

# 无成交额数据时的典型价格（(H+L+C)/3）与(vwap)近似；用于流动性代理。
# 因子多数仅依赖 close，故其余字段仅供少数因子参考。
PRICE_PROXY_NOTICE = (
    "【价格代理】该因子为学术定义的时序近似，仅用单标的价格/成交额计算，"
    "不等价于真实横截面因子，仅作研究参考。"
)


def _ret(df: pd.DataFrame) -> pd.Series:
    """收盘日收益（对数或简单 pct_change 均可，这里用简单收益）。"""
    return df["close"].pct_change()


def _market_return(df: pd.DataFrame, win: int = 60) -> pd.Series:
    """『市场』价格代理：自身滚动均价的收益率。

    单标的时序语境下没有外部指数，故以 close 的滚动均值收益近似『市场组合』。
    前导 NaN 以 0 填充，仅影响预热期，不造成前视。
    """
    close = df["close"]
    mkt = close.rolling(win, min_periods=20).mean().pct_change()
    return mkt.fillna(0.0)


def _roll_beta(y: pd.Series, x: pd.Series, d: int) -> pd.Series:
    """滚动 OLS 斜率 beta = Cov(y,x)/Var(x)（与 _reg_beta 等价的时序实现）。

    用滚动协方差/方差直接计算，避免 wq._reg_beta 在 DataFrame.rolling.apply
    上的一维退化问题。仅用历史窗口，无前视。
    """
    denom = x.rolling(d, min_periods=max(5, d // 3)).var().replace(0, np.nan)
    return _cov(y, x, d) / denom


def _roll_alpha(beta: pd.Series, y: pd.Series, x: pd.Series, d: int) -> pd.Series:
    """滚动 OLS 截距 alpha = mean(y) - beta*mean(x)。"""
    return y.rolling(d, min_periods=max(5, d // 3)).mean() - beta * x.rolling(
        d, min_periods=max(5, d // 3)
    ).mean()



# ----------------------------- 学术因子公式（价格代理版） -----------------------------

def acad_mom_12m_1m(df: pd.DataFrame) -> pd.Series:
    """动量 12-1（Carhart 1997 动量因子价格代理）。

    学术定义：过去 12 个月收益扣除最近 1 个月（避免短期反转污染），截面排序做多/做空。
    价格代理：用 rolling 收益率 ``close/close.shift(12*21)-1`` 再减去最近一个月
    ``close/close.shift(21)-1``。单标的下表征长中期动量强度。
    """
    close = df["close"]
    # 12 个月（约 21*12=252 个交易日）复合收益
    mom12 = close / close.shift(252) - 1.0
    # 最近 1 个月（约 21 个交易日）收益
    ret1 = close / close.shift(21) - 1.0
    return mom12 - ret1


def acad_short_term_reversal(df: pd.DataFrame) -> pd.Series:
    """短期反转（Jegadeesh 1990 / Lehmann 1990 价格代理）。

    学术定义：短期（1 个月）赢家继续走低的动量反转效应，做空最近强势者。
    价格代理：取最近 1 个月收益的相反数 ``-(close/close.shift(21)-1)``。
    """
    return -(_ret(df).rolling(21, min_periods=1).sum())


def acad_vol_20(df: pd.DataFrame) -> pd.Series:
    """波动率（20 日，Fama-French 市场波动/风险的价格代理）。

    学术定义：横截面/市场波动率用于风险定价。价格代理：20 日日收益的滚动标准差（_std）。
    """
    return _std(_ret(df), 20)


def acad_value_proxy(df: pd.DataFrame) -> pd.Series:
    """价值代理 / 价格位置（Fama-French HML 价值因子的价格代理）。

    学术定义：账面市值比 B/M（HML 做多高 B/M、做空低 B/M）。单标的不含基本面，
    故以「价格在滚动区间中的相对低位」作为『便宜/超跌』代理：
    值 = (close - rolling_min) / (rolling_max - rolling_min)，取值范围 [0,1]，
    数值越小说明价格越接近区间低点，越『便宜』（价值风格倾向）。
    """
    close = df["close"]
    lo = _ts_min(close, 250)
    hi = _ts_max(close, 250)
    rng = (hi - lo).replace(0, np.nan)
    return (close - lo) / rng


def acad_beta(df: pd.DataFrame) -> pd.Series:
    """市场 beta（Fama-French 市场因子 beta 的价格代理）。

    学术定义：个股收益对市场组合收益的回归斜率。单标的下以自身滚动均值作为
    『市场』代理，用 _roll_beta（滚动协方差/方差）回归 close 收益对自身滚动均值收益
    得 beta。
    """
    ret = _ret(df)
    mkt = _market_return(df)
    return _roll_beta(ret, mkt, 60)


def acad_bab(df: pd.DataFrame) -> pd.Series:
    """Betting-Against-Beta（Frazzini & Pedersen 2014）价格代理。

    学术定义：低 beta 组合做多、高 beta 组合做空的 BAB。真实 BAB 需截面（跨资产
    排序）。单标的下退化为「负的滚动 beta」：beta 越低，因子值越高，倾向做多低 beta
    （低波动异象）。即 -_roll_beta（对自身滚动均值市场代理回归）。
    """
    ret = _ret(df)
    mkt = _market_return(df)
    beta = _roll_beta(ret, mkt, 60)
    return -beta


def acad_idio_vol(df: pd.DataFrame) -> pd.Series:
    """特质波动率（Ang, Hodrick, Xing & Zhang 2006）价格代理。

    学术定义：用市场收益回归个股，取残差标准差（特质波动）。真实需横截面回归。
    单标的下：以 _roll_beta 把 close 收益对自身滚动均值『市场』代理回归，取残差，
    再对该残差取滚动标准差即特质波动代理。
    """
    ret = _ret(df)
    mkt = _market_return(df)
    beta = _roll_beta(ret, mkt, 60)
    alpha = _roll_alpha(beta, ret, mkt, 60)
    resi = ret - (alpha + beta * mkt)
    return _std(resi, 20)


def acad_profit_growth(df: pd.DataFrame) -> pd.Series:
    """盈利/成长代理（Fama-French RMW 盈利因子价格代理）。

    学术定义：RMW 做多高盈利、做空低盈利；此处以价格动量趋势作为盈利/成长的价格
    代理：对未来盈利改善的预期常反映在趋势斜率上，用 60 日 close 的滚动回归斜率
    _slope（正斜率 ≈ 高盈利成长）。注明为纯价格代理。
    """
    return _slope(df["close"], 60)


def acad_liquidity_20(df: pd.DataFrame) -> pd.Series:
    """流动性（Amihud 2002 非流动性的价格代理，取倒数）。

    学术定义：Amihud 非流动性 = |收益| / 成交额（越大越不流动）；流动性 = 倒数。
    价格代理：无成交额字段时以 成交额≈volume*close 近似，流动性 = 成交额/价格
    （正比于换手活跃度）。此处用 _adv(20) 滚动平均成交额，再除以 close，表征流动性。
    """
    adv20 = _adv(df, 20)
    return adv20 / df["close"].replace(0, np.nan)


def acad_downside_vol(df: pd.DataFrame) -> pd.Series:
    """下行风险 / 下行波动（Sortino 比率分母；下偏波动价格代理）。

    学术定义：仅负收益的波动率（下行偏差）。价格代理：把正收益置 0，对负收益序列
    取 20 日滚动标准差。
    """
    ret = _ret(df)
    downside = ret.where(ret < 0, 0.0)
    return _std(downside, 20)


def acad_skew_20(df: pd.DataFrame) -> pd.Series:
    """偏度（收益三阶矩，价格代理）。

    学术定义：收益分布偏度（负偏度风险溢价）。价格代理：20 日日收益的滚动偏度。
    """
    return _ret(df).rolling(20, min_periods=5).skew()


_ACADEMIC_FUNCS: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "acad_mom_12m_1m": acad_mom_12m_1m,
    "acad_short_term_reversal": acad_short_term_reversal,
    "acad_vol_20": acad_vol_20,
    "acad_value_proxy": acad_value_proxy,
    "acad_beta": acad_beta,
    "acad_bab": acad_bab,
    "acad_idio_vol": acad_idio_vol,
    "acad_profit_growth": acad_profit_growth,
    "acad_liquidity_20": acad_liquidity_20,
    "acad_downside_vol": acad_downside_vol,
    "acad_skew_20": acad_skew_20,
}


class AcademicFactor(Factor):
    """学术风格因子（价格代理版）。

    借鉴 Fama-French / Carhart / Frazzini-Pedersen / Ang 等文献，全部以单标的
    价格/成交额的滚动统计实现，**避免前视**（仅用历史已知窗口）。
    """

    def __init__(self, name: str) -> None:
        if name not in _ACADEMIC_FUNCS:
            raise KeyError(f"未知学术因子: {name}")
        self._name = name
        self.meta = FactorMeta(
            name=name,
            category="academic",
            description=f"学术因子(价格代理): {name}",
        )
        self.params = {"name": name}

    def compute(self, bars: List[BarData]) -> pd.Series:
        df = bars_to_df(bars)
        if df.empty:
            return pd.Series(dtype=float)
        out = _ACADEMIC_FUNCS[self._name](df)
        # 清理无穷并填充预热期 NaN；确保与 bars 等长
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def list_academic() -> List[str]:
    """返回全部学术因子名（排序）。"""
    return sorted(_ACADEMIC_FUNCS.keys())


def build_academic_factor(name: str) -> AcademicFactor:
    """按名字构造学术因子。"""
    return AcademicFactor(name)
