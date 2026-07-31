"""期货席位因子 F1–F8（参考 quantskills 思路用 pandas 重实现）。

这些因子刻画**期货主力合约上各席位(期货公司)的净持仓结构与变化**，是商品期货
量价之外一类重要 alpha 来源（持仓排名靠前的席位往往具备信息与资金优势）。

输入：``seat_df`` —— 一个 ``DataFrame``，行=交易日(升序)，列=各席位，值=该席位
当日的**净持仓**（多头持仓 - 空头持仓，可为负）。另可传入 ``total_oi``（全市场
总持仓，1-D Series，与 seat_df 行对齐）用于占比类因子。

⚠️ 数据依赖：QuantMind 离线 MockFeed 只产出单标的 OHLCV，**不含席位级持仓**。
生产环境需接入期货席位持仓数据源（如交易所每日会员持仓排名、或第三方数据商的
seat_open_interest 接口）后构建 ``seat_df``。本模块提供 `make_synthetic_seat_df`
便于离线演示与测试。

因子定义（透明、可复核）：
  F1  净持仓          = seat_df（各席位净持仓矩阵）
  F2  净持仓变化       = seat_df.diff()
  F3  净持仓占比       = seat_df / total_oi（缺失 total_oi 时退化为除以全席位净持仓绝对值和）
  F4  多空持仓比代理   = |净持仓| / (净持仓绝对值和)（某席位资金集中度）
  F5  净持仓变化率     = seat_df.pct_change()
  F6  净持仓二阶变化   = seat_df.diff().diff()（加速度）
  F7  净持仓 Z-score   = (净持仓 - 滚动均值) / 滚动标准差（截面标准化）
  F8  席位情绪综合     = 各席位 rank 后的均值（>0.5 偏多，<0.5 偏空）

为得到可在单标的因子评估中使用的 1-D 序列，除 F1/F2/F5/F6 返回矩阵外，
`compute_seat_factors` 额外提供 `aggregate=True` 时把多席位**等权聚合**成单列
（取各席位均值），方便直接喂给 ``FactorEvaluator``。
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from ...core.object import BarData
from .base import Factor, FactorMeta, bars_to_df


def make_synthetic_seat_df(n_days: int = 250, n_seats: int = 8, seed: int = 0) -> pd.DataFrame:
    """生成合成席位净持仓（离线演示/测试用）。

    用随机游走生成各席位的净持仓，使因子有可计算结构，但不含任何真实信息。
    """
    rng = np.random.default_rng(seed)
    idx = pd.RangeIndex(n_days)
    data = {}
    for i in range(n_seats):
        walk = np.cumsum(rng.normal(0, 100, n_days))
        # 部分席位带轻微趋势，制造结构差异
        if i % 3 == 0:
            walk = walk + np.linspace(0, 500, n_days)
        data[f"seat{i}"] = walk
    return pd.DataFrame(data, index=idx)


def compute_seat_factors(
    seat_df: pd.DataFrame,
    total_oi: Optional[pd.Series] = None,
    aggregate: bool = True,
) -> Dict[str, pd.Series]:
    """计算 F1–F8。

    返回 dict：键为因子名，值为 1-D Series（aggregate=True，多席位等权聚合）；
    若 aggregate=False，F1/F2/F5/F6 返回矩阵（DataFrame），其余仍为 1-D。
    """
    seat_df = seat_df.astype(float)
    total_abs = seat_df.abs().sum(axis=1).replace(0, np.nan)
    denom = total_oi if total_oi is not None else total_abs

    f1 = seat_df
    f2 = seat_df.diff()
    f3 = seat_df.div(denom, axis=0)
    # F4: 某席位净持仓占全席位净持仓绝对值和的比例（集中度代理）
    f4 = seat_df.abs().div(total_abs, axis=0)
    f5 = seat_df.pct_change()
    f6 = seat_df.diff().diff()
    f7 = (seat_df - seat_df.rolling(20, min_periods=5).mean()) / seat_df.rolling(20, min_periods=5).std()
    # F8: 各席位截面 rank 后取均值（情绪综合）
    f8 = seat_df.rank(axis=1, pct=True).mean(axis=1)

    out: Dict[str, pd.Series] = {}
    if aggregate:
        out["F1_net_position"] = f1.mean(axis=1)
        out["F2_net_change"] = f2.mean(axis=1)
        out["F3_net_ratio"] = f3.mean(axis=1)
        out["F4_concentration"] = f4.mean(axis=1)
        out["F5_net_change_rate"] = f5.mean(axis=1)
        out["F6_net_accel"] = f6.mean(axis=1)
        out["F7_net_zscore"] = f7.mean(axis=1)
        out["F8_seat_sentiment"] = f8
    else:
        out.update({
            "F1_net_position": f1, "F2_net_change": f2, "F3_net_ratio": f3,
            "F4_concentration": f4, "F5_net_change_rate": f5, "F6_net_accel": f6,
            "F7_net_zscore": f7, "F8_seat_sentiment": f8,
        })
    # 统一填 0 避免 NaN 干扰评估
    return {k: v.fillna(0.0) for k, v in out.items()}


# ----------------------------- Factor 封装 -----------------------------
_SEAT_FACTORS = ["F1_net_position", "F2_net_change", "F3_net_ratio", "F4_concentration",
                 "F5_net_change_rate", "F6_net_accel", "F7_net_zscore", "F8_seat_sentiment"]


class SeatFactor(Factor):
    """期货席位因子。需要席位净持仓矩阵（通过 ``seat_df`` 参数提供）。

    典型用法（在策略/研究中先拿到 seat_df 再计算）：
        f = SeatFactor("F7_net_zscore", seat_df=my_seat_df, total_oi=my_oi)
        series = f.compute(bars)   # bars 仅用于对齐长度
    """

    def __init__(self, name: str, seat_df: Optional[pd.DataFrame] = None,
                 total_oi: Optional[pd.Series] = None) -> None:
        if name not in _SEAT_FACTORS:
            raise KeyError(f"未知席位因子: {name}（可选: {_SEAT_FACTORS}）")
        self._name = name
        self.seat_df = seat_df
        self.total_oi = total_oi
        self.meta = FactorMeta(name=name, category="futures_seat",
                               description=f"期货席位因子 {name}（需席位净持仓数据）")
        self.params = {"name": name}

    def set_data(self, seat_df: pd.DataFrame, total_oi: Optional[pd.Series] = None) -> "SeatFactor":
        self.seat_df = seat_df
        self.total_oi = total_oi
        return self

    def compute(self, bars: List[BarData]) -> pd.Series:
        if self.seat_df is None:
            raise RuntimeError(
                "SeatFactor 需要席位净持仓数据：先调用 set_data(seat_df, total_oi) "
                "或构造时传入 seat_df。当前数据源(MockFeed)不含席位级持仓。"
            )
        res = compute_seat_factors(self.seat_df, self.total_oi, aggregate=True)
        s = res[self._name]
        # 对齐到输入 bars 长度（多退少补 0）
        n = len(bars)
        if len(s) >= n:
            return s.iloc[:n].reset_index(drop=True)
        out = pd.Series([0.0] * n)
        out.iloc[: len(s)] = s.values
        return out
