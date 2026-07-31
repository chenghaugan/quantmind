"""配对交易(Pairs Trading)策略模板。

配对交易经典做法：对两支高度协整的品种构造价差 ``spread = P_leg1 - P_leg2``，
当价差偏离其均值（z-score）超过阈值时反向入场，回归均值时离场。把价差当作一个
**合成标的**后，即可复用 QuantMind 的单标的回测/模拟/实盘「切换路线」框架。

本模块提供：
  - ``build_spread_bars(bars_a, bars_b)``：由两条 legs 构造价差合成 K 线
    （close = a.close - b.close，high/low/open 同理，volume = min(a,b)）；
  - ``PairTradingStrategy``：在价差合成标的上做 z-score 均值回复交易。

说明：这里「价差合成标的」已内含对冲关系（多价差 = 多 leg1 + 空 leg2）。若需要
在 leg1/leg2 上**同时**下两腿真实订单（而非交易合成标的），需扩展 Portfolio 级
下单接口——当前单标的 StrategyContext 下，价差合成标的是最贴合且可端到端运行的方案。
"""
from __future__ import annotations

from typing import List

import pandas as pd

from ..core.constant import Exchange, Interval
from ..core.object import BarData
from .base import CtaTemplate


def _date_key(b: BarData):
    """用日历日期对齐（忽略日内时刻/微秒差异，兼容不同数据源时间戳）。"""
    dt = b.datetime
    try:
        return dt.date()
    except Exception:  # noqa: BLE001
        return dt


def build_spread_bars(bars_a: List[BarData], bars_b: List[BarData],
                      name: str = "SPREAD") -> List[BarData]:
    """由两条 legs 构造价差合成日线（按日历日期对齐，取交集）。

    用日期（而非精确 datetime）对齐，以兼容不同数据源在日内时刻/微秒上的差异。
    """
    da = {_date_key(b): b for b in bars_a}
    db = {_date_key(b): b for b in bars_b}
    out: List[BarData] = []
    for d in sorted(set(da) & set(db)):
        a, b = da[d], db[d]
        out.append(BarData(
            symbol=name, exchange=a.exchange,
            datetime=a.datetime, interval=Interval.DAILY,
            open_price=a.open_price - b.open_price,
            high_price=a.high_price - b.high_price,
            low_price=a.low_price - b.low_price,
            close_price=a.close_price - b.close_price,
            volume=min(a.volume, b.volume),
            turnover=0.0, open_interest=0.0,
        ))
    return out


class PairTradingStrategy(CtaTemplate):
    """价差 z-score 均值回复配对交易（在价差合成标的上运行）。"""

    author = "QuantMind"
    parameters = ["window", "entry_z", "exit_z", "size", "max_pos"]

    def __init__(self, context, setting=None) -> None:
        self.window = 30          # 价差均值/标准差估计窗口
        self.entry_z = 1.5        # 开仓 z 阈值
        self.exit_z = 0.3         # 平仓 z 阈值
        self.size = 1
        self.max_pos = 1.0
        self._z: pd.Series = None
        self._idx = 0
        super().__init__(context, setting)

    def on_init(self) -> None:
        vt = self.vt_symbols[0]
        bars = self.context.get_history(vt, 9999)
        closes = pd.Series([b.close_price for b in bars], dtype=float)
        mean = closes.rolling(self.window, min_periods=10).mean()
        sd = closes.rolling(self.window, min_periods=10).std().replace(0, pd.NA)
        self._z = ((closes - mean) / sd).fillna(0.0)
        self.context.write_log(f"配对交易就绪：价差均值回复，窗口 {self.window}，开仓±{self.entry_z}")

    def on_bar(self, bar: BarData) -> None:
        if self._z is None or self._idx >= len(self._z):
            return
        z = float(self._z.iloc[self._idx])
        self._idx += 1
        if z < -self.entry_z:
            target = self.max_pos          # 价差偏低 -> 做多价差（多 leg1/空 leg2）
        elif z > self.entry_z:
            target = -self.max_pos         # 价差偏高 -> 做空价差
        elif abs(z) < self.exit_z:
            target = 0.0                   # 回归 -> 平仓
        else:
            return  # 持仓中、未触发出场阈值，保持
        self.set_target(bar.vt_symbol, target * self.size)
        self.pos = target * self.size
