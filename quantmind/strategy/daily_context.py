# -*- coding: utf-8 -*-
"""日线级上下文查询对象（供分钟级策略查询日线指标）。

设计动机（借鉴 LLM 策略挖掘实践）：很多策略思想天然是「日线定方向、
分钟找入场」的多时间框架结构。回测引擎只喂单一周期的 bar，本对象在
回测启动前把日线数据预计算成查询表，注入策略实例（``self.daily``），
供 ``on_bar`` 按当前 bar 的日期查询日线级指标。

防前视语义：**所有指标只使用「当前交易日之前」的已完成日线**——
在 D 日任意分钟时刻查询，得到的是截至 D-1 交易日的指标，
当日尚未收盘的日线绝不参与计算。
"""
from __future__ import annotations

from bisect import bisect_left
from datetime import datetime, date as _date
from typing import List, Optional

from ..core.object import BarData


class DailyContext:
    """日线指标查询表。

    用法（生成的策略 ``on_bar`` 内）::

        ph = self.daily.prev_high(bar.datetime)      # 前一交易日高点
        ma = self.daily.sma(20, bar.datetime)        # 截至前一交易日的20日均线
        a = self.daily.atr(14, bar.datetime)         # 截至前一交易日的14日ATR

    所有查询基于「当前日期之前」的已完成日线，天然无前视。
    """

    def __init__(self, daily_bars: List[BarData]) -> None:
        self._bars = sorted(daily_bars, key=lambda b: b.datetime)
        self._dates = [b.datetime.date() for b in self._bars]
        self._closes = [b.close_price for b in self._bars]
        self._opens = [b.open_price for b in self._bars]
        self._highs = [b.high_price for b in self._bars]
        self._lows = [b.low_price for b in self._bars]

    # ------------------------------------------------------------------
    def _pos(self, current_dt) -> int:
        """返回「严格早于 current_dt 日期」的最后一根日线的下标；无则 -1。"""
        d = current_dt.date() if isinstance(current_dt, datetime) else current_dt
        if isinstance(d, _date):
            i = bisect_left(self._dates, d)
            return i - 1
        return -1

    def _n_pos(self, n: int, current_dt) -> List[int]:
        """返回截至前一交易日的最近 n 根日线下标（不足 n 根时返回可用部分）。"""
        p = self._pos(current_dt)
        if p < 0:
            return []
        return list(range(max(0, p - n + 1), p + 1))

    # ---------------------------- 单日价格 ----------------------------
    def prev_high(self, current_dt, offset: int = 0) -> Optional[float]:
        """前一（offset=0）或更早 offset 个交易日的最高价。"""
        idx = self._pos(current_dt) - offset
        return self._bars[idx].high_price if 0 <= idx else None

    def prev_low(self, current_dt, offset: int = 0) -> Optional[float]:
        idx = self._pos(current_dt) - offset
        return self._bars[idx].low_price if 0 <= idx else None

    def prev_close(self, current_dt, offset: int = 0) -> Optional[float]:
        idx = self._pos(current_dt) - offset
        return self._bars[idx].close_price if 0 <= idx else None

    def prev_open(self, current_dt, offset: int = 0) -> Optional[float]:
        idx = self._pos(current_dt) - offset
        return self._bars[idx].open_price if 0 <= idx else None

    # ---------------------------- 滚动指标 ----------------------------
    def sma(self, n: int, current_dt) -> Optional[float]:
        """截至前一交易日的 n 日简单均线（不足 n 根返回 None）。"""
        idxs = self._n_pos(n, current_dt)
        if len(idxs) < n:
            return None
        return sum(self._closes[i] for i in idxs) / n

    def highest(self, n: int, current_dt) -> Optional[float]:
        """截至前一交易日的近 n 日最高价。"""
        idxs = self._n_pos(n, current_dt)
        if not idxs:
            return None
        return max(self._highs[i] for i in idxs)

    def lowest(self, n: int, current_dt) -> Optional[float]:
        """截至前一交易日的近 n 日最低价。"""
        idxs = self._n_pos(n, current_dt)
        if not idxs:
            return None
        return min(self._lows[i] for i in idxs)

    def atr(self, n: int = 14, current_dt=None) -> Optional[float]:
        """截至前一交易日的 n 日平均真实波幅（Wilder 简化：算术平均）。"""
        idxs = self._n_pos(n + 1, current_dt)
        if len(idxs) < 2:
            return None
        trs = []
        for a, b in zip(idxs, idxs[1:]):
            tr = max(
                self._highs[b] - self._lows[b],
                abs(self._highs[b] - self._closes[a]),
                abs(self._lows[b] - self._closes[a]),
            )
            trs.append(tr)
        if len(trs) < n:
            return None
        return sum(trs[-n:]) / n
