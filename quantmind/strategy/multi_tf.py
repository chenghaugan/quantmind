# -*- coding: utf-8 -*-
"""多周期上下文（MultiTFContext）：供单周期策略查询更高周期数据。

第一性原理（三条不变量）：
1. 无前视：时刻 t 的视图只包含 close_time ≤ t 的已完成 bar
   （bar 的 close_time = 桶内最后一根成员 bar 的 open_time + 基础周期）。
2. 会话安全：重采样用「锚定推进」而非日历分桶——期货节间休息/夜盘 gap
   不会产生幻影桶；代价是数据缺失时桶锚点会漂移（对指标计算无影响）。
3. 失败显式：引用不存在的周期 → KeyError（由调用方失败闭合捕获为显式错误）；
   数据深度不足 → 指标返回 None（预热语义，由生成代码判空跳过）。

不做任何数据量假设：可用 = 数据存在；深度由实际数据决定。
"""
from __future__ import annotations

from bisect import bisect_right
from datetime import timedelta
from typing import Dict, List, Optional

from ..core.object import BarData

# 基础周期 → 可重采样的更高周期（同源推导，保证与回测数据流对齐）
RESAMPLE_CANDIDATES: Dict[str, List[str]] = {
    "1m": ["5m", "15m", "30m", "1h"],
    "5m": ["15m", "30m", "1h"],
    "15m": ["30m", "1h"],
    "30m": ["1h"],
    "1h": [],
    "1d": ["1w"],
}

_INTERVAL_SECONDS = {
    "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "1d": 86400, "1w": 7 * 86400,
}


def resample_bars(bars: List[BarData], target: str,
                  base_interval: str) -> "tuple[List[BarData], List[datetime]]":
    """把基础周期 bars 锚定推进聚合为 target 周期。

    桶锚定：新桶当 (bar.open_time - anchor) >= target 周期时开启；
    桶的 close_time = 桶内最后一根成员 bar 的 open_time + 基础周期 delta。
    注意：数据缺失时桶锚点会相对整点漂移（对指标计算无影响，已测试锁定）。
    """
    if not bars:
        return [], []
    base_sec = _INTERVAL_SECONDS[base_interval]
    target_sec = _INTERVAL_SECONDS[target]
    if target_sec <= base_sec:
        raise ValueError(f"resample 目标周期 {target} 必须大于基础周期")

    out: List[BarData] = []
    close_times: List[datetime] = []
    anchor = None
    cur: Optional[BarData] = None
    last_open = None
    for b in bars:
        if cur is None or (b.datetime - anchor).total_seconds() >= target_sec:
            if cur is not None:
                out.append(cur)
                # 桶 close_time = 桶内最后一根成员 bar 的 open_time + 基础周期
                close_times.append(last_open + timedelta(seconds=base_sec))
            anchor = b.datetime
            last_open = b.datetime
            cur = BarData(
                symbol=b.symbol, exchange=b.exchange, datetime=b.datetime,
                interval=b.interval,
                open_price=b.open_price, high_price=b.high_price,
                low_price=b.low_price, close_price=b.close_price,
                volume=b.volume, open_interest=b.open_interest,
            )
        else:
            cur.high_price = max(cur.high_price, b.high_price)
            cur.low_price = min(cur.low_price, b.low_price)
            cur.close_price = b.close_price
            cur.volume += b.volume
            cur.open_interest = b.open_interest
        last_open = b.datetime
    if cur is not None:
        out.append(cur)
        close_times.append(last_open + timedelta(seconds=base_sec))
    return out, close_times


class TFView:
    """某周期在锚定时刻 t 的「已完成 bar」视图。

    所有指标只使用 close_time ≤ t 的数据（结构性防前视）；
    数据深度不足时指标返回 None（预热语义）。
    """

    def __init__(self, name: str, bars: List[BarData], count: int,
                 close_times: List) -> None:
        self.name = name
        self._bars = bars
        self._count = count          # 可用 bar 数（bisect 结果）
        self._close_times = close_times

    # ---- 原始序列（切片，用于自定义指标） ----
    @property
    def close(self) -> List[float]:
        return [b.close_price for b in self._bars[:self._count]]

    @property
    def high(self) -> List[float]:
        return [b.high_price for b in self._bars[:self._count]]

    @property
    def low(self) -> List[float]:
        return [b.low_price for b in self._bars[:self._count]]

    # ---- 单值查询 ----
    def prev_high(self, offset: int = 0) -> Optional[float]:
        i = self._count - 1 - offset
        return self._bars[i].high_price if i >= 0 else None

    def prev_low(self, offset: int = 0) -> Optional[float]:
        i = self._count - 1 - offset
        return self._bars[i].low_price if i >= 0 else None

    def prev_close(self, offset: int = 0) -> Optional[float]:
        i = self._count - 1 - offset
        return self._bars[i].close_price if i >= 0 else None

    def prev_open(self, offset: int = 0) -> Optional[float]:
        i = self._count - 1 - offset
        return self._bars[i].open_price if i >= 0 else None

    # ---- 滚动指标（截至锚定时刻，窗口 = 最近 n 根已完成 bar）----
    def sma(self, n: int) -> Optional[float]:
        if self._count < n:
            return None
        return sum(b.close_price for b in self._bars[self._count - n:self._count]) / n

    def highest(self, n: int) -> Optional[float]:
        if self._count < n:
            return None
        return max(b.high_price for b in self._bars[self._count - n:self._count])

    def lowest(self, n: int) -> Optional[float]:
        if self._count < n:
            return None
        return min(b.low_price for b in self._bars[self._count - n:self._count])

    def atr(self, n: int = 14) -> Optional[float]:
        if self._count < n + 1:
            return None
        trs = []
        for i in range(self._count - n, self._count):
            prev_c = self._bars[i - 1].close_price
            trs.append(max(
                self._bars[i].high_price - self._bars[i].low_price,
                abs(self._bars[i].high_price - prev_c),
                abs(self._bars[i].low_price - prev_c),
            ))
        return sum(trs) / n if trs else None


class MultiTFContext:
    """多周期上下文：tf(name, current_dt) 返回该周期截至 t 的已完成 bar 视图。"""

    def __init__(self) -> None:
        self._tf_bars: Dict[str, List[BarData]] = {}
        self._tf_close_times: Dict[str, List] = {}

    def add(self, name: str, bars: List[BarData],
            close_times: Optional[List] = None) -> None:
        """注册一个周期的数据。close_times 缺省时用 bar.datetime（即按开盘时间）。"""
        if not bars:
            return
        self._tf_bars[name] = bars
        self._tf_close_times[name] = (
            close_times if close_times is not None else [b.datetime for b in bars])

    def has(self, name: str) -> bool:
        return name in self._tf_bars

    def tf(self, name: str, current_dt) -> TFView:
        """返回周期 name 截至锚定时刻的视图；未注册周期 → KeyError。"""
        if name not in self._tf_bars:
            raise KeyError(f"周期 {name} 不在可用多周期上下文中（可用：{list(self._tf_bars)}）")
        ct = self._tf_close_times[name]
        n = bisect_right(ct, current_dt)
        return TFView(name, self._tf_bars[name], n, ct)
