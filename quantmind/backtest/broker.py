"""模拟经纪：撮合价格与手续费计算。

回测/模拟撮合采用「下一根 K 线开盘价 + 滑点」成交，避免前视偏差。
"""
from __future__ import annotations

from typing import List

from ..core.constant import Direction
from ..core.object import BarData


def fill_price(bar: BarData, direction: Direction, slippage: float = 0.0) -> float:
    """下一根 K 线的开盘价加减滑点。"""
    if direction == Direction.LONG:
        return bar.open_price + slippage
    return bar.open_price - slippage


def commission(amount: float, rate: float) -> float:
    return abs(amount) * rate
