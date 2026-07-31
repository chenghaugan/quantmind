"""离线兜底数据源：合成随机游走 K 线。

注册为 Fallback Chain 的最后一级——当所有真实源不可用（无网络/未装库）时，
仍能产出可用的合成数据，使研究→回测→Web 全链路在离线环境也能跑通演示。
联网且装库时，真实源（AKShare/mootdx/Yahoo）优先级更高，会先命中。
"""
from __future__ import annotations

import hashlib
import random
from datetime import datetime, timedelta
from typing import List

from .base import BaseDataFeed, HistoryRequest
from ...core.constant import Interval
from ...core.object import BarData


def _stable_seed(symbol: str, exchange: str) -> int:
    """稳定的整数种子（不依赖 Python 进程盐化的 hash()，保证跨进程可复现）。"""
    h = hashlib.md5(f"{symbol}.{exchange}".encode("utf-8")).hexdigest()
    return int(h, 16) % (2 ** 31)


class MockFeed(BaseDataFeed):
    name = "mock"

    async def fetch_bar_data(self, req: HistoryRequest) -> List[BarData]:
        end = req.end or datetime.now()
        start = req.start or (end - timedelta(days=30))
        days = max(1, (end - start).days)
        bars: List[BarData] = []
        seed = _stable_seed(req.symbol, req.exchange.value)
        price = 100.0 + seed % 50
        random.seed(seed)
        for i in range(days):
            dt = start + timedelta(days=i)
            drift = random.uniform(-2, 2)
            price = max(1.0, price + drift)
            bar = self._make_bar(
                symbol=req.symbol,
                exchange=req.exchange,
                dt=dt,
                interval=req.interval,
                o=price - 0.5,
                h=price + 1.0,
                l=price - 1.0,
                c=price,
                v=random.uniform(1000, 10000),
                oi=random.uniform(0, 5000),
            )
            bars.append(bar)
        return bars
