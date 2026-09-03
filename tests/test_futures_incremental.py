"""期货智能增量拉取测试：合约映射、按需根数计算、增量选源策略。"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# TqSdk 合约映射（含商品期货主连泛化）
def test_tqsdk_symbol_stock_index():
    from quantmind.data.feed.tqsdk_feed import _get_tqsdk_symbol
    from quantmind.core.constant import Exchange
    assert _get_tqsdk_symbol("IF0", Exchange.CFFEX) == "KQ.m@CFFEX.IF"
    assert _get_tqsdk_symbol("IM0", Exchange.CFFEX) == "KQ.m@CFFEX.IM"


def test_tqsdk_symbol_commodity_main_continuous():
    from quantmind.data.feed.tqsdk_feed import _get_tqsdk_symbol
    from quantmind.core.constant import Exchange
    assert _get_tqsdk_symbol("rb0", Exchange.SHFE) == "KQ.m@SHFE.rb"
    assert _get_tqsdk_symbol("hc0", Exchange.SHFE) == "KQ.m@SHFE.hc"
    assert _get_tqsdk_symbol("cu0", Exchange.SHFE) == "KQ.m@SHFE.cu"
    assert _get_tqsdk_symbol("TA0", Exchange.CZCE) == "KQ.m@CZCE.TA"
    assert _get_tqsdk_symbol("sc0", Exchange.INE) == "KQ.m@INE.sc"


def test_tqsdk_symbol_dce_lowercase():
    """大商所产品代码在 TqSdk 中为小写：PP0 → KQ.m@DCE.pp。"""
    from quantmind.data.feed.tqsdk_feed import _get_tqsdk_symbol
    from quantmind.core.constant import Exchange
    assert _get_tqsdk_symbol("PP0", Exchange.DCE) == "KQ.m@DCE.pp"
    assert _get_tqsdk_symbol("L0", Exchange.DCE) == "KQ.m@DCE.l"
    assert _get_tqsdk_symbol("m0", Exchange.DCE) == "KQ.m@DCE.m"


def test_tqsdk_symbol_specific_contract():
    from quantmind.data.feed.tqsdk_feed import _get_tqsdk_symbol
    from quantmind.core.constant import Exchange
    assert _get_tqsdk_symbol("IF2609", Exchange.CFFEX) == "CFFEX.IF2609"


# ---------------------------------------------------------------------------
# TqSdk 按需根数计算
def test_compute_data_length_first_time():
    """无 start（首次）→ 拉满 8000。"""
    from quantmind.data.feed.tqsdk_feed import TqSdkFeed
    assert TqSdkFeed._compute_data_length(None, 60) == 8000


def test_compute_data_length_small_gap():
    """缺口 1 小时（1m 周期）→ ~70 根，不拉 8000。"""
    from quantmind.data.feed.tqsdk_feed import TqSdkFeed
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=1)
    n = TqSdkFeed._compute_data_length(start, 60)
    assert 60 <= n <= 100


def test_compute_data_length_huge_gap_capped():
    """缺口极大 → 封顶 8000。"""
    from quantmind.data.feed.tqsdk_feed import TqSdkFeed
    start = datetime.now(timezone.utc) - timedelta(days=365)
    assert TqSdkFeed._compute_data_length(start, 60) == 8000


# ---------------------------------------------------------------------------
# 增量选源策略：缺口小 → akshare；缺口大 → TqSdk(start=latest)
def _make_bars(latest, count, step_seconds, symbol="rb0", exchange="SHFE"):
    from quantmind.core.object import BarData
    from quantmind.core.constant import Exchange as Ex, Interval
    return [
        BarData(
            symbol=symbol, exchange=Ex(exchange),
            datetime=latest - timedelta(seconds=step_seconds * i),
            interval=Interval.MINUTE,
            open_price=1, high_price=1, low_price=1, close_price=1,
            volume=0, open_interest=0, turnover=0,
        )
        for i, step_seconds in enumerate([0] + [step_seconds] * (count - 1))
    ]


def test_fetch_by_strategy_small_gap_uses_akshare():
    """缺口小于新浪窗口 → 走 akshare（不发 TqSdk 请求）。"""
    from quantmind.api import futures_download as fd
    from quantmind.data.feed.base import HistoryRequest
    from quantmind.core.constant import Exchange, Interval

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    latest = now - timedelta(minutes=30)  # 缺口 30 分钟 → 小
    cached = _make_bars(latest, 10, 60)

    calls = {"ak": 0, "tq": 0}

    class FakeAK:
        async def fetch_bar_data(self, req):
            calls["ak"] += 1
            return _make_bars(now, 5, 60)

    class FakeTQ:
        async def fetch_bar_data(self, req):
            calls["tq"] += 1
            return []

    with patch("quantmind.data.feed.akshare_future.AkShareFuturesFeed.fetch_bar_data",
               FakeAK.fetch_bar_data), \
         patch.object(fd, "_get_reusable_tqsdk", return_value=FakeTQ()):
        req = HistoryRequest(symbol="rb0", exchange=Exchange.SHFE, interval=Interval.MINUTE)
        bars, source = asyncio.run(fd._fetch_by_strategy(req, cached))

    assert source == "akshare_future"
    assert calls["ak"] == 1 and calls["tq"] == 0  # 未动用 TqSdk


def test_fetch_by_strategy_big_gap_uses_tqsdk_with_start():
    """缺口超过新浪窗口 → 走 TqSdk 且 start=本地最新。"""
    from quantmind.api import futures_download as fd
    from quantmind.data.feed.base import HistoryRequest
    from quantmind.core.constant import Exchange, Interval

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    latest = now - timedelta(days=10)  # 缺口 10 天 ≈ 14400 根 1m → 超窗口
    cached = _make_bars(latest, 10, 60)

    captured = {}

    class FakeAK:
        async def fetch_bar_data(self, req):
            return []

    class FakeTQ:
        async def fetch_bar_data(self, req):
            captured["start"] = req.start
            return _make_bars(now, 5, 60)

    with patch("quantmind.data.feed.akshare_future.AkShareFuturesFeed.fetch_bar_data",
               FakeAK.fetch_bar_data), \
         patch.object(fd, "_get_reusable_tqsdk", return_value=FakeTQ()):
        req = HistoryRequest(symbol="rb0", exchange=Exchange.SHFE, interval=Interval.MINUTE)
        bars, source = asyncio.run(fd._fetch_by_strategy(req, cached))

    assert source == "tqsdk"
    passed = captured["start"]
    if passed.tzinfo is None:
        passed = passed.replace(tzinfo=timezone.utc)
    assert passed == latest.replace(tzinfo=timezone.utc)  # 从本地最新起按需拉
