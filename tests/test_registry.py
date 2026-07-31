"""Fallback Chain（Loader Registry）测试：单源失败自动降级。"""
from datetime import datetime

from quantmind.data.feed.base import BaseDataFeed, HistoryRequest
from quantmind.data.feed.registry import DataFeedRegistry, DataUnavailable
from quantmind.core import BarData, Exchange, Interval


class FailingFeed(BaseDataFeed):
    name = "fail"

    async def fetch_bar_data(self, req: HistoryRequest):
        raise RuntimeError("network down")


class OkFeed(BaseDataFeed):
    name = "ok"

    async def fetch_bar_data(self, req: HistoryRequest):
        return [
            BarData(
                symbol=req.symbol,
                exchange=req.exchange,
                datetime=datetime(2024, 1, 1),
                interval=req.interval,
                open_price=1,
                high_price=1,
                low_price=1,
                close_price=1,
            )
        ]


async def test_fallback_to_next_source():
    reg = DataFeedRegistry()
    reg.register(FailingFeed(), priority=10)
    reg.register(OkFeed(), priority=20)
    bars = await reg.get_bar_data(HistoryRequest(symbol="x", exchange=Exchange.SHFE))
    assert len(bars) == 1


async def test_all_fail_raises():
    reg = DataFeedRegistry()
    reg.register(FailingFeed(), priority=10)
    try:
        await reg.get_bar_data(HistoryRequest(symbol="x", exchange=Exchange.SHFE))
        assert False, "should raise DataUnavailable"
    except DataUnavailable:
        pass


async def test_default_registry_has_mock_fallback():
    from quantmind.data import build_default_registry

    reg = build_default_registry()
    # mock 为最后一级（priority 100），保证离线可跑
    assert "mock" in reg.list_feeds()
    # 离线环境下，真实源失败后会落到 mock
    bars = await reg.get_bar_data(
        HistoryRequest(symbol="rb0", exchange=Exchange.SHFE, interval=Interval.DAILY)
    )
    assert len(bars) >= 1
