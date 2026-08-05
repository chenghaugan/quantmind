"""DiskBarCache + DataManager 本地行情仓库（Parquet 写缓存）测试。"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from quantmind.core.constant import Exchange, Interval
from quantmind.core.object import BarData
from quantmind.data import DataManager, InMemoryStore
from quantmind.data.feed.base import BaseDataFeed, HistoryRequest
from quantmind.data.feed.registry import DataFeedRegistry
from quantmind.data.store.disk_cache import DiskBarCache

UTC = timezone.utc


def _bars(symbol: str, exchange: Exchange, n: int, start: datetime) -> list:
    out = []
    for i in range(n):
        out.append(BarData(
            symbol=symbol, exchange=exchange, interval=Interval.DAILY,
            datetime=start + timedelta(days=i),
            open_price=100.0 + i, high_price=101.0 + i, low_price=99.0 + i,
            close_price=100.5 + i, volume=1000.0 + i, open_interest=10.0,
            turnover=1e5 + i,
        ))
    return out


class _FakeRealFeed(BaseDataFeed):
    """模拟真实源：每次返回 n 根历史；记录调用次数。"""
    name = "fake_real"

    def __init__(self, n: int = 30) -> None:
        self.n = n
        self.calls = 0

    async def fetch_bar_data(self, req: HistoryRequest):
        self.calls += 1
        return _bars(req.symbol, req.exchange, self.n, datetime(2023, 1, 1, tzinfo=UTC))


def _make_dm(root, feed):
    reg = DataFeedRegistry()
    reg.register(feed, priority=10)
    store = InMemoryStore()
    dc = DiskBarCache(root)
    dm = DataManager(reg, store, disk_cache=dc)
    return dm


def test_roundtrip_real_source_populates_cache(tmp_path):
    feed = _FakeRealFeed()
    dm = _make_dm(str(tmp_path), feed)

    req = HistoryRequest(symbol="rb0", exchange=Exchange.SHFE, interval=Interval.DAILY)
    sink: dict = {}
    # 首次：经过真实源拉取
    bars1 = asyncio.run(dm.get_bar_data(req, source_sink=sink))
    assert len(bars1) == feed.n
    assert feed.calls == 1
    assert sink["rb0"] == "fake_real"
    # 已落盘
    assert len(list(tmp_path.glob("*.parquet"))) == 1

    # 二次：直接命中磁盘缓存，真实源不再被调用
    sink2: dict = {}
    bars2 = asyncio.run(dm.get_bar_data(req, source_sink=sink2))
    assert len(bars2) == feed.n
    assert feed.calls == 1  # 未再调用
    assert sink2["rb0"] == "disk_cache"


def test_window_filter_from_cache(tmp_path):
    feed = _FakeRealFeed(n=60)
    dm = _make_dm(str(tmp_path), feed)
    req = HistoryRequest(symbol="rb0", exchange=Exchange.SHFE, interval=Interval.DAILY)
    asyncio.run(dm.get_bar_data(req))  # 预热落盘
    feed.calls = 999

    start = datetime(2023, 1, 10, tzinfo=UTC)
    end = datetime(2023, 1, 15, tzinfo=UTC)
    req2 = HistoryRequest(symbol="rb0", exchange=Exchange.SHFE, interval=Interval.DAILY,
                          start=start, end=end)
    bars = asyncio.run(dm.get_bar_data(req2))
    # 1月10~15 共 6 天
    assert len(bars) == 6
    assert bars[0].datetime == start
    assert bars[-1].datetime == end
    assert feed.calls == 999  # 未调用真实源


def test_mock_not_cached(tmp_path):
    """mock 合成数据不回写仓库，避免污染。"""

    class _MockFeed(BaseDataFeed):
        name = "mock"

        async def fetch_bar_data(self, req: HistoryRequest):
            return _bars(req.symbol, req.exchange, 10, datetime(2023, 1, 1, tzinfo=UTC))

    reg = DataFeedRegistry()
    reg.register(_MockFeed(), priority=100)
    store = InMemoryStore()
    dc = DiskBarCache(str(tmp_path))
    dm = DataManager(reg, store, disk_cache=dc)

    req = HistoryRequest(symbol="x", exchange=Exchange.SHFE, interval=Interval.DAILY)
    sink: dict = {}
    bars = asyncio.run(dm.get_bar_data(req, source_sink=sink))
    assert len(bars) == 10
    assert sink["x"] == "mock"
    # mock 不落盘
    assert len(list(tmp_path.glob("*.parquet"))) == 0


def test_refresh_skips_disk_cache_but_store_may_serve(tmp_path):
    """refresh=True 跳过磁盘缓存；若持久库已有则从库返回（仍秒级，不联网）。"""
    feed = _FakeRealFeed(n=30)
    dm = _make_dm(str(tmp_path), feed)
    req = HistoryRequest(symbol="rb0", exchange=Exchange.SHFE, interval=Interval.DAILY)
    asyncio.run(dm.get_bar_data(req))  # 落盘 + 写库
    assert feed.calls == 1
    # 默认（无 refresh 的缓存命中）-> 磁盘缓存
    sink = {}
    asyncio.run(dm.get_bar_data(req, source_sink=sink))
    assert sink["rb0"] == "disk_cache"
    assert feed.calls == 1


def test_save_merges_across_fetches(tmp_path):
    """同一标的两次真实拉取（用全新空库，仅依赖磁盘保存）合并补全，无重复。"""
    feed = _FakeRealFeed(n=30)
    req = HistoryRequest(symbol="rb0", exchange=Exchange.SHFE, interval=Interval.DAILY)
    # 第一次：全新空库，走真实源 -> 落盘
    dm1 = _make_dm(str(tmp_path), feed)
    asyncio.run(dm1.get_bar_data(req))
    assert feed.calls == 1
    # 第二次：另一个全新空库（模拟重启后库为空），但磁盘缓存仍在 -> 命中缓存
    dm2 = _make_dm(str(tmp_path), feed)
    asyncio.run(dm2.get_bar_data(req))
    assert feed.calls == 1  # 未再调用真实源
    # 缓存文件无重复（30 行）
    df = pytest.importorskip("pandas").read_parquet(list(tmp_path.glob("*.parquet"))[0])
    assert len(df) == 30


def test_dc_stats(tmp_path):
    feed = _FakeRealFeed()
    dm = _make_dm(str(tmp_path), feed)
    req = HistoryRequest(symbol="rb0", exchange=Exchange.SHFE, interval=Interval.DAILY)
    asyncio.run(dm.get_bar_data(req))
    stats = dm.disk_cache.stats()
    assert stats["files"] == 1
    assert stats["rows"] == feed.n
    assert stats["last_datetime"] is not None


def test_no_cache_preserves_old_behavior():
    """未配置 disk_cache 时行为不变（兼容旧调用方）。"""
    reg = DataFeedRegistry()
    reg.register(_FakeRealFeed(), priority=10)
    dm = DataManager(reg, InMemoryStore())  # 无 disk_cache

    req = HistoryRequest(symbol="rb0", exchange=Exchange.SHFE, interval=Interval.DAILY)
    sink: dict = {}
    bars = asyncio.run(dm.get_bar_data(req, source_sink=sink))
    assert len(bars) == 30
    assert sink["rb0"] == "fake_real"
