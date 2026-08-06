"""任务调度器测试（APScheduler 封装 + 内置任务 + /scheduler API）。

注意：这些测试依赖 ``apscheduler``（已加入 pyproject 依赖）。若 apscheduler
在运行时缺失，``scheduler.available`` 应为 False 且调度功能优雅降级——第一条用例
据此断言，其余用例在 apscheduler 缺失时跳过。
"""
from __future__ import annotations

import asyncio

import pytest

from quantmind.api.scheduler import (
    QuantMindScheduler,
    build_scheduler,
    build_default_jobs,
)


def _aps_available() -> bool:
    return getattr(__import__("quantmind.api.scheduler", fromlist=["_APSCHEDULER_AVAILABLE"]),
                   "_APSCHEDULER_AVAILABLE", False)


def test_available_flag():
    """apscheduler 可用性标志：可用时调度器可 start/list；缺失时为 False 不抛错。"""
    sched = QuantMindScheduler()
    assert sched.available is _aps_available()
    # 无论如何实例化都不抛错
    assert sched.list_jobs() == []


@pytest.mark.skipif(not _aps_available(), reason="apscheduler 未安装")
@pytest.mark.asyncio
async def test_interval_task_triggers():
    """interval 任务应周期性触发。"""
    sched = QuantMindScheduler()
    calls = []

    async def _tick(**kw):
        calls.append(kw)

    ok = sched.register("tick", _tick, interval=0.05, kwargs={"v": 1})
    assert ok
    sched.start()
    await asyncio.sleep(0.2)
    sched.stop()
    assert len(calls) >= 1
    assert calls[0]["v"] == 1


@pytest.mark.skipif(not _aps_available(), reason="apscheduler 未安装")
def test_cron_and_missing_both():
    """cron 注册成功；cron/interval 都缺时应拒绝。"""
    sched = QuantMindScheduler()

    def _noop(**kw):
        pass

    assert sched.register("c1", _noop, cron="0 12 * * 1-5")
    assert not sched.register("c2", _noop)  # 两者都缺
    assert "c1" in [j["name"] for j in sched.list_jobs()]
    assert "c2" not in [j["name"] for j in sched.list_jobs()]


@pytest.mark.skipif(not _aps_available(), reason="apscheduler 未安装")
def test_duplicate_name_overrides():
    """同名任务重复注册应覆盖（旧任务被移除，不留双副本）。"""
    sched = QuantMindScheduler()

    def _noop(**kw):
        pass

    assert sched.register("dup", _noop, interval=60)
    assert sched.register("dup", _noop, interval=120)
    names = [j["name"] for j in sched.list_jobs()]
    assert names.count("dup") == 1


@pytest.mark.skipif(not _aps_available(), reason="apscheduler 未安装")
def test_build_default_jobs_four():
    """内置任务表应含：health_check / risk_day_rotation / data_sync / cache_refresh。"""
    specs = build_default_jobs({"dm": None, "ee": None})
    names = {s["name"] for s in specs}
    assert {"health_check", "risk_day_rotation", "data_sync", "cache_refresh"} <= names


@pytest.mark.skipif(not _aps_available(), reason="apscheduler 未安装")
def test_build_scheduler_registers_defaults():
    """build_scheduler 注册内置任务，且列表含健康检查。"""
    sys_state = {"dm": None, "ee": None, "lifecycle": None}
    sched = build_scheduler(sys_state, register_defaults=True)
    try:
        names = {j["name"] for j in sched.list_jobs()}
        assert "health_check" in names
        assert "data_sync" in names
    finally:
        sched.stop()


@pytest.mark.skipif(not _aps_available(), reason="apscheduler 未安装")
def test_job_health_check_skips_when_no_dm():
    """健康检查任务在无数据管理器时应输出 inactive 而不抛错。"""
    from quantmind.api.scheduler import _job_health_check
    res = _job_health_check({"dm": None, "ee": None, "lifecycle": None})
    assert res["components"]["data_manager"] == "inactive"


@pytest.mark.skipif(not _aps_available(), reason="apscheduler 未安装")
def test_api_scheduler_endpoint():
    """/scheduler REST：返回 available 与 jobs 列表。"""
    from fastapi.testclient import TestClient
    from quantmind.api.app import app

    with TestClient(app) as c:
        r = c.get("/scheduler")
        assert r.status_code == 200
        body = r.json()
        assert "available" in body
        assert "jobs" in body


@pytest.mark.parametrize("skipped_reason", ["no_dm", "empty_cache"])
def test_cache_refresh_skips_gracefully(tmp_path, skipped_reason):
    """无仓库 / 空仓库时 cache_refresh 应优雅跳过，不抛错。"""
    import pytest as _pt
    from quantmind.api.scheduler import _job_cache_refresh

    if skipped_reason == "no_dm":
        res = asyncio.run(_job_cache_refresh({"dm": None}))
        assert res["skipped"] is True
        assert "本地行情仓库未启用" in res["reason"]
    else:
        from quantmind.data import DataManager, InMemoryStore, DiskBarCache
        from quantmind.data.feed.registry import DataFeedRegistry
        dc = DiskBarCache(str(tmp_path))
        dm = DataManager(DataFeedRegistry(), InMemoryStore(), disk_cache=dc)
        res = asyncio.run(_job_cache_refresh({"dm": dm}))
        assert res["skipped"] is True
        assert "仓库为空" in res["reason"]


def test_cache_refresh_walks_and_refreshes(tmp_path):
    """cache_refresh 应在 refresh 模式下把每个缓存标的走真实源重拉并回写。"""
    from datetime import datetime, timezone
    from quantmind.api.scheduler import _job_cache_refresh
    from quantmind.core.constant import Exchange, Interval
    from quantmind.core.object import BarData
    from quantmind.data import DataManager, InMemoryStore, DiskBarCache
    from quantmind.data.feed.base import BaseDataFeed, HistoryRequest
    from quantmind.data.feed.registry import DataFeedRegistry

    class _FakeRealFeed(BaseDataFeed):
        name = "fake_real"

        async def fetch_bar_data(self, req: HistoryRequest):
            return [BarData(
                symbol=req.symbol, exchange=req.exchange, interval=req.interval,
                datetime=datetime(2023, 1, 1, tzinfo=timezone.utc),
                open_price=100.0, high_price=101.0, low_price=99.0,
                close_price=100.5, volume=1000.0,
            )]

    reg = DataFeedRegistry()
    feed = _FakeRealFeed()
    reg.register(feed, priority=10)
    dc = DiskBarCache(str(tmp_path))
    dm = DataManager(reg, InMemoryStore(), disk_cache=dc)

    # 先灌一个缓存键（rb0）
    req = HistoryRequest(symbol="rb0", exchange=Exchange.SHFE, interval=Interval.DAILY)
    asyncio.run(dm.get_bar_data(req))
    assert dc.list_keys() == [{"symbol": "rb0", "exchange": "SHFE", "interval": "1d"}]

    # 第二次调用默认命中磁盘缓存，不再走真实源
    asyncio.run(dm.get_bar_data(req))

    # cache_refresh 应强制 refresh 重走真实源
    res = asyncio.run(_job_cache_refresh({"dm": dm}))
    assert res["refreshed"] >= 1
    assert res["failed"] == 0
    assert res["results"][0]["key"]["symbol"] == "rb0"
