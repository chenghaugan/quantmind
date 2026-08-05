"""方案 B（调度器）真实端到端验收。

与 ``test_scheduler.py``（单元级）不同，本文件验证三条此前未覆盖的闭环：

  1. **内置任务真实调度产出**：注册内置 ``_job_health_check`` 为极小 interval 任务，
     配合 APScheduler 的 ``add_listener`` 捕获 ``EVENT_JOB_EXECUTED``，拿到任务
     *返回值*——证明「调度触发 → 内置任务执行 → 结果可观测」，而非仅注册成功。
  2. **真实 lifespan 下的运行态**：TestClient 走真实 ``lifespan``（app.state.scheduler
     被挂载并启动），断言 /scheduler 返回 ``available=true``、``running=true``、
     jobs 含 health_check 且 next_run 非空。
  3. **data_sync / risk_day_rotation 在内置状态下产出正确结构**：验证内置任务的
     降级跳过与交易日计算。

依赖 apscheduler；缺失时整体跳过（与 test_scheduler.py 策略一致）。
"""
from __future__ import annotations

import asyncio
import datetime

import pytest

from quantmind.api.scheduler import QuantMindScheduler


def _aps_available() -> bool:
    return getattr(
        __import__("quantmind.api.scheduler", fromlist=["_APSCHEDULER_AVAILABLE"]),
        "_APSCHEDULER_AVAILABLE",
        False,
    )


class _FakeDM:
    """最小 DataManager 桩，满足 _job_health_check 的 registry.list_feeds()。"""

    class _registry:
        @staticmethod
        def list_feeds():
            return ["rb0.SHFE"]

    registry = _registry()


@pytest.mark.skipif(not _aps_available(), reason="apscheduler 未安装")
@pytest.mark.asyncio
async def test_e2e_builtin_health_check_real_execution():
    """闭环 1：内置 health_check 任务被真实调度执行，执行轨迹经 Listener 可观测。

    注册内置 ``_job_health_check`` 为极小 interval 任务，挂 APScheduler Listener。
    调度器真实运行时必然捕获到 >=1 次 JOB_EXECUTED，且 ``evt.retval`` 的结构
    与内置实现一致（含注入 FakeDM 的 feeds 与 components 状态）。
    """
    from apscheduler.events import EVENT_JOB_EXECUTED
    from quantmind.api.scheduler import _job_health_check

    sys_state = {"dm": _FakeDM(), "ee": None, "lifecycle": None}
    sched = QuantMindScheduler()

    executed: list = []

    def _on_event(evt):
        executed.append(evt.retval)

    sched._sched.add_listener(_on_event, EVENT_JOB_EXECUTED)

    # 用同步包装器固定 sys_state，注册为高频 interval 触发任务
    async def wrapper(**kw):
        return await asyncio.to_thread(_job_health_check, kw["sys_state"])

    assert sched.register(
        "health_check", wrapper, interval=0.05, kwargs={"sys_state": sys_state}
    )
    sched.start()
    try:
        await asyncio.sleep(0.25)
    finally:
        sched.stop()

    assert executed, "调度器运行后应至少捕获一次 health_check 执行事件"
    last = executed[-1]
    assert last["feeds"] == ["rb0.SHFE"]  # 使用了注入的 FakeDM.registry
    assert last["components"]["data_manager"] == "active"
    assert last["components"]["event_engine"] == "stopped"


@pytest.mark.skipif(not _aps_available(), reason="apscheduler 未安装")
def test_e2e_api_returns_running_scheduler_with_realtime_next_run():
    """闭环 2：真实 lifespan 下，调度器启动且 /scheduler 反映运行态与 next_run。

    现有 test_api_scheduler_endpoint 只断言字段存在；这里补强为运行态 + next_run
    时间戳真实非空，证明 app.state.scheduler 确实被启动并调度着内置任务。
    """
    from fastapi.testclient import TestClient
    from quantmind.api.app import app

    with TestClient(app) as c:
        r = c.get("/scheduler")
        assert r.status_code == 200
        body = r.json()
        assert body["available"] is True
        assert body["running"] is True
        job_names = {j["name"] for j in body["jobs"]}
        assert {"health_check", "data_sync", "risk_day_rotation"} <= job_names
        for j in body["jobs"]:
            if j["name"] == "health_check":
                assert j.get("next_run"), "health_check 应有下一次触发时间"


@pytest.mark.skipif(not _aps_available(), reason="apscheduler 未安装")
def test_e2e_builtin_misc_jobs_output_structure():
    """闭环 3：data_sync 无标的时跳过、risk_day_rotation 产出交易日结构。"""
    from quantmind.api.scheduler import _job_data_sync, _job_risk_day_rotation

    sys_state = {"dm": None, "sync_symbols": []}
    sync = _job_data_sync(sys_state)
    assert sync["action"] == "data_sync"
    assert sync["skipped"] is True
    assert "未配置" in sync["reason"]

    rot = _job_risk_day_rotation({"dm": None})
    assert rot["action"] == "risk_day_reset"
    assert "trading_day" in rot
    # trading_day 为有效 ISO 日期
    datetime.date.fromisoformat(rot["trading_day"])
    assert "is_trading_day" in rot


@pytest.mark.skipif(not _aps_available(), reason="apscheduler 未安装")
def test_e2e_data_sync_with_symbols_lists_targets():
    """闭环 3b：data_sync 配置标的时列出抓取对象（不发起真实网络请求）。"""
    from quantmind.api.scheduler import _job_data_sync

    sys_state = {
        "dm": _FakeDM(),
        "sync_symbols": [{"symbol": "rb2010", "exchange": "SHFE", "interval": "1d"}],
    }
    res = _job_data_sync(sys_state)
    assert res["skipped"] is False
    assert res["symbols"] == [
        {"symbol": "rb2010", "exchange": "SHFE", "interval": "1d"}
    ]
