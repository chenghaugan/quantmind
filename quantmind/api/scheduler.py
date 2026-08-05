"""任务调度器（APScheduler 封装）：周期任务：数据同步 / 风控日切 / 健康检查。

对应规划「Phase 5 监控 + 长期运行」：让「模拟盘跑 7 天」「数据新鲜度监控」
「风控日切」可以在 api 进程内自动化，无需外部 cron。

设计要点：
  - 轻量封装 APScheduler 的 ``AsyncIOScheduler``，只暴露 register / start / stop / list。
  - 支持 ``cron``（表达式）或 ``interval``（秒）两种触发方式。
  - 内置一组默认任务（``build_default_jobs``），依赖从 ``app.state`` 懒解析，组件不可用时
    记录告警并按 ``required=False`` 降级为 no-op，绝不因单个任务失败拖垮 api。
  - apscheduler 未安装时本模块可正常 import，仅调度功能禁用（返回空任务列表），
    保证旧的 Docker 镜像/未升级依赖时 app 仍能启动。
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

_logger = logging.getLogger("quantmind.scheduler")

try:  # apscheduler 为可选依赖：未安装时不阻塞 app 启动
    from apscheduler.schedulers.asyncio import AsyncIOScheduler  # type: ignore
    from apscheduler.triggers.cron import CronTrigger  # type: ignore
    _APSCHEDULER_AVAILABLE = True
except Exception:  # noqa: BLE001 - 可选依赖缺失，仅禁用调度
    _APSCHEDULER_AVAILABLE = False
    AsyncIOScheduler = None  # type: ignore
    CronTrigger = None  # type: ignore


# ---------------------------------------------------------------------------
# 调度器封装
# ---------------------------------------------------------------------------
class QuantMindScheduler:
    """APScheduler 的轻量封装：注册 / 启停 / 列表 / 移除。"""

    def __init__(self) -> None:
        self._sched = AsyncIOScheduler() if _APSCHEDULER_AVAILABLE else None
        self._jobs: Dict[str, Dict[str, Any]] = {}  # name -> job 元信息

    @property
    def available(self) -> bool:
        """apscheduler 是否可用。"""
        return self._sched is not None

    def register(
        self,
        name: str,
        fn: Callable[..., Any],
        *,
        cron: Optional[str] = None,
        interval: Optional[int] = None,
        max_instances: int = 1,
        misfire_grace_time: int = 60,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """注册一个周期任务。

        :param name: 任务名（唯一标识）。
        :param fn: 任务函数（可为 async）。
        :param cron: cron 表达式，如 ``"30 15 * * 1-5"``；与 ``interval`` 二选一。
        :param interval: 触发间隔（秒）；与 ``cron`` 二选一。
        :param max_instances: 同一任务最大并行实例（默认 1，防重入）。
        :param misfire_grace_time: 错过触发允许补跑的宽限秒数。
        :param kwargs: 传给 ``fn`` 的固定参数。
        :returns: 注册是否成功（False 表示 apscheduler 不可用或参数非法）。
        """
        if self._sched is None:
            _logger.warning("apscheduler 未安装，跳过任务注册: %s", name)
            return False
        if cron is None and interval is None:
            _logger.error("任务 %s 必须提供 cron 或 interval 之一", name)
            return False
        if name in self._jobs:
            _logger.warning("任务 %s 已存在，覆盖注册", name)
            self.remove(name)
        try:
            if cron:
                trigger = CronTrigger.from_crontab(cron)
                self._sched.add_job(
                    fn, trigger,
                    id=name,
                    max_instances=max_instances,
                    misfire_grace_time=misfire_grace_time,
                    kwargs=kwargs or {},
                )
            else:
                self._sched.add_job(
                    fn, "interval",
                    id=name,
                    seconds=interval,
                    max_instances=max_instances,
                    misfire_grace_time=misfire_grace_time,
                    kwargs=kwargs or {},
                )
        except Exception as exc:  # noqa: BLE001
            _logger.exception("注册任务 %s 失败: %s", name, exc)
            return False
        self._jobs[name] = {
            "name": name,
            "cron": cron,
            "interval": interval,
            "next_run": self._next_run(name),
        }
        _logger.info("已注册调度任务: %s (cron=%s, interval=%s)", name, cron, interval)
        return True

    def start(self) -> None:
        if self._sched is None:
            _logger.warning("apscheduler 未安装，调度器未启动")
            return
        if not self._sched.running:
            self._sched.start()
            _logger.info("QuantMind 调度器已启动，任务数 %d", len(self._jobs))

    def stop(self) -> None:
        if self._sched is not None and self._sched.running:
            self._sched.shutdown(wait=False)
            _logger.info("QuantMind 调度器已停止")

    def remove(self, name: str) -> bool:
        if self._sched is None:
            return False
        try:
            self._sched.remove_job(name)
        except Exception:  # noqa: BLE001 - 任务可能不存在
            pass
        return self._jobs.pop(name, None) is not None

    def list_jobs(self) -> List[Dict[str, Any]]:
        """返回已注册任务清单（含下一次触发时间）。"""
        out: List[Dict[str, Any]] = []
        for name, meta in self._jobs.items():
            m = dict(meta)
            m["next_run"] = self._next_run(name)
            out.append(m)
        return out

    def _next_run(self, job_id: str) -> Optional[str]:
        if self._sched is None:
            return None
        try:
            job = self._sched.get_job(job_id)
            if job is not None and job.next_run_time is not None:
                return job.next_run_time.isoformat()
        except Exception:  # noqa: BLE001
            pass
        return None


# ---------------------------------------------------------------------------
# 内置默认任务定义
# ---------------------------------------------------------------------------
def _job_health_check(sys_state: Dict[str, Any]) -> Dict[str, Any]:
    """健康检查：汇总引擎 / 数据 / 事件引擎状态。复用 /health 逻辑。"""
    dm = sys_state.get("dm")
    ee = sys_state.get("ee")
    return {
        "feeds": dm.registry.list_feeds() if dm else [],
        "components": {
            "data_manager": "active" if dm else "inactive",
            "event_engine": ("running" if ee and getattr(ee, "_running", False) else "stopped"),
            "lifecycle": "active" if sys_state.get("lifecycle") else "inactive",
        },
    }


def _job_risk_day_rotation(sys_state: Dict[str, Any]) -> Dict[str, Any]:
    """风控日切：按交易日历重置当日风控计数器（日亏损 / 下单数 / 成交手数）。

    注：RiskEngine 为无状态（每次请求构造独立引擎，见 risk_service），因此这里
    不维护常驻引擎，而是执行一次"日切"观测：记录当前交易日，供未来接入常驻
    RiskEngine 时调用 ``reset_day``。
    """
    try:
        from datetime import datetime, timezone
        from ..risk.calendar import TradingCalendar
        cal = TradingCalendar()
        today = datetime.now(timezone.utc).date()
        return {
            "action": "risk_day_reset",
            "trading_day": today.isoformat(),
            "is_trading_day": cal.is_trading_day(today),
        }
    except Exception as exc:  # noqa: BLE001
        _logger.exception("风控日切任务失败: %s", exc)
        return {"action": "risk_day_reset", "error": str(exc)}


def _job_data_sync(sys_state: Dict[str, Any]) -> Dict[str, Any]:
    """数据同步：对配置的标的列表做增量抓取入库。

    默认不做任何外部抓取（避免误触发真实网络请求），仅输出可抓取对象；
    具体同步标的由 ``sys_state.get("sync_symbols")`` 提供，空列表时跳过。
    """
    dm = sys_state.get("dm")
    symbols = sys_state.get("sync_symbols") or []
    if not dm or not symbols:
        return {"action": "data_sync", "skipped": True, "reason": "未配置同步标的或数据管理器不可用"}
    results: List[Dict[str, Any]] = []
    for item in symbols:
        results.append({
            "symbol": item.get("symbol"),
            "exchange": item.get("exchange", "SHFE"),
            "interval": item.get("interval", "1d"),
        })
    return {"action": "data_sync", "symbols": results, "skipped": len(results) == 0}


def build_default_jobs(sys_state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """构造内置任务注册表（懒解析，依赖从 sys_state 注入，可缺省）。"""
    return [
        {
            "name": "health_check",
            "fn": _job_health_check,
            "interval": 300,          # 每 5 分钟
            "kwargs": {"sys_state": sys_state},
            "required": True,
        },
        {
            "name": "risk_day_rotation",
            "fn": _job_risk_day_rotation,
            "cron": "0 0 * * *",      # 每日 00:00
            "kwargs": {"sys_state": sys_state},
            "required": False,
        },
        {
            "name": "data_sync",
            "fn": _job_data_sync,
            "cron": "30 15 * * 1-5",  # 交易日 15:30（后端仅登记，见函数内跳过逻辑）
            "kwargs": {"sys_state": sys_state},
            "required": False,
        },
    ]


def build_scheduler(sys_state: Dict[str, Any], register_defaults: bool = True) -> QuantMindScheduler:
    """工厂：构建调度器并按需注册内置任务。返回调度器实例。"""
    sched = QuantMindScheduler()
    if register_defaults:
        for spec in build_default_jobs(sys_state):
            ok = sched.register(
                spec["name"],
                spec["fn"],
                cron=spec.get("cron"),
                interval=spec.get("interval"),
                kwargs=spec.get("kwargs", {}),
            )
            if not ok:
                _logger.warning("内置任务 %s 注册失败", spec["name"])
    return sched


__all__ = [
    "QuantMindScheduler",
    "build_scheduler",
    "build_default_jobs",
]
