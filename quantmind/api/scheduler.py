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

import asyncio
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
        timezone: Optional[str] = None,
        max_instances: int = 1,
        misfire_grace_time: int = 60,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """注册一个周期任务。

        :param name: 任务名（唯一标识）。
        :param fn: 任务函数（可为 async）。
        :param cron: cron 表达式，如 ``"30 15 * * 1-5"``；与 ``interval`` 二选一。
        :param interval: 触发间隔（秒）；与 ``cron`` 二选一。
        :param timezone: IANA 时区名（如 ``"Asia/Shanghai"``），用于 cron 触发时间。
                         未指定时使用系统本地时区。
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
                # 解析时区
                tz = None
                if timezone:
                    try:
                        from zoneinfo import ZoneInfo
                        tz = ZoneInfo(timezone)
                    except Exception as exc:  # noqa: BLE001
                        _logger.warning("时区 %s 无效，使用系统本地时区: %s", timezone, exc)
                
                trigger = CronTrigger.from_crontab(cron, timezone=tz)
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
            "timezone": timezone,
            "next_run": self._next_run(name),
        }
        _logger.info("已注册调度任务: %s (cron=%s, interval=%s, tz=%s)", name, cron, interval, timezone)
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


async def _job_cache_refresh(sys_state: Dict[str, Any]) -> Dict[str, Any]:
    """本地行情仓库刷新：把已缓存标的从真实源重新拉取并回写，自动追新。

    识别仓库里现有的 symbol.exchange.interval 键，逐个用 refresh 模式重拉
    （绕过磁盘缓存 → 真实源 → 回写）。无缓存或未启用时跳过。
    """
    dm = sys_state.get("dm")
    dc = getattr(dm, "disk_cache", None) if dm else None
    if dm is None or dc is None:
        return {"action": "cache_refresh", "skipped": True, "reason": "本地行情仓库未启用"}
    try:
        from ..data.feed.base import HistoryRequest
        from ..core.constant import Exchange, Interval
    except Exception as exc:  # noqa: BLE001
        _logger.exception("cache_refresh 导入失败: %s", exc)
        return {"action": "cache_refresh", "error": str(exc)}

    keys = dc.list_keys()
    if not keys:
        return {"action": "cache_refresh", "skipped": True, "reason": "仓库为空"}

    # 临时开启 refresh，使 dm.get_bar_data 跳过磁盘缓存、强制走真实源并回写
    was_refresh = bool(getattr(dc, "refresh", False))
    dc.refresh = True
    try:
        results: List[Dict[str, Any]] = []
        refreshed, failed = 0, 0
        for k in keys:
            try:
                exch = Exchange(k["exchange"].upper())
                interv = Interval(k["interval"])
            except Exception:  # noqa: BLE001
                failed += 1
                results.append({"key": k, "status": "bad_key"})
                continue
            req = HistoryRequest(symbol=k["symbol"], exchange=exch, interval=interv)
            try:
                sink: Dict[str, Any] = {}
                bars = await dm.get_bar_data(req, source_sink=sink)
                ok = bool(bars)
                results.append({
                    "key": k, "status": "ok" if ok else "empty",
                    "source": sink.get(k["symbol"], ""),
                    "n": len(bars),
                })
                _record_refresh(dc, k, exch, interv, bars, ok,
                                sink.get(k["symbol"], ""))
                refreshed += 1 if ok else 0
                failed += 0 if ok else 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                results.append({"key": k, "status": "error", "error": str(exc)[:120]})
                _record_refresh_error(dc, k, exch, interv, str(exc)[:120])
        return {
            "action": "cache_refresh", "refreshed": refreshed, "failed": failed,
            "results": results,
        }
    finally:
        dc.refresh = was_refresh


def _record_refresh(dc, key, exch, interv, bars, ok, source):
    """把一次成功/空的刷新写入仓库刷新日志（与手动 warm/refresh 一致）。"""
    try:
        latest = bars[-1].datetime.isoformat() if bars else None
        dc.record_refresh(
            symbol=key["symbol"], exchange=str(exch.value), interval=interv.value,
            rows=len(bars), latest=latest,
            status="ok" if ok else "empty", detail=(source or "empty"),
        )
    except Exception:  # noqa: BLE001
        pass


def _record_refresh_error(dc, key, exch, interv, detail):
    try:
        dc.record_refresh(
            symbol=key["symbol"], exchange=str(exch.value), interval=interv.value,
            rows=0, latest=None, status="error", detail=detail,
        )
    except Exception:  # noqa: BLE001
        pass


# _MARKET_EXCHANGES: 全市场预热只关心这些交易所（A股沪深 + 港股）
_MARKET_EXCHANGES = {"SSE", "SZSE", "HKEX"}


async def _warm_single(dm, dc, vt: str) -> bool:
    """对单个 vt-symbol 拉日线并落盘；成功返回 True。

    A股/港股走**快速直连源**（腾讯日线 / 新浪港股），绕开通用多源回退链——
    那条链里有 akshare_future/efinance 等对 A股会**无超时挂死**的期货源，
    曾导致预热任务卡住数小时、web 请求 http 超时。统一加 45s 兜底超时防挂死。
    """
    from ..data.feed.base import HistoryRequest
    from ..core.constant import Exchange, Interval

    symbol, exch = vt.rsplit(".", 1)
    try:
        exchange = Exchange(exch)
        if exchange in (Exchange.SSE, Exchange.SZSE):
            # A股：腾讯日线（与「更新A股数据」同源，~1s）
            from .stock_download import _fetch_stock_bars
            bars = await asyncio.wait_for(
                _fetch_stock_bars(symbol, exchange, "1d"), timeout=45)
            source = "tencent"
        elif exchange == Exchange.HKEX:
            # 港股：新浪日线（akshare stock_hk_daily，GET 快）
            from ..data.feed.em_hk import EmHkFeed
            bars = await asyncio.wait_for(
                EmHkFeed().fetch_bar_data(
                    HistoryRequest(symbol=symbol, exchange=exchange, interval=Interval.DAILY)),
                timeout=45)
            source = "em_hk"
        else:
            # 期货：沿用通用回退链，但加超时防挂死
            sink: Dict[str, Any] = {}
            bars = await asyncio.wait_for(
                dm.get_bar_data(HistoryRequest(symbol=symbol, exchange=exchange,
                                               interval=Interval.DAILY), source_sink=sink),
                timeout=45)
            source = sink.get(symbol, "")

        ok = bool(bars)
        if bars:  # 落盘（幂等合并）；parquet 读写较重，放线程池避免阻塞事件循环
            await asyncio.to_thread(dc.save, bars)
        _record = await asyncio.to_thread(
            _record_refresh,
            dc, {"symbol": symbol, "exchange": exch, "interval": "1d"},
            exchange, Interval.DAILY, bars, ok, source)
        return ok
    except Exception as exc:  # noqa: BLE001
        await asyncio.to_thread(
            _record_refresh_error,
            dc, {"symbol": vt.split(".")[0], "exchange": vt.split(".")[-1],
                 "interval": "1d"},
            Exchange(vt.split(".")[-1]), Interval.DAILY, str(exc)[:120])
        return False


def _resolve_warm_markets(settings, markets) -> List[str]:
    """把市场选择解析为规范列表（'A'/'HK'）；缺省取配置 market_warm_markets。"""
    from ..data.feed.market_universe import MARKET_EXCHANGES
    if markets:
        return [m for m in markets if m in MARKET_EXCHANGES]
    cfg = getattr(settings, "market_warm_markets", ["A", "HK"]) or ["A", "HK"]
    return [m for m in cfg if m in MARKET_EXCHANGES]


async def _job_market_warm(
    sys_state: Dict[str, Any],
    markets: Optional[List[str]] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    full: bool = False,
) -> Dict[str, Any]:
    """全市场（A股+港股）自动预热：把未缓存的标的分批拉入本地行情仓库。

    A 股与港股**分开处理**：按市场（``markets``，'A'/'HK'）各自发现清单、各自取差集、
    各自按 batch 取前 N 个逐个 ``dm.get_bar_data`` 拉到真实源并落盘；已在缓存的不重复拉（增量），
    跑完本趟即返回，剩余留给下一趟（自推进 catch-up，无持久队列，KISS）。
    返回结果含聚合统计与 ``by_market`` 分市场明细，便于前端分别展示。

    默认关闭（``QM_MARKET_WARM_ENABLED`` 为 False 时跳过），保证开箱离线/测试不受影响。
    """
    from ..config import get_settings
    from ..data.feed.market_universe import discover_market, MARKET_EXCHANGES

    dm = sys_state.get("dm")
    dc = getattr(dm, "disk_cache", None) if dm is not None else None
    if dm is None or dc is None:
        return {"action": "market_warm", "skipped": True, "reason": "数据管理器或行情仓库未启用"}

    try:
        settings = get_settings()
        if not settings.market_warm_enabled:
            return {"action": "market_warm", "skipped": True, "reason": "未启用(settings.market_warm_enabled=False)"}
        batch = int(settings.market_warm_batch or 50)
        cap = int(settings.market_warm_max_symbols or 5000)
    except Exception as exc:  # noqa: BLE001
        _logger.exception("market_warm 读取配置失败: %s", exc)
        return {"action": "market_warm", "skipped": True, "reason": f"配置读取失败: {exc}"}

    selected = _resolve_warm_markets(settings, markets)
    if not selected:
        return {"action": "market_warm", "skipped": True,
                "reason": "未选择任何市场(markets 应为 'A'/'HK')"}

    # 已缓存键（所有市场交易所），用于各市场取差集
    try:
        cached_keys = {
            k["symbol"] + "." + k["exchange"].upper()
            for k in dc.list_keys() if k.get("exchange", "").upper() in _MARKET_EXCHANGES
        }
    except Exception as exc:  # noqa: BLE001
        _logger.warning("market_warm 读取已缓存键失败: %s", exc)
        cached_keys = set()

    by_market: Dict[str, Dict[str, Any]] = {}
    summary: Dict[str, Any] = {"target": 0, "warmed": 0, "failed": 0,
                               "pending_left": 0, "done": True}

    # 先按市场各自发现清单 / 取差集 / 截断 batch，汇总出「待建任务」扁平列表
    todos: List[tuple] = []  # (market, vt)
    for m in selected:
        exchs = MARKET_EXCHANGES.get(m, set())
        # discover_market 内为同步 akshare（可能慢/含网络重试），放线程池避免阻塞事件循环
        universe = await asyncio.to_thread(discover_market, m, cap)
        universe = [u for u in universe if u.split(".")[-1] in exchs]
        if not universe:
            by_market[m] = {"target": 0, "warmed": 0, "failed": 0,
                            "pending_left": 0, "done": False, "skipped": True,
                            "reason": "清单为空(akshare 不可用?)"}
            continue

        pending = [u for u in universe if u not in cached_keys]
        # full=True：一次把所有待建标的全建完（内部仍逐只拉取+限速，不阻塞事件循环）；
        # 否则（调度/手动单趟）按 batch 截断、剩余下趟继续（自推进 catch-up）。
        todo = pending if full else pending[:batch]
        left = max(len(pending) - len(todo), 0)
        by_market[m] = {
            "target": len(todo),
            "warmed": 0,
            "failed": 0,
            "pending_left": left,
            "done": left == 0,   # 本趟后是否已无待建（全部建库）
        }
        summary["target"] += len(todo)
        summary["pending_left"] += left
        summary["done"] = summary["done"] and by_market[m]["done"]
        todos.extend((m, vt) for vt in todo)

    if not todos:
        if not by_market:
            return {"action": "market_warm", "skipped": True,
                    "reason": "所选市场清单均为空(akshare 不可用?)"}
        # 有市场但无待建（可能全部已缓存）
        summary["action"] = "market_warm"
        summary["markets"] = selected
        summary["by_market"] = by_market
        return summary

    # 逐任务拉取落盘，并按市场归堆统计
    for i, (m, vt) in enumerate(todos, start=1):
        if progress_callback:
            progress_callback(i, len(todos), f"预热 {vt} (1d)")
        if await _warm_single(dm, dc, vt):
            by_market[m]["warmed"] += 1
            summary["warmed"] += 1
        else:
            by_market[m]["failed"] += 1
            summary["failed"] += 1

    summary["action"] = "market_warm"
    summary["markets"] = selected
    summary["by_market"] = by_market
    return summary


def build_default_jobs(sys_state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """构造内置任务注册表（懒解析，依赖从 sys_state 注入，可缺省）。"""
    from .futures_download import _job_futures_auto_download
    from .stock_download import _job_stock_auto_download
    from .services.market_update_settings_service import MarketUpdateSettingsService

    # 股票市场自动更新配置（A股/港股/美股各自独立开关与时间）
    try:
        mu_service = sys_state.get("market_update_settings") or MarketUpdateSettingsService()
    except Exception:  # noqa: BLE001
        mu_service = None

    def _stock_cron(key: str, fallback: str) -> str:
        if mu_service is None:
            return fallback
        try:
            return mu_service.get_market(key).get("schedule_cron", fallback)
        except Exception:  # noqa: BLE001
            return fallback

    def _stock_job(market: str) -> Dict[str, Any]:
        return {
            "name": {"A": "stock_download", "HK": "hk_download", "US": "us_download"}[market],
            "fn": _job_stock_auto_download,
            "cron": _stock_cron(
                {"A": "a_stock", "HK": "hk_stock", "US": "us_stock"}[market],
                {"A": "0 17 * * 1-5", "HK": "0 23 * * 1-5", "US": "0 5 * * 1-5"}[market]),
            "timezone": "Asia/Shanghai",
            "kwargs": {"sys_state": sys_state, "market": market},
            "required": False,
        }

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
            "timezone": "Asia/Shanghai",
            "kwargs": {"sys_state": sys_state},
            "required": False,
        },
        {
            "name": "data_sync",
            "fn": _job_data_sync,
            "cron": "30 15 * * 1-5",  # 交易日 15:30（后端仅登记，见函数内跳过逻辑）
            "timezone": "Asia/Shanghai",
            "kwargs": {"sys_state": sys_state},
            "required": False,
        },
        {
            "name": "cache_refresh",
            "fn": _job_cache_refresh,
            "cron": "0 17 * * 1-5",   # 交易日 17:00 收盘后刷新本地行情仓库
            "timezone": "Asia/Shanghai",
            "kwargs": {"sys_state": sys_state},
            "required": False,
        },
        {
            "name": "market_warm",
            "fn": _job_market_warm,
            # 自推进 catch-up：每趟预热 batch 个未缓存标的，剩余留给下一趟；
            # 间隔由配置决定（默认 15 分钟），apscheduler 未装时忽略该条。
            "interval": 15 * 60,
            "max_instances": 1,
            "kwargs": {"sys_state": sys_state},
            "required": False,
        },
        {
            # 期货数据自动下载：收盘后增量更新
            # 品种、周期、调度时间从配置文件读取，可在 Web 界面配置
            "name": "futures_download",
            "fn": _job_futures_auto_download,
            "cron": sys_state.get("futures_download_cron", "30 16 * * 1-5"),  # 默认交易日 16:30
            "timezone": "Asia/Shanghai",
            "kwargs": {"sys_state": sys_state},
            "required": False,
        },
        _stock_job("A"),
        _stock_job("HK"),
        _stock_job("US"),
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
                timezone=spec.get("timezone"),
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
