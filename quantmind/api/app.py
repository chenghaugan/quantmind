"""FastAPI 应用（Web 统一入口后端，接入真实引擎）。

功能：
  - /research  AI 研究（idea -> 规格/因子/策略代码）
  - /factor    因子计算 + 有效性评估（IC/IR/衰减/分位收益）
  - /backtest  回测 / 模拟 / 实盘（同一策略，按 mode 切换路线）
  - /strategies 可用策略清单
  - /order     手动下单（广播委托意图，驱动监控）
  - /lifecycle 策略生命周期晋升闸门
  - /ws        WebSocket 实时推送引擎事件（bar/signal/position/trade/account/log）
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import math
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import uuid

from ..config import get_settings
from ..data import build_default_registry, DataManager, TimescaleStore, InMemoryStore
from ..data.store.disk_cache import DiskBarCache
from ..core.engine import EventEngine
from ..core.constant import Exchange, Interval
from ..core.event import Event, EventType
from ..ai import build_provider
from ..knowledge import KnowledgeStore
from ..paper.promotion import LifecycleManager, LifecycleState
from ..research.knowledge_loop import judge_strategy, run_strategy_knowledge_loop
from ..monitoring import Notifier
from ..research import FactorRegistry
from ..research.decay import FactorDecayScanner, DecayConfig, FactorState

from .schemas import (
    ResearchRequest, ResearchResult, FactorRequest, FactorResult,
    BacktestRequest, WalkForwardRequest, StrategyInfo, OrderRequestSchema, LifecycleRequest,
    OptimizeRequest, RiskCheckRequest,
    SeatFactorRequest, SeatFactorResult,
    ExprEvalRequest, ExprEvalBatchRequest, FactorSearchRequest,
    FactorDedupRequest, ExpressionBacktestRequest, FactorPipelineRequest,
    FactorE2ERequest, KnowledgeIngestRequest, KnowledgeSearchRequest,
    StrategyRegisterRequest, StrategyValidateRequest, PaperRunRequest,
    StrategyDraftRequest, StrategyMiningRequest, AutoBacktestRequest,
)
from .ws import manager
from .services import (
    DataService, FactorService, BacktestService, LifecycleService, ResearchService,
    RiskService, OptimizeService, SettingsService, SeatService,
    DataSettingsService, DataAdminService, AlertSettingsService, SearchService,
    KnowledgeService, StrategyMiningService, FuturesDownloadSettingsService,
    MarketUpdateSettingsService,
)
from .logging_config import setup_api_logger
from .routes_auth import router as auth_router
from .routes_profile import router as profile_router
from .routes_ml import router as ml_router
from .scheduler import QuantMindScheduler, build_scheduler
from .services.ml_service import MLService
from .services.profile_service import ProfileService

_logger = setup_api_logger("INFO")

# 全局实例（用于 WebSocket 广播）
_ee: Optional[EventEngine] = None

# AI 设置允许更新的字段白名单
_AI_ALLOWED = {"provider", "api_key", "base_url", "model", "temperature"}

# ---------------------------------------------------------------- 长任务存储
# 端到端流水线等长耗时任务用「启动 + 轮询」模式，避免单次 HTTP 请求超时。
# 内存态 task_id -> {task, status, message, result}
_E2E_TASKS: Dict[str, Dict[str, Any]] = {}
_FUTURES_DOWNLOAD_TASKS: Dict[str, Dict[str, Any]] = {}
_STOCK_DOWNLOAD_TASKS: Dict[str, Dict[str, Any]] = {}
_MARKET_WARM_TASKS: Dict[str, Dict[str, Any]] = {}


def _prune_tasks(store: Dict[str, Dict[str, Any]], max_items: int = 100) -> None:
    """裁剪已完成的后台任务记录（最早已完成者先删），防止常驻进程内存无限增长。"""
    if len(store) <= max_items:
        return
    for k in list(store):
        if len(store) <= max_items:
            break
        if store[k].get("status") != "running":
            store.pop(k, None)


def _submit_task(coro, progress: Optional[Dict[str, Any]] = None) -> str:
    """把协程提交为后台任务，返回 task_id。

    ``progress`` 可选：任务内部可变字典（current/total/message），
    由任务自身更新，状态接口原样返回供前端展示真实进度。
    """
    task_id = uuid.uuid4().hex
    record: Dict[str, Any] = {
        "task": None,
        "status": "running",
        "message": "任务已提交，正在初始化…",
        "result": None,
        "progress": progress if progress is not None
        else {"current": 0, "total": 0, "message": ""},
    }
    _E2E_TASKS[task_id] = record
    _prune_tasks(_E2E_TASKS)

    async def _run() -> None:
        try:
            record["result"] = await coro
            record["status"] = "success"
            record["message"] = "任务完成"
        except asyncio.CancelledError:  # noqa: PERF203
            record["status"] = "cancelled"
            record["message"] = "任务已取消"
            raise
        except Exception as exc:  # noqa: BLE001
            _logger.exception("后台任务失败: %s", exc)
            record["status"] = "error"
            record["message"] = str(exc)
        finally:
            record["task"] = None

    record["task"] = asyncio.create_task(_run())
    return task_id


def get_ml_service() -> MLService:
    """供 routes_ml 等模块延迟获取 ML 服务（app 实例化后才可用）。"""
    return app.state.ml_service


def _jsonable(o: Any) -> Any:
    """把事件数据转成可 JSON 序列化结构。"""
    if isinstance(o, datetime):
        return o.isoformat()
    if isinstance(o, Enum):
        return o.value
    if isinstance(o, float):
        # 非有限值（inf/nan）在 JSON(allow_nan=False) 下会抛错，统一转 None
        return o if math.isfinite(o) else None
    if hasattr(o, "to_dict"):
        return o.to_dict()
    if dataclasses.is_dataclass(o):
        return {k: _jsonable(v) for k, v in dataclasses.asdict(o).items()}
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(x) for x in o]
    return o


_BROADCAST_TASKS: set = set()


def _broadcast(e: Event) -> None:
    if _ee is None:
        return
    msg = {"type": e.type.value, "data": _jsonable(e.data)}
    try:
        t = asyncio.ensure_future(manager.broadcast(msg))
        # 保留强引用避免 task 被 GC；完成后自动移除
        _BROADCAST_TASKS.add(t)
        t.add_done_callback(_BROADCAST_TASKS.discard)
    except RuntimeError:
        pass


# ------------------------------------------------------------------ 策略级 AI 沉淀
# 策略注册/回测/模拟盘逐步落库到 lifecycle：用真实指标做 AI 判读（judge_strategy）、
# 把 status/reason/brief 写回 KnowledgeStore。只修改持久层，失败仅 warning 不阻断主流程。
_STRATEGY_GATE = {"min_sharpe": 0.5, "min_drawdown": -0.30}


def _llm_provider() -> Any:
    """取当前 LLM Provider（供 AI 判读；不可用回退规则判读）。"""
    svc = getattr(app.state, "search_service", None)
    if svc is not None and getattr(svc, "provider", None) is not None:
        return svc.provider
    res = getattr(app.state, "research_service", None)
    if res is not None and getattr(res, "provider", None) is not None:
        return res.provider
    return None


async def _persist_backtest_lifecycle(strategy_id: str, risk_xray: Any) -> Optional[dict]:
    """回填回测真实指标 + 策略级判读（state=BACKTEST）。失败仅 warning。"""
    try:
        ks = KnowledgeStore()
        if (risk_xray or {}).get("return", {}).get("sharpe") is None:
            return None
        rx_return = risk_xray.get("return", {})
        rx_risk = risk_xray.get("risk", {})
        sharpe = rx_return.get("sharpe")
        mdd = rx_risk.get("max_drawdown")
        ks.update_strategy_state(
            strategy_id, state="BACKTEST",
            sharpe=sharpe, max_drawdown=mdd, status="", reason="",
        )
        judged = await judge_strategy(
            _llm_provider(),
            {"sharpe": sharpe, "max_drawdown": mdd, "state": "BACKTEST"},
            gate=_STRATEGY_GATE, fallback_rules=True,
        )
        ks.update_strategy_state(
            strategy_id, status=judged.get("status"), reason=judged.get("reason"),
        )
        _logger.info("策略生命周期回测判读落库: %s -> %s (sharpe=%s)", strategy_id,
                     judged.get("status"), sharpe)
        return judged
    except Exception as exc:  # noqa: BLE001
        _logger.warning("策略回测生命周期回填/判读失败(%s): %s", strategy_id, exc)
        return None


async def _persist_paper_lifecycle(strategy_id: str, metrics: Dict, idea: str = "") -> Optional[dict]:
    """模拟盘判读落库：state=PAPER（含真实 sharpe/回撤）+ 经验 brief。失败仅 warning。"""
    try:
        ks = KnowledgeStore()
        loop = await run_strategy_knowledge_loop(
            ks, _llm_provider(),
            [{
                "strategy_id": strategy_id, "state": "PAPER",
                "sharpe": metrics.get("sharpe"),
                "max_drawdown": metrics.get("max_drawdown"),
                "status": "paper",
            }],
            idea=idea,
        )
        judged = (loop.get("judged") or [{}])[0] if loop.get("judged") else {
            "status": "active", "reason": "", "tags": [],
        }
        ks.update_strategy_state(
            strategy_id,
            status=judged.get("status"), reason=judged.get("reason"),
            brief=loop.get("brief") or "",
        )
        _logger.info("策略生命周期模拟盘判读落库: %s -> %s", strategy_id, judged.get("status"))
        return judged
    except Exception as exc:  # noqa: BLE001
        _logger.warning("策略模拟盘生命周期判读/落库失败(%s): %s", strategy_id, exc)
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _ee
    settings = get_settings()
    registry = build_default_registry(
        local_data_root=settings.local_data_root or None,
        local_stock_root=settings.local_stock_root or None,
        local_hk_root=settings.local_hk_root or None,
        local_option_root=settings.local_option_root or None,
    )
    try:
        store = TimescaleStore(settings.db_url)
        await store.connect()
        _logger.info("使用 TimescaleDB 存储")
    except Exception as exc:  # noqa: BLE001
        _logger.warning("TimescaleDB 不可用，降级内存存储: %s", exc)
        store = InMemoryStore()
        await store.connect()

    dm = DataManager(registry, store)
    await dm.connect()

    # 本地行情仓库（Parquet 写缓存）：真实源结果落盘，后续请求秒级返回。
    # 未显式配置 local_cache_root 时，默认用项目根 data_cache/（避免每次联网拉 akshare）。
    _cache_root = (settings.local_cache_root or "").strip()
    if not _cache_root:
        _cache_root = str(Path(__file__).resolve().parent.parent.parent / "data_cache")
    disk_cache = DiskBarCache(_cache_root)
    dm.disk_cache = disk_cache
    _logger.info("本地行情仓库挂载: %s", _cache_root)

    _ee = EventEngine()
    await _ee.start()
    _ee.register_general(_broadcast)
    notifier = Notifier()
    notifier.attach(_ee)
    lifecycle_mgr = LifecycleManager(store=KnowledgeStore())
    settings_service = SettingsService()
    provider = settings_service.rebuild_provider()

    # 初始化 Service 层
    app.state.data_service = DataService(dm)
    app.state.factor_service = FactorService(dm)
    app.state.backtest_service = BacktestService(dm, _ee)
    app.state.lifecycle_service = LifecycleService(lifecycle_mgr, _ee)
    app.state.research_service = ResearchService(provider)
    app.state.risk_service = RiskService()
    app.state.optimize_service = OptimizeService(
        dm, resolver=lambda n: app.state.backtest_service._resolve_strategy_class(n)
    )
    app.state.settings_service = settings_service
    app.state.search_service = SearchService(dm, provider)
    app.state.knowledge_service = KnowledgeService()
    app.state.decay_scanner = FactorDecayScanner()
    app.state.seat_service = SeatService(dm)
    data_settings = DataSettingsService()
    app.state.data_settings_service = data_settings
    app.state.data_admin_service = DataAdminService(dm, data_settings)
    app.state.alert_settings_service = AlertSettingsService()
    app.state.futures_download_settings_service = FuturesDownloadSettingsService()
    app.state.market_update_settings_service = MarketUpdateSettingsService()

    # 初始化 ML 服务和 Profile 服务
    app.state.ml_service = MLService()
    app.state.profile_service = ProfileService()

    # 初始化 LLM 策略挖掘服务
    app.state.strategy_mining_service = StrategyMiningService(dm, lifecycle_mgr, provider)

    app.state.dm = dm
    app.state.ee = _ee
    app.state.lifecycle = lifecycle_mgr

    # ---- 任务调度器（APScheduler）----
    # 读取期货下载配置 + 股票市场自动更新配置（A股/港股/美股）
    futures_dl_settings = FuturesDownloadSettingsService()
    futures_dl_config = futures_dl_settings.get()
    market_update_settings = app.state.market_update_settings_service
    sys_state = {
        "dm": dm,
        "ee": _ee,
        "lifecycle": lifecycle_mgr,
        "settings": settings,
        "futures_download_settings": futures_dl_settings,
        "futures_download_cron": futures_dl_config.get("schedule_cron", "30 16 * * 1-5"),
        "market_update_settings": market_update_settings,
    }
    scheduler = build_scheduler(sys_state, register_defaults=True)
    scheduler.start()
    app.state.scheduler = scheduler

    yield

    if app.state.scheduler:
        app.state.scheduler.stop()
    if dm:
        await dm.close()
    if _ee:
        await _ee.stop()


app = FastAPI(title="QuantMind API", version="0.2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """请求日志中间件：记录所有请求的方法、路径、状态码和耗时"""
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        start_time = asyncio.get_event_loop().time()
        
        try:
            response = await call_next(request)
            duration = asyncio.get_event_loop().time() - start_time
            _logger.info(f"[{request_id}] {request.method} {request.url.path} - {response.status_code} - {duration:.3f}s")
            return response
        except Exception as e:
            duration = asyncio.get_event_loop().time() - start_time
            _logger.error(f"[{request_id}] {request.method} {request.url.path} - ERROR - {duration:.3f}s - {str(e)}")
            raise


app.add_middleware(RequestLoggingMiddleware)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理：返回统一错误格式，避免暴露内部细节"""
    _logger.exception(f"未捕获的异常: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "服务器内部错误", "detail": str(exc) if _logger.level <= logging.DEBUG else "请联系管理员"}
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """参数校验失败统一格式（覆盖 FastAPI 默认的 422 响应）"""
    return JSONResponse(
        status_code=422,
        content={"error": "参数校验失败", "detail": exc.errors()},
    )

# 注册路由
app.include_router(auth_router)
app.include_router(profile_router)
app.include_router(ml_router)


@app.get("/")
async def root():
    return {"name": "QuantMind API", "version": "0.2.0", "docs": "/docs"}


@app.get("/health")
async def health():
    from datetime import datetime, timezone
    dm: DataManager = app.state.dm
    ee: EventEngine = app.state.ee
    
    # 检查数据管理器状态
    feeds = dm.registry.list_feeds() if dm else []
    
    # 检查事件引擎状态
    ee_status = "running" if ee and ee._running else "stopped"
    
    # 检查生命周期管理器
    lifecycle_status = "active" if hasattr(app.state, 'lifecycle') else "inactive"
    
    return {
        "status": "ok",
        "feeds": feeds,
        "components": {
            "data_manager": "active" if dm else "inactive",
            "event_engine": ee_status,
            "lifecycle": lifecycle_status
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/feeds")
async def feeds():
    dm: DataManager = app.state.dm
    return {"feeds": dm.registry.list_feeds() if dm else []}


@app.get("/factors")
async def list_factors():
    reg = FactorRegistry()
    return {"factors": reg.list_factors()}


@app.get("/data")
async def get_data(
    symbol: str = Query(...), exchange: str = Query("SHFE"),
    interval: str = Query("1d"), start: str = Query(None), end: str = Query(None),
    page: int = Query(1, ge=1), page_size: int = Query(100, ge=1, le=1000),
):
    service: DataService = app.state.data_service
    return await service.query_bars(symbol, exchange, interval, start, end, page, page_size)


@app.get("/data/cache")
async def get_data_cache_stats():
    """本地行情仓库（Parquet 写缓存）概览：文件数 / 行数 / 最新交易日 / 根目录 + 聚合桶。

    几千只标的下逐标的明细不再全量下发；这里只返回：总览 KPI + 聚合桶
    （``agg.by_exchange / by_interval / freshness / top_rows / stale_top``）。
    逐标的明细走 ``GET /data/cache/symbols``（过滤 + 分页）。
    """
    dm: DataManager = app.state.dm
    if dm is None or dm.disk_cache is None:
        return {"enabled": False}
    try:
        # 扫描几千个 parquet 较重，放线程池避免阻塞事件循环（否则全站接口都会被卡住）
        stats = await asyncio.to_thread(
            dm.disk_cache.stats, include_symbols=False, aggregate=True)
        return {"enabled": True, **stats}
    except Exception as exc:  # noqa: BLE001
        _logger.warning("读取行情仓库统计失败: %s", exc)
        return {"enabled": True, "error": str(exc)}


@app.get("/data/cache/symbols")
async def get_data_cache_symbols(
    exchange: str = Query(""),
    market: str = Query(""),
    interval: str = Query(""),
    freshness: str = Query("", description="fresh | stale"),
    q: str = Query("", description="标的模糊搜索"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
):
    """行情仓库逐标的明细分页（总览页下钻用），按交易所/市场/周期/新鲜度/关键词过滤。"""
    dm: DataManager = app.state.dm
    if dm is None or dm.disk_cache is None:
        return {"enabled": False}
    try:
        return await asyncio.to_thread(
            dm.disk_cache.symbol_page,
            exchange=exchange, market=market, interval=interval,
            freshness=freshness, q=q, page=page, page_size=page_size,
        )
    except Exception as exc:  # noqa: BLE001
        _logger.warning("读取行情仓库标的分页失败: %s", exc)
        return {"error": str(exc)}


@app.delete("/data/cache")
async def purge_data_cache():
    """清空本地行情仓库（删除全部 .parquet）。下次请求将重新从真实源拉取并重建。"""
    dm: DataManager = app.state.dm
    if dm is None or dm.disk_cache is None:
        return {"ok": False, "error": "本地行情仓库未启用"}
    removed = 0
    for p in dm.disk_cache.root.glob("*.parquet"):
        try:
            p.unlink()
            removed += 1
        except Exception as exc:  # noqa: BLE001
            _logger.warning("删除 %s 失败: %s", p, exc)
    return {"ok": True, "removed": removed}


@app.post("/data/cache/warm")
async def warm_data_cache(payload: Dict[str, Any]):
    """预热本地行情仓库：把给定标的多真实源拉取并落盘（后续 /factor/pipeline 秒级）。

    payload: {symbols: [..], exchange: "SHFE", start?: "YYYY-MM-DD", end?: "YYYY-MM-DD"}
    与 ``/data/cache/refresh`` 共用统一落盘/日志核心 :func:`_warm_cache_symbols`，
    返回相同 schema（``refreshed`` / ``failed`` / ``results[{key,status,n,source,latest,error}]``）。
    """
    dm: DataManager = app.state.dm
    if dm is None or dm.disk_cache is None:
        return {"ok": False, "error": "本地行情仓库未启用"}
    symbols = [s for s in (payload.get("symbols") or []) if s and str(s).strip()]
    if not symbols:
        return {"ok": False, "error": "至少提供 1 个标的"}
    exch = Exchange(str(payload.get("exchange", "SHFE")).upper())
    start = datetime.fromisoformat(payload["start"]) if payload.get("start") else None
    end = datetime.fromisoformat(payload["end"]) if payload.get("end") else None

    todos: List[Tuple[str, str, str, Optional[datetime], Optional[datetime]]] = [
        (str(s).strip(), exch.value, Interval.DAILY.value, start, end) for s in symbols
    ]
    refreshed, failed, results = await _warm_cache_symbols(dm, todos)
    return {"ok": True, "refreshed": refreshed, "failed": failed, "results": results}


async def _warm_cache_symbols(
    dm: DataManager,
    todos: List[Tuple[str, str, str, Optional[datetime], Optional[datetime]]],
) -> Tuple[int, int, List[Dict[str, Any]]]:
    """统一落盘核心：把一组 (symbol, exchange, interval, start, end) 逐标的真实源拉取并落盘。

    由 ``/data/cache/warm``（显式标的）与 ``/data/cache/refresh``（已缓存键）共用，
    消除二者重复的 fetch/record_refresh 逻辑与结果 schema 差异。每个标的返回统一记录：
    ``{key:{symbol,exchange,interval}, status: ok|empty|error|bad_key, n, source, latest, error}``。
    """
    from ..data.feed.base import HistoryRequest

    dc = getattr(dm, "disk_cache", None)
    results: List[Dict[str, Any]] = []
    refreshed, failed = 0, 0
    for (symbol, exch_str, interv_str, start, end) in todos:
        key = {"symbol": symbol, "exchange": exch_str, "interval": interv_str}
        try:
            exch = Exchange(exch_str.upper())
            interv = Interval(interv_str)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            results.append({"key": key, "status": "bad_key", "n": 0, "error": str(exc)[:80]})
            continue
        try:
            req = HistoryRequest(
                symbol=symbol, exchange=exch, interval=interv, start=start, end=end
            )
            sink: Dict[str, str] = {}
            bars = await dm.get_bar_data(req, source_sink=sink)
            ok = bool(bars)
            latest = bars[-1].datetime.isoformat() if bars else None
            source = sink.get(symbol, "") or ("empty" if not ok else "")
            results.append({
                "key": key, "status": "ok" if ok else "empty", "n": len(bars),
                "source": source, "latest": latest,
            })
            refreshed += 1 if ok else 0
            failed += 0 if ok else 1
            if dc is not None:
                dc.record_refresh(
                    symbol=symbol, exchange=str(exch.value), interval=interv.value,
                    rows=len(bars), latest=latest,
                    status="ok" if ok else "empty", detail=source,
                )
        except Exception as exc:  # noqa: BLE001
            failed += 1
            results.append({"key": key, "status": "error", "n": 0, "error": str(exc)[:120]})
            if dc is not None:
                dc.record_refresh(
                    symbol=symbol, exchange=str(exch.value), interval=interv.value,
                    rows=0, latest=None, status="error", detail=str(exc)[:120],
                )
    return refreshed, failed, results


@app.get("/data/cache/history")
async def get_data_cache_history(limit: int = Query(50, ge=1, le=500)):
    """本地行情仓库刷新执行历史（最新在前）。"""
    dm: DataManager = app.state.dm
    if dm is None or dm.disk_cache is None:
        return {"enabled": False}
    return {"enabled": True, "history": dm.disk_cache.refresh_history(limit=limit)}


@app.post("/data/cache/refresh")
async def refresh_data_cache():
    """手动触发全量刷新：把仓库内所有已缓存标的从真实源重拉回写（追新）。

    等价于 scheduler 的 ``cache_refresh`` 任务；与 ``/data/cache/warm`` 共用统一落盘核心
    :func:`_warm_cache_symbols`，返回相同 schema。
    """
    dm: DataManager = app.state.dm
    if dm is None or dm.disk_cache is None:
        return {"ok": False, "error": "本地行情仓库未启用"}
    dc = dm.disk_cache
    keys = dc.list_keys()
    if not keys:
        return {"ok": True, "refreshed": 0, "failed": 0, "results": [], "skipped": True}

    todos: List[Tuple[str, str, str, None, None]] = [
        (k["symbol"], str(k["exchange"]), str(k["interval"]), None, None) for k in keys
    ]
    was_refresh = bool(getattr(dc, "refresh", False))
    dc.refresh = True
    try:
        refreshed, failed, results = await _warm_cache_symbols(dm, todos)
    finally:
        dc.refresh = was_refresh
    return {"ok": True, "refreshed": refreshed, "failed": failed, "results": results}


@app.post("/data/cache/market-warm")
async def warm_market_cache(payload: Dict[str, Any] = None):
    """手动触发全市场预热：逐市场（A股/港股）增量建库，可指定覆盖哪些市场。

    与调度任务共用 ``_job_market_warm``，便于点按手动推进与后台自动维护共用单一路径。
    可选参数：
        - markets: 市场列表（"A"=A股, "HK"=港股）；缺省=取配置（默认两者都预热）

    返回聚合统计 + ``by_market`` 分市场明细。
    """
    from .scheduler import _job_market_warm

    payload = payload or {}
    markets = payload.get("markets")

    sys_state = {
        "dm": app.state.dm,
        "ee": app.state.ee,
        "lifecycle": app.state.lifecycle,
        "settings": get_settings(),
    }
    return await _job_market_warm(sys_state, markets=markets)


@app.post("/data/cache/market-warm/start")
async def warm_market_start(payload: Dict[str, Any] = None):
    """异步启动全市场预热任务，立即返回 task_id，后台逐市场增量建库。

    与期货/A股下载一致：页面可通过 ``GET /data/cache/market-warm/status/{task_id}``
    轮询实时进度（current/total/message），避免预热期间页面卡死/无进度。
    可选参数 markets："A"/"HK" 列表，缺省取配置（默认两者）。
    """
    from .scheduler import _job_market_warm

    payload = payload or {}
    markets = payload.get("markets")
    full = bool(payload.get("full", False))

    task_id = uuid.uuid4().hex
    record: Dict[str, Any] = {
        "task": None,
        "status": "running",
        "message": "任务已提交，正在初始化…",
        "result": None,
        "progress": {"current": 0, "total": 0, "message": "准备中"},
    }
    _MARKET_WARM_TASKS[task_id] = record
    _prune_tasks(_MARKET_WARM_TASKS)

    def progress_callback(current: int, total: int, message: str):
        record["progress"]["current"] = current
        record["progress"]["total"] = total
        record["progress"]["message"] = message
        record["message"] = message

    async def _run():
        try:
            sys_state = {
                "dm": app.state.dm,
                "ee": app.state.ee,
                "lifecycle": app.state.lifecycle,
                "settings": get_settings(),
            }
            record["result"] = await _job_market_warm(
                sys_state, markets=markets, progress_callback=progress_callback, full=full
            )
            record["status"] = "success"
            record["message"] = "全市场预热完成"
        except asyncio.CancelledError:
            record["status"] = "cancelled"
            record["message"] = "任务已取消"
            raise
        except Exception as exc:
            _logger.exception("全市场预热任务失败: %s", exc)
            record["status"] = "error"
            record["message"] = str(exc)
        finally:
            record["task"] = None

    record["task"] = asyncio.create_task(_run())
    return {"task_id": task_id, "status": "running"}


@app.get("/data/cache/market-warm/status/{task_id}")
async def warm_market_status(task_id: str):
    """查询全市场预热任务状态（running / success / error / cancelled / not_found）。

    任务成功后 ``result`` 携带完整结果（含分市场 ``by_market`` 明细）。
    """
    rec = _MARKET_WARM_TASKS.get(task_id)
    if rec is None:
        return JSONResponse(
            status_code=404,
            content={"task_id": task_id, "status": "not_found",
                     "message": "任务不存在（可能后端已重启）"},
        )
    done = rec["status"] in ("success", "error", "cancelled")
    return {
        "task_id": task_id,
        "status": rec["status"],
        "message": rec["message"],
        "progress": rec["progress"],
        "result": rec["result"] if done else None,
    }


@app.post("/data/futures/download")
async def download_futures_data():
    """手动触发期货数据下载（同步，向后兼容）。

    注意：同步模式受单次请求超时限制，长跑请改用
    ``POST /data/futures/download/start`` + ``GET /data/futures/download/status/{task_id}``。
    """
    from .futures_download import _job_futures_auto_download

    sys_state = {
        "dm": app.state.dm,
        "ee": app.state.ee,
        "lifecycle": app.state.lifecycle,
        "settings": get_settings(),
        "futures_download_settings": app.state.futures_download_settings_service,
    }
    return await _job_futures_auto_download(sys_state)


@app.post("/data/futures/download/start")
async def download_futures_data_start(payload: Dict[str, Any] = None):
    """异步启动期货数据下载任务。

    立即返回 ``{"task_id": ...}``，任务在后台执行，客户端通过
    ``GET /data/futures/download/status/{task_id}`` 轮询进度/结果。
    避免页面切换导致下载中断。

    可选参数：
        - symbols: 品种列表（如 ["IF0", "IC0", "rb0"]）
        - intervals: 周期列表（如 ["1d", "60m", "15m"]）
    """
    from .futures_download import _job_futures_auto_download

    payload = payload or {}
    symbols = payload.get("symbols")
    intervals = payload.get("intervals")

    task_id = uuid.uuid4().hex
    record: Dict[str, Any] = {
        "task": None,
        "status": "running",
        "message": "任务已提交，正在初始化…",
        "result": None,
        "progress": {
            "current": 0,
            "total": 0,
            "message": "准备中",
        },
    }
    _FUTURES_DOWNLOAD_TASKS[task_id] = record
    _prune_tasks(_FUTURES_DOWNLOAD_TASKS)

    def progress_callback(current: int, total: int, message: str):
        record["progress"]["current"] = current
        record["progress"]["total"] = total
        record["progress"]["message"] = message
        record["message"] = message

    async def _run():
        try:
            sys_state = {
                "dm": app.state.dm,
                "ee": app.state.ee,
                "lifecycle": app.state.lifecycle,
                "settings": get_settings(),
                "futures_download_settings": app.state.futures_download_settings_service,
            }
            record["result"] = await _job_futures_auto_download(
                sys_state,
                symbols=symbols,
                intervals=intervals,
                progress_callback=progress_callback,
            )
            record["status"] = "success"
            record["message"] = "下载完成"
        except asyncio.CancelledError:
            record["status"] = "cancelled"
            record["message"] = "任务已取消"
            raise
        except Exception as exc:
            _logger.exception("期货下载任务失败: %s", exc)
            record["status"] = "error"
            record["message"] = str(exc)
        finally:
            record["task"] = None

    record["task"] = asyncio.create_task(_run())
    return {"task_id": task_id, "status": "running"}


@app.get("/data/futures/download/status/{task_id}")
async def download_futures_data_status(task_id: str):
    """查询期货数据下载任务状态。

    ``status``: running / success / error / cancelled / not_found。
    任务成功后 ``result`` 携带完整结果。
    ``progress`` 包含实时进度信息。
    """
    rec = _FUTURES_DOWNLOAD_TASKS.get(task_id)
    if rec is None:
        return JSONResponse(
            status_code=404,
            content={"task_id": task_id, "status": "not_found", "message": "任务不存在（可能后端已重启）"},
        )
    done = rec["status"] in ("success", "error", "cancelled")
    return {
        "task_id": task_id,
        "status": rec["status"],
        "message": rec["message"],
        "progress": rec["progress"],
        "result": rec["result"] if done else None,
    }


@app.post("/data/stock/download/start")
async def download_stock_data_start(payload: Dict[str, Any] = None):
    """异步启动 A股数据更新任务（日线/60m/30m）。

    立即返回 task_id，后台执行，可安全切换页面。
    可选参数：
        - symbols: 标的列表（如 ["600000.SSE"]），缺省=更新仓库内全部 A股
        - intervals: 周期列表（"1d"/"1h"/"30m"），缺省=三档
        - manual: 是否手动触发（跳过启用开关检查，默认 True）
    """
    from .stock_download import _job_stock_auto_download

    payload = payload or {}
    task_id = uuid.uuid4().hex
    record: Dict[str, Any] = {
        "task": None,
        "status": "running",
        "message": "任务已提交，正在初始化…",
        "result": None,
        "progress": {"current": 0, "total": 0, "message": "准备中"},
    }
    _STOCK_DOWNLOAD_TASKS[task_id] = record
    _prune_tasks(_STOCK_DOWNLOAD_TASKS)

    def progress_callback(current: int, total: int, message: str):
        record["progress"]["current"] = current
        record["progress"]["total"] = total
        record["progress"]["message"] = message
        record["message"] = message

    async def _run():
        try:
            sys_state = {
                "dm": app.state.dm,
                "ee": app.state.ee,
                "lifecycle": app.state.lifecycle,
                "settings": get_settings(),
            }
            record["result"] = await _job_stock_auto_download(
                sys_state,
                symbols=payload.get("symbols"),
                intervals=payload.get("intervals"),
                manual=payload.get("manual", True),
                progress_callback=progress_callback,
            )
            record["status"] = "success"
            record["message"] = "A股数据更新完成"
        except asyncio.CancelledError:
            record["status"] = "cancelled"
            record["message"] = "任务已取消"
            raise
        except Exception as exc:
            _logger.exception("A股数据更新任务失败: %s", exc)
            record["status"] = "error"
            record["message"] = str(exc)
        finally:
            record["task"] = None

    record["task"] = asyncio.create_task(_run())
    return {"task_id": task_id, "status": "running"}


@app.get("/data/stock/download/status/{task_id}")
async def download_stock_data_status(task_id: str):
    """查询 A股数据更新任务状态。"""
    rec = _STOCK_DOWNLOAD_TASKS.get(task_id)
    if rec is None:
        return JSONResponse(
            status_code=404,
            content={"task_id": task_id, "status": "not_found", "message": "任务不存在（可能后端已重启）"},
        )
    done = rec["status"] in ("success", "error", "cancelled")
    return {
        "task_id": task_id,
        "status": rec["status"],
        "message": rec["message"],
        "progress": rec["progress"],
        "result": rec["result"] if done else None,
    }


@app.post("/research", response_model=ResearchResult)
async def research(req: ResearchRequest):
    service: ResearchService = app.state.research_service
    return await service.research(req)


# --------------------------------------------------------------------------
# AI 模型设置（API Key / Base URL / 模型 / 温度）
# --------------------------------------------------------------------------
@app.get("/settings/ai")
async def get_ai_settings():
    """读取当前 AI 模型配置（不含明文 key 校验，前端按需展示）。"""
    return app.state.settings_service.get()


@app.put("/settings/ai")
async def put_ai_settings(payload: Dict[str, Any]):
    """保存 AI 模型配置并即时重建 Provider，无需重启。"""
    data = app.state.settings_service.save(payload)
    provider = app.state.settings_service.rebuild_provider()
    # 同步更新所有持有 provider 引用的 service
    app.state.research_service.provider = provider
    app.state.search_service.provider = provider
    synced = data.pop("synced_env", False)
    return {"ok": True, "settings": data, "provider": provider.name, "synced_env": synced}


@app.post("/settings/ai/test")
async def test_ai_settings(payload: Optional[Dict[str, Any]] = None):
    """用当前（或传入临时）配置发一条测试请求，验证连通性。"""
    svc: SettingsService = app.state.settings_service
    if payload:
        tmp = SettingsService()
        tmp.data = {**svc.get(), **{k: v for k, v in payload.items() if k in _AI_ALLOWED}}
        return await tmp.test()
    return await svc.test()


# --------------------------------------------------------------------------
# 本地数据路径配置（期货 / A股 / 港股 / 期权 / 席位）
# --------------------------------------------------------------------------
@app.get("/settings/data")
async def get_data_settings():
    """读取 5 个本地数据根目录配置。"""
    return app.state.data_settings_service.get()


@app.put("/settings/data")
async def put_data_settings(payload: Dict[str, Any]):
    """保存本地数据根目录配置并同步回 .env。"""
    return app.state.data_settings_service.save(payload)


@app.get("/data/files")
async def data_files():
    """扫描本地数据根目录，返回 parquet/csv 文件清单。"""
    return app.state.data_admin_service.list_files()


# --------------------------------------------------------------------------
# 告警通知配置
# --------------------------------------------------------------------------
@app.get("/settings/alert")
async def get_alert_settings():
    """读取告警通知配置。"""
    return app.state.alert_settings_service.get()


@app.put("/settings/alert")
async def put_alert_settings(payload: Dict[str, Any]):
    """保存告警通知配置。"""
    return app.state.alert_settings_service.save(payload)


@app.get("/settings/futures-download")
async def get_futures_download_settings():
    """读取期货数据自动下载配置。"""
    return app.state.futures_download_settings_service.get()


@app.put("/settings/futures-download")
async def put_futures_download_settings(payload: Dict[str, Any]):
    """保存期货数据自动下载配置。"""
    result = app.state.futures_download_settings_service.save(payload)
    # 如果调度时间变更，需要更新调度器
    if "schedule_cron" in payload and app.state.scheduler:
        from .futures_download import _job_futures_auto_download
        sched: QuantMindScheduler = app.state.scheduler
        sched.remove("futures_download")
        sched.register(
            "futures_download",
            _job_futures_auto_download,
            cron=result.get("schedule_cron"),
            timezone="Asia/Shanghai",
            kwargs={"sys_state": {
                "dm": app.state.dm,
                "ee": app.state.ee,
                "lifecycle": app.state.lifecycle,
                "settings": get_settings(),
            }},
        )
    return result


@app.get("/settings/market-update")
async def get_market_update_settings():
    """读取股票市场（A股/港股/美股）自动更新配置。"""
    return app.state.market_update_settings_service.get()


@app.put("/settings/market-update")
async def put_market_update_settings(payload: Dict[str, Any]):
    """保存股票市场自动更新配置，并按需重新注册对应调度任务。"""
    result = app.state.market_update_settings_service.save(payload)
    # 涉及调度时间变更的市场重新注册对应任务
    job_by_key = {"a_stock": "stock_download", "hk_stock": "hk_download", "us_stock": "us_download"}
    market_by_key = {"a_stock": "A", "hk_stock": "HK", "us_stock": "US"}
    if app.state.scheduler:
        from .stock_download import _job_stock_auto_download
        sched: QuantMindScheduler = app.state.scheduler
        for key, job_name in job_by_key.items():
            if key in payload:
                sched.remove(job_name)
                sched.register(
                    job_name,
                    _job_stock_auto_download,
                    cron=result.get(key, {}).get("schedule_cron"),
                    timezone="Asia/Shanghai",
                    kwargs={"sys_state": {
                        "dm": app.state.dm,
                        "ee": app.state.ee,
                        "lifecycle": app.state.lifecycle,
                        "settings": get_settings(),
                        "market_update_settings": app.state.market_update_settings_service,
                    }, "market": market_by_key[key]},
                )
    return result


@app.post("/factor", response_model=FactorResult)
async def factor(req: FactorRequest):
    service: FactorService = app.state.factor_service
    return await service.evaluate(req)


@app.post("/factor/evaluate")
async def factor_evaluate(req: FactorRequest):
    """因子研究统一评估：多标的（symbols≥2）→ 截面契约；单个 → 单标时序契约。

    替代旧 ``/cross-section``（香除独立 CrossSectionService）。
    """
    service: FactorService = app.state.factor_service
    return await service.evaluate_dict(req)


@app.get("/factor/cs-factors")
async def factor_cs_factors():
    """多标（截面）模式可用的截面 Alpha 因子清单（原 /cross-section/factors）。"""
    return {"factors": FactorService.cs_factors()}



@app.post("/factor/expr-eval")
async def factor_expr_eval(req: ExprEvalRequest):
    """表达式截面评估（多标的面板 → IC/RankIC/衰减…）。"""
    service: SearchService = app.state.search_service
    try:
        return await service.evaluate_expression(
            req.expression, req.symbols, req.exchange, req.interval,
            req.start, req.end, req.forward_periods, req.market,
        )
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:  # noqa: BLE001
        _logger.exception("因子表达式评估失败")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/factor/expr-batch")
async def factor_expr_batch(req: ExprEvalBatchRequest):
    """批量评估多个表达式。"""
    service: SearchService = app.state.search_service
    try:
        return await service.evaluate_expressions_batch(
            req.expressions, req.symbols, req.exchange, req.interval,
            req.start, req.end, req.forward_periods, req.market,
        )
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:  # noqa: BLE001
        _logger.exception("批量表达式评估失败")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/factor/search")
async def factor_search(req: FactorSearchRequest):
    """因子迭代搜索（co 链式精炼 / ea 进化 / tot 树状，seed → best + 轨迹）。"""
    service: SearchService = app.state.search_service
    try:
        return await service.search(
            req.seed, req.symbols, req.exchange, req.interval,
            req.start, req.end, algo=req.algo, rounds=req.rounds,
            forward_periods=req.forward_periods, market=req.market,
            val_symbols=req.val_symbols, val_start=req.val_start, val_end=req.val_end,
        )
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:  # noqa: BLE001
        _logger.exception("因子搜索失败")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/factor/dedup")
async def factor_dedup(req: FactorDedupRequest):
    """因子相关性聚类去冗余：输入一批表达式，返回每簇代表性因子。"""
    service: SearchService = app.state.search_service
    try:
        th = max(0.0, min(1.0, req.correlation_threshold))
        return await service.dedup(
            req.expressions, req.symbols, req.exchange, req.interval,
            req.start, req.end, th, req.min_abs_metric,
            req.forward_periods, req.market, req.compute_ic,
        )
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:  # noqa: BLE001
        _logger.exception("因子去冗余失败")
        return JSONResponse(status_code=500, content={"error": str(e)})



@app.post("/strategy/validate")
async def strategy_validate(req: StrategyValidateRequest):
    """策略思想测试：策略思路 →（LLM 预编程 或 预置模板）→ 多品种真实回测 → 门槛 → 有效策略库。

    ``llm_code=True``（默认）：把 ``idea``（策略思想，如布林带回穿规则全文）交给 LLM
    预编程为 CtaTemplate 策略代码，AST 沙箱校验后逐品种回测；
    ``llm_code=False``：用 ``strategy`` 预置模板（momentum/chan_*/bollinger_recover/dual_ma）。
    返回：LLM 生成的代码 + 每品种回测报告 + 净值曲线 + gate 判定 +（promote 时）入库结果。
    """
    service: BacktestService = app.state.backtest_service
    try:
        provider = _llm_provider()
        return await service.validate_strategy(req, provider=provider)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:  # noqa: BLE001
        _logger.exception("策略验证失败")
        return JSONResponse(status_code=500, content={"error": str(e)})


_DRAFT_STATE_FILE = Path("data_cache") / "strategy_workflow_state.json"


@app.get("/strategy/draft/state")
async def get_draft_state():
    """读取 LLM 策略挖掘工作流状态（服务器端持久化，跨会话/刷新/重连不丢失）。"""
    if _DRAFT_STATE_FILE.exists():
        try:
            return json.loads(_DRAFT_STATE_FILE.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            _logger.warning("读取工作流状态失败: %s", e)
    return {}


@app.put("/strategy/draft/state")
async def save_draft_state(payload: Dict[str, Any]):
    """保存 LLM 策略挖掘工作流状态（策略思想/代码/品种/周期等）。"""
    try:
        _DRAFT_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        payload["saved_at"] = datetime.now().isoformat(timespec="seconds")
        _DRAFT_STATE_FILE.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"ok": True}
    except Exception as e:  # noqa: BLE001
        _logger.exception("保存工作流状态失败")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.delete("/strategy/draft/state")
async def clear_draft_state():
    """清除 LLM 策略挖掘工作流状态（配合前端“清空重新开始”）。"""
    try:
        if _DRAFT_STATE_FILE.exists():
            _DRAFT_STATE_FILE.unlink()
        return {"ok": True}
    except Exception as e:  # noqa: BLE001
        _logger.exception("清除工作流状态失败")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/strategy/draft/start")
async def strategy_draft_start(req: StrategyDraftRequest):
    """对话式 LLM 策略编程：生成/修改代码草稿（**只出码不回测**）。

    无状态多轮：前端每次携带完整 history（user=思想/修改意见，assistant=上轮代码），
    服务端不存会话。后台任务跑（LLM 调用 10~60s），前端轮询 status。
    """
    service: BacktestService = app.state.backtest_service
    try:
        provider = _llm_provider()
        progress: Dict[str, Any] = {"current": 0, "total": 1,
                                    "message": "LLM 编程中…"}
        task_id = _submit_task(
            service.draft_strategy_code(
                provider, req.idea,
                [m.model_dump() for m in req.history]),
            progress=progress)
        return {"task_id": task_id, "status": "running"}
    except Exception as e:  # noqa: BLE001
        _logger.exception("策略代码草稿启动失败")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/strategy/draft/status/{task_id}")
async def strategy_draft_status(task_id: str):
    """查询对话式策略编程草稿任务状态（running / success / error / not_found）。"""
    rec = _E2E_TASKS.get(task_id)
    if rec is None:
        return JSONResponse(status_code=404, content={
            "task_id": task_id, "status": "not_found",
            "message": "任务不存在（后端可能已重启），请重新生成。",
        })
    return {
        "task_id": task_id,
        "status": rec["status"],
        "message": rec["message"],
        "progress": rec.get("progress"),
        "result": rec.get("result") if rec["status"] == "success" else None,
    }


@app.post("/strategy/validate/start")
async def strategy_validate_start(req: StrategyValidateRequest):
    """策略验证的**后台非阻塞**入口：立即返回 task_id，长任务不占同步请求。

    与 :func:`strategy_validate` 等价，但跑在后台任务（客户端离开/切页不中断），
    前端通过 ``GET /strategy/validate/status/{task_id}`` 轮询进度。
    用于 LLM 策略挖掘页：几十个品种回测可能跑几分钟，避免同步请求被切页打断。
    """
    service: BacktestService = app.state.backtest_service
    try:
        provider = _llm_provider()
        progress: Dict[str, Any] = {"current": 0, "total": 0,
                                    "message": "任务已提交，正在初始化…"}

        def _prog_cb(msg: str, cur: int = 0, tot: int = 0) -> None:
            progress["message"] = msg
            progress["current"] = int(cur)
            progress["total"] = int(tot)

        task_id = _submit_task(
            service.validate_strategy(req, provider=provider, progress=_prog_cb),
            progress=progress)
        return {"task_id": task_id, "status": "running"}
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:  # noqa: BLE001
        _logger.exception("策略验证后台启动失败")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/strategy/validate/status/{task_id}")
async def strategy_validate_status(task_id: str):
    """查询后台策略验证任务状态（running / success / error / cancelled / not_found）。"""
    rec = _E2E_TASKS.get(task_id)
    if rec is None:
        return JSONResponse(status_code=404, content={
            "task_id": task_id, "status": "not_found",
            "message": "任务不存在（后端可能已重启），请重新运行。",
        })
    return {
        "task_id": task_id,
        "status": rec["status"],
        "message": rec["message"],
        "result": rec["result"],
        "progress": rec.get("progress") or {"current": 0, "total": 0, "message": ""},
    }


@app.delete("/strategy/validate/history/{run_id}")
async def strategy_validate_history_delete(run_id: str):
    """删除一条策略验证历史记录，并尝试删除对应的 lifecycle 入库记录。

    用于清理错误运行（如周期与策略不匹配导致的误入库）。
    """
    service: BacktestService = app.state.backtest_service
    try:
        result = service.delete_validation_history(run_id)
        return result
    except Exception as e:  # noqa: BLE001
        _logger.exception("删除策略验证历史失败")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/strategy/draft/validate")
async def strategy_draft_validate(payload: Dict[str, Any]):
    """沙箱校验策略代码（用于前端编辑后重新检验）。

    请求体：
        {"code": "策略代码字符串"}

    返回：
        {"ok": bool, "error": str, "errors": list}
    """
    code = payload.get("code", "")
    if not code.strip():
        return {"ok": False, "error": "代码为空", "errors": ["代码为空"]}
    try:
        from ..ai.sandbox import compile_strategy
        ok, err, errors = compile_strategy(code, require_base="CtaTemplate")
        return {"ok": ok, "error": err, "errors": errors}
    except Exception as e:  # noqa: BLE001
        _logger.exception("沙箱校验失败")
        return {"ok": False, "error": str(e), "errors": [str(e)]}


@app.post("/factor/backtest")
async def expression_backtest(req: ExpressionBacktestRequest):
    """对挖掘出的 DSL 因子表达式直接做截面多空组合回测（研究→组合闭环）。"""
    service: SearchService = app.state.search_service
    try:
        return await service.backtest_expression(
            req.expression, req.symbols, req.exchange, req.interval,
            req.start, req.end, req.forward_periods, req.n_groups,
            req.long_short, req.cost_rate,
        )
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:  # noqa: BLE001
        _logger.exception("表达式回测失败")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/factor/pipeline")
async def factor_pipeline(req: FactorPipelineRequest):
    """端到端因子挖掘流水线：挖掘 → 去冗余 → 逐因子OOS回测 → 复合组合。"""
    service: SearchService = app.state.search_service
    try:
        th = max(0.0, min(1.0, req.dedup_threshold))
        result = await service.pipeline(
            req.seeds, req.symbols, req.exchange, req.interval,
            req.start, req.end, algo=req.algo, rounds=req.rounds,
            forward_periods=req.forward_periods, market=req.market,
            dedup_threshold=th, min_abs_ic=req.min_abs_ic,
            train_frac=req.train_frac, val_frac=req.val_frac,
            run_composite=req.run_composite,
            composite_scheme=req.composite_scheme,
            n_groups=req.n_groups, long_short=req.long_short,
            cost_rate=req.cost_rate, max_candidates=req.max_candidates,
            net_gate=req.net_gate, max_turnover=req.max_turnover,
            min_net_sharpe=req.min_net_sharpe,
        )
        return result
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:  # noqa: BLE001
        _logger.exception("因子挖掘流水线失败")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/factor/e2e/start")
async def factor_e2e_start(req: FactorE2ERequest):
    """端到端因子研究（异步启动）：AI 证据 → 挖掘 → OOS 复合 alpha → 策略代码。

    立即返回 ``{"task_id": ...}``，任务在后台执行，客户端通过
    ``GET /factor/e2e/status/{task_id}`` 轮询进度/结果。避免长跑超时。
    """
    service: SearchService = app.state.search_service
    try:
        # 先做轻量参数/数据校验，尽早把明显错误抛给客户端（不进入后台任务）
        if not (req.idea or "").strip():
            return JSONResponse(status_code=400, content={"error": "投资想法不能为空"})
        symbols = [s for s in (req.symbols or []) if s and s.strip()]
        if len(symbols) < 2:
            return JSONResponse(status_code=400,
                                content={"error": "端到端流水线至少需要 2 个标的"})
        _e2e_progress: Dict[str, Any] = {"current": 0, "total": 4, "message": "准备中"}
        task_id = _submit_task(service.e2e(
            req, ingest=req.ingest_knowledge,
            bt_service=app.state.backtest_service,
            progress=_e2e_progress), progress=_e2e_progress)
        return {"task_id": task_id, "status": "running"}
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:  # noqa: BLE001
        _logger.exception("端到端任务提交失败")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/factor/e2e/status/{task_id}")
async def factor_e2e_status(task_id: str):
    """查询端到端流水线后台任务状态。

    ``status``: running / success / error / cancelled / not_found。
    任务成功后 ``result`` 携带完整结果（与 /factor/e2e 返回体一致）。
    """
    rec = _E2E_TASKS.get(task_id)
    if rec is None:
        return JSONResponse(status_code=404,
                            content={"task_id": task_id, "status": "not_found",
                                     "message": "任务不存在（可能后端已重启）"})
    done = rec["status"] in ("success", "error", "cancelled")
    return {
        "task_id": task_id,
        "status": rec["status"],
        "message": rec["message"],
        "result": rec["result"] if done else None,
    }


@app.post("/factor/e2e")
async def factor_e2e(req: FactorE2ERequest):
    """端到端因子研究（同步，向后兼容）：AI 证据 → 挖掘 → OOS 复合 alpha → 策略代码。

    注意：同步模式受单次请求超时限制，长跑请改用
    ``POST /factor/e2e/start`` + ``GET /factor/e2e/status/{task_id}``。
    """
    service: SearchService = app.state.search_service
    try:
        return await service.e2e(req, ingest=req.ingest_knowledge,
                                 bt_service=app.state.backtest_service)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:  # noqa: BLE001
        _logger.exception("端到端因子研究失败")
        return JSONResponse(status_code=500, content={"error": str(e)})


# --------------------------------------------------------------------------
# 因子衰减监控（对标 Vibe-Trading strategy-dev-manager）
# --------------------------------------------------------------------------
_FUTURES_CODE_EXCHANGE = {
    # CFFEX 金融期货
    "IF": "CFFEX", "IH": "CFFEX", "IC": "CFFEX", "IM": "CFFEX",
    "TS": "CFFEX", "TF": "CFFEX", "T": "CFFEX",
    # SHFE 上期所
    "RB": "SHFE", "HC": "SHFE", "SS": "SHFE", "WR": "SHFE", "AG": "SHFE",
    "AU": "SHFE", "BU": "SHFE", "CU": "SHFE", "AL": "SHFE", "ZN": "SHFE",
    "PB": "SHFE", "NI": "SHFE", "SN": "SHFE", "SP": "SHFE", "AO": "SHFE",
    "FU": "SHFE", "RU": "SHFE",
    # INE 能源中心
    "SC": "INE", "NR": "INE", "LU": "INE", "EC": "INE", "BC": "INE",
    # DCE 大商所
    "I": "DCE", "J": "DCE", "JM": "DCE", "M": "DCE", "Y": "DCE", "P": "DCE",
    "L": "DCE", "V": "DCE", "PP": "DCE", "EG": "DCE", "EB": "DCE", "PG": "DCE",
    "LH": "DCE", "RR": "DCE", "BB": "DCE", "FB": "DCE", "A": "DCE", "C": "DCE",
    "CS": "DCE", "JD": "DCE",
    # CZCE 郑商所
    "MA": "CZCE", "TA": "CZCE", "FG": "CZCE", "SA": "CZCE", "UR": "CZCE",
    "OI": "CZCE", "PK": "CZCE", "CF": "CZCE", "SR": "CZCE", "RM": "CZCE",
    "AP": "CZCE", "SF": "CZCE", "SM": "CZCE", "CY": "CZCE", "ZC": "CZCE",
    "SH": "CZCE", "PF": "CZCE", "PX": "CZCE", "WH": "CZCE", "RS": "CZCE",
    # GFEX 广期所
    "SI": "GFEX", "LC": "GFEX",
}


def _resolve_symbol_exchange(symbol: str, dm) -> Tuple[str, str]:
    """把裸代码解析为 (code, exchange)。

    优先读 ``code.EX`` 自带后缀 → 其次在本地行情仓库探测 ``{code}.{EXCH}.1d.parquet``
    → 最后按期货代码映射兑底。
    """
    s = (symbol or "").strip()
    if not s:
        return s, "SHFE"
    if "." in s:
        head, _, exch = s.rpartition(".")
        if head and exch:
            return head.strip(), exch.strip().upper()
    # 探测本地行情仓库（真实可用数据为准）
    heads = [s, s.upper()]
    if dm is not None and dm.disk_cache is not None:
        for h in heads:
            for exch in ("CFFEX", "SHFE", "DCE", "CZCE", "INE", "GFEX", "SSE", "SZSE", "HKEX"):
                if (dm.disk_cache.root / f"{h}.{exch}.1d.parquet").exists():
                    return h, exch
    # 期货代码映射兑底
    code = s.upper().rstrip("0123456789")
    exch = _FUTURES_CODE_EXCHANGE.get(code) or _FUTURES_CODE_EXCHANGE.get(s.upper())
    return s, (exch or "SHFE")


@app.get("/factors/decay")
async def factors_decay_list():
    """列出所有已扫描因子的衰减状态。"""
    scanner: FactorDecayScanner = app.state.decay_scanner
    return {"factors": [m.to_dict() for m in scanner.list_records()]}


@app.post("/factors/decay/scan")
async def factors_decay_scan():
    """触发全量因子衰减扫描（基于知识库中已沉淀因子的**真实 IC 时序**）。

    对每个 active 因子：用其 expression 在真实行情面板上逐日截面 rank IC
    （无 expression/标的或真实数据不足时标记为无法评估，不伪造时序）。
    """
    import pandas as pd

    from ..data.feed.base import HistoryRequest
    from ..research.factors.alpha_cs import Panel
    from ..research.pipeline import _daily_ic_series

    scanner: FactorDecayScanner = app.state.decay_scanner
    ks = app.state.knowledge_service.store
    dm = app.state.dm
    items = ks.list_items(kind="factor", limit=500)
    active_factors = [it for it in items if it.get("metadata", {}).get("status") == "active"]

    factor_ic_map: Dict[str, pd.Series] = {}
    current_states: Dict[str, FactorState] = {}
    for it in active_factors[:30]:  # 限制数量避免超时
        fid = it["kb_id"]
        meta = it.get("metadata", {})
        expr = str(meta.get("expression") or "").strip()
        symbols = [s for s in (meta.get("symbols") or []) if s]

        empty = pd.Series(dtype=float)
        factor_ic_map[fid] = empty
        current_states[fid] = FactorState.ACTIVE
        if not expr or len(symbols) < 2:
            continue  # 无表达式/标的 → 无法评估，compute_metrics 会给出"数据不足"

        # 仅使用本地行情仓库真实落盘数据（不上网、不 fallback 到 mock），
        # 保证衰减判定全部基于真实行情，且避免网络超时拖慢整仓扫描。
        cached = []
        for s in symbols:
            if dm is not None and dm.disk_cache is not None:
                _code, exch = _resolve_symbol_exchange(s, dm)
                if (dm.disk_cache.root / f"{_code}.{exch}.1d.parquet").exists():
                    cached.append((s, exch))
        if len(cached) < 2:
            continue  # 真实仓库中不足 2 个可评估标的

        tasks = [dm.get_bar_data(
            HistoryRequest(symbol=s, exchange=Exchange(exch), interval=Interval.DAILY),
        ) for s, exch in cached]
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception:  # noqa: BLE001
            continue
        bars_by_symbol: Dict[str, list] = {}
        for (s, _exch), res in zip(cached, results):
            if isinstance(res, Exception) or not res:
                continue
            bars_by_symbol[s] = res
        if len(bars_by_symbol) < 2:
            continue  # 真实数据不足以构面面板
        try:
            panel = Panel.from_bars(bars_by_symbol)
            # 衰减检测只需近端窗口（history 252 / recent 60），裁到最近 ~300 个交易日，
            # 避免对全历史（数千日）逐日 spearman 计算导致接口卡死。
            _n = min(300, len(panel.dates))
            if _n < 20:
                continue
            panel = dataclasses.replace(
                panel,
                close=panel.close.iloc[-_n:], open=panel.open.iloc[-_n:],
                high=panel.high.iloc[-_n:], low=panel.low.iloc[-_n:],
                volume=panel.volume.iloc[-_n:], amount=panel.amount.iloc[-_n:],
            )
            ic_list = _daily_ic_series(expr, panel, forward_periods=1)
        except Exception:  # noqa: BLE001
            continue
        valid = [x for x in (ic_list or []) if x is not None and x == x]
        if len(valid) < 20:
            continue  # 真实 IC 样本过少，不做衰减判定
        if ic_list and len(ic_list) == len(panel.dates):
            factor_ic_map[fid] = pd.Series(ic_list, index=panel.dates)

    results = scanner.scan_all(factor_ic_map, current_states)
    # 未评估（空/不足）的因子打上明确标记，避免与正常扫描混淆
    for m in results:
        if m.n_samples_history == 0:
            m.notes = ["无真实 IC 时序（缺表达式/标的或真实数据不足），未做衰减评估"]
    return {
        "scanned": len(results),
        "factors": [m.to_dict() for m in results],
    }


# --------------------------------------------------------------------------
# 知识库（knowledge）：因子 / 策略 / 研究日志 / 方法论 沉淀 + 检索 + 列表
# --------------------------------------------------------------------------
@app.post("/knowledge/ingest")
async def knowledge_ingest(req: KnowledgeIngestRequest):
    """手动写入一条知识库记录（factor | strategy | research_log | methodology）。"""
    service: KnowledgeService = app.state.knowledge_service
    try:
        return service.ingest(req)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.post("/knowledge/search")
async def knowledge_search(req: KnowledgeSearchRequest):
    """知识库轻量关键词检索。"""
    service: KnowledgeService = app.state.knowledge_service
    return service.search(req)


@app.get("/knowledge")
async def knowledge_list(
    kind: Optional[str] = Query(None), limit: int = Query(50, ge=1, le=500),
):
    """知识库列表（最新在前），可按 kind 过滤。"""
    service: KnowledgeService = app.state.knowledge_service
    return service.list(kind=kind, limit=limit)


@app.get("/runs")
async def runs_list(limit: int = Query(30, ge=1, le=500)):
    """端到端运行历史（最新在前）：run_id/idea/复合统计/AI brief/status。"""
    return {"runs": KnowledgeStore().list_runs(limit=limit)}


@app.get("/runs/{run_id}")
async def runs_detail(run_id: str):
    """单次运行详情：run 摘要 + 该 run 的全部因子试验明细（factor_trials）。"""
    ks = KnowledgeStore()
    run = ks.get_run(run_id)
    if run is None:
        return JSONResponse(status_code=404, content={"error": f"运行不存在: {run_id}"})
    run["trials"] = ks.trials_for_run(run_id)
    return run


@app.post("/backtest")
async def backtest(req: BacktestRequest):
    service: BacktestService = app.state.backtest_service
    result = await service.run_backtest(req)
    if "error" not in result:
        # 若该策略已存在于生命周期表，回填真实回测指标 + AI 判读（失败不阻断返回）
        try:
            if KnowledgeStore().get_strategy_lifecycle(req.strategy) is not None:
                if "lifecycle" not in result:  # 勿覆盖既有 key
                    result["lifecycle"] = {}
                _judged = await _persist_backtest_lifecycle(req.strategy, result.get("risk_xray"))
                _rec = KnowledgeStore().get_strategy_lifecycle(req.strategy)
                if _rec is not None:
                    result["lifecycle"] = {
                        "state": _rec.get("state"),
                        "status": _rec.get("status"),
                        "reason": _rec.get("reason"),
                        "sharpe": _rec.get("sharpe"),
                        "max_drawdown": _rec.get("max_drawdown"),
                    }
        except Exception as exc:  # noqa: BLE001
            _logger.warning("策略回测生命周期接线失败(%s): %s", req.strategy, exc)
    return result


@app.post("/walkforward")
async def walkforward(req: WalkForwardRequest):
    service: BacktestService = app.state.backtest_service
    return await service.run_walkforward(req)


@app.get("/strategies", response_model=List[StrategyInfo])
async def strategies():
    return app.state.backtest_service.list_strategies()


@app.post("/strategies/register")
async def strategy_register(req: StrategyRegisterRequest):
    """注册 AI 生成策略入模拟盘：沙箱校验 + 实例池 + 生命周期(IDEA->RESEARCH)。"""
    bs: BacktestService = app.state.backtest_service
    ok, err, info = bs.register_generated_strategy(req.name, req.code)
    if not ok:
        return {"ok": False, "error": err or "注册失败"}
    lc: LifecycleManager = app.state.lifecycle
    rec = lc.get_or_create(req.name)
    try:
        lc.promote(req.name, LifecycleState.RESEARCH, metrics={}, note=f"AI生成:{req.idea[:40]}")
    except Exception:  # noqa: BLE001 晋升失败不阻断注册
        pass
    # 关联 run_id + 写策略元信息（只在新行/尚无 run_id 时 upsert，避免覆盖已有回测指标）
    try:
        ks = KnowledgeStore()
        existing = ks.get_strategy_lifecycle(req.name)
        if existing is None or not (existing.get("run_id") or ""):
            ks.upsert_strategy_lifecycle(
                strategy_id=req.name,
                run_id=req.run_id or "",
                idea=req.idea,
                state="RESEARCH",
                source="AI生成",
                code=req.code,
                code_safe=True,       # 沙箱已通过
                symbols=None,
            )
            if req.composite_fwd_ic is not None:
                ks.update_strategy_state(req.name, composite_fwd_ic=req.composite_fwd_ic)
    except Exception:  # noqa: BLE001 落库失败不阻断注册
        _logger.warning("策略注册落库失败: %s", req.name)
    info["lifecycle"] = rec.state.value
    return {
        "ok": True,
        "strategy_id": req.name,
        "run_id": req.run_id or "",
        "info": info,
        "lifecycle": info.get("lifecycle"),
        "state": rec.state.value,
    }


@app.post("/strategies/paper")
async def strategy_paper(req: PaperRunRequest):
    """模拟盘实跑：把已注册/内置策略跑 PaperEngine 历史回放，晋升生命周期到 PAPER。"""
    bs: BacktestService = app.state.backtest_service
    result = await bs.run_paper(req)
    if "error" in result:
        return JSONResponse(status_code=400, content=result)

    lc: LifecycleManager = app.state.lifecycle
    metrics = result.get("metrics", {})
    try:
        lc.promote(
            req.strategy,
            LifecycleState.PAPER,
            metrics=metrics,
            note=f"模拟盘实跑: {result['vt_symbol']} {result['bars']}bars {result['trade_count']}笔",  # noqa: E501
        )
    except Exception:  # noqa: BLE001 晋升失败不阻断实跑返回
        pass
    rec = lc.get_or_create(req.strategy)
    # 策略级 AI 判读 + 经验 brief 落库（失败不阻断返回）
    _persisted = await _persist_paper_lifecycle(req.strategy, metrics)
    if "lifecycle" not in result:  # 勿覆盖既有 key
        result["lifecycle"] = {}
    result["lifecycle"] = {
        "state": rec.state.value,
        "status": (_persisted or {}).get("status"),
        "reason": (_persisted or {}).get("reason"),
    }
    return result


@app.post("/order")
async def order(req: OrderRequestSchema):
    service: LifecycleService = app.state.lifecycle_service
    result = await service.place_order(req)
    if "error" in result:
        return JSONResponse(status_code=400, content=result)
    return result


@app.get("/orders")
async def orders():
    """查询本会话（内存）的订单历史。"""
    return app.state.lifecycle_service.list_orders()


@app.get("/positions")
async def positions():
    """查询当前净持仓（由订单台账推导）。"""
    return app.state.lifecycle_service.list_positions()


@app.delete("/order/{order_id}")
async def cancel_order(order_id: str):
    """撤销一笔手动订单。"""
    return app.state.lifecycle_service.cancel_order(order_id)


@app.post("/lifecycle")
async def lifecycle(req: LifecycleRequest):
    service: LifecycleService = app.state.lifecycle_service
    return await service.promote(req)


# --------------------------------------------------------------------------
# 任务调度器
# --------------------------------------------------------------------------
@app.get("/scheduler")
async def scheduler_list():
    """列出已注册的周期任务及其下次触发时间。"""
    sched: QuantMindScheduler = app.state.scheduler
    return {
        "available": sched.available,
        "running": sched._sched.running if sched._sched else False,
        "jobs": sched.list_jobs(),
    }


@app.post("/scheduler/start")
async def scheduler_start():
    """启动调度器（若未运行）。"""
    sched: QuantMindScheduler = app.state.scheduler
    if not sched.available:
        return {"ok": False, "error": "apscheduler 未安装"}
    sched.start()
    return {"ok": True, "running": sched._sched.running}


@app.post("/scheduler/stop")
async def scheduler_stop():
    """停止调度器。"""
    sched: QuantMindScheduler = app.state.scheduler
    if not sched.available:
        return {"ok": False, "error": "apscheduler 未安装"}
    sched.stop()
    return {"ok": True, "running": False}


# --------------------------------------------------------------------------
# 数据质量
# --------------------------------------------------------------------------
@app.get("/data/quality")
async def data_quality(
    symbol: str = Query(...), exchange: str = Query("SHFE"),
    interval: str = Query("1d"), start: str = Query(None), end: str = Query(None),
    freshness_days: int = Query(None, ge=1, le=365),
):
    """数据质量体检：间隙 / 异常尖峰 / 换月跳变 / 新鲜度 + 0-100 评分。"""
    service: DataService = app.state.data_service
    try:
        return await service.quality_report(symbol, exchange, interval, start, end, freshness_days)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


# --------------------------------------------------------------------------
# 风控
# --------------------------------------------------------------------------
@app.get("/risk/profiles")
async def risk_profiles():
    """风控限额档位（default / conservative / unlimited）及字段中文标签。"""
    service: RiskService = app.state.risk_service
    return service.profiles()


@app.post("/risk/check")
async def risk_check(req: RiskCheckRequest):
    """委托风控预检：不真正下单，只跑一遍闸门看会不会被拦。"""
    service: RiskService = app.state.risk_service
    result = service.check_order(req)
    if "error" in result:
        return JSONResponse(status_code=400, content=result)
    return result


@app.get("/risk/calendar")
async def risk_calendar(
    day: str = Query(None), symbol: str = Query("rb0"),
    exchange: str = Query("SHFE"), horizon: int = Query(14, ge=1, le=60),
):
    """交易日历：交易日 / 夜盘 / 当前时段 / 未来 N 天排期。"""
    service: RiskService = app.state.risk_service
    result = service.calendar_info(day, symbol, exchange, horizon)
    if "error" in result:
        return JSONResponse(status_code=400, content=result)
    return result


# --------------------------------------------------------------------------
# 参数寻优
# --------------------------------------------------------------------------
@app.get("/optimize/space")
async def optimize_space(strategy: str = Query("dual_ma")):
    """返回某策略的推荐搜索空间与可选目标指标。"""
    return {
        "strategy": strategy,
        "param_space": app.state.optimize_service.param_space_of(strategy),
        "metrics": OptimizeService.metrics(),
    }


@app.post("/optimize")
async def optimize(req: OptimizeRequest):
    """网格搜索最优参数。"""
    service: OptimizeService = app.state.optimize_service
    try:
        return await service.optimize(req)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.post("/optimize/optuna")
async def optimize_optuna(req: OptimizeRequest):
    """Optuna 贝叶斯寻优（强制 method=optuna）。"""
    req = req.model_copy(update={"method": "optuna"})
    service: OptimizeService = app.state.optimize_service
    try:
        return await service.optimize(req)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


# --------------------------------------------------------------------------
# LLM 策略挖掘
# --------------------------------------------------------------------------
@app.post("/strategy-mining/architect")
async def strategy_mining_architect(req: StrategyMiningRequest):
    """LLM 策略架构师：从因子设计策略规格。"""
    service: StrategyMiningService = app.state.strategy_mining_service
    return await service.architect(req)


@app.post("/strategy-mining/auto-backtest")
async def strategy_mining_auto_backtest(req: AutoBacktestRequest):
    """自动回测循环：编译 → 回测 → 评估 → 调整（迭代）。"""
    service: StrategyMiningService = app.state.strategy_mining_service
    return await service.auto_backtest(req)


# --------------------------------------------------------------------------
# 席位因子（商品期货独有）
# --------------------------------------------------------------------------
@app.get("/seat-factors")
async def seat_factor_list():
    """期货席位因子 F1-F8 清单与简介。"""
    service: SeatService = app.state.seat_service
    return {"factors": service.list_factors()}


@app.post("/seat-factor", response_model=SeatFactorResult)
async def seat_factor(req: SeatFactorRequest):
    """按品种加载席位 CSV → 计算所选因子 → 评估 IC / 分组收益。"""
    service: SeatService = app.state.seat_service
    return await service.compute(req)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        await manager.send_personal(ws, {"type": "hello", "msg": "QuantMind WebSocket 已连接"})
        while True:
            data = await ws.receive_json()
            await manager.send_personal(ws, {"type": "echo", "data": data})
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        # 非法 JSON（JSONDecodeError）/对端异常断开（RuntimeError）等也会跳出循环，
        # 必须清理，否则死连接永久残留 manager.active 导致 broadcast 逐次变慢
        _logger.debug("WebSocket 异常断开: %s", exc)
    finally:
        manager.disconnect(ws)
