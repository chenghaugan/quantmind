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
import logging
import math
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import uuid

from ..config import get_settings
from ..data import build_default_registry, DataManager, TimescaleStore, InMemoryStore
from ..core.engine import EventEngine
from ..core.event import Event, EventType
from ..ai import build_provider
from ..paper.promotion import LifecycleManager
from ..monitoring import Notifier
from ..research import FactorRegistry

from .schemas import (
    ResearchRequest, ResearchResult, FactorRequest, FactorResult,
    BacktestRequest, WalkForwardRequest, StrategyInfo, OrderRequestSchema, LifecycleRequest,
    OptimizeRequest, CrossSectionRequest, RiskCheckRequest,
    SeatFactorRequest, SeatFactorResult, DataDownloadRequest,
)
from .ws import manager
from .services import (
    DataService, FactorService, BacktestService, LifecycleService, ResearchService,
    RiskService, OptimizeService, CrossSectionService, SettingsService, SeatService,
    DataSettingsService, DataAdminService, AlertSettingsService,
)
from .logging_config import setup_api_logger
from .routes_auth import router as auth_router

_logger = setup_api_logger("INFO")

# 全局实例（用于 WebSocket 广播）
_ee: Optional[EventEngine] = None

# AI 设置允许更新的字段白名单
_AI_ALLOWED = {"provider", "api_key", "base_url", "model", "temperature"}


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


def _broadcast(e: Event) -> None:
    if _ee is None:
        return
    msg = {"type": e.type.value, "data": _jsonable(e.data)}
    try:
        asyncio.ensure_future(manager.broadcast(msg))
    except RuntimeError:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _ee
    settings = get_settings()
    registry = build_default_registry()
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

    _ee = EventEngine()
    await _ee.start()
    _ee.register_general(_broadcast)
    notifier = Notifier()
    notifier.attach(_ee)
    lifecycle_mgr = LifecycleManager()
    settings_service = SettingsService()
    provider = settings_service.rebuild_provider()

    # 初始化 Service 层
    app.state.data_service = DataService(dm)
    app.state.factor_service = FactorService(dm)
    app.state.backtest_service = BacktestService(dm, _ee)
    app.state.lifecycle_service = LifecycleService(lifecycle_mgr, _ee)
    app.state.research_service = ResearchService(provider)
    app.state.risk_service = RiskService()
    app.state.optimize_service = OptimizeService(dm)
    app.state.cross_section_service = CrossSectionService(dm)
    app.state.settings_service = settings_service
    app.state.seat_service = SeatService(dm)
    data_settings = DataSettingsService()
    app.state.data_settings_service = data_settings
    app.state.data_admin_service = DataAdminService(dm, data_settings)
    app.state.alert_settings_service = AlertSettingsService()

    app.state.dm = dm
    app.state.ee = _ee
    app.state.lifecycle = lifecycle_mgr

    yield

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


@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """404 处理"""
    return JSONResponse(status_code=404, content={"error": "接口不存在"})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """参数校验失败统一格式（覆盖 FastAPI 默认的 422 响应）"""
    return JSONResponse(
        status_code=422,
        content={"error": "参数校验失败", "detail": exc.errors()},
    )

# 注册路由
app.include_router(auth_router)


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
    app.state.research_service.provider = provider
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


@app.post("/data/download")
async def data_download(req: DataDownloadRequest):
    """下载指定标的数据并入库（拉取 + 回写持久存储）。"""
    try:
        return await app.state.data_admin_service.download(
            req.symbol, req.exchange, req.interval, req.start or "", req.end or ""
        )
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


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


@app.post("/factor", response_model=FactorResult)
async def factor(req: FactorRequest):
    service: FactorService = app.state.factor_service
    return await service.evaluate(req)


@app.post("/backtest")
async def backtest(req: BacktestRequest):
    service: BacktestService = app.state.backtest_service
    return await service.run_backtest(req)


@app.post("/walkforward")
async def walkforward(req: WalkForwardRequest):
    service: BacktestService = app.state.backtest_service
    return await service.run_walkforward(req)


@app.get("/strategies", response_model=List[StrategyInfo])
async def strategies():
    return BacktestService.list_strategies()


@app.post("/order")
async def order(req: OrderRequestSchema):
    service: LifecycleService = app.state.lifecycle_service
    return await service.place_order(req)


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
        "param_space": OptimizeService.param_space_of(strategy),
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
# 截面研究
# --------------------------------------------------------------------------
@app.get("/cross-section/factors")
async def cross_section_factors():
    """可用截面 Alpha 因子清单。"""
    return {"factors": CrossSectionService.factors()}


@app.post("/cross-section")
async def cross_section(req: CrossSectionRequest):
    """多标的截面因子 IC 评估 + 多空组合回测。"""
    service: CrossSectionService = app.state.cross_section_service
    try:
        return await service.run(req)
    except (ValueError, KeyError) as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


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
        manager.disconnect(ws)
