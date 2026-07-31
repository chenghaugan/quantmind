"""FastAPI 应用（Web 统一入口后端，接入真实引擎）。

功能：
  - /research  AI 研究（idea -> 规格/因子/策略代码）
  - /factor    因子计算 + 有效性评估（IC/IR/衰减/分位收益）
  - /backtest  回测 / 模拟 / 实盘（同一策略，按 mode 切换路线）
  - /strategies 可用策略清单与运行
  - /order     手动下单（广播委托意图，驱动监控）
  - /lifecycle 策略生命周期晋升闸门
  - /ws        WebSocket 实时推送引擎事件（bar/signal/position/trade/account/log）
"""
from __future__ import annotations

import asyncio
import dataclasses
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..config import get_settings
from ..data import build_default_registry, DataManager, TimescaleStore, InMemoryStore
from ..data.feed.base import HistoryRequest
from ..core.constant import Exchange, Interval, Direction, Offset
from ..core.engine import EventEngine
from ..core.contracts import default_size
from ..core.object import BarData, OrderData, Status
from ..core.event import Event, EventType
from ..research import (
    MomentumFactor, FactorRegistry, FactorEvaluator,
    eval_factor_expression, build_factor_registry, FactorSpec,
)
from ..strategy import run_strategy, MultiFactorStrategy, DualMaStrategy, VolTargetStrategy, PairTradingStrategy
from ..ai import ResearchAgent, build_provider
from ..paper.promotion import LifecycleManager, LifecycleState
from ..monitoring import Notifier
from .schemas import (
    BarOut, ResearchRequest, ResearchResult, FactorRequest, FactorResult,
    BacktestRequest, StrategyInfo, OrderRequestSchema, LifecycleRequest,
)
from .ws import manager

_logger = logging.getLogger("quantmind.api")

dm: Optional[DataManager] = None
_ee: Optional[EventEngine] = None
_lifecycle: Optional[LifecycleManager] = None
_provider = None

_STRATEGY_MAP = {
    "dual_ma": DualMaStrategy,
    "multifactor": MultiFactorStrategy,
    "vol_target": VolTargetStrategy,
    "pair": PairTradingStrategy,
}


def _jsonable(o: Any) -> Any:
    """把事件数据转成可 JSON 序列化结构。"""
    if isinstance(o, datetime):
        return o.isoformat()
    if isinstance(o, Enum):
        return o.value
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


def _default_size(vt_symbol: str) -> float:
    return default_size(vt_symbol)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global dm, _ee, _lifecycle, _provider
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
    _lifecycle = LifecycleManager()
    _provider = build_provider(settings.llm_provider)

    app.state.dm = dm
    app.state.ee = _ee
    app.state.lifecycle = _lifecycle
    yield
    if dm:
        await dm.close()
    if _ee:
        await _ee.stop()


app = FastAPI(title="QuantMind API", version="0.2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/")
async def root():
    return {"name": "QuantMind API", "version": "0.2.0", "docs": "/docs"}


@app.get("/health")
async def health():
    return {"status": "ok", "feeds": dm.registry.list_feeds() if dm else []}


@app.get("/feeds")
async def feeds():
    return {"feeds": dm.registry.list_feeds() if dm else []}


@app.get("/factors")
async def list_factors():
    reg = FactorRegistry()
    return {"factors": reg.list_factors()}


@app.get("/data", response_model=List[BarOut])
async def get_data(
    symbol: str = Query(...), exchange: str = Query("SHFE"),
    interval: str = Query("1d"), start: str = Query(None), end: str = Query(None),
):
    exch = Exchange(exchange.upper())
    interv = Interval(interval)
    req = HistoryRequest(symbol=symbol, exchange=exch, interval=interv,
                         start=datetime.fromisoformat(start) if start else None,
                         end=datetime.fromisoformat(end) if end else None)
    bars = await dm.get_bar_data(req)
    return [BarOut(symbol=b.symbol, exchange=b.exchange.value, datetime=b.datetime.isoformat(),
                   interval=b.interval.value, open=b.open_price, high=b.high_price,
                   low=b.low_price, close=b.close_price, volume=b.volume) for b in bars]


@app.post("/research", response_model=ResearchResult)
async def research(req: ResearchRequest):
    agent = ResearchAgent(_provider)
    out = await agent.research(req.idea, req.asset_class or "")
    return ResearchResult(
        idea=out.spec.idea, asset_class=out.spec.asset_class, hypothesis=out.spec.hypothesis,
        suggested_factors=out.spec.suggested_factors, risk_notes=out.spec.risk_notes,
        generated_factors=[{"name": f.name, "kind": f.kind, "window": f.window, "weight": f.weight}
                           for f in out.factors],
        code_safe=out.code_safe, code_errors=out.code_errors,
    )


@app.post("/factor", response_model=FactorResult)
async def factor(req: FactorRequest):
    exch = Exchange(req.exchange.upper())
    interv = Interval(req.interval)
    bars = await dm.get_bar_data(HistoryRequest(symbol=req.symbol, exchange=exch, interval=interv))
    if not bars:
        return FactorResult(factor_name=req.factor, n_samples=0)
    # 解析因子
    if req.expression:
        from ..research.factors.base import bars_to_df
        df = bars_to_df(bars)
        series = eval_factor_expression(req.expression, df)
        name = req.expression
    else:
        f = _resolve_factor(req.factor, req.window)
        series = f.compute(bars)
        name = f.meta.name
    series.name = name
    rep = FactorEvaluator().evaluate(series, bars, forward_periods=req.forward_periods)
    return FactorResult(
        factor_name=rep.factor_name, ic_mean=rep.ic_mean, ir=rep.ir, ic_std=rep.ic_std,
        ic_positive_ratio=rep.ic_positive_ratio, ic_decay=rep.ic_decay,
        top_quantile_return=rep.top_quantile_return, long_short_return=rep.long_short_return,
        n_samples=rep.n_samples,
    )


@app.post("/backtest")
async def backtest(req: BacktestRequest):
    strat_class = _STRATEGY_MAP.get(req.strategy, MultiFactorStrategy)
    vt = f"{req.symbol}.{req.exchange.upper()}"
    bars = await dm.get_bar_data(HistoryRequest(symbol=req.symbol, exchange=Exchange(req.exchange.upper()),
                                                interval=Interval("1d")))
    if not bars:
        return {"error": "无数据"}
    sizes = dict(req.sizes) or {vt: _default_size(vt)}
    result = await asyncio.to_thread(
        run_strategy, req.mode, strat_class, vt, dict(req.setting), bars,
        _ee, sizes, req.gateway,
    )
    return result


@app.get("/strategies", response_model=List[StrategyInfo])
async def strategies():
    return [
        StrategyInfo(name="dual_ma", description="双均线趋势/动量策略",
                     parameters={"fast": 5, "slow": 20, "size": 1, "max_pos": 1.0}),
        StrategyInfo(name="multifactor", description="多因子组合策略（动量+均值回复+波动率）",
                     parameters={"specs": "see research", "threshold": 0.3, "size": 1, "max_pos": 1.0}),
        StrategyInfo(name="vol_target", description="全天候风格：波动率目标+动量过滤(单标的风险平价)",
                     parameters={"lookback": 20, "target_vol": 0.20, "momentum_win": 60, "size": 1, "max_pos": 1.0}),
        StrategyInfo(name="pair", description="配对交易：价差合成标的 z-score 均值回复",
                     parameters={"window": 30, "entry_z": 1.5, "exit_z": 0.3, "size": 1, "max_pos": 1.0}),
    ]


@app.post("/order")
async def order(req: OrderRequestSchema):
    if _ee is None:
        return {"ok": False, "msg": "引擎未启动"}
    direction = Direction(req.direction)
    offset = Offset(req.offset)
    sym, exch = req.vt_symbol.rsplit(".", 1)
    od = OrderData(symbol=sym, exchange=Exchange(exch), order_id="WEB-MANUAL",
                   direction=direction, offset=offset, price=req.price, volume=req.volume,
                   status=Status.SUBMITTED)
    _ee.put_event(EventType.EVENT_ORDER, od)
    from ..core.object import LogData
    _ee.put_event(EventType.EVENT_LOG, LogData(msg=f"手动下单: {req.vt_symbol} {req.direction} x{req.volume}"))
    return {"ok": True, "order": req.vt_symbol}


@app.post("/lifecycle")
async def lifecycle(req: LifecycleRequest):
    if _lifecycle is None:
        return {"ok": False}
    try:
        to = LifecycleState(req.to)
    except ValueError:
        return {"ok": False, "msg": f"非法状态: {req.to}"}
    ok, reasons = _lifecycle.promote(req.strategy_id, to, req.metrics, req.note)
    return {"ok": ok, "state": _lifecycle.get_or_create(req.strategy_id).state.value, "reasons": reasons}


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


def _resolve_factor(name: str, window: int):
    """按名称解析内置因子（支持 'momentum_20' / 'mean_reversion' 等形式）。"""
    reg = FactorRegistry()
    try:
        return reg.get(name)
    except KeyError:
        pass
    if "_" in name:
        kind, _, w = name.rpartition("_")
        try:
            w = int(w)
        except ValueError:
            w = window
        from ..research.technical import build_factor
        return build_factor(kind, w)
    from ..research.technical import build_factor
    return build_factor(name, window)
