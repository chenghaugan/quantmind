"""事件引擎与主引擎（参考 vnpy.trader.engine，升级为 asyncio）。

- ``EventEngine``：异步队列 + 类型化路由，驱动策略与 WebSocket 推送。
- ``MainEngine``：聚合事件引擎、网关、数据管理，对外提供高层 API。
- ``OmsEngine``：从事件流维护持仓/账户。
- ``LogEngine``：统一日志输出（可被 Web/CLI 订阅）。
"""
from __future__ import annotations

import asyncio
import logging
import threading
from typing import Awaitable, Callable, Dict, List, Optional

from .event import Event, EventType
from .gateway import BaseGateway
from .object import LogData

EventHandler = Callable[[Event], None]

_logger = logging.getLogger("quantmind.engine")


class EventEngine:
    """异步事件引擎。"""

    def __init__(self, queue_max: int = 100_000) -> None:
        self._queue: asyncio.Queue = asyncio.Queue(queue_max)
        self._handlers: Dict[EventType, List[EventHandler]] = {}
        self._general_handlers: List[EventHandler] = []
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ---- 注册 ----
    def register(self, event_type: EventType, handler: EventHandler) -> None:
        self._handlers.setdefault(event_type, []).append(handler)

    def unregister(self, event_type: EventType, handler: EventHandler) -> None:
        if event_type in self._handlers:
            try:
                self._handlers[event_type].remove(handler)
            except ValueError:
                pass

    def register_general(self, handler: EventHandler) -> None:
        self._general_handlers.append(handler)

    # ---- 入队 ----
    def put(self, event: Event) -> None:
        """线程安全入队：跨线程（如 to_thread 工作线程）投递经事件循环调度。"""
        loop = self._loop
        if (loop is not None and loop.is_running()
                and threading.current_thread() is not threading.main_thread()):
            try:
                loop.call_soon_threadsafe(self._put_nowait, event)
                return
            except RuntimeError:
                pass  # 事件循环已关闭：退回直接入队
        self._put_nowait(event)

    def _put_nowait(self, event: Event) -> None:
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            _logger.warning("事件队列已满，丢弃事件 %s", event.type)

    def put_event(self, event_type: EventType, data: object) -> None:
        self.put(Event(event_type, data))

    # ---- 生命周期 ----
    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._loop = asyncio.get_running_loop()
        self._task = asyncio.create_task(self._run(), name="event-engine")
        _logger.info("EventEngine 启动")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        _logger.info("EventEngine 停止")

    async def _run(self) -> None:
        while self._running:
            event = await self._queue.get()
            await self._process(event)

    async def _process(self, event: Event) -> None:
        for handler in self._handlers.get(event.type, []):
            await self._maybe_await(handler, event)
        for handler in self._general_handlers:
            await self._maybe_await(handler, event)

    @staticmethod
    async def _maybe_await(handler: EventHandler, event: Event) -> None:
        try:
            result = handler(event)
            if asyncio.iscoroutine(result):
                await result
        except Exception as exc:  # noqa: BLE001
            _logger.exception("事件处理异常: %s", exc)


class LogEngine:
    """日志引擎：订阅 EVENT_LOG 并输出；其余模块用 ``write_log`` 发日志。"""

    def __init__(self, event_engine: EventEngine) -> None:
        self.event_engine = event_engine

    def write_log(self, msg: str, level: int = logging.INFO) -> None:
        self.event_engine.put_event(
            EventType.EVENT_LOG, LogData(msg=msg, level=level)
        )


class OmsEngine:
    """订单/持仓管理引擎（从事件流聚合）。"""

    def __init__(self, event_engine: EventEngine) -> None:
        self.event_engine = event_engine
        self.positions: Dict[str, object] = {}
        self.accounts: Dict[str, object] = {}
        event_engine.register(EventType.EVENT_POSITION, self._on_position)
        event_engine.register(EventType.EVENT_ACCOUNT, self._on_account)

    def _on_position(self, event: Event) -> None:
        pos = event.data
        self.positions[pos.vt_position_id] = pos

    def _on_account(self, event: Event) -> None:
        acc = event.data
        self.accounts[acc.vt_account_id] = acc


class MainEngine:
    """主引擎：聚合事件引擎、网关、数据管理。"""

    def __init__(self) -> None:
        self.event_engine = EventEngine()
        self.log_engine = LogEngine(self.event_engine)
        self.oms_engine = OmsEngine(self.event_engine)
        self.gateways: Dict[str, BaseGateway] = {}
        self.data_manager = None  # 由外部注入（避免循环依赖）

    def write_log(self, msg: str, level: int = logging.INFO) -> None:
        self.log_engine.write_log(msg, level)

    def add_gateway(self, gateway: BaseGateway) -> None:
        self.gateways[gateway.gateway_name] = gateway
        self.write_log(f"网关已注册: {gateway.gateway_name}")

    def get_gateway(self, name: str) -> Optional[BaseGateway]:
        return self.gateways.get(name)

    async def start(self) -> None:
        await self.event_engine.start()

    async def stop(self) -> None:
        for gw in self.gateways.values():
            gw.close()
        await self.event_engine.stop()
