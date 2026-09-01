"""WebSocket 客户端（Streamlit 用）——计划方案 C（WS 前端消费）落地。

设计：
  - 基于已声明的 ``websockets``（asyncio）客户端，在**后台 daemon 线程**里跑一个
    独立的 asyncio 事件循环，连接后端 ``/ws`` 并把收到的消息 **put 进 Queue**——
    主线程（Streamlit）只读 Queue，避免跨线程操作 Streamlit / asyncio 内部状态。
  - **自动重连**：连接断开/握手异常后按指数退避重连，直到显式 ``stop()``。
  - 与本项目已有的 ``7_实时监控`` 线程模型一致（后台线程只写 Queue），并抽出为
    可复用模块，供任意页面订阅事件流。

线程模型：
  WSClient.start() -> 启动后台线程运行 ``_run`` (asyncio 事件循环)
  后台线程: 循环 connect -> recv -> messages.put(msg)；异常 -> 退避重连
  WSClient.messages: queue.Queue，主线程在 ``st.fragment(run_every=N)`` 里只读
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
from typing import Any, Optional

logger = logging.getLogger("quantmind.web.ws_client")

# 默认后端 WS 端点（与 api_client.API_URL 同源）
WS_URL = "ws://127.0.0.1:8000/ws"


class WSClient:
    """异步 WebSocket 客户端（后台线程 + 自动重连）。

    :param url: 后端 `/ws` 端点。
    :param max_queue: 消息队列上限；超限时丢弃最旧消息，防止内存增长。
    :param reconnect_delay: 首次重连等待秒数，之后按 2 的幂退避至 max_delay。
    :param max_delay: 最大重连间隔（秒）。
    """

    def __init__(
        self,
        url: str = WS_URL,
        max_queue: int = 2000,
        reconnect_delay: float = 1.0,
        max_delay: float = 30.0,
    ) -> None:
        self.url = url
        self.max_queue = max_queue
        self.reconnect_delay = reconnect_delay
        self.max_delay = max_delay
        self.messages: "queue.Queue[dict]" = queue.Queue(maxsize=max_queue)
        # 供主线程只读的状态
        self._connected = False
        self._error: Optional[str] = None
        self._stop = False
        self._thread: Optional[threading.Thread] = None
        self._active_ws: Any = None
        self._lock = threading.Lock()

    # ---- 只读状态（主线程轮询）----
    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def error(self) -> Optional[str]:
        return self._error

    # ---- 生命周期 ----
    def start(self) -> None:
        """启动后台线程。已有线程在跑则忽略。"""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="qm-ws-client", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """停止后台线程与连接（主动关闭当前连接并取消后台任务）。"""
        self._stop = True
        self._set_connected(False)
        self._set_error(None)
        # 主动关闭当前 websocket，解除 recv 阻塞
        self._close_active()
        # 取消后台 _loop 任务，即使正阻塞在 connect/recv 也能立刻退出
        if self._thread is not None:
            loop = getattr(self._thread, "_loop", None)
            task = getattr(self._thread, "_task", None)
            if loop is not None and task is not None:
                try:
                    loop.call_soon_threadsafe(_schedule_cancel, task)
                except Exception:  # noqa: BLE001
                    pass
        self._wakeup()

    def _close_active(self) -> None:
        """在后台线程的 asyncio loop 中主动关闭当前连接。"""
        if self._thread is None:
            return
        # 通过 loop.call_soon_threadsafe 让后台 loop 去 close，线程安全
        loop = getattr(self._thread, "_loop", None)
        if loop is not None:
            ws = self._active_ws
            if ws is not None:
                try:
                    loop.call_soon_threadsafe(_schedule_close, loop, ws)
                except Exception:  # noqa: BLE001
                    pass

    def _wakeup(self) -> None:
        try:
            self.messages.put({"type": "_stop"})
        except queue.Full:
            # 队列满：丢最旧一条腾位置
            try:
                self.messages.get_nowait()
            except queue.Empty:
                pass
            try:
                self.messages.put({"type": "_stop"})
            except queue.Full:
                pass

    def clear_messages(self) -> None:
        while not self.messages.empty():
            try:
                self.messages.get_nowait()
            except queue.Empty:
                break

    # ---- 内部（后台线程）----
    def _set_connected(self, val: bool) -> None:
        with self._lock:
            self._connected = val

    def _set_error(self, val: Optional[str]) -> None:
        with self._lock:
            self._error = val

    def _run(self) -> None:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._thread._loop = loop  # type: ignore[attr-defined]
            task = loop.create_task(self._loop())
            self._thread._task = task  # type: ignore[attr-defined]
            try:
                loop.run_until_complete(task)
            finally:
                loop.close()
        except Exception as exc:  # noqa: BLE001 - 后台线程兜底，绝不抛到主线程
            logger.exception("WS 后台线程异常退出: %s", exc)
            self._set_error(str(exc))
            self._set_connected(False)

    async def _loop(self) -> None:
        import websockets

        delay = self.reconnect_delay
        while not self._stop:
            try:
                async with websockets.connect(self.url, ping_interval=20, ping_timeout=20,
                                              open_timeout=10, close_timeout=5) as ws:
                    self._active_ws = ws
                    self._set_connected(True)
                    self._set_error(None)
                    delay = self.reconnect_delay  # 重连成功后重置退避
                    # 握手后服务端回 hello
                    await self._recv_loop(ws)
                    # 服务端优雅关连接（正常退出 recv_loop）也要清状态，
                    # 否则退避重连期间页面误显示“已连接”
                    self._set_connected(False)
            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001
                if self._stop:
                    break
                self._set_connected(False)
                self._set_error(str(exc))
                logger.warning("WS 连接失败，%ss 后重连: %s", delay, exc)
            finally:
                self._active_ws = None
            # 等待重连（期间可被 stop 打断）
            if self._stop:
                break
            await asyncio.sleep(delay)
            delay = min(delay * 2, self.max_delay)

    async def _recv_loop(self, ws) -> None:
        """持续接收消息直到连接关闭或停止。"""
        try:
            async for raw in ws:
                if self._stop:
                    break
                try:
                    msg = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
                except Exception:
                    msg = {"type": "raw", "data": raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw}
                self._put(msg)
        except Exception:
            # 连接中断：外层循环负责重连
            raise

    def _put(self, msg: dict) -> None:
        if msg.get("type") == "_stop":
            return
        try:
            self.messages.put_nowait(msg)
        except queue.Full:
            # 丢弃最旧消息，保持队列新鲜
            try:
                self.messages.get_nowait()
            except queue.Empty:
                pass
            try:
                self.messages.put_nowait(msg)
            except queue.Full:
                pass


def connect_ws(url: str = WS_URL) -> WSClient:
    """便捷工厂：创建并启动一个 WSClient。"""
    client = WSClient(url)
    client.start()
    return client


def _schedule_close(loop, ws) -> None:
    """在目标事件循环里调度关闭 websocket（供 thread-safe 关闭接收循环）。"""
    try:
        asyncio.run_coroutine_threadsafe(_close(ws), loop)
    except Exception:  # noqa: BLE001
        pass


def _schedule_cancel(task) -> None:
    """在目标事件循环里取消后台任务（供 thread-safe 停线程）。"""
    task.cancel()


async def _close(ws) -> None:
    try:
        await ws.close()
    except Exception:  # noqa: BLE001
        pass


__all__ = ["WSClient", "connect_ws", "WS_URL"]
