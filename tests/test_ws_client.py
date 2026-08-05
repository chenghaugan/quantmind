"""方案 C：WebSocket 客户端（ws_client.py）测试。

验证 M1 行为：
  - 连接后端 /ws 并收到 hello 握手（进入消息队列）。
  - 断线/连接失败后自动按退避重连。
  - stop() 干净停止后台线程。

用本地极简 WS echo 服务器（后台 asyncio 线程）做真实连接测试，不依赖完整后端。
"""
from __future__ import annotations

import asyncio
import threading
import time

import pytest
import websockets

from quantmind.web.utils.ws_client import WSClient


async def _echo_handler(ws):
    """极简 echo 服务器：连接即回 hello，收到 ping 回 pong。"""
    await ws.send('{"type":"hello","msg":"connected"}')
    async for raw in ws:
        if isinstance(raw, (str, bytes)) and (raw if isinstance(raw, str) else raw.decode()).startswith("ping"):
            await ws.send('{"type":"echo","data":"pong"}')


def _serve(port: int, stop_event: threading.Event):
    async def _main():
        async with websockets.serve(_echo_handler, "127.0.0.1", port):
            # 等待 stop_event，之后退出并关闭服务器释放端口
            while not stop_event.is_set():
                await asyncio.sleep(0.1)

    asyncio.run(_main())


@pytest.fixture()
def echo_server():
    port = 9779
    stop_event = threading.Event()
    t = threading.Thread(target=_serve, args=(port, stop_event), daemon=True)
    t.start()
    time.sleep(0.4)
    yield f"ws://127.0.0.1:{port}/ws"
    stop_event.set()  # 关闭服务器，释放端口
    t.join(timeout=2)


def _wait_connected(client, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline and not client.connected:
        time.sleep(0.05)
    return client.connected


def test_ws_client_connects_and_receives_hello(echo_server):
    """连接成功且收到 hello 握手进入队列。"""
    client = WSClient(echo_server)
    client.start()
    try:
        assert _wait_connected(client), "应成功连接"
        msgs = []
        deadline = time.time() + 3
        while time.time() < deadline and not msgs:
            while not client.messages.empty():
                msgs.append(client.messages.get_nowait())
            time.sleep(0.05)
        assert any(m.get("type") == "hello" for m in msgs), "应收到 hello"
    finally:
        client.stop()


def test_ws_client_auto_reconnects_after_failure():
    """后端不可达时进入重连；随后后端起来能连上（自动重连）。"""
    client = WSClient("ws://127.0.0.1:9771/ws", reconnect_delay=0.2, max_delay=0.5)
    client.start()
    stop_server = threading.Event()
    server_thread: threading.Thread | None = None
    try:
        # 无后端时应处于未连接状态且不崩溃
        time.sleep(0.6)
        assert client._thread is not None
        assert client.connected is False
        # 起一个服务器，客户端应能自动连上
        server_thread = threading.Thread(target=_serve, args=(9771, stop_server), daemon=True)
        server_thread.start()
        assert _wait_connected(client, timeout=6.0), "自动重连应成功"
    finally:
        client.stop()
        stop_server.set()
        if server_thread is not None:
            server_thread.join(timeout=2)


def test_ws_client_stop_terminates_thread(echo_server):
    """stop() 后后台线程终止、连接断开。"""
    client = WSClient(echo_server)
    client.start()
    assert _wait_connected(client)
    client.stop()
    time.sleep(1.2)  # 给后台循环一个退避周期退出
    assert client.connected is False
    assert client._thread.is_alive() is False
