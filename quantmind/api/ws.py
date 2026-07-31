"""WebSocket 连接管理：向所有客户端广播实时事件（bar/signal/position/risk/order）。

由 EventEngine 的事件驱动——引擎产生事件后调用 ``broadcast`` 推送至 Web 前端。
"""
from __future__ import annotations

import logging
from typing import Dict, List

from fastapi import WebSocket

_logger = logging.getLogger("quantmind.api.ws")


class ConnectionManager:
    def __init__(self) -> None:
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.append(ws)
        _logger.info("WebSocket 客户端连接，当前 %d", len(self.active))

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, message: dict) -> None:
        dead: List[WebSocket] = []
        for ws in self.active:
            try:
                await ws.send_json(message)
            except Exception:  # noqa: BLE001
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    async def send_personal(self, ws: WebSocket, message: dict) -> None:
        try:
            await ws.send_json(message)
        except Exception:  # noqa: BLE001
            self.disconnect(ws)


# 全局单例（由 app 启动）
manager = ConnectionManager()
