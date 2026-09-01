"""WebSocket 连接管理：向所有客户端广播实时事件（bar/signal/position/risk/order）。

由 EventEngine 的事件驱动——引擎产生事件后调用 ``broadcast`` 推送至 Web 前端。
"""
from __future__ import annotations

import asyncio
import logging
from typing import Dict, List

from fastapi import WebSocket

_logger = logging.getLogger("quantmind.api.ws")

#: 单客户端发送超时（秒）：慢/半死客户端不拖住请求路径
_BROADCAST_TIMEOUT = 5.0


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
        # 快照并发送：避免遍历中 connect/disconnect 修改列表导致漏发
        snapshot = list(self.active)
        if not snapshot:
            return
        results = await asyncio.gather(
            *(self._send_with_timeout(ws, message) for ws in snapshot)
        )
        for ws, ok in zip(snapshot, results):
            if not ok:
                self.disconnect(ws)

    @staticmethod
    async def _send_with_timeout(ws: WebSocket, message: dict) -> bool:
        try:
            await asyncio.wait_for(ws.send_json(message), timeout=_BROADCAST_TIMEOUT)
            return True
        except Exception:  # noqa: BLE001
            return False

    async def send_personal(self, ws: WebSocket, message: dict) -> None:
        try:
            await ws.send_json(message)
        except Exception:  # noqa: BLE001
            self.disconnect(ws)


# 全局单例（由 app 启动）
manager = ConnectionManager()
