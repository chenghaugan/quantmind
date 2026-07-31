"""Redis 缓存（最新 Bar / 信号 / 头寸 + pub/sub 给模拟交易）。

延迟导入 redis，无 Redis 时不影响导入。所有方法对连接失败做容错（返回 None）。
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Optional

from ...core.object import BarData
from ...core.event import Event

_logger = logging.getLogger("quantmind.data.cache")


class RedisStore:
    """Redis 缓存层。"""

    def __init__(self, url: str) -> None:
        self.url = url
        self._client = None

    async def connect(self) -> None:
        import redis.asyncio as aioredis

        self._client = aioredis.from_url(self.url, decode_responses=True)
        await self._client.ping()
        _logger.info("Redis 连接已建立")

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()

    def _key_bar(self, vt_symbol: str, interval: str) -> str:
        return f"bar:{vt_symbol}:{interval}"

    async def set_latest_bar(self, bar: BarData) -> None:
        if self._client is None:
            return
        try:
            await self._client.set(
                self._key_bar(bar.vt_symbol, bar.interval.value),
                json.dumps(bar.to_dict()),
            )
        except Exception as exc:  # noqa: BLE001
            _logger.warning("Redis 写最新 bar 失败: %s", exc)

    async def get_latest_bar(self, vt_symbol: str, interval: str) -> Optional[dict]:
        if self._client is None:
            return None
        try:
            raw = await self._client.get(self._key_bar(vt_symbol, interval))
            return json.loads(raw) if raw else None
        except Exception:  # noqa: BLE001
            return None

    async def publish(self, channel: str, message: Any) -> None:
        if self._client is None:
            return
        try:
            await self._client.publish(channel, json.dumps(message, default=str))
        except Exception as exc:  # noqa: BLE001
            _logger.warning("Redis 发布失败: %s", exc)

    async def subscribe(self, channel: str, handler: Callable[[dict], None]) -> None:
        if self._client is None:
            return
        try:
            pubsub = self._client.pubsub()
            await pubsub.subscribe(channel)

            async def _loop() -> None:
                async for msg in pubsub.listen():
                    if msg and msg.get("type") == "message":
                        try:
                            handler(json.loads(msg["data"]))
                        except Exception:  # noqa: BLE001
                            pass

            import asyncio

            asyncio.create_task(_loop())
        except Exception as exc:  # noqa: BLE001
            _logger.warning("Redis 订阅失败: %s", exc)
