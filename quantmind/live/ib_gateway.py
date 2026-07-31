"""IB 网关桩（Interactive Brokers，港股 + 港股期权）。接口对齐 vnpy ``BaseGateway``。"""
from __future__ import annotations

import logging

from ..core.gateway import BaseGateway, SubscribeRequest, OrderRequest, CancelRequest

_logger = logging.getLogger("quantmind.live.ib")


class IbGateway(BaseGateway):
    """IB 网关桩。"""

    def __init__(self, event_engine, gateway_name: str = "IB") -> None:
        super().__init__(event_engine, gateway_name)
        self.connected = False
        self._seq = 0

    def connect(self, settings: dict) -> None:
        _logger.info("[IB] 连接 (host=%s) —— 桩实现", settings.get("host", "127.0.0.1"))
        self.connected = True

    def subscribe(self, req: SubscribeRequest) -> None:
        _logger.info("[IB] 订阅 %s.%s", req.symbol, req.exchange.value)

    def send_order(self, req: OrderRequest) -> str:
        self._seq += 1
        oid = f"IB-{self._seq}"
        _logger.info("[IB] 发单 %s %s %.2f x%.0f", oid, req.direction.value, req.price, req.volume)
        return oid

    def cancel_order(self, req: CancelRequest) -> None:
        _logger.info("[IB] 撤单 %s", req.order_id)

    def query_position(self) -> None:
        _logger.info("[IB] 查询持仓")

    def query_account(self) -> None:
        _logger.info("[IB] 查询账户")

    def close(self) -> None:
        self.connected = False
