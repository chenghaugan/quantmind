"""CTP 网关桩（期货 + 期货期权 + 指数期权）。

MVP 阶段为 stub：实现 vnpy ``BaseGateway`` 接口，连接/发单仅打日志并模拟即时回执，
填真实凭证或接 simnow/openctp 后即可启用实盘。平今/平昨由 ``OffsetConverter`` 处理。
"""
from __future__ import annotations

import logging
from typing import Dict

from ..core.constant import Exchange
from ..core.gateway import BaseGateway, SubscribeRequest, OrderRequest, CancelRequest
from ..core.offset import OffsetConverter

_logger = logging.getLogger("quantmind.live.ctp")


class CtpGateway(BaseGateway):
    """CTP 网关桩。"""

    def __init__(self, event_engine, gateway_name: str = "CTP") -> None:
        super().__init__(event_engine, gateway_name)
        self.converter = OffsetConverter()
        self.connected = False
        self._seq = 0

    def connect(self, settings: dict) -> None:
        user = settings.get("user", "")
        _logger.info("[CTP] 连接 (user=%s) —— 桩实现，未真正连交易所", user)
        self.connected = True

    def subscribe(self, req: SubscribeRequest) -> None:
        _logger.info("[CTP] 订阅 %s.%s", req.symbol, req.exchange.value)

    def send_order(self, req: OrderRequest) -> str:
        req = self.converter.convert_order_req(req)
        self._seq += 1
        oid = f"CTP-{self._seq}"
        _logger.info("[CTP] 发单 %s %s %s %.2f x%.0f [%s]",
                     oid, req.direction.value, req.offset.value, req.price, req.volume, req.exchange.value)
        # 桩：模拟交易所回执（实际应走 on_order / on_trade 回调）
        from ..core.event import EventType
        from ..core.object import OrderData, TradeData, Status
        self.event_engine.put_event(EventType.EVENT_ORDER, OrderData(
            symbol=req.symbol, exchange=req.exchange, order_id=oid, direction=req.direction,
            offset=req.offset, price=req.price, volume=req.volume, status=Status.SUBMITTED))
        return oid

    def cancel_order(self, req: CancelRequest) -> None:
        _logger.info("[CTP] 撤单 %s", req.order_id)

    def query_position(self) -> None:
        _logger.info("[CTP] 查询持仓")

    def query_account(self) -> None:
        _logger.info("[CTP] 查询账户")

    def close(self) -> None:
        self.connected = False
        _logger.info("[CTP] 断开")
