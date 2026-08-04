"""CTP 网关桩（期货 + 期货期权 + 指数期权）。

MVP 阶段为 stub：实现 vnpy ``BaseGateway`` 接口，连接/发单仅打日志并模拟即时回执，
填真实凭证或接 simnow/openctp 后即可启用实盘。平今/平昨由 ``OffsetConverter`` 处理。

「半可用」增强：发单通过 :func:`~quantmind.live.sim.simulate_one_trade` 产出
真实的 OrderData/TradeData 回报（而非仅打日志），使 ``LiveEngine``+``OrderManager``
+``reconcile`` 的完整闭环可在**离线**状态下演练；真正的 simnow/openctp 回执
留待接入 provider 后由回调驱动。
"""
from __future__ import annotations

import logging
from typing import Dict

from ..core.constant import Exchange
from ..core.gateway import BaseGateway, SubscribeRequest, OrderRequest, CancelRequest
from ..core.offset import OffsetConverter
from .sim import simulate_one_trade

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
        # 半可用：模拟成交回报（离线演练用；接 simnow 后改为真实回调）
        simulate_one_trade(self.event_engine, oid, req,
                           price=req.price if req.price > 0 else 1.0,
                           gateway_name=self.gateway_name)
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
