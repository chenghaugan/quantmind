"""实盘路由引擎（继承 StrategyContext）。

与回测/模拟**共用同一份策略代码**：只需把策略绑定到 ``LiveEngine``，
其 ``send_order`` 就会把委托路由到真实网关（CTP/XTP/IB 桩），实现「切换路线即可跑实盘」。
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from ..core.constant import Direction, Exchange
from ..core.gateway import BaseGateway, OrderRequest
from ..core.object import BarData, PositionData
from ..strategy.context import StrategyContext

_logger = logging.getLogger("quantmind.live")


class LiveEngine(StrategyContext):
    """实盘引擎：把策略委托路由到网关。"""

    mode = "live"

    def __init__(
        self,
        gateway: BaseGateway,
        event_engine=None,
        history: Optional[Dict[str, List[BarData]]] = None,
    ) -> None:
        self.gateway = gateway
        self.event_engine = event_engine
        self._history = history or {}
        self.positions: Dict[str, PositionData] = {}

    def send_order(self, req: OrderRequest) -> str:
        _logger.info("[LIVE] 路由委托至网关 %s", self.gateway.gateway_name)
        return self.gateway.send_order(req)

    def get_position(self, vt_symbol: str) -> PositionData:
        return self.positions.get(
            vt_symbol, PositionData(symbol=vt_symbol.split(".")[0],
                                    exchange=Exchange(vt_symbol.split(".")[1]),
                                    direction=Direction.NET, volume=0.0)
        )

    def get_history(self, vt_symbol: str, count: int) -> List[BarData]:
        return self._history.get(vt_symbol, [])[-count:]

    def connect(self, settings: dict) -> None:
        self.gateway.connect(settings)

    def close(self) -> None:
        self.gateway.close()
