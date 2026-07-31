"""网关抽象（参考 vnpy.trader.gateway）。

所有实时/实盘网关（CTP/XTP/IB）均继承 ``BaseGateway``，接口对齐 vnpy。
MVP 阶段这些网关以桩（stub）形式存在，填凭证或接模拟盘（simnow/openctp）即可启用。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from .constant import Exchange, Direction, Offset, Interval
if TYPE_CHECKING:
    from .engine import EventEngine


@dataclass
class SubscribeRequest:
    """订阅请求。"""

    symbol: str
    exchange: Exchange


@dataclass
class OrderRequest:
    """委托请求。"""

    symbol: str
    exchange: Exchange
    direction: Direction
    offset: Offset
    volume: float
    price: float = 0.0
    type: str = "LIMIT"


@dataclass
class CancelRequest:
    """撤单请求。"""

    order_id: str
    symbol: str = ""
    exchange: Optional[Exchange] = None


class BaseGateway(ABC):
    """网关基类。"""

    def __init__(self, event_engine: EventEngine, gateway_name: str) -> None:
        self.event_engine = event_engine
        self.gateway_name = gateway_name

    @abstractmethod
    def connect(self, settings: dict) -> None:
        """连接网关（填凭证/模拟盘配置）。"""

    @abstractmethod
    def subscribe(self, req: SubscribeRequest) -> None:
        """订阅行情。"""

    @abstractmethod
    def send_order(self, req: OrderRequest) -> str:
        """发单，返回 order_id。"""

    @abstractmethod
    def cancel_order(self, req: CancelRequest) -> None:
        """撤单。"""

    @abstractmethod
    def query_position(self) -> None:
        """查询持仓。"""

    @abstractmethod
    def query_account(self) -> None:
        """查询账户。"""

    def close(self) -> None:
        """断开（默认空实现，子类覆盖）。"""
