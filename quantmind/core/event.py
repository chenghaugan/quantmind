"""事件定义（参考 vnpy.event）。"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class EventType(Enum):
    """事件类型。"""

    EVENT_TICK = "eTick"
    EVENT_BAR = "eBar"
    EVENT_SIGNAL = "eSignal"
    EVENT_ORDER = "eOrder"
    EVENT_TRADE = "eTrade"
    EVENT_POSITION = "ePosition"
    EVENT_ACCOUNT = "eAccount"
    EVENT_CONTRACT = "eContract"
    EVENT_RISK = "eRisk"
    EVENT_LOG = "eLog"
    EVENT_EXCEPTION = "eException"


@dataclass
class Event:
    """事件载体。"""

    type: EventType
    data: Any = None
