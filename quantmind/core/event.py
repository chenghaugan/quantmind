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
    # 回测/模拟盘生命周期事件（WebSocket 实时推送）
    EVENT_BACKTEST_START = "eBacktestStart"
    EVENT_BACKTEST_PROGRESS = "eBacktestProgress"
    EVENT_BACKTEST_COMPLETE = "eBacktestComplete"
    EVENT_BACKTEST_ERROR = "eBacktestError"


@dataclass
class Event:
    """事件载体。"""

    type: EventType
    data: Any = None
