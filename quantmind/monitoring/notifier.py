"""监控与告警：订阅 EventEngine 事件，输出日志并按规则触发告警。

告警渠道：控制台（默认）+ Telegram（可选桩，配置 webhook 即可启用）。
对应规划「监控 Web + Telegram 同步」。
"""
from __future__ import annotations

import logging
from typing import Callable, List, Optional

from ..core.event import Event, EventType

_logger = logging.getLogger("quantmind.monitoring")


class Notifier:
    """事件订阅者：记录日志、触发告警、可回调外部（Telegram/Webhook）。"""

    def __init__(
        self,
        alert_rules: Optional[List[Callable[[Event], Optional[str]]]] = None,
        on_alert: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.alert_rules = alert_rules or [self._default_rule]
        self.on_alert = on_alert

    def attach(self, event_engine) -> None:
        event_engine.register_general(self._on_event)

    def _on_event(self, event: Event) -> None:
        if event.type == EventType.EVENT_LOG:
            _logger.info("[LOG] %s", getattr(event.data, "msg", event.data))
            return
        if event.type == EventType.EVENT_TRADE:
            _logger.info("[TRADE] %s", self._fmt_trade(event.data))
        for rule in self.alert_rules:
            msg = rule(event)
            if msg:
                self._alert(msg)

    def _alert(self, msg: str) -> None:
        _logger.warning("[ALERT] %s", msg)
        if self.on_alert:
            try:
                self.on_alert(msg)
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def _default_rule(event: Event) -> Optional[str]:
        # 风险事件直接告警
        if event.type == EventType.EVENT_RISK:
            return f"风险事件: {event.data}"
        # 账户权益异常（简化：由上层注入 EVENT_RISK）
        return None

    @staticmethod
    def _fmt_trade(trade) -> str:
        return f"{trade.vt_symbol} {trade.direction.value} @ {trade.price:.2f} x{trade.volume}"


# ---- Telegram 桩（配置 webhook 后启用） ----
def make_telegram_alerter(webhook_url: str, chat_id: str) -> Callable[[str], None]:
    """返回一个告警回调；真实发送需 requests（此处仅记录，避免强制依赖）。"""
    def _send(msg: str) -> None:
        _logger.info("[TELEGRAM-stub] 拟发送 -> %s : %s", chat_id, msg)
        # 真实实现：requests.post(webhook_url, json={...})
    return _send
