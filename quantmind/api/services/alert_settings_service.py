"""AlertSettingsService：告警通知配置（Telegram Webhook/社区渠道）持久化。

告警通道由后端 monitoring.notifier 消费。当前提供 Telegram 风格的通知配置，
保存到 ``config/alert_settings.json``，可在「设置」页维护，供后续告警回调使用。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

# 允许前端更新的告警配置字段白名单
_ALLOWED = ["enabled", "channel", "webhook_url", "chat_id", "secret"]


class AlertSettingsService:
    def __init__(self) -> None:
        self.path = Path(__file__).resolve().parent.parent.parent / "config" / "alert_settings.json"
        self.data: Dict[str, Any] = self._load()

    def _defaults(self) -> Dict[str, Any]:
        return {
            "enabled": False,
            "channel": "telegram",
            "webhook_url": "",
            "chat_id": "",
            "secret": "",
        }

    def _load(self) -> Dict[str, Any]:
        data = self._defaults()
        if self.path.exists():
            try:
                saved = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(saved, dict):
                    data.update({k: v for k, v in saved.items() if k in _ALLOWED})
            except Exception:  # noqa: BLE001
                pass
        return data

    def get(self) -> Dict[str, Any]:
        out = {k: self.data.get(k) for k in _ALLOWED}
        out["source"] = "json" if self.path.exists() else "default"
        return out

    def save(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if "enabled" in payload:
            self.data["enabled"] = bool(payload["enabled"])
        for key in ("channel", "webhook_url", "chat_id", "secret"):
            if key in payload and payload[key] is not None:
                self.data[key] = str(payload[key])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return self.get()

    def notify_payload(self) -> Dict[str, Any]:
        """返回供 Notifier 使用的通知配置参数。"""
        return {
            "enabled": bool(self.data.get("enabled")),
            "channel": self.data.get("channel", "telegram"),
            "webhook_url": self.data.get("webhook_url", ""),
            "chat_id": self.data.get("chat_id", ""),
        }
