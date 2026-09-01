"""MarketUpdateSettingsService：A股/港股/美股数据自动更新配置的持久化与读取。

与期货的 ``FuturesDownloadSettingsService`` 对齐，按市场分别管理：
  - a_stock: A股（沪深）每日增量更新
  - hk_stock: 港股每日增量更新
  - us_stock: 美股每日增量更新

每个市场一个独立配置：{"enabled": bool, "schedule_cron": str}。
读取优先链：运行时 JSON 覆盖文件(config/market_update_settings.json) > 代码默认值；
JSON 不存在时，a_stock 的初始值回退到环境变量配置（stock_autoupdate_*，向后兼容）。
保存时写入 JSON，使配置在重启后仍生效。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

# 市场标识 -> 显示名（web 端复用）
MARKETS = {
    "a_stock": "A股（沪深）",
    "hk_stock": "港股",
    "us_stock": "美股",
}

# 默认配置
_DEFAULTS = {
    # A股：默认交易日 23:00（收盘后充分时间）；启用状态回退环境变量
    "a_stock": {"enabled": False, "schedule_cron": "0 23 * * 1-5"},
    # 港股：默认交易日 23:00（16:00 收盘后）
    "hk_stock": {"enabled": False, "schedule_cron": "0 23 * * 1-5"},
    # 美股：北京时间凌晨 05:00（美股 16:00 ET 收盘后）
    "us_stock": {"enabled": False, "schedule_cron": "0 5 * * 1-5"},
}


class MarketUpdateSettingsService:
    def __init__(self) -> None:
        self.path = Path(__file__).resolve().parent.parent.parent / "config" / "market_update_settings.json"
        self.data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        """加载配置，优先 JSON 文件，否则用默认值（含环境变量向后兼容）。"""
        data = {k: dict(v) for k, v in _DEFAULTS.items()}
        if self.path.exists():
            try:
                saved = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(saved, dict):
                    for key in _DEFAULTS.keys():
                        if key in saved and isinstance(saved[key], dict):
                            data[key].update({
                                kk: saved[key][kk] for kk in ("enabled", "schedule_cron")
                                if kk in saved[key]
                            })
            except Exception:  # noqa: BLE001 - 配置损坏时静默用默认值
                pass
        else:
            # JSON 不存在（首次使用）：A 股启用状态/时间沿用旧环境变量配置（向后兼容）
            try:
                from ...config import get_settings
                s = get_settings()
                data["a_stock"]["enabled"] = bool(getattr(s, "stock_autoupdate_enabled", False))
                data["a_stock"]["schedule_cron"] = getattr(s, "stock_autoupdate_cron", data["a_stock"]["schedule_cron"])
            except Exception:  # noqa: BLE001 - 配置不可用时静默用默认值
                pass
        return data

    def get(self) -> Dict[str, Any]:
        """获取当前配置（含来源标记）。"""
        return {
            key: {
                "enabled": bool(cfg.get("enabled", False)),
                "schedule_cron": cfg.get("schedule_cron", _DEFAULTS[key]["schedule_cron"]),
            }
            for key, cfg in self.data.items()
        } | {"source": "json" if self.path.exists() else "default"}

    def save(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """保存配置到 JSON 文件（只允许已知市场/字段）。"""
        for key in _DEFAULTS.keys():
            if key not in payload or not isinstance(payload[key], dict):
                continue
            for field in ("enabled", "schedule_cron"):
                if field in payload[key]:
                    self.data[key][field] = payload[key][field]

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return self.get()

    def get_market(self, market_key: str) -> Dict[str, Any]:
        """获取单个市场配置。"""
        cfg = self.data.get(market_key, {})
        return {
            "enabled": bool(cfg.get("enabled", False)),
            "schedule_cron": cfg.get("schedule_cron", _DEFAULTS.get(market_key, {}).get("schedule_cron", "0 17 * * 1-5")),
        }
