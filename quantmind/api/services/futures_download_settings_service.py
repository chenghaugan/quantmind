"""FuturesDownloadSettingsService：期货数据自动下载配置的持久化与读取。

管理自动下载任务的配置：
  - schedule_cron: 调度时间（cron 表达式）
  - symbols: 要下载的期货品种列表
  - intervals: 要下载的数据周期列表
  - enabled: 是否启用自动下载

读取优先链：运行时 JSON 覆盖文件(config/futures_download_settings.json) > 代码默认值。
保存时写入 JSON，使配置在重启后仍生效。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


# 默认配置
_DEFAULTS = {
    "enabled": True,
    "schedule_cron": "30 16 * * 1-5",  # 交易日 16:30
    "symbols": ["IF0", "IC0", "IH0", "IM0"],  # 默认下载股指期货
    "intervals": ["1d", "60m", "30m", "15m", "5m", "1m"],  # 默认全周期
}


class FuturesDownloadSettingsService:
    def __init__(self) -> None:
        self.path = Path(__file__).resolve().parent.parent.parent / "config" / "futures_download_settings.json"
        self.data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        """加载配置，优先 JSON 文件，否则用默认值。"""
        data = _DEFAULTS.copy()
        if self.path.exists():
            try:
                saved = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(saved, dict):
                    # 只允许已知字段
                    for key in _DEFAULTS.keys():
                        if key in saved:
                            data[key] = saved[key]
            except Exception:  # noqa: BLE001
                pass
        return data

    def get(self) -> Dict[str, Any]:
        """获取当前配置。"""
        return {
            "enabled": self.data.get("enabled", True),
            "schedule_cron": self.data.get("schedule_cron", _DEFAULTS["schedule_cron"]),
            "symbols": self.data.get("symbols", _DEFAULTS["symbols"]),
            "intervals": self.data.get("intervals", _DEFAULTS["intervals"]),
            "source": "json" if self.path.exists() else "default",
        }

    def save(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """保存配置到 JSON 文件。"""
        # 只允许已知字段
        allowed_keys = ["enabled", "schedule_cron", "symbols", "intervals"]
        for key in allowed_keys:
            if key in payload:
                self.data[key] = payload[key]
        
        # 写入 JSON
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        
        return self.get()
