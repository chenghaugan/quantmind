"""DataAdminService：本地数据入库管理。

能力：
  - list_files()    扫描各本地数据根目录，列出 parquet/csv 数据文件与概览
  - download()      触发指定标的全量/增量抓取，经 DataManager 拉取并回写存储

「下载即入库」：DataManager.get_bar_data 在数据源命中后会回写持久存储
（Timescale/内存缓存），因此一次显式的长时间区间抓取即可完成入库。
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from ...core.constant import Exchange, Interval
from ...data import DataManager
from ...data.feed.base import HistoryRequest
from .data_settings_service import DataSettingsService

_logger = logging.getLogger(__name__)

# 本地数据根目录 -> 目录标签
_TYPE_INFO = [
    ("local_data_root", "本地数据（公共）"),
    ("local_stock_root", "A 股数据"),
    ("local_hk_root", "港股数据"),
    ("local_option_root", "期权数据"),
    ("seat_data_root", "期货席位数据"),
]


class DataAdminService:
    def __init__(self, dm: DataManager, settings: DataSettingsService) -> None:
        self.dm = dm
        self.settings = settings

    # ----------------------------------------------------------------- 文件清单
    def list_files(self) -> Dict[str, Any]:
        roots = self.settings.get()
        groups: List[Dict[str, Any]] = []
        total_files = 0
        for field, label in _TYPE_INFO:
            root = roots.get(field) or ""
            entries: List[Dict[str, Any]] = []
            if root and Path(root).exists():
                p = Path(root)
                for f in p.rglob("*"):
                    if f.is_file() and f.suffix.lower() in (".parquet", ".csv", ".pq"):
                        size_mb = round(f.stat().st_size / 1024 / 1024, 2)
                        entries.append({
                            "path": str(f.relative_to(p)),
                            "size_mb": size_mb,
                            "modified": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                        })
                total_files += len(entries)
            groups.append({
                "key": field,
                "label": label,
                "root": root,
                "files": entries[:200],   # 限制返回量，避免超大目录拖慢
                "count": len(entries),
            })
        return {"groups": groups, "total_files": total_files}

    # ----------------------------------------------------------------- 下载入库
    async def download(
        self,
        symbol: str,
        exchange: str = "SHFE",
        interval: str = "1d",
        start: str = "",
        end: str = "",
    ) -> Dict[str, Any]:
        exch = Exchange(exchange.upper())
        req = HistoryRequest(
            symbol=symbol,
            exchange=exch,
            interval=Interval(interval or "1d"),
            start=datetime.fromisoformat(start) if start else None,
            end=datetime.fromisoformat(end) if end else None,
        )
        bars = await self.dm.get_bar_data(req)
        if not bars:
            return {"symbol": symbol, "exchange": exch.value, "downloaded": 0, "error": "未取到数据"}
        _logger.info("数据下载完成: %s.%s %s -> %d 根", symbol, exch.value, interval, len(bars))
        return {
            "symbol": symbol,
            "exchange": exch.value,
            "interval": interval,
            "bars": len(bars),
            "start": bars[0].datetime.isoformat(),
            "end": bars[-1].datetime.isoformat(),
            "downloaded": len(bars),
        }
