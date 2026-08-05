"""本地日频 Parquet 行情仓库（透明读缓存 + 回写）。

解决实时数据源（AKShare 等）每次请求都走网络/限频导致 ``/factor/pipeline``
动辄数分钟的问题：一旦某个标的从**真实源**成功拉取，就把它落盘到本地
``.parquet`` 文件（按 ``symbol.exchange.interval`` 分文件、存全量历史），
后续任意时间窗口的请求都直接从磁盘秒级返回，无需再次联网。

写入只发生在数据来自*真实*源（非 mock）；来自 mock 的合成数据**不落盘**，
以免污染仓库。时区/字段与体系一致（UTC naive，字段 ``datetime/open/high/low/close``
``/volume/open_interest/turnover``）。
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from ...core.constant import Exchange, Interval
from ...core.object import BarData
from ..feed.base import HistoryRequest

_logger = logging.getLogger("quantmind.data.disk_cache")

_BAR_COLUMNS = [
    "datetime", "open", "high", "low", "close",
    "volume", "open_interest", "turnover",
]


class DiskBarCache:
    """本地 Parquet 行情仓库。

    :param root_dir: 仓库根目录；路径不存在时自动创建。
    :param refresh: 若为 True，读取阶段跳过缓存、强制走真实源并回写（默认 False）。
    """

    def __init__(self, root_dir: str, refresh: bool = False) -> None:
        self.root = Path(root_dir)
        self.refresh = refresh
        self.root.mkdir(parents=True, exist_ok=True)
        _logger.info("本地行情仓库初始化: %s (refresh=%s)", self.root, refresh)

    # ----------------------------------------------------------------- 路径
    @staticmethod
    def _safe_name(s: str) -> str:
        """把 symbol/exchange 处理为安全的文件名片段（避免非法字符）。"""
        return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in s)

    def _path(self, req: HistoryRequest) -> Path:
        return (
            self.root
            / f"{self._safe_name(req.symbol)}.{self._safe_name(req.exchange.value)}.{req.interval.value}.parquet"
        )

    # ----------------------------------------------------------------- 读取
    def _bars_to_df(self, bars: List[BarData]) -> pd.DataFrame:
        rows = [
            {
                "datetime": b.datetime,
                "open": b.open_price,
                "high": b.high_price,
                "low": b.low_price,
                "close": b.close_price,
                "volume": b.volume,
                "open_interest": b.open_interest,
                "turnover": b.turnover,
            }
            for b in bars
        ]
        df = pd.DataFrame(rows, columns=_BAR_COLUMNS)
        if not df.empty:
            df = df.drop_duplicates(subset="datetime").sort_values("datetime").reset_index(drop=True)
        return df

    def load(self, req: HistoryRequest) -> List[BarData]:
        """从磁盘读该请求的全量历史，再按 start/end 切窗返回。

        未命中返回空列表。self.refresh 时直接返回空（强制走真实源）。
        """
        if self.refresh:
            return []
        path = self._path(req)
        if not path.exists():
            return []
        try:
            df = pd.read_parquet(path)
        except Exception as exc:  # noqa: BLE001
            _logger.warning("读取本地行情仓库 %s 失败: %s", path, exc)
            return []
        if df.empty:
            return []
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        if req.start is not None:
            df = df[df["datetime"] >= pd.Timestamp(req.start)]
        if req.end is not None:
            df = df[df["datetime"] <= pd.Timestamp(req.end)]
        bars: List[BarData] = []
        for _, row in df.iterrows():
            dt = row["datetime"]
            if isinstance(dt, pd.Timestamp):
                dt = dt.to_pydatetime()
            bars.append(BarData(
                symbol=req.symbol,
                exchange=req.exchange,
                datetime=dt,
                interval=req.interval,
                open_price=float(row.get("open", 0) or 0),
                high_price=float(row.get("high", 0) or 0),
                low_price=float(row.get("low", 0) or 0),
                close_price=float(row.get("close", 0) or 0),
                volume=float(row.get("volume", 0) or 0),
                open_interest=float(row.get("open_interest", 0) or 0),
                turnover=float(row.get("turnover", 0) or 0),
            ))
        return bars

    # ----------------------------------------------------------------- 回写
    def save(self, bars: List[BarData]) -> int:
        """把真实源返回的全量 bars 落盘（追加合并历史 + 新增）。

        返回写入条数。空输入直接返回 0。
        """
        if not bars:
            return 0
        req = HistoryRequest(
            symbol=bars[0].symbol,
            exchange=bars[0].exchange,
            interval=bars[0].interval,
        )
        new_df = self._bars_to_df(bars)
        path = self._path(req)
        merged = new_df
        if path.exists():
            try:
                old = pd.read_parquet(path)
                old["datetime"] = pd.to_datetime(old["datetime"], errors="coerce")
                merged = (
                    pd.concat([old, new_df], ignore_index=True)
                    .drop_duplicates(subset="datetime")
                    .sort_values("datetime")
                    .reset_index(drop=True)
                )
            except Exception as exc:  # noqa: BLE001
                _logger.warning("合并现有仓库文件失败，覆盖写入: %s", exc)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            merged.to_parquet(path, index=False)
        except Exception as exc:  # noqa: BLE001
            _logger.warning("写本地行情仓库 %s 失败: %s", path, exc)
            return 0
        _logger.info(
            "本地行情仓库回写 %s.%s.%s: %d 根", bars[0].symbol, bars[0].exchange.value,
            bars[0].interval.value, len(merged),
        )
        return len(merged)

    # ----------------------------------------------------------------- 元信息
    def stats(self) -> dict:
        """仓库概览：文件数、总行数、最大最后交易日。"""
        n_files = 0
        n_rows = 0
        last = None
        for p in self.root.glob("*.parquet"):
            n_files += 1
            try:
                df = pd.read_parquet(p, columns=["datetime"])
                n_rows += len(df)
                mx = pd.to_datetime(df["datetime"]).max()
                if last is None or (mx is not None and mx > last):
                    last = mx
            except Exception:  # noqa: BLE001
                continue
        return {
            "root": str(self.root),
            "files": n_files,
            "rows": n_rows,
            "last_datetime": last.isoformat() if last is not None else None,
        }
