"""本地文件数据源：从 CSV/Parquet 读取历史 K 线（真实数据离线落地）。

用途：把 china-futures-5min、astock-data-toolkit 等落地的 CSV/Parquet 接进 feed 层，
使回测/因子评估直接吃真实数据，而不依赖实时 API 的限频与稳定性。

时间约定：文件时间为北京时间(naive)，统一转为 UTC 后存入 BarData（与体系一致）。
按 UTC 自然日聚合等价于中国交易日（UTC 日界 ≈ 北京时间 08:00，位于夜盘结束后、日盘开始前）。
"""
from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from .base import BaseDataFeed, HistoryRequest
from ...core.constant import Interval
from ...core.object import BarData

_logger = logging.getLogger("quantmind.data.local_file")

UTC = timezone.utc

# 常见列名别名 -> 规范内部列名（不区分大小写）
_COLUMN_ALIASES = {
    "datetime": ["datetime", "date", "time", "trade_time", "trading_time", "timestamp", "dt"],
    "open": ["open", "open_price", "o"],
    "high": ["high", "high_price", "h"],
    "low": ["low", "low_price", "l"],
    "close": ["close", "close_price", "c", "last", "last_price"],
    "volume": ["volume", "vol", "v"],
    "open_interest": ["open_interest", "oi", "hold", "positions", "interest"],
    "turnover": ["turnover", "amount"],
}


def map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """把任意常见列名映射为规范列名（不区分大小写）。"""
    lower = {c.lower(): c for c in df.columns}
    rename = {}
    for std, aliases in _COLUMN_ALIASES.items():
        for a in aliases:
            if a in lower:
                rename[lower[a]] = std
                break
    return df.rename(columns=rename)


class LocalFileFeed(BaseDataFeed):
    """本地文件通用数据源基类。

    子类只需实现 :meth:`_resolve_paths` 返回该请求对应的文件列表与主连标志。
    其余（列映射、时区转换、5min→日频降采样、过滤、转 BarData）由基类统一处理。
    """

    name = "local_file"

    def __init__(self, root_dir: str, name: str = "local_file", tz_offset_hours: int = 8) -> None:
        self.root = Path(root_dir)
        self.name = name
        self.tz_offset_hours = tz_offset_hours

    # ---- 子类实现 ----
    def _resolve_paths(self, req: HistoryRequest) -> Tuple[List[Path], bool]:
        """返回 (文件路径列表, 是否为主连/需拼接)。"""
        raise NotImplementedError

    # ---- 通用处理 ----
    def _read_file(self, path: Path) -> pd.DataFrame:
        if path.suffix.lower() in (".csv", ".txt"):
            return pd.read_csv(path)
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        raise ValueError(f"不支持的文件类型: {path.suffix}")

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """解析时间列并把北京时间(naive)转为 UTC(naive)。"""
        if "datetime" not in df.columns:
            raise ValueError("数据中缺少时间列（datetime/date/time 等）")
        dt = pd.to_datetime(df["datetime"], errors="coerce")
        if dt.isna().any():
            _logger.warning("存在无法解析的时间值，已置为 NaT")
        dt = dt - pd.Timedelta(hours=self.tz_offset_hours)
        out = df.copy()
        out["datetime"] = dt
        return out

    def _expiry_end_day(self, path: Path) -> datetime:
        """从文件名提取 YYMM 交割月月末最后一天（如 IC2401 -> 2024-01-31）。"""
        m = re.search(r"(\d{4})", path.stem)
        if m:
            yymm = m.group(1)
            yy = int(yymm[:2])
            mm = int(yymm[2:])
            year = 2000 + yy if yy < 70 else 1900 + yy
            try:
                nxt = datetime(year + 1, 1, 1) if mm == 12 else datetime(year, mm + 1, 1)
                return nxt - timedelta(days=1)
            except ValueError:
                pass
        return datetime(2099, 12, 31)

    def _build_continuous(self, pieces: List[Tuple[datetime, pd.DataFrame]]) -> pd.DataFrame:
        """简单主力连续：按交割月月末窗口衔接拼接（不做换月调整，可能产生跳变）。

        第 i 个合约取 (上一合约月末, 本合约月末] 区间；相邻合约在交割月末无缝衔接，
        重叠期（两合约同时交易）归近月合约，符合主力连续直觉。
        """
        pieces = sorted(pieces, key=lambda x: x[0])
        out_frames = []
        for i, (exp_end, df) in enumerate(pieces):
            start = pieces[i - 1][0] if i > 0 else datetime(1900, 1, 1)
            mask = (df["datetime"] > start) & (df["datetime"] <= exp_end)
            piece = df.loc[mask]
            if len(piece):
                out_frames.append(piece)
        if not out_frames:
            return pieces[0][1].iloc[0:0]
        return pd.concat(out_frames).sort_values("datetime").reset_index(drop=True)

    def _resample_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """任意分钟频 -> 日频。按交易日聚合，时间戳归一为交易日 00:00（UTC）。

        夜盘归属：UTC 12:00 之后（北京时间 20:00 之后）的 bar 属于下一交易日，
        与交易所官方日线口径一致（此前按 UTC 自然日聚合会把夜盘并入当日）。
        """
        df = df.sort_values("datetime").copy()
        df["_date"] = df["datetime"].apply(self._trading_day)
        rows = []
        for _d, sub in df.groupby("_date"):
            rows.append({
                "datetime": pd.Timestamp(_d),
                "open": float(sub["open"].iloc[0]),
                "high": float(sub["high"].max()),
                "low": float(sub["low"].min()),
                "close": float(sub["close"].iloc[-1]),
                "volume": float(sub["volume"].sum()) if "volume" in sub else 0.0,
                "open_interest": float(sub["open_interest"].iloc[-1]) if "open_interest" in sub else 0.0,
                "turnover": float(sub["turnover"].sum()) if "turnover" in sub else 0.0,
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _trading_day(dt) -> "date":
        """UTC bar 时间 -> 交易日（夜盘归属下一**交易日**）。

        用交易日历取夜盘（UTC 12:00 之后，即北京时间 20:00 之后）之后的
       下一个交易日（周五夜盘归周一、节前夜盘归节后首日）；
        日历不可用时退化为自然日+1。
        """
        if dt.hour >= 12:
            candidate = (dt + pd.Timedelta(days=1)).date()
            try:
                from ...risk.calendar import TradingCalendar
                cal = TradingCalendar()
                nxt = cal.next_trading_day(candidate - timedelta(days=1), max_search=15)
                if nxt is not None:
                    return nxt
            except Exception:  # noqa: BLE001
                pass
            return candidate
        return dt.date()

    def _to_bars(self, df: pd.DataFrame, req: HistoryRequest) -> List[BarData]:
        bars: List[BarData] = []
        for _, row in df.iterrows():
            dt = row["datetime"]
            if isinstance(dt, pd.Timestamp):
                dt = dt.to_pydatetime()
            bars.append(self._make_bar(
                symbol=req.symbol,
                exchange=req.exchange,
                dt=dt,
                interval=req.interval,
                o=float(row.get("open", 0) or 0),
                h=float(row.get("high", 0) or 0),
                l=float(row.get("low", 0) or 0),
                c=float(row.get("close", 0) or 0),
                v=float(row.get("volume", 0) or 0),
                oi=float(row.get("open_interest", 0) or 0),
                turnover=float(row.get("turnover", 0) or 0),
            ))
        return bars

    async def fetch_bar_data(self, req: HistoryRequest) -> List[BarData]:
        paths, is_main = self._resolve_paths(req)
        if not paths:
            _logger.warning("本地源 %s 未找到 %s.%s 的文件", self.name, req.symbol, req.exchange.value)
            return []
        pieces: List[Tuple[datetime, pd.DataFrame]] = []
        for p in paths:
            if not p.exists():
                continue
            try:
                raw = self._read_file(p)
            except Exception as exc:  # noqa: BLE001
                _logger.warning("读取 %s 失败: %s", p, exc)
                continue
            raw = map_columns(raw)
            raw = self._normalize(raw)
            exp_end = self._expiry_end_day(p)
            pieces.append((exp_end, raw))
        if not pieces:
            return []
        if is_main and len(pieces) > 1:
            combined = self._build_continuous(pieces)
        else:
            combined = pd.concat([p[1] for p in pieces]).sort_values("datetime").reset_index(drop=True)
        # 频率：5m 原样返回（供分钟级研究），其他降采样到日频
        if req.interval != Interval.MINUTE_5:
            combined = self._resample_daily(combined)
        if req.start is not None:
            combined = combined[combined["datetime"] >= pd.Timestamp(req.start)]
        if req.end is not None:
            combined = combined[combined["datetime"] <= pd.Timestamp(req.end)]
        combined = combined.dropna(subset=["close"])
        if combined.empty:
            return []
        return self._to_bars(combined, req)
