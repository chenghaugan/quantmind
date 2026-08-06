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

import json
import logging
from datetime import date, datetime, timezone
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

_REFRESH_LOG_NAME = "refresh_log.json"


def _utcnow_iso() -> str:
    """当前 UTC 时间的 ISO 字符串（naive，与体系一致）。"""
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()


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
    # -- 刷新执行日志（refresh_log.json） -------------------------------
    @property
    def refresh_log_path(self) -> Path:
        return self.root / _REFRESH_LOG_NAME

    def _load_refresh_log(self) -> List[dict]:
        """读取刷新执行历史（倒序，最新在前）。文件缺失/损坏返回空列表。"""
        path = self.refresh_log_path
        if not path.exists():
            return []
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                return data
        except Exception as exc:  # noqa: BLE001
            _logger.warning("读取刷新日志失败 %s: %s", path, exc)
        return []

    def _append_refresh_log(self, entry: dict, max_entries: int = 200) -> None:
        """写入一条刷新记录（追加到最新位置，超出上限裁剪）。"""
        try:
            entries = self._load_refresh_log()
            entries.insert(0, entry)
            if len(entries) > max_entries:
                entries = entries[:max_entries]
            self.refresh_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.refresh_log_path, "w", encoding="utf-8") as fh:
                json.dump(entries, fh, ensure_ascii=False, indent=2)
        except Exception as exc:  # noqa: BLE001
            _logger.warning("写刷新日志失败: %s", exc)

    def record_refresh(
        self,
        *,
        symbol: str,
        exchange: str,
        interval: str,
        rows: int,
        latest: Optional[str],
        status: str = "ok",
        detail: Optional[str] = None,
    ) -> None:
        """记录一次刷新（成功/失败/空）到日志，供状态页展示历史。"""
        self._append_refresh_log({
            "ts": _utcnow_iso(),
            "symbol": symbol,
            "exchange": exchange,
            "interval": interval,
            "rows": int(rows),
            "latest": latest,
            "status": status,
            "detail": (detail or "")[:200],
        })

    def refresh_history(self, limit: int = 50) -> List[dict]:
        """返回最近 ``limit`` 条刷新记录（最新在前）。"""
        return self._load_refresh_log()[:limit]

    # -- 新鲜度（staleness） -------------------------------------------
    @staticmethod
    def _as_date(v) -> Optional[date]:
        """把 ISO 字符串/pd.Timestamp 规范化为 date。"""
        if v is None:
            return None
        try:
            if isinstance(v, pd.Timestamp):
                return v.date()
            return pd.Timestamp(v).date()
        except Exception:  # noqa: BLE001
            return None

    def staleness(
        self,
        latest_date,
        ref_trading_day: Optional[date] = None,
    ) -> Tuple[Optional[int], Optional[bool]]:
        """计算一个标的的最新交易日相对于基准交易日落后几个交易日。

        ``ref_trading_day`` 缺省时取仓库内所有标的的最大最后交易日（作为"最新可用日"）
        或当前日历最近的交易日。返回 ``(staleness_days, up_to_date)``；
        无 latest 数据返回 ``(None, None)``。
        """
        latest = self._as_date(latest_date)
        if latest is None:
            return None, None
        try:
            from ...risk.calendar import TradingCalendar
            cal = TradingCalendar()
        except Exception:  # noqa: BLE001
            cal = None

        if ref_trading_day is None:
            # 缺省基准：取最接近"今天"的交易日（严格意义的最新数据日）
            ref = date.today()
            if cal is not None:
                ref = cal.prev_trading_day(ref, max_search=15) or ref
        else:
            ref = self._as_date(ref_trading_day)
        if ref is None:
            return None, None

        if cal is not None:
            # 用交易日历数两个日期之间的交易日数（不含 latest 本身）
            count = 0
            cur = latest
            while cur < ref:
                cur = cal.next_trading_day(cur, max_search=15)
                if cur is None or cur > ref:
                    break
                count += 1
        else:
            count = (ref - latest).days
        return count, count <= 0

    def list_keys(self) -> List[dict]:
        """返回仓库内所有标的键：{symbol, exchange, interval}（由文件名解析）。"""
        out: List[dict] = []
        for p in self.root.glob("*.parquet"):
            # 文件名形如 {symbol}.{exchange}.{interval}.parquet
            stem = p.stem
            parts = stem.split(".")
            if len(parts) == 3:
                out.append({
                    "symbol": parts[0],
                    "exchange": parts[1],
                    "interval": parts[2],
                })
        return out

    def stats(self, include_symbols: bool = True) -> dict:
        """仓库概览：文件数、总行数、最大最后交易日，及逐标的明细。

        ``include_symbols`` 为 True 时额外返回 ``symbols`` 列表，每项含
        ``{file, symbol, exchange, interval, rows, start, end, last}``，
        供「行情仓库总览」页展示各标的覆盖区间与行数。
        """
        n_files = 0
        n_rows = 0
        last = None
        per: List[dict] = []
        for p in self.root.glob("*.parquet"):
            n_files += 1
            parts = p.stem.split(".")
            key = {"symbol": parts[0], "exchange": parts[1], "interval": parts[2]} \
                if len(parts) == 3 else {}
            info = {"file": p.name, "rows": 0, "start": None, "end": None, "last": None}
            info.update(key)
            try:
                df = pd.read_parquet(p, columns=["datetime"])
                n_rows += len(df)
                dt = pd.to_datetime(df["datetime"], errors="coerce").dropna()
                if len(dt):
                    info["rows"] = int(len(dt))
                    info["start"] = dt.min().isoformat()
                    info["end"] = dt.max().isoformat()
                    info["last"] = dt.max().isoformat()
                    if last is None or dt.max() > last:
                        last = dt.max()
                    _sd, _up = self.staleness(dt.max())
                    info["staleness_days"] = _sd
                    info["up_to_date"] = _up
            except Exception:  # noqa: BLE001
                continue
            per.append(info)
        # 每个标的最末一次刷新记录
        hist = self._load_refresh_log()
        last_by_key: dict = {}
        for h in hist:
            key = (h.get("symbol"), h.get("exchange", "").upper(), h.get("interval"))
            if key not in last_by_key:
                last_by_key[key] = h
        for info in per:
            _h = last_by_key.get((info.get("symbol"), info.get("exchange", "").upper(),
                                  info.get("interval")))
            if _h:
                info["last_refresh"] = {
                    "ts": _h.get("ts"),
                    "rows": _h.get("rows"),
                    "latest": _h.get("latest"),
                    "status": _h.get("status"),
                }
        out: dict = {
            "root": str(self.root),
            "files": n_files,
            "rows": n_rows,
            "last_datetime": last.isoformat() if last is not None else None,
            "refresh_log": _REFRESH_LOG_NAME,
        }
        if include_symbols:
            out["symbols"] = per
        return out
