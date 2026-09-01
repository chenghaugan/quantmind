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
import os
import tempfile
import time
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

#: 交易所 -> 市场归类（总览聚合用）
_MARKET_OF = {
    "CFFEX": "期货", "SHFE": "期货", "DCE": "期货", "CZCE": "期货",
    "INE": "期货", "GFEX": "期货",
    "SSE": "A股", "SZSE": "A股", "HKEX": "港股",
    "NASDAQ": "美股", "NYSE": "美股", "AMEX": "美股",
    "COMEX": "美股", "CME": "美股", "NYMEX": "美股", "CBOT": "美股",
}


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
        # 文件名形如 {code}.{exchange}.{interval}.parquet（与仓库既有文件/list_keys 解析一致）。
        # req.symbol 通常已含交易所后缀（如 cu0.SHFE），此处剥去后缀取裸 code，避免重复写交易所。
        sym = req.symbol
        code = sym.rsplit(".", 1)[0] if "." in sym else sym
        return (
            self.root
            / f"{self._safe_name(code)}.{self._safe_name(req.exchange.value)}.{req.interval.value}.parquet"
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
        df = df.dropna(subset=["datetime"])
        # 覆盖检查：文件未覆盖请求窗口时视为未命中，回源补齐后由 save 合并，
        # 避免早期小窗口请求留下的切片被当作全量历史返回（数据完整性）。
        # tz 归一与 manager._clip_bars 同口径，避免 naive/aware 混比抛 TypeError
        _start = pd.Timestamp(req.start) if req.start is not None else None
        _end = pd.Timestamp(req.end) if req.end is not None else None
        if _start is not None and _start.tzinfo is not None:
            _start = _start.tz_convert("UTC").tz_localize(None)
        if _end is not None and _end.tzinfo is not None:
            _end = _end.tz_convert("UTC").tz_localize(None)
        if _start is not None and df["datetime"].min() > _start:
            return []
        if _end is not None and df["datetime"].max() < _end:
            return []
        if _start is not None:
            df = df[df["datetime"] >= _start]
        if _end is not None:
            df = df[df["datetime"] <= _end]
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

    def latest_datetime(self, req: HistoryRequest) -> Optional[datetime]:
        """轻量读取本地最新 bar 时间戳（只读 datetime 列，供预检跳过用）。

        相比 ``load`` 不构建 BarData 列表，大文件读取快一个量级。
        naive 时间戳视为 UTC（与仓库写入约定一致）。无缓存返回 None。
        """
        path = self._path(req)
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path, columns=["datetime"])
            if df.empty:
                return None
            s = pd.to_datetime(df["datetime"], errors="coerce").dropna()
            if s.empty:
                return None
            dt = s.max().to_pydatetime()
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception as exc:  # noqa: BLE001
            _logger.warning("读取本地最新时间戳 %s 失败: %s", path, exc)
            return None

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
                    .drop_duplicates(subset="datetime", keep="last")
                    .sort_values("datetime")
                    .reset_index(drop=True)
                )
            except Exception as exc:  # noqa: BLE001
                _logger.warning("合并现有仓库文件失败，覆盖写入: %s", exc)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # 原子写（临时文件 + rename）：多进程共享目录时避免读到写了一半的文件
            fd, tmp_path = tempfile.mkstemp(
                prefix=path.stem + ".", suffix=".tmp", dir=str(path.parent))
            os.close(fd)
            try:
                merged.to_parquet(tmp_path, index=False)
                os.replace(tmp_path, path)
            except Exception:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise
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

    def stats(self, include_symbols: bool = True, aggregate: bool = False) -> dict:
        """仓库概览：文件数、总行数、最大最后交易日，及逐标的明细。

        ``include_symbols`` 为 True 时额外返回 ``symbols`` 列表，每项含
        ``{file, symbol, exchange, interval, rows, start, end, last}``。
        ``aggregate`` 为 True 时额外返回 ``agg`` 聚合桶（by_exchange / by_interval /
        freshness / top_rows / stale_top），供总览页默认展示，避免逐标的铺开。
        """
        per = self._scan_symbols()
        n_files = len(per)
        n_rows = sum(s.get("rows", 0) for s in per)
        last = None
        for s in per:
            _l = s.get("last")
            if _l and (last is None or _l > last):
                last = _l
        out: dict = {
            "root": str(self.root),
            "files": n_files,
            "rows": n_rows,
            "last_datetime": last,
            "refresh_log": _REFRESH_LOG_NAME,
        }
        if include_symbols:
            out["symbols"] = per
        if aggregate:
            out["agg"] = self._buckets(per)
        return out

    # --------------------------------------------------------- 聚合 / 分页
    _SCAN_TTL = 20.0

    def _ensure_cache(self) -> None:
        if not getattr(self, "_scan_cache", None):
            self._scan_cache = {"ts": 0.0, "per": [], "sig": None}

    def _dir_signature(self) -> tuple:
        """仓库目录签名：(文件名, mtime_ns, size) 的排序元组。

        os.scandir 只碰目录项（几千文件 ~ 几十 ms），不读文件内容；仓库无变化时
        直接复用上次扫描结果，避免总览页反复全量扫 parquet（几千文件要数十秒~分钟级）。
        """
        sig = []
        try:
            with os.scandir(self.root) as it:
                for e in it:
                    if e.is_file() and e.name.endswith(".parquet"):
                        st = e.stat()
                        sig.append((e.name, st.st_mtime_ns, st.st_size))
        except FileNotFoundError:
            pass
        sig.sort()
        return tuple(sig)

    def _scan_symbols(self, force: bool = False) -> List[dict]:
        """扫描仓库全部 parquet 头标：目录签名失效缓存，总览与明细下钻共享同一快照。"""
        self._ensure_cache()
        if not force:
            try:
                sig = self._dir_signature()
                if (self._scan_cache.get("sig") == sig
                        and self._scan_cache.get("per") is not None):
                    return self._scan_cache["per"]
            except Exception:  # noqa: BLE001 - 签名失败退回 TTL 行为
                pass
        per = self._do_scan()
        try:
            sig = self._dir_signature()
        except Exception:  # noqa: BLE001
            sig = None
        self._scan_cache = {"ts": time.time(), "per": per, "sig": sig}
        return per

    @staticmethod
    def _file_summary(p: Path) -> Optional[dict]:
        """从 parquet footer 统计读 (rows, start, end)：不读数据页。

        依赖写文件时记录的 datetime 列 min/max 统计（pyarrow 默认写入）；
        任一 row group 缺统计或类型不支持时返回 None，由调用方回退慢路径。
        """
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(p)
        rows = int(pf.metadata.num_rows)
        idx = pf.schema_arrow.get_field_index("datetime")
        if idx < 0:
            return None
        mn = mx = None
        for rg in range(pf.metadata.num_row_groups):
            st = pf.metadata.row_group(rg).column(idx).statistics
            if st is None or not st.has_min_max:
                return None
            mn = st.min if mn is None else min(mn, st.min)
            mx = st.max if mx is None else max(mx, st.max)
        if mn is None or mx is None:
            return None
        try:
            mn = pd.Timestamp(mn)
            mx = pd.Timestamp(mx)
        except (ValueError, TypeError):  # 统计给出的是裸 int 等不可靠类型 → 回退
            return None
        return {"rows": rows, "start": mn.isoformat(),
                "end": mx.isoformat(), "last": mx.isoformat()}

    def _file_summary_slow(self, p: Path) -> Optional[dict]:
        """旧行为：读 datetime 列算 min/max（慢，仅 footer 统计缺失时兜底）。"""
        df = pd.read_parquet(p, columns=["datetime"])
        dt = pd.to_datetime(df["datetime"], errors="coerce").dropna()
        if not len(dt):
            return {"rows": 0, "start": None, "end": None, "last": None}
        return {"rows": int(len(dt)), "start": dt.min().isoformat(),
                "end": dt.max().isoformat(), "last": dt.max().isoformat()}

    def _do_scan(self) -> List[dict]:
        """真正扫描仓库：读每个 parquet 的 footer 元数据（行数 + datetime 列 min/max 统计）。

        footer 统计不读数据页，单文件毫秒级，几千只标的秒级完成；
        统计缺失（未写入 min/max）的文件回退为读 datetime 列（旧行为）。
        """
        per: List[dict] = []
        for p in self.root.glob("*.parquet"):
            parts = p.stem.split(".")
            key = {"symbol": parts[0], "exchange": parts[1], "interval": parts[2]} \
                if len(parts) == 3 else {}
            info = {"file": p.name, "rows": 0, "start": None, "end": None, "last": None}
            info.update(key)
            try:
                summary = self._file_summary(p)
                if summary is None:  # footer 统计不可用 → 回退旧路径
                    summary = self._file_summary_slow(p)
                if summary:
                    info.update(summary)
                    last = pd.Timestamp(summary["end"]) if summary.get("end") else None
                    if last is not None:
                        _sd, _up = self.staleness(last)
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
        return per

    @staticmethod
    def _market_of(exchange: str) -> str:
        return _MARKET_OF.get((exchange or "").upper(), "其他")

    def _buckets(self, per: List[dict]) -> dict:
        """把逐标的明细聚合成总览桶（交易所/周期/新鲜度 + Top-N）。"""
        by_exch: dict = {}
        by_int: dict = {}
        for s in per:
            exch = (s.get("exchange") or "?")
            intv = s.get("interval") or "?"
            rows = s.get("rows", 0)
            s_start = s.get("start")
            s_last = s.get("last")
            bee = by_exch.setdefault(exch, {"exchange": exch,
                                            "market": self._market_of(exch),
                                            "symbols": 0, "rows": 0,
                                            "coverage_start": None, "coverage_end": None})
            bee["symbols"] += 1
            bee["rows"] += rows
            if s_start and (bee["coverage_start"] is None or s_start < bee["coverage_start"]):
                bee["coverage_start"] = s_start
            if s_last and (bee["coverage_end"] is None or s_last > bee["coverage_end"]):
                bee["coverage_end"] = s_last
            bi = by_int.setdefault(intv, {"interval": intv, "symbols": 0, "rows": 0})
            bi["symbols"] += 1
            bi["rows"] += rows
        fresh = stale_1_3 = stale_gt3 = 0
        for s in per:
            up = s.get("up_to_date")
            sd = s.get("staleness_days")
            if up is True:
                fresh += 1
            elif up is False:
                if sd is not None and sd > 3:
                    stale_gt3 += 1
                else:
                    stale_1_3 += 1
        top_rows = sorted(per, key=lambda x: -(x.get("rows") or 0))[:50]
        stale_top = sorted((s for s in per if s.get("up_to_date") is False),
                           key=lambda x: -(x.get("staleness_days") or 0))[:50]
        def _pick(s, keys):
            return {k: s.get(k) for k in keys}
        return {
            "by_exchange": sorted(by_exch.values(), key=lambda x: -x["symbols"]),
            "by_interval": sorted(by_int.values(), key=lambda x: -x["symbols"]),
            "freshness": {"fresh": fresh, "stale_1_3d": stale_1_3, "stale_gt3d": stale_gt3},
            "markets": sorted({self._market_of(s.get("exchange")) for s in per}),
            "exchanges": sorted({(s.get("exchange") or "") for s in per}),
            "intervals": sorted({(s.get("interval") or "") for s in per}),
            "top_rows": [_pick(s, ("symbol", "exchange", "interval", "rows", "last"))
                          for s in top_rows],
            "stale_top": [_pick(s, ("symbol", "exchange", "interval",
                                    "staleness_days", "last")) for s in stale_top],
        }

    def symbol_page(
        self,
        exchange: str = "",
        market: str = "",
        interval: str = "",
        freshness: str = "",
        q: str = "",
        page: int = 1,
        page_size: int = 50,
    ) -> dict:
        """逐标的明细分页（供下钻）：按交易所/市场/周期/新鲜度/关键词过滤。"""
        per = self._scan_symbols()
        out: List[dict] = []
        for s in per:
            if exchange and (s.get("exchange") or "").upper() != exchange.upper():
                continue
            if market and self._market_of(s.get("exchange")) != market:
                continue
            if interval and (s.get("interval") or "") != interval:
                continue
            if freshness:
                up = s.get("up_to_date")
                if freshness == "stale":
                    if up is not False:
                        continue
                elif freshness == "fresh":
                    if up is not True:
                        continue
            if q:
                key = f"{s.get('symbol')}.{s.get('exchange')}"
                if q.lower() not in key.lower():
                    continue
            out.append(s)
        total = len(out)
        page = max(1, page)
        page_size = max(1, min(page_size, 500))
        start = (page - 1) * page_size
        agg = self._buckets(per)
        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "symbols": out[start:start + page_size],
            "markets": agg["markets"],
            "exchanges": agg["exchanges"],
            "intervals": agg["intervals"],
            "freshness": agg["freshness"],
        }
