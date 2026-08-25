"""下载股指期货主力连续（IF0/IC0/IH0/IM0）全周期数据 → 本地 parquet 仓库。

数据源：
  - akshare（新浪）：日线主力连续约 2017 至今（~9.5 年），分钟数据受 1023 根上限限制
  - efinance（东方财富）：分钟数据无 1023 根限制，可能获取更长历史
  - auto：分钟数据优先 efinance，失败回退 akshare；日线用 akshare

落盘：``data_cache/{symbol}.{exchange}.{interval}.parquet``
  （与 ``DiskBarCache.save`` 格式一致：datetime(UTC)/open/high/low/close/volume/
   open_interest/turnover，幂等合并去重）。

用法：
    .\\venv\\Scripts\\python.exe scripts\\download_index_futures.py [--periods 1d,60m,30m,15m,5m,1m] [--source akshare|efinance|auto]
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from quantmind.core.constant import Exchange, Interval  # noqa: E402
from quantmind.core.object import BarData  # noqa: E402
from quantmind.data.store.disk_cache import DiskBarCache  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

#: 股指期货主力连续（CFFEX）
SYMBOLS = ["IF0", "IC0", "IH0", "IM0"]

#: (名称, Interval, 新浪 period 参数)
PERIODS: List[tuple] = [
    ("1d", Interval.DAILY, "1d"),
    ("60m", Interval.HOUR, "60"),
    ("30m", Interval.MINUTE_30, "30"),
    ("15m", Interval.MINUTE_15, "15"),
    ("5m", Interval.MINUTE_5, "5"),
    ("1m", Interval.MINUTE, "1"),
]


def _parse_dt(value, is_minute: bool):
    """解析新浪日期/时间 → UTC tz-aware Timestamp（与仓库既有数据一致）。

    日线：纯日期 "2017-01-17" → 转 UTC 午夜（naive 直接 localize UTC）。
    分钟：北京时间带时刻 → Asia/Shanghai → UTC（-8h）。
    """
    if isinstance(value, datetime):
        dt = value
    else:
        s = str(value)
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(s[: len(fmt)], fmt)
                break
            except ValueError:
                continue
        else:
            dt = pd.to_datetime(s).to_pydatetime()
    if dt.tzinfo is None:
        if is_minute:
            dt = dt.replace(tzinfo=timezone.utc) - pd.Timedelta(hours=8) if False else dt
            # 新浪分钟时间戳为北京时间：先标 Asia/Shanghai 再转 UTC
            return pd.Timestamp(dt, tz="Asia/Shanghai").tz_convert("UTC").to_pydatetime()
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _df_to_bars(df, symbol: str, interval) -> List[BarData]:
    """新浪 DataFrame → BarData 列表（列别名对齐 resolve_ohlc_columns）。"""
    if df is None or df.empty:
        return []
    cols = {c: c for c in df.columns}
    date_col = next((c for c in df.columns if str(c).lower() in ("date", "datetime", "day")), None)
    o = cols.get("open")
    h = cols.get("high")
    l = cols.get("low")
    c = cols.get("close")
    v = cols.get("volume")
    oi = cols.get("hold") or cols.get("open_interest")
    to = cols.get("turnover") or cols.get("amount")
    if not (date_col and o and h and l and c):
        return []
    is_minute = interval != Interval.DAILY
    bars = []
    for _, row in df.iterrows():
        try:
            bars.append(BarData(
                symbol=symbol,
                exchange=Exchange.CFFEX,
                datetime=_parse_dt(row[date_col], is_minute),
                interval=interval,
                open_price=float(row[o]),
                high_price=float(row[h]),
                low_price=float(row[l]),
                close_price=float(row[c]),
                volume=float(row[v]) if v and pd.notna(row[v]) else 0.0,
                open_interest=float(row[oi]) if oi and pd.notna(row[oi]) else 0.0,
                turnover=float(row[to]) if to and pd.notna(row[to]) else 0.0,
            ))
        except (TypeError, ValueError):
            continue
    return bars


async def fetch_one(symbol: str, interval, period: str, source: str = "akshare") -> List[BarData]:
    """拉单个 品种×周期。source: akshare/efinance/auto"""
    import akshare as ak
    
    # auto 模式：分钟数据优先 efinance，日线用 akshare
    if source == "auto":
        if interval == Interval.DAILY:
            source = "akshare"
        else:
            source = "efinance"  # 分钟数据优先尝试 efinance
    
    # efinance 数据源
    if source == "efinance":
        try:
            from quantmind.data.feed.efinance_feed import EfinanceFeed
            feed = EfinanceFeed()
            from quantmind.data.feed.base import HistoryRequest
            req = HistoryRequest(
                symbol=symbol,
                exchange=Exchange.CFFEX,
                interval=interval,
            )
            bars = await feed.fetch_bar_data(req)
            if bars:
                return bars
            # efinance 返回空数据，回退到 akshare
            if source == "efinance":
                print(f"    efinance 返回空数据，回退 akshare")
        except Exception as exc:
            print(f"    efinance 失败: {str(exc)[:60]}，回退 akshare")
        source = "akshare"  # 回退到 akshare
    
    # akshare 数据源
    if interval == Interval.DAILY:
        df = await asyncio.to_thread(ak.futures_zh_daily_sina, symbol=symbol)
    else:
        df = await asyncio.to_thread(ak.futures_zh_minute_sina, symbol=symbol, period=period)
    return _df_to_bars(df, symbol, interval)


async def main() -> int:
    ap = argparse.ArgumentParser(description="下载股指期货主力连续全周期数据")
    ap.add_argument("--periods", type=str, default="1d,60m,30m,15m,5m,1m",
                    help="逗号分隔周期（1d/60m/30m/15m/5m/1m）")
    ap.add_argument("--source", type=str, default="auto", choices=["akshare", "efinance", "auto"],
                    help="数据源：akshare（新浪）/ efinance（东方财富）/ auto（分钟优先efinance，失败回退akshare）")
    args = ap.parse_args()

    want = {p.strip() for p in args.periods.split(",") if p.strip()}
    periods = [(name, iv, p) for name, iv, p in PERIODS if name in want]
    cache = DiskBarCache(str(PROJECT / "data_cache"))

    print(f"品种: {SYMBOLS}  周期: {[name for name, _, _ in periods]}  数据源: {args.source}")
    print("=" * 70)

    for sym in SYMBOLS:
        for name, interval, period in periods:
            label = f"{sym} {name}"
            try:
                bars = await fetch_one(sym, interval, period, source=args.source)
            except Exception as exc:  # noqa: BLE001
                print(f"  ✗ {label:<12} 拉取失败: {str(exc)[:80]}")
                continue
            if not bars:
                print(f"  ✗ {label:<12} 空数据")
                continue
            # 落盘（幂等合并）
            n = cache.save(bars)
            start = bars[0].datetime.strftime("%Y-%m-%d %H:%M")
            end = bars[-1].datetime.strftime("%Y-%m-%d %H:%M")
            print(f"  ✓ {label:<12} 本次{bars[0].interval.value:<4} 拉取 {len(bars):<6} 根 "
                  f"合并后 {n:<6} 根 | {start} ~ {end}")
            await asyncio.sleep(0.3)  # 温和限速，避免限流

    print("=" * 70)
    print("完成。数据位于 data_cache/{symbol}.CFFEX.{interval}.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
