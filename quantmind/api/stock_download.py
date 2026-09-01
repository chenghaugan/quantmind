"""A股数据自动下载（每日增量更新：日线 / 60分钟 / 30分钟）。

与期货 ``futures_download`` 的设计对齐：
  - 只更新**已在本地行情仓库里的 A 股标的**（从 disk_cache 读已有键），天然自适应规模，
    不会对全市场 5000+ 只做全量拉取（开销过大、不现实）。
  - 首次建库仍靠「全市场预热」（market_warm，日线），本任务负责之后每日追新，
    并补齐 60/30 分钟周期的首次拉取。
  - 数据源复用现有回退链：A股腾讯/mootdx，均走 ``dm.get_bar_data`` 自动落盘。

周期：1d（腾讯日线，完整历史）、60m / 30m（腾讯分钟，各约 1970 根上限）。

设计要点：
  - 每个标的 × 每个周期：只从源拉取 - 过滤出比本地最新更新的 bar 落盘（幂等合并）
  - 已是最新的跳过；分钟首次（仓库无该周期）直接全量
  - 预检跳过：本地最新 bar 已覆盖到最近收盘边界（11:30/15:00 北京时间，日线为交易日）
    时直接不发源请求——重复点击不再浪费带宽；节假日保守回退为照常拉源，宁可多拉不漏
  - 默认关闭（QM_STOCK_AUTOUPDATE_ENABLED 或配置 stock_autoupdate_enabled），需要时启用
"""
from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from .services.market_update_settings_service import MarketUpdateSettingsService

_logger = logging.getLogger("quantmind.scheduler.stock")

# A股下载周期（按顺序）
STOCK_INTERVALS = ["1d", "1h", "30m"]

# 周期字符串 -> Interval value（注意：60分钟在 Interval 枚举中为 HOUR，value="1h"）
_INTERVAL_MAP = {
    "1d": "1d",
    "1h": "1h",
    "30m": "30m",
}

# 北京时区（UTC+8，A股收盘 15:00）
_CST = timezone(timedelta(hours=8))


def _as_utc(dt: datetime) -> datetime:
    """naive 视为 UTC，统一成 aware 便于比较（parquet 读回可能丢时区）。"""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _last_weekday(d: date) -> date:
    """最近的一个周一～周五（含当天）。"""
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def _expected_latest_daily(now_utc: datetime) -> datetime:
    """最近一个已收盘交易日的日线时间戳（保守估计，不考虑节假日）。

    A股日线 bar 时间戳为交易日 UTC 零点；北京时间 15:00 收盘后当天才算已收盘。
    节假日会让该估计偏"新"→ 判定未到最新 → 照常拉源（宁可多拉不漏，只损失一点效率）。
    """
    now_cst = _as_utc(now_utc).astimezone(_CST)
    d = now_cst.date()
    if d.weekday() >= 5 or now_cst.time() < time(15, 0):  # 周末或盘中
        d = _last_weekday(d - timedelta(days=1))
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)


def _expected_latest_minute(now_utc: datetime, interval_str: str) -> datetime:
    """最近一个已收盘的分钟线边界（最近一次收盘 11:30 / 15:00 北京时间）。

    预留一个 bar 时长容差（腾讯分钟 bar 时间戳存在起始/结束两种约定），
    保证收盘后更新的仓库恰好命中跳过条件；缺整段行情时不会误跳。
    """
    tol = {"1h": timedelta(hours=1), "30m": timedelta(minutes=30)}[interval_str]
    now_cst = _as_utc(now_utc).astimezone(_CST)
    d = now_cst.date()
    while True:
        if d.weekday() < 5:
            for hh, mm in ((15, 0), (11, 30)):  # 倒序找最近已收盘时点
                close_cst = datetime(d.year, d.month, d.day, hh, mm, tzinfo=_CST)
                if close_cst <= now_cst:
                    return (close_cst - tol).astimezone(timezone.utc)
        d -= timedelta(days=1)


def _is_up_to_date(latest: Optional[datetime], interval_str: str,
                   now_utc: Optional[datetime] = None) -> bool:
    """本地最新 bar 是否已覆盖到最近收盘边界（命中则跳过源请求）。

    保守策略：不依赖交易日历——节假日会判定为未到最新而照常拉源，宁可多拉不漏。
    """
    if latest is None:
        return False
    latest = _as_utc(latest)
    now = _as_utc(now_utc) if now_utc else datetime.now(timezone.utc)
    if interval_str == "1d":
        return latest >= _expected_latest_daily(now)
    if interval_str in ("1h", "30m"):
        return latest >= _expected_latest_minute(now, interval_str)
    return False  # 其他周期不预检，照常拉源


def _fetch_stock_bars(symbol: str, exchange, interval_str: str) -> list:
    """A股专用拉取：按周期调用正确接口（绕开 mootdx 分钟缺陷与复杂回退链）。

    **同步函数**——调用方必须用 ``asyncio.to_thread`` 包裹执行：
    akshare 网络请求 + 行构建（iterrows）都是长同步段，放事件循环会拖死状态轮询。

    周期 -> 接口：
      - 1d   : akshare stock_zh_a_daily（腾讯日线，完整历史）
      - 1h/30m: akshare stock_zh_a_minute（腾讯分钟，约 1970 根上限）
    """
    import akshare as ak

    prefix = "sh" if exchange.value == "SSE" else "sz"

    if interval_str == "1d":
        df = ak.stock_zh_a_daily(symbol=f"{prefix}{symbol}", adjust="qfq")
        col_d, col_o, col_h, col_l, col_c, col_v = "date", "open", "high", "low", "close", "volume"
        rows = []
        for _, r in df.iterrows():
            rows.append({
                "datetime": r[col_d], "open": float(r[col_o]), "high": float(r[col_h]),
                "low": float(r[col_l]), "close": float(r[col_c]),
                "volume": float(r[col_v]) if col_v in df.columns else 0.0,
            })
        return _to_bars(rows, symbol, exchange, "DAILY", is_minute=False)

    # 分钟：腾讯 stock_zh_a_minute
    period = {"1h": "60", "30m": "30"}[interval_str]
    df = ak.stock_zh_a_minute(symbol=f"{prefix}{symbol}", period=period, adjust="qfq")
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "datetime": r["day"], "open": float(r["open"]), "high": float(r["high"]),
            "low": float(r["low"]), "close": float(r["close"]),
            "volume": float(r["volume"]) if "volume" in df.columns else 0.0,
        })
    return _to_bars(rows, symbol, exchange, "HOUR" if interval_str == "1h" else "MINUTE_30",
                    is_minute=True)


def _to_bars(rows: list, symbol: str, exchange, interval_name: str, is_minute: bool) -> list:
    """rows -> BarData 列表（时间升序、去重）。"""
    from ..core.constant import Interval
    from ..core.object import BarData
    from datetime import timezone

    interval = Interval({"DAILY": "1d", "HOUR": "1h", "MINUTE_30": "30m"}[interval_name])
    bars = []
    seen = set()
    for r in rows:
        dt = r["datetime"]
        if isinstance(dt, str):
            dt = pd.to_datetime(dt).to_pydatetime()
        elif not hasattr(dt, "tzinfo"):  # datetime.date 或 pandas date
            dt = pd.to_datetime(dt).to_pydatetime()
        if dt.tzinfo is None and is_minute:
            # 腾讯分钟为北京时间 → UTC
            dt = pd.Timestamp(dt, tz="Asia/Shanghai").tz_convert("UTC").to_pydatetime()
        elif dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        key = dt.isoformat()
        if key in seen:
            continue
        seen.add(key)
        bars.append(BarData(
            symbol=symbol, exchange=exchange, datetime=dt, interval=interval,
            open_price=r["open"], high_price=r["high"], low_price=r["low"],
            close_price=r["close"], volume=r["volume"], open_interest=0.0, turnover=0.0,
        ))
    bars.sort(key=lambda b: b.datetime)
    return bars


# 市场定义：交易所集合 + 默认周期（A=沪深 / HK=港股 / US=美股）
_MARKET_DEFS = {
    "A": {"exchanges": {"SSE", "SZSE"}, "intervals": ["1d", "1h", "30m"],
         "settings_key": "a_stock", "label": "A股"},
    "HK": {"exchanges": {"HKEX"}, "intervals": ["1d"],
           "settings_key": "hk_stock", "label": "港股"},
    "US": {"exchanges": {"NYSE", "NASDAQ"}, "intervals": ["1d"],
           "settings_key": "us_stock", "label": "美股"},
}


async def _fetch_market_bars(market: str, symbol: str, exchange, interval_str: str) -> tuple:
    """按市场拉取股票 bar（异步，内部处理线程池/超时）。返回 (bars, source)。"""
    if market == "A":
        # A股：腾讯日线/分钟（同步 akshare，放线程池）
        bars = await asyncio.to_thread(_fetch_stock_bars, symbol, exchange, interval_str)
        return bars, "tencent"
    from ..data.feed.base import HistoryRequest
    from ..core.constant import Interval
    req = HistoryRequest(symbol=symbol, exchange=exchange,
                         interval=Interval(_INTERVAL_MAP[interval_str]))
    if market == "HK":
        # 港股：新浪/东财/yfinance 回退链（feed 内部已 to_thread 包网络请求）
        from ..data.feed.em_hk import EmHkFeed
        bars = await asyncio.wait_for(EmHkFeed().fetch_bar_data(req), timeout=45)
        return bars, "em_hk"
    if market == "US":
        # 美股：yfinance（日线）
        from ..data.feed.yfinance_us import YFinanceUSFeed
        bars = await asyncio.wait_for(YFinanceUSFeed().fetch_bar_data(req), timeout=45)
        return bars, "yfinance_us"
    return [], "unknown"


async def _job_stock_auto_download(
    sys_state: Dict[str, Any],
    symbols: Optional[List[str]] = None,
    intervals: Optional[List[str]] = None,
    manual: bool = False,
    market: str = "A",
    progress_callback: Optional[callable] = None,
) -> Dict[str, Any]:
    """股票数据增量更新：按市场（A=沪深 / HK=港股 / US=美股）更新到最新。

    执行流程：
    1. 确定标的池：传入 symbols（手动触发）或仓库内该市场已缓存标的；否则跳过
    2. 对每个标的，遍历周期
    3. 每个周期从真实源拉取，过滤出新增 bar 落盘
    4. 返回统计

    Args:
        symbols: 待更新标的（如 ["600000.SSE"]）。None 则更新该市场全部已缓存标的。
        intervals: 周期列表 ("1d"/"1h"/"30m")，None 用该市场默认档。
        manual: True 手动触发（跳过启用开关检查）。
        market: 市场标识（"A" / "HK" / "US"）。
        progress_callback: 进度回调(current, total, message)。
    """
    dm = sys_state.get("dm")
    if dm is None:
        return {"action": "stock_download", "skipped": True, "reason": "数据管理器不可用"}
    dc = getattr(dm, "disk_cache", None)
    if dc is None:
        return {"action": "stock_download", "skipped": True, "reason": "行情仓库未启用"}

    mdef = _MARKET_DEFS.get(market)
    if mdef is None:
        return {"action": "stock_download", "skipped": True, "reason": f"未知市场: {market}"}
    label = mdef["label"]

    # 自动调度才检查启用开关；手动触发不检查
    if not manual:
        try:
            service = sys_state.get("market_update_settings") or MarketUpdateSettingsService()
            mcfg = service.get_market(mdef["settings_key"])
            if not mcfg.get("enabled", False):
                return {"action": "stock_download", "skipped": True,
                        "reason": f"未启用({label}自动更新已关闭)"}
        except Exception as exc:  # noqa: BLE001
            _logger.exception("stock_download 配置读取失败: %s", exc)
            return {"action": "stock_download", "error": str(exc)}

    _EXCHANGES = mdef["exchanges"]

    # 确定标的池
    if symbols:
        pool = set(symbols)
    else:
        try:
            pool = {
                k["symbol"] + "." + k["exchange"].upper()
                for k in dc.list_keys()
                if (k.get("exchange") or "").upper() in _EXCHANGES
            }
        except Exception as exc:  # noqa: BLE001
            _logger.warning("stock_download 读取仓库键失败: %s", exc)
            pool = set()

    if not pool:
        return {"action": "stock_download", "skipped": True,
                "reason": f"仓库无 {label} 缓存（请先运行「全市场预热」建库，或手动指定标的）"}

    try:
        from ..data.feed.base import HistoryRequest
        from ..core.constant import Exchange, Interval
    except Exception as exc:  # noqa: BLE001
        _logger.exception("stock_download 导入失败: %s", exc)
        return {"action": "stock_download", "error": str(exc)}

    # 确定本次周期：默认用该市场周期档；港股/美股仅日线（数据源限制）
    want = intervals or mdef["intervals"]
    if market != "A":
        want = [iv for iv in want if iv == "1d"] or ["1d"]
    interval_enum = {k: Interval(v) for k, v in _INTERVAL_MAP.items() if k in want}

    results: List[Dict[str, Any]] = []
    updated, failed, up_to_date = 0, 0, 0
    total_tasks = len(pool) * len(interval_enum)
    current_task = 0

    for vt in sorted(pool):
        symbol, exch = vt.rsplit(".", 1)
        try:
            exchange = Exchange(exch)
        except ValueError:
            failed += 1
            continue

        for interval_str, interval in interval_enum.items():
            current_task += 1
            if progress_callback:
                progress_callback(current_task, total_tasks, f"更新 {vt} {interval_str}")
            key = f"{symbol}.{exch}.{interval_str}"
            try:
                # 预检：轻量读本地最新时间戳（只读 datetime 列，放线程池避免阻塞事件循环）
                req_check = HistoryRequest(symbol=symbol, exchange=exchange, interval=interval)
                latest_cached = await asyncio.to_thread(dc.latest_datetime, req_check)

                # 预检跳过：本地已覆盖到最近收盘边界 → 不发源请求（幂等，省流量省时间）
                if _is_up_to_date(latest_cached, interval_str):
                    results.append({"key": key, "status": "up_to_date", "new_rows": 0,
                                    "total_rows": None,
                                    "latest": latest_cached.isoformat() if latest_cached else None,
                                    "skipped": True})
                    up_to_date += 1
                    await asyncio.sleep(0)  # 让出事件循环，保证状态轮询可用
                    continue

                # 按市场拉取：A股腾讯日线/分钟；港股新浪/东财链；美股 yfinance（日线）。
                # 全部带 45s 兜底超时，事件循环不再被长同步段阻塞
                bars, source = await _fetch_market_bars(market, symbol, exchange, interval_str)

                if not bars:
                    results.append({"key": key, "status": "up_to_date", "new_rows": 0,
                                    "total_rows": None,
                                    "latest": latest_cached.isoformat() if latest_cached else None})
                    up_to_date += 1
                    await asyncio.sleep(0)
                    continue

                new_bars = [b for b in bars if latest_cached is None or b.datetime > latest_cached]
                if new_bars:
                    n = await asyncio.to_thread(dc.save, new_bars)
                    results.append({"key": key, "status": "ok", "source": source,
                                    "new_rows": len(new_bars), "total_rows": n,
                                    "latest": new_bars[-1].datetime.isoformat()})
                    updated += 1
                else:
                    results.append({"key": key, "status": "up_to_date", "new_rows": 0,
                                    "total_rows": None,
                                    "latest": latest_cached.isoformat() if latest_cached else None})
                    up_to_date += 1

                await asyncio.sleep(0.2)  # 温和限速
            except Exception as exc:  # noqa: BLE001
                failed += 1
                results.append({"key": key, "status": "error", "error": str(exc)[:100]})
                _logger.error("%s增量更新失败 %s: %s", label, key, exc)

    return {
        "action": "stock_download",
        "market": market,
        "updated": updated,
        "failed": failed,
        "up_to_date": up_to_date,
        "total": total_tasks,
        "results": results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


__all__ = ["_job_stock_auto_download"]
