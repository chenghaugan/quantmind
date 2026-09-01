"""期货数据自动下载任务。

支持股指期货 + 商品期货，可自定义品种、交易所、周期。
调度时间从配置文件读取，支持在 Web 界面动态修改。

设计要点：
  - 增量更新：DiskBarCache.save 自动合并去重
  - 失败回退：单源失败不影响整体，自动降级到下一数据源
  - 可配置：品种、周期、调度时间均可在 Web 界面配置
  - 进度追踪：支持实时进度更新，前端可轮询查看
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_logger = logging.getLogger("quantmind.scheduler.futures")


# ---------------------------------------------------------------------------
# 期货品种配置（完整列表）
# ---------------------------------------------------------------------------
# 交易所映射（品种代码 -> 交易所代码）
FUTURES_EXCHANGE = {
    # 股指期货（CFFEX）
    "IF0": "CFFEX", "IC0": "CFFEX", "IH0": "CFFEX", "IM0": "CFFEX",
    # 商品期货-黑色系（SHFE/DCE）
    "rb0": "SHFE", "hc0": "SHFE", "i0": "DCE", "j0": "DCE", "jm0": "DCE",
    # 商品期货-有色金属（SHFE）
    "cu0": "SHFE", "al0": "SHFE", "zn0": "SHFE", "pb0": "SHFE",
    "ni0": "SHFE", "sn0": "SHFE",
    # 商品期货-贵金属（SHFE）
    "au0": "SHFE", "ag0": "SHFE",
    # 商品期货-能源化工（SHFE/INE/CZCE/DCE）
    "sc0": "INE", "fu0": "SHFE", "lu0": "SHFE", "bu0": "SHFE", "ru0": "SHFE",
    "MA0": "CZCE", "TA0": "CZCE", "PP0": "DCE", "L0": "DCE", "V0": "DCE",
    "eg0": "DCE", "eb0": "DCE",
    # 商品期货-农产品（DCE/CZCE）
    "m0": "DCE", "y0": "DCE", "a0": "DCE", "p0": "DCE",
    "OI0": "CZCE", "RM0": "CZCE", "SR0": "CZCE", "CF0": "CZCE", "AP0": "CZCE",
}


_REUSABLE_TQSDK = None


def _get_reusable_tqsdk():
    """返回进程级复用的 TqSdkFeed 单例（避免每次新建/关闭连接）。"""
    global _REUSABLE_TQSDK
    if _REUSABLE_TQSDK is None:
        from ..data.feed.tqsdk_feed import TqSdkFeed
        _REUSABLE_TQSDK = TqSdkFeed()
    return _REUSABLE_TQSDK


async def _fetch_by_strategy(
    req: HistoryRequest,
    cached_bars: Optional[List[Any]],
) -> tuple:
    """按缓存状态选择数据源拉取，返回 (bars, source_name)。

    策略：
      - 首次（缓存为空）：用 TqSdk，拉完整历史（最多 8000 根，纵深长）
      - 增量（缓存有数据）：用 akshare，只拉最新 1023 根，本地过滤新增

    注意：TqSdk 连接在进程内复用（模块级单例），不随每次调用新建/关闭，
    避免反复握手导致的缓慢与卡顿。
    """
    is_first = (not cached_bars)
    
    if is_first:
        # 首次：TqSdk 完整历史（复用连接）
        from ..data.feed.tqsdk_feed import TqSdkFeed
        feed = _get_reusable_tqsdk()
        try:
            bars = await feed.fetch_bar_data(req)
            if bars:
                return bars, "tqsdk"
        except Exception as exc:  # noqa: BLE001
            _logger.warning("TqSdk 首次拉取失败(%s)，回退 akshare: %s", req.symbol, exc)
    
    # 增量（或首次 TqSdk 失败）：akshare 最新数据
    from ..data.feed.akshare_future import AkShareFuturesFeed
    feed = AkShareFuturesFeed()
    bars = await feed.fetch_bar_data(req)
    if bars:
        return bars, "akshare_future"
    # akshare 也拿不到，回退 TqSdk（增量场景兜底，复用连接）
    if not is_first:
        from ..data.feed.tqsdk_feed import TqSdkFeed
        feed2 = _get_reusable_tqsdk()
        try:
            bars2 = await feed2.fetch_bar_data(req)
            if bars2:
                return bars2, "tqsdk"
        except Exception:  # noqa: BLE001
            pass
    return [], ""
    return [], ""


async def _job_futures_auto_download(
    sys_state: Dict[str, Any],
    symbols: Optional[List[str]] = None,
    intervals: Optional[List[str]] = None,
    progress_callback: Optional[callable] = None,
) -> Dict[str, Any]:
    """期货数据自动下载：收盘后增量更新数据。
    
    执行流程：
    1. 从配置文件读取要下载的品种和周期（或使用传入参数）
    2. 遍历所有品种 × 周期
    3. 检查本地缓存的最新时间戳
    4. 只下载新增数据（增量更新）
    5. 通过 DataManager 拉取数据（自动回退：TqSdk → akshare）
    6. 写入 DiskBarCache（幂等合并）
    7. 记录下载日志
    
    Args:
        sys_state: 系统状态（包含 dm, ee 等）
        symbols: 要下载的品种列表（None 则从配置读取）
        intervals: 要下载的周期列表（None 则从配置读取）
        progress_callback: 进度回调函数，签名 callback(current, total, message)
    
    Returns:
        dict: 包含下载统计信息
    """
    dm = sys_state.get("dm")
    if dm is None:
        return {"action": "futures_download", "skipped": True, "reason": "数据管理器不可用"}
    
    dc = getattr(dm, "disk_cache", None)
    if dc is None:
        return {"action": "futures_download", "skipped": True, "reason": "行情仓库未启用"}
    
    # 读取配置或使用传入参数
    if symbols is None or intervals is None:
        settings_service = sys_state.get("futures_download_settings")
        if settings_service is None:
            symbols = symbols or ["IF0", "IC0", "IH0", "IM0"]
            intervals = intervals or ["1d", "60m", "30m", "15m", "5m", "1m"]
        else:
            config = settings_service.get()
            symbols = symbols or config.get("symbols", ["IF0", "IC0", "IH0", "IM0"])
            intervals = intervals or config.get("intervals", ["1d", "60m", "30m", "15m", "5m", "1m"])
    
    try:
        from ..data.feed.base import HistoryRequest
        from ..core.constant import Exchange, Interval
    except Exception as exc:
        _logger.exception("futures_download 导入失败: %s", exc)
        return {"action": "futures_download", "error": str(exc)}
    
    # 周期映射
    interval_enum_map = {
        "1d": Interval.DAILY,
        "60m": Interval.HOUR,
        "30m": Interval.MINUTE_30,
        "15m": Interval.MINUTE_15,
        "5m": Interval.MINUTE_5,
        "3m": Interval.MINUTE_3,
        "1m": Interval.MINUTE,
        "2h": Interval.HOUR_2,
        "4h": Interval.HOUR_4,
    }
    
    # 周期间隔（秒）
    interval_seconds_map = {
        "1d": 86400,
        "60m": 3600,
        "30m": 1800,
        "15m": 900,
        "5m": 300,
        "3m": 180,
        "1m": 60,
        "2h": 7200,
        "4h": 14400,
    }

    results: List[Dict[str, Any]] = []
    downloaded, failed, skipped, up_to_date = 0, 0, 0, 0
    
    # 计算总任务数
    total_tasks = len(symbols) * len(intervals)
    current_task = 0
    
    # 遍历所有品种 × 周期
    for symbol in symbols:
        exchange_str = FUTURES_EXCHANGE.get(symbol)
        if not exchange_str:
            _logger.warning("未知品种 %s，跳过", symbol)
            failed += 1
            results.append({"symbol": symbol, "status": "unknown_symbol"})
            continue
        
        try:
            exchange = Exchange(exchange_str)
        except ValueError:
            failed += 1
            results.append({"symbol": symbol, "status": "bad_exchange"})
            continue
        
        for interval_str in intervals:
            interval = interval_enum_map.get(interval_str)
            if interval is None:
                continue
            
            current_task += 1
            key = f"{symbol}.{exchange_str}.{interval_str}"
            
            # 进度回调
            if progress_callback:
                progress_callback(current_task, total_tasks, f"检查 {key}...")
            
            try:
                # 检查本地缓存的最新时间戳
                req_check = HistoryRequest(
                    symbol=symbol,
                    exchange=exchange,
                    interval=interval,
                )
                cached_bars = dc.load(req_check)
                
                latest_cached = None
                if cached_bars:
                    latest_cached = max(b.datetime for b in cached_bars)
                    _logger.debug("%s 缓存最新: %s (%d 根)", key, latest_cached, len(cached_bars))
                
                # 按策略拉取：首次用 TqSdk（完整历史），增量用 akshare（最新数据）
                req = HistoryRequest(
                    symbol=symbol,
                    exchange=exchange,
                    interval=interval,
                )
                try:
                    bars, source = await _fetch_by_strategy(req, cached_bars)
                except Exception as exc:  # noqa: BLE001
                    bars, source = [], ""
                    _logger.error("按策略拉取失败 %s: %s", key, exc)
                
                if bars:
                    # 过滤出新增数据（比 latest_cached 更新的）
                    if latest_cached:
                        new_bars = [b for b in bars if b.datetime > latest_cached]
                    else:
                        new_bars = bars  # 首次下载，全部是新数据
                    
                    if new_bars:
                        # 写入仓库（幂等合并）
                        n = dc.save(new_bars)
                        results.append({
                            "key": key,
                            "status": "ok",
                            "source": source,
                            "new_rows": len(new_bars),
                            "total_rows": n,
                            "latest": new_bars[-1].datetime.isoformat(),
                            "cached_before": len(cached_bars) if cached_bars else 0,
                        })
                        downloaded += 1
                        _logger.info("期货增量更新: %s 新增 %d 根 (总计 %d 根, 来源:%s)", 
                                    key, len(new_bars), n, source)
                    else:
                        # 没有新数据
                        results.append({
                            "key": key,
                            "status": "up_to_date",
                            "source": "disk_cache",
                            "new_rows": 0,
                            "total_rows": len(cached_bars),
                            "latest": latest_cached.isoformat(),
                        })
                        up_to_date += 1
                        _logger.info("期货数据已是最新: %s (%d 根)", key, len(cached_bars))
                else:
                    if latest_cached:
                        # 数据源返回空，但本地有缓存
                        results.append({
                            "key": key,
                            "status": "up_to_date",
                            "source": "disk_cache",
                            "new_rows": 0,
                            "total_rows": len(cached_bars),
                            "latest": latest_cached.isoformat(),
                        })
                        up_to_date += 1
                    else:
                        # 首次下载且数据源返回空
                        results.append({"key": key, "status": "empty"})
                        skipped += 1
                    _logger.debug("期货下载无新数据: %s", key)
                
                # 温和限速，避免触发数据源限流
                await asyncio.sleep(0.3)
                
            except Exception as exc:
                failed += 1
                results.append({"key": key, "status": "error", "error": str(exc)[:100]})
                _logger.error("期货下载失败: %s - %s", key, exc)
    
    return {
        "action": "futures_download",
        "downloaded": downloaded,
        "failed": failed,
        "skipped": skipped,
        "up_to_date": up_to_date,
        "total": total_tasks,
        "results": results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


__all__ = ["_job_futures_auto_download"]
