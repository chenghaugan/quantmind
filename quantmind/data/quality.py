"""数据质量检查（间隙/异常尖峰/新鲜度/换月跳变标记）。"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List

from ..core.constant import Interval
from ..core.object import BarData

_EXPECTED_GAP = {
    Interval.DAILY: timedelta(days=1),
    Interval.MINUTE: timedelta(minutes=1),
    Interval.MINUTE_5: timedelta(minutes=5),
    Interval.HOUR: timedelta(hours=1),
}


@dataclass
class QualityReport:
    symbol: str
    total: int = 0
    gaps: int = 0
    outliers: int = 0
    rollover_jumps: int = 0
    last_ts: datetime | None = None
    stale: bool = False
    issues: List[str] = field(default_factory=list)


def check_bars(bars: List[BarData], interval: Interval, freshness: timedelta | None = None) -> QualityReport:
    """对一组 bar 做基础质量检查。"""
    rep = QualityReport(symbol=bars[0].vt_symbol if bars else "")
    if not bars:
        rep.issues.append("空数据")
        return rep

    rep.total = len(bars)
    rep.last_ts = bars[-1].datetime
    expected_gap = _EXPECTED_GAP.get(interval, timedelta(days=1))

    for i in range(1, len(bars)):
        prev, cur = bars[i - 1], bars[i]
        # 间隙
        if cur.datetime - prev.datetime > expected_gap * 3:
            rep.gaps += 1
        # 异常尖峰（单根涨跌 > 20% 视为可疑，主连换月常见，单独标记）
        if prev.close_price:
            chg = abs(cur.close_price - prev.close_price) / prev.close_price
            if chg > 0.2:
                if chg > 0.5:
                    rep.rollover_jumps += 1
                else:
                    rep.outliers += 1

    # 新鲜度
    if freshness is not None:
        # bar 约定为 naive UTC，用本地时钟在 UTC+8 环境会误判“年轻”8 小时
        now = datetime.now(timezone.utc).replace(tzinfo=None) \
            if bars[-1].datetime.tzinfo is None else datetime.now(tz=bars[-1].datetime.tzinfo)
        if now - bars[-1].datetime > freshness:
            rep.stale = True
            rep.issues.append("数据过期")

    if rep.gaps:
        rep.issues.append(f"发现 {rep.gaps} 处时间间隙")
    if rep.outliers:
        rep.issues.append(f"发现 {rep.outliers} 个异常尖峰")
    if rep.rollover_jumps:
        rep.issues.append(f"发现 {rep.rollover_jumps} 处疑似换月跳变")
    return rep
