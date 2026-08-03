"""交易日历与交易时段（中国市场，含期货夜盘）。

为什么实盘必须有这一层
----------------------
回测里「有 K 线就能成交」，实盘不行：非交易时段下单会被交易所直接拒绝，
更糟的是**集合竞价/收盘后残留的委托**会在下一交易日以完全不同的价格成交。
因此实盘路径必须在发单前做时段闸门。

时间约定
--------
框架内部时间一律 **UTC**（见 ``core.object``）。本模块统一把传入时间换算为
**北京时间（UTC+8）** 再判断时段。传入 naive datetime 时视为 UTC。

节假日数据
----------
``HOLIDAYS`` 内置 2024–2026 年中国大陆法定休市日（**参考值，须按交易所公告校准**）。
可通过三种方式覆盖：

  1. ``TradingCalendar(holidays={date(2026, 1, 1), ...})``
  2. 环境变量 ``QM_HOLIDAY_FILE`` 指向每行一个 ``YYYY-MM-DD`` 的文本文件
  3. ``TradingCalendar.from_file(path)``
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Set, Tuple

UTC = timezone.utc
CST = timezone(timedelta(hours=8))  # 北京时间

# --------------------------------------------------------------------------
# 节假日（中国大陆休市日，含法定节假日调休放假日；周末自动休市不列入）
# 注意：调休「补班」的周六/周日交易所**不开市**，故周末一律视为非交易日。
# --------------------------------------------------------------------------
_HOLIDAY_STR = """
2024-01-01
2024-02-09 2024-02-12 2024-02-13 2024-02-14 2024-02-15 2024-02-16
2024-04-04 2024-04-05
2024-05-01 2024-05-02 2024-05-03
2024-06-10
2024-09-16 2024-09-17
2024-10-01 2024-10-02 2024-10-03 2024-10-04 2024-10-07
2025-01-01
2025-01-28 2025-01-29 2025-01-30 2025-01-31 2025-02-03 2025-02-04
2025-04-04
2025-05-01 2025-05-02 2025-05-05
2025-05-31 2025-06-02
2025-10-01 2025-10-02 2025-10-03 2025-10-06 2025-10-07 2025-10-08
2026-01-01 2026-01-02
2026-02-16 2026-02-17 2026-02-18 2026-02-19 2026-02-20 2026-02-23
2026-04-06
2026-05-01 2026-05-04 2026-05-05
2026-06-19
2026-09-25
2026-10-01 2026-10-02 2026-10-05 2026-10-06 2026-10-07 2026-10-08
"""

HOLIDAYS: Set[date] = {
    date.fromisoformat(tok) for tok in _HOLIDAY_STR.split() if tok.strip()
}

# --------------------------------------------------------------------------
# 日盘时段（北京时间，[start, end) 半开区间）
# --------------------------------------------------------------------------
_T = time

# 商品期货（SHFE/DCE/CZCE/INE/GFEX）通用日盘
_COMMODITY_DAY: List[Tuple[time, time]] = [
    (_T(9, 0), _T(10, 15)),
    (_T(10, 30), _T(11, 30)),
    (_T(13, 30), _T(15, 0)),
]

# 中金所股指期货（IF/IC/IH/IM）
_CFFEX_INDEX_DAY: List[Tuple[time, time]] = [
    (_T(9, 30), _T(11, 30)),
    (_T(13, 0), _T(15, 0)),
]
# 中金所国债期货（T/TF/TS/TL）收盘 15:15
_CFFEX_BOND_DAY: List[Tuple[time, time]] = [
    (_T(9, 30), _T(11, 30)),
    (_T(13, 0), _T(15, 15)),
]

# A 股 / ETF / 股票期权
_STOCK_DAY: List[Tuple[time, time]] = [
    (_T(9, 30), _T(11, 30)),
    (_T(13, 0), _T(15, 0)),
]

# 港股
_HK_DAY: List[Tuple[time, time]] = [
    (_T(9, 30), _T(12, 0)),
    (_T(13, 0), _T(16, 0)),
]

DAY_SESSIONS: Dict[str, List[Tuple[time, time]]] = {
    "SHFE": _COMMODITY_DAY,
    "DCE": _COMMODITY_DAY,
    "CZCE": _COMMODITY_DAY,
    "INE": _COMMODITY_DAY,
    "GFEX": _COMMODITY_DAY,
    "CFFEX": _CFFEX_INDEX_DAY,
    "SSE": _STOCK_DAY,
    "SZSE": _STOCK_DAY,
    "HKEX": _HK_DAY,
    "SGE": _COMMODITY_DAY,
}

# --------------------------------------------------------------------------
# 夜盘收盘时间（按品种前缀）。缺省 = 无夜盘。
#   23:00 —— 黑色/化工/多数农产品
#   01:00 —— 有色金属
#   02:30 —— 贵金属 + 原油
# --------------------------------------------------------------------------
_NIGHT_2300 = {
    "RB", "HC", "SS", "BU", "FU", "SP", "RU", "NR", "BR",
    "I", "J", "JM", "PP", "L", "V", "EG", "EB", "PG",
    "M", "Y", "P", "A", "B", "C", "CS", "RR",
    "TA", "MA", "SR", "CF", "CY", "OI", "RM", "FG", "SA", "UR", "ZC",
    "SF", "SM", "PX", "SH", "PF", "PR",
}
_NIGHT_0100 = {"CU", "AL", "ZN", "PB", "NI", "SN", "AO", "BC", "AD"}
_NIGHT_0230 = {"AU", "AG", "SC"}

# 明确无夜盘：JD LH AP CJ PK SI LC PS EC WH RS RI LR PM JR 及全部中金所品种
_CFFEX_PREFIXES = {"IF", "IC", "IH", "IM", "T", "TF", "TS", "TL", "IO", "MO", "HO"}


def _product_prefix(symbol: str) -> str:
    """从合约代码提取品种前缀（字母部分，大写）。

    ``rb2410`` → ``RB``；``rb0`` → ``RB``；``IF2409`` → ``IF``；``600000`` → ``""``。
    """
    s = symbol.strip().upper()
    out = []
    for ch in s:
        if ch.isalpha():
            out.append(ch)
        else:
            break
    return "".join(out)


def night_close_time(symbol: str, exchange: str) -> Optional[time]:
    """返回该品种的夜盘收盘时间；无夜盘返回 ``None``。"""
    if exchange.upper() in {"CFFEX", "SSE", "SZSE", "HKEX"}:
        return None
    prefix = _product_prefix(symbol)
    if not prefix or prefix in _CFFEX_PREFIXES:
        return None
    if prefix in _NIGHT_0230:
        return time(2, 30)
    if prefix in _NIGHT_0100:
        return time(1, 0)
    if prefix in _NIGHT_2300:
        return time(23, 0)
    return None


NIGHT_OPEN = time(21, 0)


def day_sessions_for(symbol: str, exchange: str) -> List[Tuple[time, time]]:
    """返回该合约的日盘时段列表。"""
    exch = exchange.upper()
    if exch == "CFFEX":
        prefix = _product_prefix(symbol)
        if prefix in {"T", "TF", "TS", "TL"}:
            return _CFFEX_BOND_DAY
        return _CFFEX_INDEX_DAY
    return DAY_SESSIONS.get(exch, _COMMODITY_DAY)


# --------------------------------------------------------------------------
# 交易日历
# --------------------------------------------------------------------------
@dataclass
class TradingCalendar:
    """交易日历（周末 + 节假日休市）。"""

    holidays: Set[date] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.holidays is None:
            self.holidays = set(_load_env_holidays() or HOLIDAYS)

    # ---- 交易日 ----
    def is_trading_day(self, d: date) -> bool:
        if isinstance(d, datetime):
            d = d.date()
        if d.weekday() >= 5:  # 周六=5 周日=6
            return False
        return d not in self.holidays

    def next_trading_day(self, d: date, max_search: int = 30) -> Optional[date]:
        if isinstance(d, datetime):
            d = d.date()
        cur = d
        for _ in range(max_search):
            cur = cur + timedelta(days=1)
            if self.is_trading_day(cur):
                return cur
        return None

    def prev_trading_day(self, d: date, max_search: int = 30) -> Optional[date]:
        if isinstance(d, datetime):
            d = d.date()
        cur = d
        for _ in range(max_search):
            cur = cur - timedelta(days=1)
            if self.is_trading_day(cur):
                return cur
        return None

    def has_night_session(self, d: date) -> bool:
        """``d`` 日晚是否开夜盘：``d`` 是交易日且**次日也是交易日**。

        节假日前最后一个交易日晚上不开夜盘（夜盘归属下一交易日）。
        """
        if not self.is_trading_day(d):
            return False
        return self.is_trading_day(d + timedelta(days=1))

    # ---- 交易时段 ----
    def is_trading_time(
        self,
        dt: datetime,
        symbol: str = "",
        exchange: str = "SHFE",
    ) -> bool:
        """判断 ``dt``（UTC 或带时区）是否处于该合约的交易时段。"""
        bj = _to_beijing(dt)
        d, t = bj.date(), bj.time()
        exch = exchange.upper()

        # 1) 日盘
        if self.is_trading_day(d):
            for start, end in day_sessions_for(symbol, exch):
                if start <= t < end:
                    return True

        # 2) 夜盘（21:00 之后，属于当日晚开的夜盘）
        night_end = night_close_time(symbol, exch)
        if night_end is None:
            return False
        # 收盘时间 <= 21:00 说明跨零点（01:00 / 02:30）；23:00 收盘则当日结束
        crosses_midnight = night_end <= NIGHT_OPEN
        if t >= NIGHT_OPEN:
            if crosses_midnight or t < night_end:
                return self.has_night_session(d)
            return False

        # 3) 跨零点夜盘（00:00 ~ night_end，属于前一日晚开的夜盘）
        if crosses_midnight and t < night_end:
            prev = d - timedelta(days=1)
            return self.has_night_session(prev)
        return False

    def session_name(self, dt: datetime, symbol: str = "", exchange: str = "SHFE") -> str:
        """返回当前所处时段名称（用于日志/告警）。"""
        bj = _to_beijing(dt)
        t = bj.time()
        if not self.is_trading_time(dt, symbol, exchange):
            return "CLOSED"
        if t >= NIGHT_OPEN or t < time(6, 0):
            return "NIGHT"
        if t < time(12, 0):
            return "MORNING"
        return "AFTERNOON"

    @classmethod
    def from_file(cls, path: str) -> "TradingCalendar":
        """从文件加载节假日（每行/每空白分隔一个 ``YYYY-MM-DD``，``#`` 开头为注释）。"""
        return cls(holidays=_parse_holiday_file(path))


def _to_beijing(dt: datetime) -> datetime:
    """UTC / naive(视为 UTC) / 带时区 → 北京时间。"""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(CST)


def _parse_holiday_file(path: str) -> Set[date]:
    out: Set[date] = set()
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.split("#", 1)[0]
            for tok in line.split():
                try:
                    out.add(date.fromisoformat(tok))
                except ValueError:
                    continue
    return out


def _load_env_holidays() -> Optional[Set[date]]:
    path = os.environ.get("QM_HOLIDAY_FILE")
    if path and os.path.exists(path):
        try:
            return _parse_holiday_file(path)
        except OSError:
            return None
    return None


# 模块级默认日历（无状态、可共享）
DEFAULT_CALENDAR = TradingCalendar()


def is_trading_day(d: date) -> bool:
    """便捷函数：使用默认日历判断交易日。"""
    return DEFAULT_CALENDAR.is_trading_day(d)


def is_trading_time(dt: datetime, symbol: str = "", exchange: str = "SHFE") -> bool:
    """便捷函数：使用默认日历判断交易时段。"""
    return DEFAULT_CALENDAR.is_trading_time(dt, symbol, exchange)


def beijing_time(dt: datetime) -> datetime:
    """便捷函数：转北京时间（供日志展示）。"""
    return _to_beijing(dt)
