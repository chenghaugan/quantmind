"""RiskService：风控限额档位、委托预检、交易日历查询。

把 :mod:`quantmind.risk` 的能力暴露给 Web，使风控闸门在上实盘前可视化、可试算：
下单前先在这里跑一遍 ``check_order``，就能看到会不会被哪一条限额拦下。
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from ...core.constant import Direction, Exchange, Offset
from ...core.contracts import default_size
from ...core.gateway import OrderRequest
from ...core.object import PositionData
from ...risk import RiskEngine, RiskLimits
from ...risk.calendar import TradingCalendar, day_sessions_for, night_close_time
from ..schemas import RiskCheckRequest

_logger = logging.getLogger("quantmind.api")

#: 档位名 -> 构造器
PROFILES = {
    "default": RiskLimits,
    "conservative": RiskLimits.conservative,
    "unlimited": RiskLimits.unlimited,
}

PROFILE_LABELS = {
    "default": "默认档（百万级中低频组合）",
    "conservative": "保守档（小资金 / 新策略首次上实盘）",
    "unlimited": "不限档（仅供测试回放，禁止实盘）",
}

#: 限额字段中文说明，供前端表格直接使用
LIMIT_LABELS: Dict[str, str] = {
    "max_order_volume": "单笔最大手数",
    "min_order_volume": "单笔最小手数",
    "volume_tick": "手数步进",
    "max_price_deviation": "限价偏离上限",
    "max_position_volume": "单合约净持仓上限",
    "max_position_value": "单合约名义市值上限",
    "max_margin_ratio": "保证金占用率上限",
    "max_daily_loss": "单日最大亏损（元）",
    "max_daily_loss_ratio": "单日最大亏损率",
    "max_drawdown_ratio": "最大回撤熔断线",
    "max_orders_per_day": "单日下单笔数上限",
    "max_orders_per_minute": "每分钟下单笔数上限",
    "max_trade_volume_per_day": "单日成交手数上限",
    "forbidden_symbols": "黑名单",
    "allowed_symbols": "白名单",
    "check_trading_session": "校验交易时段",
    "allow_open": "允许开仓",
    "self_trade_guard": "防自成交",
}

#: 拒单代码中文说明
CODE_LABELS: Dict[str, str] = {
    "PASS": "通过",
    "SYMBOL_FORBIDDEN": "合约在黑名单",
    "SYMBOL_NOT_ALLOWED": "合约不在白名单",
    "NOT_TRADING_TIME": "非交易时段",
    "OPEN_FORBIDDEN": "熔断后禁止开仓",
    "KILL_SWITCH": "全局熔断",
    "ORDER_VOLUME_TOO_LARGE": "单笔手数超限",
    "ORDER_VOLUME_TOO_SMALL": "单笔手数过小",
    "VOLUME_TICK_INVALID": "手数不符合步进",
    "PRICE_DEVIATION": "限价偏离过大",
    "PRICE_INVALID": "价格非法",
    "POSITION_LIMIT": "持仓手数超限",
    "POSITION_VALUE_LIMIT": "持仓市值超限",
    "MARGIN_LIMIT": "保证金占用超限",
    "DAILY_LOSS_LIMIT": "触发单日亏损熔断",
    "DRAWDOWN_LIMIT": "触发回撤熔断",
    "ORDER_COUNT_DAILY": "单日下单笔数超限",
    "ORDER_RATE_LIMIT": "下单频率超限",
    "TRADE_VOLUME_DAILY": "单日成交手数超限",
    "SELF_TRADE": "存在反向活动挂单（自成交）",
    "CLOSE_EXCEEDS_POSITION": "平仓量超过持仓",
}


def _build_limits(profile: str, overrides: Optional[Dict[str, Any]] = None) -> RiskLimits:
    factory = PROFILES.get(profile, RiskLimits)
    limits = factory()
    for k, v in (overrides or {}).items():
        if not hasattr(limits, k):
            continue
        if k in ("forbidden_symbols", "allowed_symbols"):
            v = set(v or [])
        setattr(limits, k, v)
    return limits


class RiskService:
    """风控查询与试算（无状态，每次请求构造独立引擎，互不污染）。"""

    def __init__(self, calendar: Optional[TradingCalendar] = None) -> None:
        self.calendar = calendar or TradingCalendar()

    # ---- 档位 ----
    def profiles(self) -> Dict[str, Any]:
        return {
            "profiles": [
                {
                    "name": name,
                    "label": PROFILE_LABELS.get(name, name),
                    "limits": factory().to_dict(),
                }
                for name, factory in PROFILES.items()
            ],
            "labels": LIMIT_LABELS,
            "codes": CODE_LABELS,
        }

    # ---- 委托预检 ----
    def check_order(self, req: RiskCheckRequest) -> Dict[str, Any]:
        try:
            symbol, _, exch = req.vt_symbol.rpartition(".")
            if not symbol:
                symbol, exch = req.vt_symbol, "SHFE"
            exchange = Exchange(exch.upper())
        except ValueError as e:
            return {"error": f"非法合约代码：{req.vt_symbol}（{e}）"}

        overrides = dict(req.overrides or {})
        overrides.setdefault("check_trading_session", req.check_session)
        limits = _build_limits(req.profile, overrides)

        engine = RiskEngine(limits=limits, initial_equity=req.equity)
        engine.state.margin_used = req.margin_used

        try:
            direction = Direction(req.direction)
            offset = Offset(req.offset)
        except ValueError as e:
            return {"error": f"非法方向/开平：{e}"}

        order = OrderRequest(
            symbol=symbol, exchange=exchange, direction=direction,
            offset=offset, volume=req.volume, price=req.price,
        )
        position = None
        if req.position_volume:
            position = PositionData(
                symbol=symbol, exchange=exchange,
                direction=Direction.NET, volume=req.position_volume,
            )

        decision = engine.check_order(
            order,
            position=position,
            last_price=req.last_price or req.price,
            equity=req.equity,
            margin_used=req.margin_used,
        )
        d = decision.to_dict()
        d["code_label"] = CODE_LABELS.get(d["code"], d["code"])

        size = default_size(req.vt_symbol)
        notional = abs(req.volume) * (req.last_price or req.price) * size
        return {
            "decision": d,
            "profile": req.profile,
            "limits": limits.to_dict(),
            "state": engine.state.to_dict(),
            "context": {
                "vt_symbol": req.vt_symbol,
                "contract_size": size,
                "notional": round(notional, 2),
                "margin_estimate": round(notional * engine._margin_rate(req.vt_symbol), 2),
                "margin_rate": engine._margin_rate(req.vt_symbol),
            },
        }

    # ---- 交易日历 ----
    def calendar_info(self, day: Optional[str] = None, symbol: str = "rb0",
                      exchange: str = "SHFE", horizon: int = 14) -> Dict[str, Any]:
        try:
            d = date.fromisoformat(day) if day else datetime.now().date()
        except ValueError:
            return {"error": f"非法日期：{day}"}

        cal = self.calendar
        sessions = [[a.strftime("%H:%M"), b.strftime("%H:%M")]
                    for a, b in day_sessions_for(symbol, exchange)]
        night_end = night_close_time(symbol, exchange)

        upcoming: List[dict] = []
        for i in range(horizon):
            cur = d + timedelta(days=i)
            upcoming.append({
                "date": cur.isoformat(),
                "weekday": cur.weekday(),
                "is_trading_day": cal.is_trading_day(cur),
                "has_night_session": cal.has_night_session(cur) if cal.is_trading_day(cur) else False,
            })

        now = datetime.now()
        nxt = cal.next_trading_day(d)
        prv = cal.prev_trading_day(d)
        return {
            "date": d.isoformat(),
            "symbol": symbol,
            "exchange": exchange.upper(),
            "is_trading_day": cal.is_trading_day(d),
            "has_night_session": cal.has_night_session(d),
            "next_trading_day": nxt.isoformat() if nxt else None,
            "prev_trading_day": prv.isoformat() if prv else None,
            "day_sessions": sessions,
            "night_close": night_end.strftime("%H:%M") if night_end else None,
            "now": now.isoformat(timespec="seconds"),
            "now_is_trading_time": cal.is_trading_time(now, symbol, exchange),
            "now_session": cal.session_name(now, symbol, exchange),
            "upcoming": upcoming,
            "holiday_count": len(cal.holidays),
        }
