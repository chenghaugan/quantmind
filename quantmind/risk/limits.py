"""风控限额定义与检查结果。

限额分五类：

1. **单笔**：单笔手数上限、手数步进（整数手）、限价单价格偏离保护。
2. **单品种**：净持仓手数上限、持仓名义市值上限。
3. **组合**：总保证金占用率上限、单日亏损、最大回撤熔断。
4. **频率**：单日下单笔数、每分钟下单笔数、单日成交手数（防程序失控/自成交）。
5. **准入**：黑白名单、交易时段、是否允许开仓。

所有阈值默认值偏**保守**；``None`` 表示该项不检查。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Set


class RiskCode(str, Enum):
    """风控拒单代码（便于日志聚合与告警分类）。"""

    PASS = "PASS"
    # 准入
    SYMBOL_FORBIDDEN = "SYMBOL_FORBIDDEN"        # 在黑名单
    SYMBOL_NOT_ALLOWED = "SYMBOL_NOT_ALLOWED"    # 不在白名单
    NOT_TRADING_TIME = "NOT_TRADING_TIME"        # 非交易时段
    OPEN_FORBIDDEN = "OPEN_FORBIDDEN"            # 熔断后禁止开仓
    KILL_SWITCH = "KILL_SWITCH"                  # 全局熔断（连平仓也停）
    # 单笔
    ORDER_VOLUME_TOO_LARGE = "ORDER_VOLUME_TOO_LARGE"
    ORDER_VOLUME_TOO_SMALL = "ORDER_VOLUME_TOO_SMALL"
    VOLUME_TICK_INVALID = "VOLUME_TICK_INVALID"
    PRICE_DEVIATION = "PRICE_DEVIATION"
    PRICE_INVALID = "PRICE_INVALID"
    # 单品种
    POSITION_LIMIT = "POSITION_LIMIT"
    POSITION_VALUE_LIMIT = "POSITION_VALUE_LIMIT"
    # 组合
    MARGIN_LIMIT = "MARGIN_LIMIT"
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"
    DRAWDOWN_LIMIT = "DRAWDOWN_LIMIT"
    # 频率
    ORDER_COUNT_DAILY = "ORDER_COUNT_DAILY"
    ORDER_RATE_LIMIT = "ORDER_RATE_LIMIT"
    TRADE_VOLUME_DAILY = "TRADE_VOLUME_DAILY"
    # 其他
    SELF_TRADE = "SELF_TRADE"                    # 同合约存在反向活动挂单
    CLOSE_EXCEEDS_POSITION = "CLOSE_EXCEEDS_POSITION"  # 平仓量超过持仓


@dataclass
class RiskDecision:
    """风控判定结果。

    ``passed=False`` 时 ``code``/``reason`` 说明原因；引擎据此拒单并告警。
    """

    passed: bool
    code: RiskCode = RiskCode.PASS
    reason: str = ""
    vt_symbol: str = ""

    def __bool__(self) -> bool:  # 允许 ``if decision:`` 写法
        return self.passed

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "code": self.code.value,
            "reason": self.reason,
            "vt_symbol": self.vt_symbol,
        }

    @classmethod
    def ok(cls, vt_symbol: str = "") -> "RiskDecision":
        return cls(passed=True, code=RiskCode.PASS, vt_symbol=vt_symbol)

    @classmethod
    def reject(cls, code: RiskCode, reason: str, vt_symbol: str = "") -> "RiskDecision":
        return cls(passed=False, code=code, reason=reason, vt_symbol=vt_symbol)


@dataclass
class RiskLimits:
    """风控限额集合。

    ``None`` = 不检查该项。默认值适用于百万级资金的中低频组合，
    **上实盘前必须按账户规模与品种重新标定**。
    """

    # ---- 1. 单笔 ----
    max_order_volume: Optional[float] = 100.0      # 单笔最大手数
    min_order_volume: Optional[float] = None       # 单笔最小手数（低于则拒）
    volume_tick: Optional[float] = 1.0             # 手数步进（1 = 必须整数手）
    max_price_deviation: Optional[float] = 0.05    # 限价单相对参考价最大偏离（5%）

    # ---- 2. 单品种 ----
    max_position_volume: Optional[float] = 500.0   # 单合约净持仓手数上限（绝对值）
    max_position_value: Optional[float] = None     # 单合约名义市值上限（元）
    allow_close_exceed_position: bool = False      # 是否允许平仓量超过持仓（默认禁止）

    # ---- 3. 组合 ----
    max_margin_ratio: Optional[float] = 0.8        # 保证金占用 / 权益 上限
    max_daily_loss: Optional[float] = None         # 单日最大亏损（元，正数）
    max_daily_loss_ratio: Optional[float] = 0.05   # 单日最大亏损率（相对日初权益）
    max_drawdown_ratio: Optional[float] = 0.20     # 最大回撤熔断线（相对权益高点）
    halt_on_daily_loss: bool = True                # 触发日亏损 → 熔断（仅允许平仓）
    halt_on_drawdown: bool = True                  # 触发回撤 → 熔断（仅允许平仓）

    # ---- 4. 频率 ----
    max_orders_per_day: Optional[int] = 1000
    max_orders_per_minute: Optional[int] = 60
    max_trade_volume_per_day: Optional[float] = None

    # ---- 5. 准入 ----
    forbidden_symbols: Set[str] = field(default_factory=set)   # 黑名单（vt_symbol 或品种前缀）
    allowed_symbols: Set[str] = field(default_factory=set)     # 白名单（空 = 不限制）
    check_trading_session: bool = True                         # 是否校验交易时段
    allow_open: bool = True                                    # 全局是否允许开仓
    self_trade_guard: bool = True                              # 防同合约反向对敲

    # ---- 便捷构造 ----
    @classmethod
    def conservative(cls) -> "RiskLimits":
        """保守档（小资金/新策略首次上实盘）。"""
        return cls(
            max_order_volume=10.0,
            max_position_volume=50.0,
            max_margin_ratio=0.3,
            max_daily_loss_ratio=0.02,
            max_drawdown_ratio=0.10,
            max_orders_per_day=200,
            max_orders_per_minute=20,
        )

    @classmethod
    def unlimited(cls) -> "RiskLimits":
        """不限档（**仅供单元测试/回放**，禁止用于实盘）。"""
        return cls(
            max_order_volume=None,
            min_order_volume=None,
            volume_tick=None,
            max_price_deviation=None,
            max_position_volume=None,
            max_position_value=None,
            allow_close_exceed_position=True,
            max_margin_ratio=None,
            max_daily_loss=None,
            max_daily_loss_ratio=None,
            max_drawdown_ratio=None,
            max_orders_per_day=None,
            max_orders_per_minute=None,
            max_trade_volume_per_day=None,
            check_trading_session=False,
            self_trade_guard=False,
        )

    def to_dict(self) -> dict:
        return {
            "max_order_volume": self.max_order_volume,
            "min_order_volume": self.min_order_volume,
            "volume_tick": self.volume_tick,
            "max_price_deviation": self.max_price_deviation,
            "max_position_volume": self.max_position_volume,
            "max_position_value": self.max_position_value,
            "max_margin_ratio": self.max_margin_ratio,
            "max_daily_loss": self.max_daily_loss,
            "max_daily_loss_ratio": self.max_daily_loss_ratio,
            "max_drawdown_ratio": self.max_drawdown_ratio,
            "max_orders_per_day": self.max_orders_per_day,
            "max_orders_per_minute": self.max_orders_per_minute,
            "max_trade_volume_per_day": self.max_trade_volume_per_day,
            "forbidden_symbols": sorted(self.forbidden_symbols),
            "allowed_symbols": sorted(self.allowed_symbols),
            "check_trading_session": self.check_trading_session,
            "allow_open": self.allow_open,
            "self_trade_guard": self.self_trade_guard,
        }
