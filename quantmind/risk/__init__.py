"""风险控制层（实盘化 P0）。

设计原则：**风控是代码强制的闸门，不是策略的自觉**。

任何委托在到达网关之前，必须先经过 :class:`~quantmind.risk.engine.RiskEngine`
的 :meth:`check_order`。风控拒单不会抛异常打断策略循环，而是返回
:class:`RiskDecision(passed=False)` 并广播 ``EVENT_RISK`` 事件——策略无法
「说服」风控放行，也无法通过参数把硬阈值调没（阈值只能在引擎装配时传入）。

模块组成
--------
  - :mod:`quantmind.risk.limits`   —— 限额定义（单笔/单品种/组合/频率/时段）
  - :mod:`quantmind.risk.calendar` —— 交易日历与交易时段（含期货夜盘）
  - :mod:`quantmind.risk.engine`   —— 风控引擎 + 熔断开关（kill switch）
  - :mod:`quantmind.risk.portfolio`—— 组合级风控（多策略暴露/集中度/相关性）
  - :mod:`quantmind.risk.turbulence`- Turbulence 市场状态检测（马氏距离）
"""
from .limits import RiskLimits, RiskCode, RiskDecision
from .calendar import TradingCalendar, is_trading_time, is_trading_day
from .engine import RiskEngine, RiskState
from .portfolio import (
    PortfolioRiskState,
    PortfolioRiskEngine,
    PortfolioLimits,
    PositionBookEntry,
    compute_strategy_correlation,
)
from .turbulence import (
    Regime,
    TurbulenceConfig,
    TurbulenceDetector,
    TurbulenceRiskAdapter,
)

__all__ = [
    "RiskLimits",
    "RiskCode",
    "RiskDecision",
    "TradingCalendar",
    "is_trading_time",
    "is_trading_day",
    "RiskEngine",
    "RiskState",
    "PortfolioRiskState",
    "PortfolioRiskEngine",
    "PortfolioLimits",
    "PositionBookEntry",
    "compute_strategy_correlation",
    "Regime",
    "TurbulenceConfig",
    "TurbulenceDetector",
    "TurbulenceRiskAdapter",
]
