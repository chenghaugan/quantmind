"""领域模型（参考 vnpy.trader.object）。

统一内部表示：时间一律以 UTC 存储；通过 ``vt_symbol`` 唯一标识合约。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from .constant import Exchange, Interval, Direction, Offset, OptionType, Status, Product

UTC = timezone.utc


def _as_utc(dt: datetime) -> datetime:
    """将 naive 时间视为 UTC，并统一为 UTC 时区。"""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


@dataclass
class BaseData:
    """所有数据基类。"""

    gateway_name: str = ""

    @property
    def vt_symbol(self) -> str:
        """``symbol.exchange`` 形式，子类覆盖。"""
        return ""


@dataclass
class BarData(BaseData):
    """K 线（OHLCV）。"""

    symbol: str = ""
    exchange: Exchange = Exchange.CFFEX
    datetime: datetime = field(default_factory=lambda: datetime.now(UTC))
    interval: Interval = Interval.DAILY
    volume: float = 0.0
    turnover: float = 0.0
    open_interest: float = 0.0
    open_price: float = 0.0
    high_price: float = 0.0
    low_price: float = 0.0
    close_price: float = 0.0

    # ---- 兼容别名（vnpy 风格）：LLM 生成的策略常写 bar.high/bar.close，----
    # 这里提供安全网，避免 AttributeError 中断回测。
    @property
    def high(self) -> float:
        return self.high_price

    @property
    def low(self) -> float:
        return self.low_price

    @property
    def close(self) -> float:
        return self.close_price

    @property
    def open(self) -> float:
        return self.open_price

    def __post_init__(self) -> None:
        self.datetime = _as_utc(self.datetime)

    @property
    def vt_symbol(self) -> str:
        return f"{self.symbol}.{self.exchange.value}"

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "exchange": self.exchange.value,
            "datetime": self.datetime.isoformat(),
            "interval": self.interval.value,
            "volume": self.volume,
            "turnover": self.turnover,
            "open_interest": self.open_interest,
            "open": self.open_price,
            "high": self.high_price,
            "low": self.low_price,
            "close": self.close_price,
        }


@dataclass
class TickData(BaseData):
    """逐笔/实时行情。"""

    symbol: str = ""
    exchange: Exchange = Exchange.CFFEX
    datetime: datetime = field(default_factory=lambda: datetime.now(UTC))
    name: str = ""
    last_price: float = 0.0
    volume: float = 0.0
    open_interest: float = 0.0
    bid_price_1: float = 0.0
    ask_price_1: float = 0.0
    bid_volume_1: float = 0.0
    ask_volume_1: float = 0.0
    # 五档（简化）
    bid_price_2: float = 0.0
    ask_price_2: float = 0.0

    def __post_init__(self) -> None:
        self.datetime = _as_utc(self.datetime)

    @property
    def vt_symbol(self) -> str:
        return f"{self.symbol}.{self.exchange.value}"


@dataclass
class ContractData(BaseData):
    """合约合约信息（乘数/保证金/最小变动等）。"""

    symbol: str = ""
    exchange: Exchange = Exchange.CFFEX
    name: str = ""
    product: Product = Product.FUTURE
    size: float = 1.0            # 合约乘数
    pricetick: float = 0.01      # 最小变动价位
    min_trade_volume: float = 1.0
    margin_rate: float = 0.1     # 保证金率（期货）
    open_limit: float = 0.0      # 涨跌停幅度（0 表示未知）
    underlying_symbol: str = ""  # 期权标的
    option_type: Optional[OptionType] = None
    strike_price: float = 0.0
    expiry: Optional[datetime] = None

    @property
    def vt_symbol(self) -> str:
        return f"{self.symbol}.{self.exchange.value}"


@dataclass
class OptionData(ContractData):
    """期权合约（继承 ContractData，明确期权属性）。"""

    product: Product = Product.OPTION


@dataclass
class OrderData(BaseData):
    """委托。"""

    symbol: str = ""
    exchange: Exchange = Exchange.CFFEX
    order_id: str = ""
    type: str = "LIMIT"
    direction: Direction = Direction.LONG
    offset: Offset = Offset.OPEN
    price: float = 0.0
    volume: float = 0.0
    traded: float = 0.0
    status: Status = Status.SUBMITTING
    datetime: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        self.datetime = _as_utc(self.datetime)

    @property
    def vt_symbol(self) -> str:
        return f"{self.symbol}.{self.exchange.value}"


@dataclass
class TradeData(BaseData):
    """成交。"""

    symbol: str = ""
    exchange: Exchange = Exchange.CFFEX
    order_id: str = ""
    trade_id: str = ""
    direction: Direction = Direction.LONG
    offset: Offset = Offset.OPEN
    price: float = 0.0
    volume: float = 0.0
    datetime: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        self.datetime = _as_utc(self.datetime)

    @property
    def vt_symbol(self) -> str:
        return f"{self.symbol}.{self.exchange.value}"


@dataclass
class PositionData(BaseData):
    """持仓（按合约+方向）。"""

    symbol: str = ""
    exchange: Exchange = Exchange.CFFEX
    direction: Direction = Direction.NET
    volume: float = 0.0
    frozen: float = 0.0
    price: float = 0.0          # 持仓均价
    pnl: float = 0.0            # 浮动盈亏
    yd_volume: float = 0.0      # 昨仓（用于平今/平昨）

    @property
    def vt_symbol(self) -> str:
        return f"{self.symbol}.{self.exchange.value}"

    @property
    def vt_position_id(self) -> str:
        return f"{self.vt_symbol}.{self.direction.value}"


@dataclass
class LogData(BaseData):
    """日志。"""

    msg: str = ""
    level: int = 20  # logging.INFO

    @property
    def vt_symbol(self) -> str:
        return ""


@dataclass
class AccountData(BaseData):
    """资金账户。"""

    account_id: str = ""
    balance: float = 0.0
    frozen: float = 0.0
    available: float = 0.0

    @property
    def vt_account_id(self) -> str:
        return self.account_id
