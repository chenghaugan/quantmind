"""多市场常量定义（参考 vnpy.trader.constant）。"""
from enum import Enum


class Exchange(Enum):
    """交易所。国内期货/股票/期权 + 预留现货。"""

    # 期货
    CFFEX = "CFFEX"   # 中国金融期货交易所（股指/国债）
    SHFE = "SHFE"     # 上海期货交易所
    DCE = "DCE"       # 大连商品交易所
    CZCE = "CZCE"     # 郑州商品交易所
    INE = "INE"       # 上海国际能源交易中心
    GFEX = "GFEX"     # 广州期货交易所
    # A 股
    SSE = "SSE"       # 上海证券交易所
    SZSE = "SZSE"     # 深圳证券交易所
    # 港股
    HKEX = "HKEX"     # 香港交易所
    # 现货（预留）
    SGE = "SGE"       # 上海黄金交易所


class Interval(Enum):
    """K 线周期。"""

    MINUTE = "1m"
    MINUTE_3 = "3m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR = "1h"
    HOUR_2 = "2h"
    HOUR_4 = "4h"
    DAILY = "1d"
    WEEKLY = "1w"


class Direction(Enum):
    """买卖方向。"""

    LONG = "多"
    SHORT = "空"
    NET = "净"


class Offset(Enum):
    """开平。期货/期权需要区分平今/平昨。"""

    NONE = ""
    OPEN = "开"               # 开仓
    CLOSE = "平"              # 平仓（交易所自动归并）
    CLOSE_TODAY = "平今"      # 平今
    CLOSE_YESTERDAY = "平昨"  # 平昨


class OptionType(Enum):
    """期权类型。"""

    CALL = "CALL"
    PUT = "PUT"


class Status(Enum):
    """委托状态。"""

    SUBMITTING = "提交中"
    SUBMITTED = "已提交"
    PARTTRADED = "部分成交"
    ALLTRADED = "全部成交"
    CANCELLED = "已撤销"
    CANCELLING = "撤销中"
    REJECTED = "拒单"


class Product(Enum):
    """合约品种类别。"""

    EQUITY = "股票"
    FUTURE = "期货"
    OPTION = "期权"
    INDEX = "指数"
    ETF = "ETF"
    SPOT = "现货"


class GatewayType(Enum):
    """网关类别。"""

    CTP = "ctp"
    XTP = "xtp"
    IB = "ib"
