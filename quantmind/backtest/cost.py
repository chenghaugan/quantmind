"""结构化交易成本模型。

回测若不计入真实交易成本，Sharpe / 收益会被严重高估。本模块提供：

  - ``CostModel``：单一合约/品种的成本参数（手续费、最低手续费、平今倍率、
    A 股印花税、滑点、保证金率、冲击成本）。
  - ``CONTRACT_COST_TABLE`` / ``default_cost_table()``：国内主要品种的常见近似
    成本预设（**数值为示例，需按交易所当期公告校准**）。
  - ``lookup_cost(vt_symbol)``：按 ``symbol -> 品种前缀 -> 交易所默认 -> 兜底``
    逐级解析成本。
  - 计算函数：``compute_commission`` / ``apply_slippage`` / ``compute_margin``。

设计原则
--------
  - 默认零成本假设不成立：期货平今（close-today）手续费与开仓差异巨大
    （如股指期货 2019 年后平今免收，此前曾高达万分之数十）；A 股卖出收印花税
    （2023-08-28 起为 0.05%，此前为 0.1%）。
  - 滑点按「最小变动价位(tick)倍数」或「成交价比例」建模，比固定点数更贴近实盘。
  - 保证金仅作占用记录与可选容量约束，不影响权益曲线数值（权益 = 余额 + 浮动盈亏）。

校准提示
--------
  本表数值为常见近似，且费率随交易所公告/税收政策变化，**回测前请按当期公告校准**：
  - A 股印花税：2023-08-28 后 0.05%（卖出）；2023-08-28 前 0.1%。
  - 港股印花税：2023-11-17 后 0.1%；此前曾多次调整。
  - 股指期货平今：2019 年后免收，此前极高——长周期回测应分段处理。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from ..core.constant import Direction
from ..core.object import BarData


@dataclass
class CostModel:
    """单一合约/品种的交易成本参数。

    所有费率均为「比率」或「每手固定」，成交时按成交额/手数计算。

    ``asset_class`` 取值：``"future"`` / ``"equity"`` / ``"option"``。
    """

    asset_class: str = "future"
    # 手续费（按成交额比例）
    commission_rate: float = 0.0002
    commission_per_lot: float = 0.0          # 每手固定手续费（加在比例之上）
    min_commission: float = 0.0             # 单笔最低手续费
    # 平今仓倍率（相对开仓费率）。0 = 平今免费；股指设 0；部分品种 > 1
    close_today_rate_multiplier: float = 1.0
    close_today_per_lot_multiplier: float = 1.0
    close_yesterday_rate_multiplier: float = 1.0   # 平昨倍率（多数=1）
    # A 股印花税（仅卖出收取）
    stamp_tax_rate: float = 0.0
    # 滑点
    slippage_ticks: float = 0.0             # 滑点 = slippage_ticks * tick_size
    slippage_rate: float = 0.0             # 滑点 = open_price * slippage_rate
    tick_size: float = 0.01                # 最小变动价位（用于 tick 滑点）
    # 保证金率（期货），仅记录/约束用
    margin_rate: float = 0.1
    # 冲击成本（按成交量线性，price * volume * size * impact_rate）
    impact_rate: float = 0.0
    # 合约乘数（与引擎 sizes 一致；此处冗余存储便于独立计算）
    multiplier: float = 1.0
    note: str = ""

    # ---- 计算 ----
    def commission_for(
        self,
        volume: float,
        price: float,
        size: float,
        is_close_today: bool,
        is_close_yesterday: bool = False,
    ) -> float:
        """单笔（部分）成交的手续费。

        ``is_close_today``：平今仓；``is_close_yesterday``：平昨（否则视为开仓/其他）。
        """
        if is_close_today:
            rate = self.commission_rate * self.close_today_rate_multiplier
            per_lot = self.commission_per_lot * self.close_today_per_lot_multiplier
        elif is_close_yesterday:
            rate = self.commission_rate * self.close_yesterday_rate_multiplier
            per_lot = self.commission_per_lot * self.close_yesterday_rate_multiplier
        else:
            rate = self.commission_rate
            per_lot = self.commission_per_lot
        fee = price * volume * size * rate + per_lot * volume
        return fee

    def stamp_tax_for(self, volume: float, price: float, size: float, direction: Direction) -> float:
        """印花税（A 股卖出时收取）。"""
        if self.stamp_tax_rate <= 0 or direction != Direction.SHORT:
            return 0.0
        return price * volume * size * self.stamp_tax_rate

    def impact_for(self, volume: float, price: float, size: float) -> float:
        return price * volume * size * self.impact_rate

    def slippage_amount(self, bar: BarData, direction: Direction) -> float:
        """返回滑点绝对金额（带符号：多 + / 空 -），用于成交价偏移。"""
        slip = 0.0
        if self.slippage_ticks:
            slip += self.slippage_ticks * self.tick_size
        if self.slippage_rate:
            slip += bar.open_price * self.slippage_rate
        return slip if direction == Direction.LONG else -slip

    def margin_for(self, volume: float, price: float, size: float) -> float:
        return abs(volume) * price * size * self.margin_rate


# ---------------------------------------------------------------------------
# 预设成本表（常见近似，需按交易所当期公告校准）
# ---------------------------------------------------------------------------
# 说明：
#   - 股指期货 IF/IC/IH：2019 年后平今免收（close_today_rate_multiplier=0）；
#     2015.8-2019 期间平今曾高达开仓费率的数十倍，长周期回测应以当时公告为准。
#   - 螺纹 RB、热卷 HC、沥青 BU 等：平今免收（交易所优惠）。
#   - 国债 TF/T：按每手固定（约 3 元/手），保证金率低。
#   - A 股：万 2.5 佣金 + 卖出千 1 印花税 + 最低 5 元。
_FUTURE_COMMON = CostModel(asset_class="future", commission_rate=0.0001,
                           close_today_rate_multiplier=1.0, tick_size=1.0,
                           margin_rate=0.1, note="商品期货通用近似（万1）")

CONTRACT_COST_TABLE: Dict[str, CostModel] = {
    # 金融期货（CFFEX）
    "IF": CostModel(asset_class="future", commission_rate=0.000023,
                    close_today_rate_multiplier=0.0, tick_size=0.2,
                    margin_rate=0.12, multiplier=300, note="沪深300股指：平今免(2019后)"),
    "IC": CostModel(asset_class="future", commission_rate=0.000023,
                    close_today_rate_multiplier=0.0, tick_size=0.2,
                    margin_rate=0.12, multiplier=200, note="中证500股指：平今免"),
    "IH": CostModel(asset_class="future", commission_rate=0.000023,
                    close_today_rate_multiplier=0.0, tick_size=0.2,
                    margin_rate=0.12, multiplier=300, note="上证50股指：平今免"),
    "TF": CostModel(asset_class="future", commission_rate=0.0,
                    commission_per_lot=3.0, tick_size=0.005,
                    margin_rate=0.02, multiplier=10000, note="五年国债：约3元/手"),
    "T": CostModel(asset_class="future", commission_rate=0.0,
                   commission_per_lot=3.0, tick_size=0.005,
                   margin_rate=0.02, multiplier=10000, note="十年国债：约3元/手"),
    "TS": CostModel(asset_class="future", commission_rate=0.0,
                    commission_per_lot=3.0, tick_size=0.005,
                    margin_rate=0.02, multiplier=20000, note="两年国债：约3元/手"),
    # 上海期货交易所 SHFE
    "RB": CostModel(asset_class="future", commission_rate=0.0001,
                    close_today_rate_multiplier=0.0, tick_size=1.0,
                    margin_rate=0.1, multiplier=10, note="螺纹钢：平今免"),
    "HC": CostModel(asset_class="future", commission_rate=0.0001,
                    close_today_rate_multiplier=0.0, tick_size=1.0,
                    margin_rate=0.1, multiplier=10, note="热卷：平今免"),
    "BU": CostModel(asset_class="future", commission_rate=0.0001,
                    tick_size=2.0, margin_rate=0.1, multiplier=10, note="沥青"),
    "CU": CostModel(asset_class="future", commission_rate=0.00005,
                    tick_size=10.0, margin_rate=0.1, multiplier=5, note="铜"),
    "AG": CostModel(asset_class="future", commission_rate=0.00005,
                    tick_size=1.0, margin_rate=0.1, multiplier=15, note="白银"),
    "AU": CostModel(asset_class="future", commission_per_lot=10.0,
                    tick_size=0.02, margin_rate=0.08, multiplier=1000, note="黄金：约10元/手"),
    "RU": CostModel(asset_class="future", commission_rate=0.000045,
                    tick_size=5.0, margin_rate=0.1, multiplier=10, note="橡胶"),
    "AL": CostModel(asset_class="future", commission_rate=0.000045,
                    tick_size=5.0, margin_rate=0.1, multiplier=5, note="铝"),
    "ZN": CostModel(asset_class="future", commission_rate=0.000045,
                    tick_size=5.0, margin_rate=0.1, multiplier=5, note="锌"),
    "NI": CostModel(asset_class="future", commission_rate=0.0001,
                    tick_size=10.0, margin_rate=0.1, multiplier=1, note="镍"),
    # 大连商品交易所 DCE
    "I": CostModel(asset_class="future", commission_rate=0.0001,
                   tick_size=0.5, margin_rate=0.1, multiplier=100, note="铁矿石"),
    "J": CostModel(asset_class="future", commission_rate=0.0001,
                   tick_size=0.5, margin_rate=0.1, multiplier=100, note="焦炭"),
    "JM": CostModel(asset_class="future", commission_rate=0.0001,
                    tick_size=0.5, margin_rate=0.1, multiplier=60, note="焦煤"),
    "M": CostModel(asset_class="future", commission_rate=0.00015,
                   tick_size=1.0, margin_rate=0.1, multiplier=10, note="豆粕"),
    "Y": CostModel(asset_class="future", commission_rate=0.0002,
                   tick_size=2.0, margin_rate=0.1, multiplier=10, note="豆油"),
    "P": CostModel(asset_class="future", commission_rate=0.00025,
                   tick_size=2.0, margin_rate=0.1, multiplier=10, note="棕榈"),
    "PP": CostModel(asset_class="future", commission_rate=0.0001,
                    tick_size=1.0, margin_rate=0.1, multiplier=5, note="PP"),
    "L": CostModel(asset_class="future", commission_rate=0.0001,
                   tick_size=1.0, margin_rate=0.1, multiplier=5, note="塑料"),
    "C": CostModel(asset_class="future", commission_rate=0.00012,
                   tick_size=1.0, margin_rate=0.1, multiplier=10, note="玉米"),
    "CS": CostModel(asset_class="future", commission_rate=0.00012,
                    tick_size=1.0, margin_rate=0.1, multiplier=10, note="玉米淀粉"),
    # 郑州商品交易所 CZCE
    "TA": CostModel(asset_class="future", commission_rate=0.000003,
                    tick_size=2.0, margin_rate=0.1, multiplier=5, note="PTA（低费率）"),
    "MA": CostModel(asset_class="future", commission_rate=0.0001,
                    tick_size=1.0, margin_rate=0.1, multiplier=10, note="甲醇"),
    "SR": CostModel(asset_class="future", commission_rate=0.00015,
                    tick_size=1.0, margin_rate=0.1, multiplier=10, note="白糖"),
    "CF": CostModel(asset_class="future", commission_rate=0.00015,
                    tick_size=5.0, margin_rate=0.1, multiplier=5, note="棉花"),
    "FG": CostModel(asset_class="future", commission_rate=0.0001,
                    tick_size=1.0, margin_rate=0.1, multiplier=20, note="玻璃"),
    "RM": CostModel(asset_class="future", commission_rate=0.00015,
                    tick_size=1.0, margin_rate=0.1, multiplier=10, note="菜粕"),
    "OI": CostModel(asset_class="future", commission_rate=0.0002,
                    tick_size=2.0, margin_rate=0.1, multiplier=10, note="菜油"),
    # 上海国际能源 INE
    "SC": CostModel(asset_class="future", commission_rate=0.00005,
                    tick_size=0.1, margin_rate=0.1, multiplier=1000, note="原油"),
    "LU": CostModel(asset_class="future", commission_rate=0.00005,
                    tick_size=1.0, margin_rate=0.1, multiplier=10, note="低硫燃料油"),
    # 广州期货交易所 GFEX
    "SI": CostModel(asset_class="future", commission_rate=0.0001,
                    tick_size=5.0, margin_rate=0.1, multiplier=5, note="工业硅"),
    "LC": CostModel(asset_class="future", commission_rate=0.0001,
                    tick_size=50.0, margin_rate=0.1, multiplier=1, note="碳酸锂"),
    # A 股（按品种前缀兜底到交易所默认，这里给通用股票成本）
    # 印花税按 2023-08-28 起现行税率 0.05%（卖出）校准；过户费已并入 note 供参考。
    "EQUITY": CostModel(asset_class="equity", commission_rate=0.00025,
                        min_commission=5.0, stamp_tax_rate=0.0005,
                        tick_size=0.01, margin_rate=1.0, multiplier=1,
                        note="A股：万2.5佣金+卖出万5印花税+最低5元(+过户费双向万0.1)"),
    # 美股：免佣金趋势 + 交易规费极低，成本以滑点为主（约 0.05%-0.1%）
    "US_EQUITY": CostModel(asset_class="equity", commission_rate=0.0,
                            stamp_tax_rate=0.0, slippage_rate=0.0005,
                            tick_size=0.01, margin_rate=0.5, multiplier=1,
                            note="美股：免佣+SEC规费可忽略，用 0.05% 滑点近似"),
    # 期权
    "OPTION": CostModel(asset_class="option", commission_per_lot=1.5,
                        tick_size=0.0001, margin_rate=0.1, multiplier=1,
                        note="期权：约1.5元/手（近似）"),
}

# 交易所默认（按资产类别兜底）
_EXCHANGE_DEFAULT: Dict[str, CostModel] = {
    "CFFEX": _FUTURE_COMMON,
    "SHFE": _FUTURE_COMMON,
    "DCE": _FUTURE_COMMON,
    "CZCE": _FUTURE_COMMON,
    "INE": _FUTURE_COMMON,
    "GFEX": _FUTURE_COMMON,
    "SSE": CONTRACT_COST_TABLE["EQUITY"],
    "SZSE": CONTRACT_COST_TABLE["EQUITY"],
    # 港股印花税按 2023-11-17 起现行税率 0.1% 校准（另有交易征费/交收费等并入 note）
    "HKEX": CostModel(asset_class="equity", commission_rate=0.0005,
                      stamp_tax_rate=0.001, tick_size=0.01, margin_rate=1.0,
                      note="港股：约万5佣金+0.1%印花税(+征费/交收费≈万0.0565)"),
    # 美股（NASDAQ/NYSE/AMEX）
    "NASDAQ": CONTRACT_COST_TABLE["US_EQUITY"],
    "NYSE": CONTRACT_COST_TABLE["US_EQUITY"],
    "AMEX": CONTRACT_COST_TABLE["US_EQUITY"],
}


def _symbol_root(symbol: str) -> str:
    """把合约 symbol 归一化为品种前缀，如 'rb0'->'RB'、'IC2401'->'IC'、'600519'->'EQUITY'。"""
    m = re.match(r"([A-Za-z]+)", symbol)
    if not m:
        return "EQUITY"  # 纯数字视为股票
    return m.group(1).upper()


def default_cost_table() -> Dict[str, CostModel]:
    """返回默认成本预设表的副本。"""
    return dict(CONTRACT_COST_TABLE)


def lookup_cost(vt_symbol: str, table: Optional[Dict[str, CostModel]] = None) -> CostModel:
    """按 vt_symbol 解析成本模型。

    解析顺序：精确 symbol -> 品种前缀（如 RB，非数字代码）-> 交易所默认 -> 商品期货通用。
    纯数字 symbol（股票代码，如 600519/00700）跳过品种前缀，直接按交易所兜底，
    这样港股数字代码（如 00700.HKEX）能命中港股成本，而非被误判为 A 股。
    """
    table = table or CONTRACT_COST_TABLE
    if "." not in vt_symbol:
        sym, exch = vt_symbol, ""
    else:
        sym, exch = vt_symbol.rsplit(".", 1)
    # 1) 精确 symbol（如 'rb0' 已被预设表收录则直接用，否则走前缀）
    if sym in table:
        return table[sym]
    # 2) 品种前缀（股票数字代码跳过，避免误判为 A 股 EQUITY）
    if not sym.isdigit():
        root = _symbol_root(sym)
        if root in table:
            return table[root]
    # 3) 交易所默认
    if exch and exch in _EXCHANGE_DEFAULT:
        return _EXCHANGE_DEFAULT[exch]
    # 4) 兜底：商品期货通用
    return _FUTURE_COMMON


def compute_commission(
    cost: CostModel,
    volume: float,
    price: float,
    size: float,
    direction: Direction,
    offset,
    close_today_volume: float = 0.0,
) -> Tuple[float, float, float]:
    """计算单笔成交的总成本现金支出。

    返回 (手续费, 印花税, 冲击成本)。
    ``offset`` 为 ``Offset`` 枚举；``close_today_volume`` 为按开仓批次判定的平今量
    （同一笔成交可能同时含平今与平昨部分，此处按比例拆分计费）。
    """
    from ..core.constant import Offset

    is_close_yesterday = (offset == Offset.CLOSE_YESTERDAY)
    ct = max(0.0, min(close_today_volume, volume))
    rest = volume - ct
    fee = 0.0
    if ct > 1e-9:
        fee += cost.commission_for(ct, price, size, True, False)
    if rest > 1e-9:
        fee += cost.commission_for(rest, price, size, False, is_close_yesterday)
    # 最低手续费（对整笔）
    if cost.min_commission > 0 and fee < cost.min_commission:
        fee = cost.min_commission
    tax = cost.stamp_tax_for(volume, price, size, direction)
    impact = cost.impact_for(volume, price, size)
    return fee, tax, impact


def apply_slippage(cost: Optional[CostModel], bar: BarData, direction: Direction,
                   fallback_slippage: float = 0.0) -> float:
    """返回成交价（含滑点）。cost 为 None 时退回固定点数滑点（兼容旧逻辑）。"""
    if cost is None:
        return bar.open_price + (fallback_slippage if direction == Direction.LONG
                                 else -fallback_slippage)
    return bar.open_price + cost.slippage_amount(bar, direction)


def compute_margin(cost: CostModel, volume: float, price: float, size: float) -> float:
    return cost.margin_for(volume, price, size)
