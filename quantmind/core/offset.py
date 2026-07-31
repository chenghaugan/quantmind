"""平今/平昨换算器（移植自 vnpy.trader.utility.OffsetConverter）。

国内期货中，上期所(SHFE)、中金所(CFFEX) 对平今仓和平昨仓手续费不同，
需要在发平仓单时正确选择 Offset。大商所(DCE)、郑商所(CZCE)、能源中心(INE)
平今视为平仓（优惠/免），用 CLOSE 即可（交易所自动归并）。
"""
from __future__ import annotations

from typing import Dict

from .constant import Exchange, Offset
from .gateway import OrderRequest
from .object import PositionData

# 需要严格区分平今/平昨的交易所
DISTINGUISH_EXCHANGES = {Exchange.SHFE, Exchange.CFFEX}


class OffsetConverter:
    """维护净持仓 → 自动选择平仓 Offset。"""

    def __init__(self) -> None:
        # vt_position_id -> PositionData
        self.positions: Dict[str, PositionData] = {}

    def update_position(self, pos: PositionData) -> None:
        self.positions[pos.vt_position_id] = pos

    def get_position(self, vt_symbol: str, direction_value: str) -> PositionData | None:
        key = f"{vt_symbol}.{direction_value}"
        return self.positions.get(key)

    def convert_order_req(self, req: OrderRequest) -> OrderRequest:
        """根据当前持仓，将平仓单的 Offset 调整为平今/平昨。"""
        if req.offset in (Offset.OPEN, Offset.NONE):
            return req

        pos = self.get_position(f"{req.symbol}.{req.exchange.value}", req.direction.value)
        if pos is None:
            # 无持仓记录，默认平今（多数交易所平今=平）
            req.offset = Offset.CLOSE_TODAY
            return req

        # 需要区分的交易所：优先平今，今仓不足再平昨
        if req.exchange in DISTINGUISH_EXCHANGES:
            if pos.volume - pos.yd_volume >= req.volume:
                req.offset = Offset.CLOSE_TODAY
            else:
                # 今仓不够，剩余平昨（简化：整体平昨）
                req.offset = Offset.CLOSE_YESTERDAY
        else:
            # 其余交易所：平仓即可，交易所自动归并
            req.offset = Offset.CLOSE
        return req
