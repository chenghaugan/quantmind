"""策略运行上下文（Runner 抽象）。

同一份策略代码通过绑定不同的 ``StrategyContext`` 实现，即可在
**回测 / 模拟 / 实盘** 三种模式下运行——这就是「切换路线即可跑实盘」的核心：
策略只调用 ``context.send_order / set_target / get_position``，不关心底层是
模拟撮合还是真实网关。

实现：
  - ``BacktestEngine``（回测，批量历史）
  - ``PaperEngine``（模拟，实时/回放）
  - ``LiveEngine``（实盘，路由到 CTP/XTP/IB 网关桩）
三者均继承 ``StrategyContext``。
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from ..core.constant import Direction, Offset
from ..core.event import EventType
from ..core.gateway import OrderRequest
from ..core.object import BarData, LogData, PositionData
from ..core.offset import OffsetConverter

_logger = logging.getLogger("quantmind.strategy.context")


def parse_vt_symbol(vt_symbol: str) -> Tuple[str, str]:
    sym, exch = vt_symbol.rsplit(".", 1)
    return sym, exch


class StrategyContext(ABC):
    """策略运行上下文（Runner）。"""

    mode: str = "base"
    event_engine = None

    @abstractmethod
    def send_order(self, req: OrderRequest) -> str:
        """发送委托，返回 order_id。"""

    @abstractmethod
    def get_position(self, vt_symbol: str) -> PositionData:
        """返回该合约净持仓（direction=NET，volume 为带符号净仓）。"""

    @abstractmethod
    def get_history(self, vt_symbol: str, count: int) -> List[BarData]:
        """返回该合约最近 ``count`` 根历史 K 线（升序）。"""

    # ---- 通用实现 ----
    def set_target(self, vt_symbol: str, target: float) -> Optional[str]:
        """把某合约净持仓调整到 ``target``（带符号：正多/负空）。

        自动计算 delta 并选择开/平 Offset；实盘由网关的 OffsetConverter 归并平今/平昨。
        返回发出的 order_id（无变化则返回 None）。
        """
        cur = self.get_position(vt_symbol)
        cur_vol = cur.volume if cur else 0.0
        delta = target - cur_vol
        if abs(delta) < 1e-9:
            return None
        sym, exch = parse_vt_symbol(vt_symbol)
        direction = Direction.LONG if delta > 0 else Direction.SHORT
        volume = abs(delta)
        if cur_vol != 0 and cur_vol * target < 0:
            # 精确反手（异号）：实际含平仓动作，按 CLOSE 计费/传网关（由 OffsetConverter
            # 拆分平仓+开仓量），按 OPEN 会让成本模型用错费率、实盘 offset 语义错误
            offset = Offset.CLOSE
        else:
            offset = Offset.OPEN if abs(target) >= abs(cur_vol) else Offset.CLOSE
        req = OrderRequest(
            symbol=sym,
            exchange=_exch_from_str(exch),
            direction=direction,
            offset=offset,
            volume=volume,
            price=0.0,
        )
        order_id = self.send_order(req)
        if self.event_engine is not None:
            self.event_engine.put_event(
                EventType.EVENT_SIGNAL,
                {"vt_symbol": vt_symbol, "target": target, "order_id": order_id},
            )
        return order_id

    def write_log(self, msg: str, level: int = logging.INFO) -> None:
        if self.event_engine is not None:
            self.event_engine.put_event(EventType.EVENT_LOG, LogData(msg=msg, level=level))
        else:
            _logger.log(level, msg)


def _exch_from_str(exch: str):
    from ..core.constant import Exchange

    return Exchange(exch)
