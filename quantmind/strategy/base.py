"""策略模板（参考 vnpy_ctastrategy.template.CtaTemplate）。

策略只与 ``StrategyContext`` 交互：通过 ``context.set_target / send_order / get_position``
下单与查询，不感知底层是回测、模拟还是实盘。同一份策略代码即可「切换路线」。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..core.object import BarData
from .context import StrategyContext


class CtaTemplate:
    """CTA 策略模板。"""

    author: str = ""
    parameters: List[str] = []
    variables: List[str] = []

    def __init__(self, context: StrategyContext, setting: Optional[Dict[str, Any]] = None) -> None:
        self.context = context
        self.vt_symbols: List[str] = []
        self.inited = False
        self.trading = False
        self.pos: float = 0.0  # 主合约净持仓（仅用于 CTA 单标的显示）

        setting = setting or {}
        for name in self.parameters:
            if name in setting:
                setattr(self, name, setting[name])
        # 允许设置任意参数
        for k, v in setting.items():
            if not hasattr(self, k):
                setattr(self, k, v)

    # ---- 生命周期 ----
    def on_init(self) -> None:
        """初始化（加载历史、预计算因子等）。"""

    def on_start(self) -> None:
        self.trading = True
        self.context.write_log(f"策略启动: {self.__class__.__name__}")

    def on_bar(self, bar: BarData) -> None:
        """收到一根 K 线。"""

    def on_stop(self) -> None:
        self.trading = False
        self.context.write_log(f"策略停止: {self.__class__.__name__}")

    # ---- 便捷下单（转发到 context） ----
    def set_target(self, vt_symbol: str, target: float) -> Optional[str]:
        return self.context.set_target(vt_symbol, target)

    def buy(self, vt_symbol: str, volume: float, price: float = 0.0) -> Optional[str]:
        from ..core.constant import Direction, Offset, Exchange
        from ..core.gateway import OrderRequest

        sym, exch = vt_symbol.rsplit(".", 1)
        return self.context.send_order(
            OrderRequest(symbol=sym, exchange=Exchange(exch), direction=Direction.LONG,
                         offset=Offset.OPEN, volume=volume, price=price)
        )

    def sell(self, vt_symbol: str, volume: float, price: float = 0.0) -> Optional[str]:
        from ..core.constant import Direction, Offset, Exchange
        from ..core.gateway import OrderRequest

        sym, exch = vt_symbol.rsplit(".", 1)
        return self.context.send_order(
            OrderRequest(symbol=sym, exchange=Exchange(exch), direction=Direction.SHORT,
                         offset=Offset.CLOSE, volume=volume, price=price)
        )

    def write_log(self, msg: str, level: int = 20) -> None:
        self.context.write_log(msg, level)
