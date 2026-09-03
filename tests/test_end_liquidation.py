"""期末强平回归测试：持仓到回测结束的策略，浮动盈亏必须计入胜率/盈亏比。"""
from datetime import datetime, timezone

from quantmind.backtest.engine import BacktestEngine
from quantmind.core.constant import Direction, Exchange, Offset
from quantmind.core.gateway import OrderRequest
from quantmind.core.object import BarData


class _BuyOnceStrategy:
    """首根K线开多后永不平仓——用于验证期末强平。"""

    def __init__(self, engine, setting=None):
        self.engine = engine

    def on_init(self):
        pass

    def on_start(self):
        pass

    def on_stop(self):
        pass

    def on_bar(self, bar):
        pos = self.engine.get_position("IC0.CFFEX")
        if pos.volume == 0:
            self.engine.send_order(OrderRequest(
                symbol="IC0", exchange=Exchange.CFFEX,
                direction=Direction.LONG, offset=Offset.OPEN,
                volume=1, price=bar.close_price))


def test_liquidation_includes_floating_pnl():
    bars = []
    price = 100.0
    for d in range(5):
        for h in (1, 2):
            price += 1.0  # 单边上涨：每天 +2 点
            bars.append(BarData(
                symbol="IC0", exchange=Exchange.CFFEX,
                datetime=datetime(2026, 1, 5 + d, h, 0, tzinfo=timezone.utc),
                open_price=price, high_price=price, low_price=price,
                close_price=price, volume=100))

    eng = BacktestEngine({"IC0.CFFEX": bars},
                         sizes={"IC0.CFFEX": 1.0}, commission=0.0,
                         slippage=0.0, warmup_bars=0)
    eng.add_strategy(_BuyOnceStrategy, "IC0.CFFEX", {})
    rep = eng.run()

    # 开仓 1 笔 + 期末强平 1 笔
    assert rep.trade_count == 2, f"应有2笔成交（开仓+期末强平），实际 {rep.trade_count}"
    # 末根收盘价 110，次bar成交价 102 → +8 点全部计入（size=1.0）
    import pytest
    assert rep.total_return == pytest.approx(8 / 1_000_000, rel=1e-6), rep.total_return  # 次bar撮合: 110-102
    assert rep.win_rate == 1.0
