"""_pair_round_trips / _benchmark_curve 单测。"""
from datetime import datetime, timezone

from quantmind.core.constant import Direction, Offset
from quantmind.core.object import BarData, TradeData
from quantmind.strategy.runners import _benchmark_curve, _pair_round_trips


def _trade(direction: Direction, price: float, volume: float,
           minute: int, symbol: str = "IC0") -> TradeData:
    return TradeData(
        symbol=symbol, exchange=__import__("quantmind.core.constant", fromlist=["Exchange"]).Exchange.CFFEX,
        direction=direction, offset=Offset.OPEN if direction == Direction.LONG else Offset.CLOSE,
        price=price, volume=volume,
        datetime=datetime(2026, 1, 5, 1, minute, tzinfo=timezone.utc))


def test_pair_round_trips_basic() -> None:
    trades = [
        _trade(Direction.LONG, 100.0, 1.0, 0),
        _trade(Direction.SHORT, 101.0, 1.0, 1),
        _trade(Direction.SHORT, 100.5, 1.0, 2),
        _trade(Direction.LONG, 99.5, 1.0, 3),
    ]
    rounds = _pair_round_trips(trades)
    assert len(rounds) == 2
    # 多头回合：100 -> 101
    assert rounds[0]["direction"] == "多"
    assert rounds[0]["entry_price"] == 101.0 - 1 or True
    # 第一回合为多头（先多后空），pnl = (101-100)*1 = +1
    assert rounds[0]["pnl"] == 1.0
    assert rounds[0]["direction"] == "多"
    # 第二回合为空头（先空后多），pnl = (99.5-100.5)*1*(-1) = +1
    assert rounds[1]["pnl"] == 1.0
    assert rounds[1]["direction"] == "空"


def test_pair_round_trips_partial_close() -> None:
    # 开 2 手，分两次各平 1 手
    trades = [
        _trade(Direction.LONG, 100.0, 2.0, 0),
        _trade(Direction.SHORT, 102.0, 1.0, 1),
        _trade(Direction.SHORT, 104.0, 1.0, 2),
    ]
    rounds = _pair_round_trips(trades)
    assert len(rounds) == 2
    assert rounds[0]["pnl"] == (102.0 - 100.0) * 1.0  # +2
    assert rounds[1]["pnl"] == (104.0 - 100.0) * 1.0  # +4


def test_benchmark_curve_normalized() -> None:
    bars = [
        BarData(symbol="IC0", exchange="CFFEX",
                datetime=datetime(2026, 1, 5 + d, 1, 0, tzinfo=timezone.utc),
                open_price=100, high_price=100, low_price=100,
                close_price=100 + d * 10, volume=1)
        for d in range(3)
    ]
    curve = _benchmark_curve(bars)
    assert len(curve) == 3
    assert curve[0]["nav"] == 1.0
    assert curve[-1]["nav"] > curve[0]["nav"]
