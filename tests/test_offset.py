"""OffsetConverter 平今/平昨测试。"""
from quantmind.core import (
    OffsetConverter,
    OrderRequest,
    Offset,
    Exchange,
    Direction,
    PositionData,
)


def _pos(volume, yd, exch=Exchange.SHFE, direction=Direction.LONG):
    return PositionData(
        symbol="rb",
        exchange=exch,
        direction=direction,
        volume=volume,
        yd_volume=yd,
    )


def test_close_today_when_today_enough():
    conv = OffsetConverter()
    conv.update_position(_pos(volume=5, yd=2))
    req = OrderRequest(symbol="rb", exchange=Exchange.SHFE, direction=Direction.LONG,
                       offset=Offset.CLOSE, volume=3)
    out = conv.convert_order_req(req)
    assert out.offset == Offset.CLOSE_TODAY


def test_close_yesterday_when_today_insufficient():
    conv = OffsetConverter()
    conv.update_position(_pos(volume=5, yd=2))  # 今仓=3
    req = OrderRequest(symbol="rb", exchange=Exchange.SHFE, direction=Direction.LONG,
                       offset=Offset.CLOSE, volume=5)
    out = conv.convert_order_req(req)
    assert out.offset == Offset.CLOSE_YESTERDAY


def test_non_distinguish_exchange_uses_close():
    conv = OffsetConverter()
    conv.update_position(_pos(volume=10, yd=10, exch=Exchange.DCE))
    req = OrderRequest(symbol="rb", exchange=Exchange.DCE, direction=Direction.LONG,
                       offset=Offset.CLOSE, volume=4)
    out = conv.convert_order_req(req)
    assert out.offset == Offset.CLOSE


def test_open_unchanged():
    conv = OffsetConverter()
    req = OrderRequest(symbol="rb", exchange=Exchange.SHFE, direction=Direction.LONG,
                       offset=Offset.OPEN, volume=1)
    assert conv.convert_order_req(req).offset == Offset.OPEN
