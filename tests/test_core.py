"""core 领域模型测试。"""
from datetime import datetime, timezone

from quantmind.core import BarData, Exchange, Interval, OptionData, OptionType, Product


def test_bar_vt_symbol():
    b = BarData(symbol="rb0", exchange=Exchange.SHFE, interval=Interval.DAILY)
    assert b.vt_symbol == "rb0.SHFE"


def test_bar_utc_normalization_naive():
    # naive 时间应被视为 UTC
    b = BarData(symbol="IF0", exchange=Exchange.CFFEX, datetime=datetime(2024, 1, 1, 9, 0, 0))
    assert b.datetime.tzinfo == timezone.utc


def test_bar_utc_normalization_aware():
    b = BarData(
        symbol="600519",
        exchange=Exchange.SSE,
        datetime=datetime(2024, 1, 1, 9, 0, 0, tzinfo=timezone.utc),
    )
    assert b.datetime.utcoffset() == timezone.utc.utcoffset(None)


def test_option_data():
    o = OptionData(
        symbol="IO2409-C-3900",
        exchange=Exchange.CFFEX,
        option_type=OptionType.CALL,
        strike_price=3900.0,
        product=Product.OPTION,
    )
    assert o.option_type == OptionType.CALL
    assert o.vt_symbol == "IO2409-C-3900.CFFEX"
