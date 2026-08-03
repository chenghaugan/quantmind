"""数据馈送适配器单测：重点验证 akshare 中文列名的解析（无需联网）。

这些测试覆盖了此前导致 A 股/期权源静默降级到 MockFeed 的列名映射 bug：
akshare 的 stock_zh_a_hist / option_* 等接口返回中文列（日期/开盘/收盘...），
而旧代码只认英文列，取数时 row[None] 抛 KeyError 使整源失败。
"""
import pandas as pd
import pytest

from quantmind.core.constant import Exchange, Interval
from quantmind.data.feed.base import resolve_ohlc_columns
from quantmind.data.feed.mootdx_astock import MootdxAStockFeed
from quantmind.data.feed.akshare_future import AkShareFuturesFeed
from quantmind.data.feed.akshare_option import AkShareOptionFeed
from quantmind.data.feed.em_hk import EmHkFeed
from quantmind.data.feed.base import HistoryRequest


def _cn_astock_df():
    """模拟 akshare stock_zh_a_hist 返回的中文列 DataFrame。"""
    return pd.DataFrame({
        "日期": ["2024-01-02", "2024-01-03"],
        "开盘": [10.5, 10.8],
        "收盘": [10.7, 10.6],
        "最高": [10.9, 10.95],
        "最低": [10.4, 10.55],
        "成交量": [10000, 12000],
        "成交额": [107000.0, 127200.0],
    })


def _en_future_df():
    """模拟 akshare futures_zh_daily_sina 返回的英文列 DataFrame。"""
    return pd.DataFrame({
        "date": ["2024-01-02", "2024-01-03"],
        "open": [3550.0, 3560.0],
        "high": [3663.0, 3665.0],
        "low": [3513.0, 3515.0],
        "close": [3561.0, 3562.0],
        "volume": [354590, 360000],
        "hold": [45548, 46000],
        "settle": [0.0, 0.0],
    })


def _cn_option_df():
    """模拟 akshare option_* 返回的中文列 DataFrame。"""
    return pd.DataFrame({
        "日期": ["2024-01-02", "2024-01-03"],
        "开盘": [0.12, 0.13],
        "最高": [0.15, 0.14],
        "最低": [0.11, 0.12],
        "收盘": [0.13, 0.125],
        "成交量": [5000, 5200],
    })


def test_resolve_ohlc_columns_chinese():
    cols = resolve_ohlc_columns(_cn_astock_df())
    assert cols["date"] == "日期"
    assert cols["open"] == "开盘"
    assert cols["close"] == "收盘"
    assert cols["turnover"] == "成交额"


def test_resolve_ohlc_columns_english():
    cols = resolve_ohlc_columns(_en_future_df())
    assert cols["date"] == "date"
    assert cols["open_interest"] == "hold"


def test_mootdx_astock_chinese_parse():
    req = HistoryRequest(symbol="600000", exchange=Exchange.SSE, interval=Interval.DAILY)
    bars = MootdxAStockFeed._df_to_bars(_cn_astock_df(), req)
    assert len(bars) == 2
    b0 = bars[0]
    assert b0.open_price == 10.5
    assert b0.high_price == 10.9
    assert b0.low_price == 10.4
    assert b0.close_price == 10.7
    assert b0.volume == 10000
    assert b0.turnover == 107000.0
    assert b0.datetime.year == 2024


def test_akshare_future_english_parse():
    req = HistoryRequest(symbol="rb0", exchange=Exchange.SHFE, interval=Interval.DAILY)
    bars = AkShareFuturesFeed._df_to_bars(_en_future_df(), "rb0", Exchange.SHFE, Interval.DAILY)
    assert len(bars) == 2
    assert bars[0].open_price == 3550.0
    assert bars[0].open_interest == 45548.0


def test_option_chinese_parse():
    req = HistoryRequest(symbol="10004230", exchange=Exchange.SSE, interval=Interval.DAILY)
    bars = AkShareOptionFeed._df_to_bars(_cn_option_df(), req)
    assert len(bars) == 2
    assert bars[0].open_price == 0.12
    assert bars[0].close_price == 0.13


def test_missing_columns_raises():
    bad = pd.DataFrame({"foo": [1, 2]})
    req = HistoryRequest(symbol="600000", exchange=Exchange.SSE, interval=Interval.DAILY)
    with pytest.raises(ValueError):
        MootdxAStockFeed._df_to_bars(bad, req)
