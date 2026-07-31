"""本地文件源（LocalFileFeed / ChinaFuturesCSVFeed）测试。

用合成 CSV 样本在离线环境验证：列名宽匹配、5min→日频降采样、北京时间→UTC、
china-futures 仓库路径解析（具体合约 + 主连拼接）、以及数据源缺失时自动降级到 mock。
"""
from datetime import datetime
from pathlib import Path

import pandas as pd

from quantmind.data.feed.local_file import LocalFileFeed, map_columns
from quantmind.data.feed.china_futures_csv import ChinaFuturesCSVFeed
from quantmind.data.feed.base import HistoryRequest
from quantmind.data.feed.registry import DataFeedRegistry
from quantmind.data.feed.mock import MockFeed
from quantmind.core import Exchange, Interval

# 合成 5min 数据（北京时间 naive）
CSV_IC2401 = """datetime,open,high,low,close,volume,open_interest
2023-12-29 09:00:00,5000,5010,4990,5005,100,50
2023-12-29 09:05:00,5005,5020,5000,5015,120,52
2024-01-02 09:00:00,5015,5030,5010,5025,200,60
2024-01-02 09:05:00,5025,5040,5020,5035,180,62
"""
CSV_IC2403 = """datetime,open,high,low,close,volume,open_interest
2024-02-01 09:00:00,5100,5110,5090,5105,150,70
2024-02-01 09:05:00,5105,5120,5100,5115,160,72
2024-03-01 09:00:00,5115,5130,5110,5125,210,80
"""


def _make_tree(root: Path) -> None:
    d = root / "5min" / "CFFEX" / "IC"
    d.mkdir(parents=True)
    (d / "IC2401.csv").write_text(CSV_IC2401)
    (d / "IC2403.csv").write_text(CSV_IC2403)


def test_map_columns_alias():
    df = pd.DataFrame({"Date": ["2024-01-01"], "Open": [1], "High": [2], "Low": [0],
                       "Close": [1.5], "Vol": [10], "Oi": [3]})
    out = map_columns(df)
    for c in ("datetime", "open", "high", "low", "close", "volume", "open_interest"):
        assert c in out.columns
    assert out["open"].iloc[0] == 1
    assert out["volume"].iloc[0] == 10


def test_resample_daily_ohlc():
    feed = LocalFileFeed("/tmp")
    df = pd.DataFrame({
        "datetime": pd.to_datetime(["2024-01-02 09:00", "2024-01-02 09:05", "2024-01-03 09:00"]),
        "open": [5000, 5005, 5015],
        "high": [5010, 5020, 5030],
        "low": [4990, 5000, 5010],
        "close": [5005, 5015, 5025],
        "volume": [100, 120, 200],
        "open_interest": [50, 52, 60],
    })
    daily = feed._resample_daily(df)
    assert len(daily) == 2
    r0 = daily.iloc[0]
    assert r0["open"] == 5000 and r0["high"] == 5020 and r0["low"] == 4990 and r0["close"] == 5015
    assert r0["volume"] == 220
    assert daily.iloc[1]["close"] == 5025


def test_bj_to_utc():
    feed = LocalFileFeed("/tmp")
    df = pd.DataFrame({"datetime": ["2024-01-02 09:00:00"], "open": [1], "high": [1],
                       "low": [1], "close": [1]})
    out = feed._normalize(df)
    assert out["datetime"].iloc[0] == pd.Timestamp("2024-01-02 01:00:00")


async def test_china_futures_specific_contract(tmp_path):
    _make_tree(tmp_path)
    feed = ChinaFuturesCSVFeed(str(tmp_path))
    bars = await feed.fetch_bar_data(HistoryRequest(
        symbol="IC2401", exchange=Exchange.CFFEX, interval=Interval.DAILY))
    assert len(bars) == 2  # 2023-12-29 与 2024-01-02 两个自然日
    # 时间已转 UTC（北京时间 09:05 -> UTC 01:05）；BarData.datetime 为 UTC aware
    assert bars[0].datetime == pd.Timestamp("2023-12-29 01:05:00", tz="UTC").to_pydatetime()
    # 2023-12-29 日线：high=max(5010,5020)=5020，volume=100+120=220
    assert bars[0].high_price == 5020
    assert bars[0].volume == 220


async def test_china_futures_continuous(tmp_path):
    _make_tree(tmp_path)
    feed = ChinaFuturesCSVFeed(str(tmp_path))
    bars = await feed.fetch_bar_data(HistoryRequest(
        symbol="IC0", exchange=Exchange.CFFEX, interval=Interval.DAILY))
    # 主连拼接：IC2401(2根) + IC2403(2根) = 4 根，按交割月窗口衔接、无重叠
    assert len(bars) == 4
    dts = [b.datetime for b in bars]
    assert dts == sorted(dts)
    assert len(set(dts)) == 4


async def test_local_feed_fallback_to_mock(tmp_path):
    reg = DataFeedRegistry()
    reg.register(ChinaFuturesCSVFeed(str(tmp_path / "nope")), priority=5)
    reg.register(MockFeed(), priority=100)
    bars = await reg.get_bar_data(HistoryRequest(
        symbol="rb0", exchange=Exchange.SHFE, interval=Interval.DAILY))
    assert len(bars) >= 1  # 本地源缺失 -> 降级 mock


async def test_lowercase_symbol_resolves(tmp_path):
    _make_tree(tmp_path)
    feed = ChinaFuturesCSVFeed(str(tmp_path))
    bars = await feed.fetch_bar_data(HistoryRequest(
        symbol="ic0", exchange=Exchange.CFFEX, interval=Interval.DAILY))
    assert len(bars) == 4  # 小写符号也应解析为主连
