"""港股 / 期权 本地 Parquet/CSV 适配器测试（复用通用 LocalDailyParquetFeed）。

离线环境无 pyarrow，主路径用 .csv 验证逻辑；Parquet 真读测试由 importorskip 守卫。
"""
from __future__ import annotations

import pandas as pd
import pytest
from pathlib import Path

from quantmind.core.constant import Exchange, Interval
from quantmind.data.feed.base import HistoryRequest
from quantmind.data.feed.local_daily import (
    ChinaHKAStockParquetFeed,
    ChinaOptionParquetFeed,
)
from quantmind.data.feed import build_default_registry


def _make_daily_df():
    dates = pd.date_range("2024-01-02", periods=5, freq="D")
    return pd.DataFrame({
        "date": dates,
        "open": [10.0, 11.0, 12.0, 13.0, 14.0],
        "high": [10.5, 11.5, 12.5, 13.5, 14.5],
        "low": [9.5, 10.5, 11.5, 12.5, 13.5],
        "close": [10.2, 11.2, 12.2, 13.2, 14.2],
        "volume": [1000, 1100, 1200, 1300, 1400],
        "amount": [10000.0, 11000.0, 12000.0, 13000.0, 14000.0],
    })


def _req(symbol, exch, interval=Interval.DAILY):
    return HistoryRequest(symbol=symbol, exchange=exch, interval=interval)


# ----------------------------- 港股 -----------------------------
async def test_hk_layout_with_suffix(tmp_path: Path):
    _make_daily_df().to_csv(tmp_path / "00700.HK.csv", index=False)
    feed = ChinaHKAStockParquetFeed(str(tmp_path))
    bars = await feed.fetch_bar_data(_req("00700", Exchange.HKEX))
    assert len(bars) == 5
    assert bars[0].exchange == Exchange.HKEX
    assert bars[0].close_price == 10.2
    assert bars[0].datetime.day == 2  # 纯日期保持


async def test_hk_layout_subdir(tmp_path: Path):
    d = tmp_path / "HK"
    d.mkdir()
    _make_daily_df().to_csv(d / "00700.csv", index=False)
    feed = ChinaHKAStockParquetFeed(str(tmp_path))
    bars = await feed.fetch_bar_data(_req("00700", Exchange.HKEX))
    assert len(bars) == 5


async def test_non_hk_exchange_returns_empty(tmp_path: Path):
    # 港股源不应接走 A 股/期货请求
    _make_daily_df().to_csv(tmp_path / "600000.SH.csv", index=False)
    feed = ChinaHKAStockParquetFeed(str(tmp_path))
    assert await feed.fetch_bar_data(_req("600000", Exchange.SSE)) == []


# ----------------------------- 期权 -----------------------------
async def test_option_layout_csv(tmp_path: Path):
    _make_daily_df().to_csv(tmp_path / "IO2409-C-3900.csv", index=False)
    feed = ChinaOptionParquetFeed(str(tmp_path))
    bars = await feed.fetch_bar_data(_req("IO2409-C-3900", Exchange.CFFEX))
    assert len(bars) == 5
    assert bars[0].exchange == Exchange.CFFEX
    assert bars[0].close_price == 10.2


async def test_option_layout_subdir_cffex(tmp_path: Path):
    d = tmp_path / "option"
    d.mkdir()
    _make_daily_df().to_csv(d / "IO2409-C-3900.csv", index=False)
    feed = ChinaOptionParquetFeed(str(tmp_path))
    bars = await feed.fetch_bar_data(_req("IO2409-C-3900", Exchange.CFFEX))
    assert len(bars) == 5


async def test_option_non_option_exchange_empty(tmp_path: Path):
    # 期权源不应接走港股（HKEX 不在期权交易所集合内）
    _make_daily_df().to_csv(tmp_path / "00700.HK.csv", index=False)
    feed = ChinaOptionParquetFeed(str(tmp_path))
    assert await feed.fetch_bar_data(_req("00700", Exchange.HKEX)) == []


# ----------------------------- registry 接线 -----------------------------
async def test_registry_wires_hk_feed(tmp_path: Path):
    _make_daily_df().to_csv(tmp_path / "00700.HK.csv", index=False)
    reg = build_default_registry(local_hk_root=str(tmp_path))
    assert "china_hk_parquet" in reg.list_feeds()
    pris = {name: p for feed, p in reg.ordered() for name in [feed.name]}
    assert pris["china_hk_parquet"] < pris["em_hk"]
    bars = await reg.get_bar_data(_req("00700", Exchange.HKEX))
    assert len(bars) == 5


async def test_registry_wires_option_feed(tmp_path: Path):
    _make_daily_df().to_csv(tmp_path / "IO2409-C-3900.csv", index=False)
    reg = build_default_registry(local_option_root=str(tmp_path))
    assert "china_option_parquet" in reg.list_feeds()
    pris = {name: p for feed, p in reg.ordered() for name in [feed.name]}
    assert pris["china_option_parquet"] < pris["akshare_option"]
    bars = await reg.get_bar_data(_req("IO2409-C-3900", Exchange.CFFEX))
    assert len(bars) == 5


def test_registry_fallback_when_unconfigured():
    reg = build_default_registry()
    assert "china_hk_parquet" not in reg.list_feeds()
    assert "china_option_parquet" not in reg.list_feeds()


async def test_parquet_real_read_hk(tmp_path: Path):
    pytest.importorskip("pyarrow", reason="需要 pyarrow 才能读 parquet")
    _make_daily_df().to_parquet(tmp_path / "00700.HK.parquet", index=False)
    feed = ChinaHKAStockParquetFeed(str(tmp_path))
    bars = await feed.fetch_bar_data(_req("00700", Exchange.HKEX))
    assert len(bars) == 5
    assert bars[0].close_price == 10.2
