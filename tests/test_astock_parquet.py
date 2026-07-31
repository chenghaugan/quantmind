"""A 股本地 Parquet/CSV 适配器测试。

离线环境无 pyarrow，故主路径用 .csv 验证全部逻辑（列映射/时区/路径探测/registry 接线）；
另附 pytest.importorskip("pyarrow") 守卫的 Parquet 真读测试，用户环境装了 pyarrow 即自动启用。
"""
from __future__ import annotations

import pandas as pd
import pytest
from pathlib import Path

from quantmind.core.constant import Exchange, Interval
from quantmind.data.feed.base import HistoryRequest
from quantmind.data.feed.astock_parquet import ChinaAStockParquetFeed
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


async def test_csv_layout_with_suffix(tmp_path: Path):
    # 布局: 600000.SH.csv
    _make_daily_df().to_csv(tmp_path / "600000.SH.csv", index=False)
    feed = ChinaAStockParquetFeed(str(tmp_path))
    bars = await feed.fetch_bar_data(_req("600000", Exchange.SSE))
    assert len(bars) == 5
    assert bars[0].symbol == "600000"
    assert bars[0].exchange == Exchange.SSE
    # 纯日期保持交易日期不变（无 UTC 日界导致的前移一天）
    assert bars[0].datetime.year == 2024 and bars[0].datetime.month == 1 and bars[0].datetime.day == 2
    assert bars[0].datetime.hour == 0
    assert bars[0].open_price == 10.0
    assert bars[0].close_price == 10.2
    assert bars[0].high_price == 10.5
    assert bars[0].low_price == 9.5
    # amount -> turnover 映射
    assert bars[0].turnover == 10000.0
    assert bars[-1].close_price == 14.2


async def test_layout_subdir_exchange(tmp_path: Path):
    # 布局: SH/600000.csv
    d = tmp_path / "SH"
    d.mkdir()
    _make_daily_df().to_csv(d / "600000.csv", index=False)
    feed = ChinaAStockParquetFeed(str(tmp_path))
    bars = await feed.fetch_bar_data(_req("600000", Exchange.SSE))
    assert len(bars) == 5
    assert bars[0].close_price == 10.2


async def test_layout_data_subdir_szse(tmp_path: Path):
    # 布局: data/000001.SZ.csv (深交所)
    d = tmp_path / "data"
    d.mkdir()
    _make_daily_df().to_csv(d / "000001.SZ.csv", index=False)
    feed = ChinaAStockParquetFeed(str(tmp_path))
    bars = await feed.fetch_bar_data(_req("000001", Exchange.SZSE))
    assert len(bars) == 5
    assert bars[0].exchange == Exchange.SZSE


async def test_symbol_with_suffix_and_szse(tmp_path: Path):
    # 直接给 000001.SZ 作为 symbol
    _make_daily_df().to_csv(tmp_path / "000001.SZ.csv", index=False)
    feed = ChinaAStockParquetFeed(str(tmp_path))
    bars = await feed.fetch_bar_data(_req("000001.SZ", Exchange.SZSE))
    assert len(bars) == 5


async def test_non_astock_exchange_returns_empty(tmp_path: Path):
    # 期货/其它交易所不应被 A 股源接走，保证 registry 正确降级
    _make_daily_df().to_csv(tmp_path / "rb0.SHFE.csv", index=False)
    feed = ChinaAStockParquetFeed(str(tmp_path))
    bars = await feed.fetch_bar_data(_req("rb0", Exchange.SHFE))
    assert bars == []


async def test_missing_symbol_returns_empty(tmp_path: Path):
    feed = ChinaAStockParquetFeed(str(tmp_path))
    bars = await feed.fetch_bar_data(_req("999999", Exchange.SSE))
    assert bars == []


async def test_date_preserved_no_off_by_one(tmp_path: Path):
    # 确认日频日期不被 -8h 推前一天
    _make_daily_df().to_csv(tmp_path / "600000.SH.csv", index=False)
    feed = ChinaAStockParquetFeed(str(tmp_path))
    bars = await feed.fetch_bar_data(_req("600000", Exchange.SSE))
    dates = [(b.datetime.year, b.datetime.month, b.datetime.day) for b in bars]
    assert dates[0] == (2024, 1, 2)
    assert dates[-1] == (2024, 1, 6)


async def test_registry_wires_astock_feed(tmp_path: Path):
    _make_daily_df().to_csv(tmp_path / "600000.SH.csv", index=False)
    reg = build_default_registry(local_stock_root=str(tmp_path))
    assert "china_astock_parquet" in reg.list_feeds()
    # 优先级应高于 mootdx(20)
    pris = {name: p for feed, p in reg.ordered() for name in [feed.name]}
    assert pris["china_astock_parquet"] < pris["mootdx_astock"]
    # 实际取数走本地 A 股源
    bars = await reg.get_bar_data(_req("600000", Exchange.SSE))
    assert len(bars) == 5
    assert bars[0].symbol == "600000"


def test_registry_fallback_when_no_local_stock():
    # 未配置 local_stock_root 时不应注册 A 股本地源
    reg = build_default_registry()
    assert "china_astock_parquet" not in reg.list_feeds()


async def test_parquet_real_read(tmp_path: Path):
    pytest.importorskip("pyarrow", reason="需要 pyarrow 才能读 parquet")
    _make_daily_df().to_parquet(tmp_path / "600000.SH.parquet", index=False)
    feed = ChinaAStockParquetFeed(str(tmp_path))
    bars = await feed.fetch_bar_data(_req("600000", Exchange.SSE))
    assert len(bars) == 5
    assert bars[0].close_price == 10.2
    assert bars[0].turnover == 10000.0
