# -*- coding: utf-8 -*-
"""A股增量更新「预检跳过」测试。

覆盖：
  1. 日线/分钟线"最近已收盘边界"推算（周末回退、盘中不追当日）；
  2. _is_up_to_date 命中/未命中（含 naive 时间戳容错）；
  3. 集成：本地已最新 → 不发源请求（fetch 不被调用）；本地陈旧 → 正常拉取过滤落盘。
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from quantmind.api.stock_download import (
    _expected_latest_daily,
    _expected_latest_minute,
    _is_up_to_date,
    _job_stock_auto_download,
)
from quantmind.core.constant import Exchange, Interval
from quantmind.core.object import BarData

# 2024-01-12 是周五；2024-01-15 是周一
FRI_POST_CLOSE = datetime(2024, 1, 12, 8, 0, tzinfo=timezone.utc)   # 周五 16:00 CST
SAT = datetime(2024, 1, 13, 2, 0, tzinfo=timezone.utc)              # 周六 10:00 CST
MON_INTRADAY = datetime(2024, 1, 15, 2, 0, tzinfo=timezone.utc)     # 周一 10:00 CST
MON_POST_CLOSE = datetime(2024, 1, 15, 8, 0, tzinfo=timezone.utc)   # 周一 16:00 CST


def _cst(y, mo, d, h, mi):
    return datetime(y, mo, d, h, mi, tzinfo=timezone(timedelta(hours=8)))


# ---------------------------------------------------------------- 边界推算
def test_expected_daily():
    """日线边界：收盘后=当天；周末/盘中=上一交易日。"""
    assert _expected_latest_daily(FRI_POST_CLOSE).date().isoformat() == "2024-01-12"
    assert _expected_latest_daily(SAT).date().isoformat() == "2024-01-12"      # 周五回退
    assert _expected_latest_daily(MON_INTRADAY).date().isoformat() == "2024-01-12"  # 盘中不追当日
    assert _expected_latest_daily(MON_POST_CLOSE).date().isoformat() == "2024-01-15"
    # 边界：15:00 整收盘即视为已收盘
    assert _expected_latest_daily(_cst(2024, 1, 15, 15, 0)).date().isoformat() == "2024-01-15"
    assert _expected_latest_daily(_cst(2024, 1, 15, 14, 59)).date().isoformat() == "2024-01-12"


def test_expected_minute():
    """分钟边界：最近收盘 11:30/15:00（留一个 bar 容差），跨周末回退。"""
    # 周五 15:30 CST → 15:00 - 30m 容差 = 14:30 CST
    assert _expected_latest_minute(FRI_POST_CLOSE, "30m") == _cst(2024, 1, 12, 14, 30)
    # 周五 15:30 CST → 15:00 - 1h 容差 = 14:00 CST
    assert _expected_latest_minute(FRI_POST_CLOSE, "1h") == _cst(2024, 1, 12, 14, 0)
    # 周五 10:00 CST（午前盘中）→ 周四 15:00 - 容差
    assert _expected_latest_minute(_cst(2024, 1, 12, 10, 0), "30m") == _cst(2024, 1, 11, 14, 30)
    # 周六 → 周五 15:00 - 容差
    assert _expected_latest_minute(SAT, "1h") == _cst(2024, 1, 12, 14, 0)
    # 周一 10:00 CST → 周五 15:00 - 容差
    assert _expected_latest_minute(MON_INTRADAY, "30m") == _cst(2024, 1, 12, 14, 30)
    # 周一 13:30 CST（午后开盘后）→ 当天 11:30 - 容差
    assert _expected_latest_minute(_cst(2024, 1, 15, 13, 30), "30m") == _cst(2024, 1, 15, 11, 0)


def test_is_up_to_date():
    """fresh 命中 / stale 未命中 / naive 时间戳视为 UTC。"""
    fresh_daily = datetime(2024, 1, 12, 0, 0, tzinfo=timezone.utc)   # 周五日线
    assert _is_up_to_date(fresh_daily, "1d", now_utc=SAT) is True
    stale_daily = datetime(2024, 1, 11, 0, 0, tzinfo=timezone.utc)   # 周四日线
    assert _is_up_to_date(stale_daily, "1d", now_utc=FRI_POST_CLOSE) is False
    # naive 视为 UTC
    assert _is_up_to_date(datetime(2024, 1, 12, 0, 0), "1d", now_utc=SAT) is True
    # None/未知周期不跳过
    assert _is_up_to_date(None, "1d", now_utc=SAT) is False
    assert _is_up_to_date(fresh_daily, "5m", now_utc=SAT) is False
    # 分钟 fresh：周五收盘后更新过（最后 bar 14:30 时间戳）
    assert _is_up_to_date(_cst(2024, 1, 12, 14, 30), "30m", now_utc=SAT) is True


# ---------------------------------------------------------------- 集成：跳过不发请求
class _FakeDC:
    def __init__(self, bars):
        self._bars = bars
        self.save_calls = []

    def load(self, req):
        return list(self._bars)

    def latest_datetime(self, req):
        return max((b.datetime for b in self._bars), default=None)

    def save(self, bars):
        self._bars = list(self._bars) + list(bars)
        self.save_calls.append(len(bars))
        return len(self._bars)


class _FakeDM:
    def __init__(self, bars):
        self.disk_cache = _FakeDC(bars)


def _bar(dt):
    return BarData(symbol="600000", exchange=Exchange.SSE, datetime=dt,
                   interval=Interval("1d"), open_price=10, high_price=10,
                   low_price=10, close_price=10, volume=1.0,
                   open_interest=0.0, turnover=0.0)


def test_job_skips_fresh_without_fetch(monkeypatch):
    """本地已最新 → 预检命中，完全不发源请求。"""
    now = datetime.now(timezone.utc)
    fresh = _expected_latest_daily(now)  # 用边界自身当最新 bar，必然命中
    calls = []

    def fake_fetch(symbol, exchange, interval_str):  # 同步：不应被调用
        calls.append(symbol)
        return []

    import quantmind.api.stock_download as sd
    monkeypatch.setattr(sd, "_fetch_stock_bars", fake_fetch)

    out = asyncio.run(_job_stock_auto_download(
        {"dm": _FakeDM([_bar(fresh)])}, symbols=["600000.SSE"],
        intervals=["1d"], manual=True))
    assert calls == []                      # 未发源请求
    assert out["up_to_date"] == 1
    assert out["results"][0]["skipped"] is True


def test_job_fetches_when_stale(monkeypatch):
    """本地陈旧 → 正常拉源，且只落盘比本地新的 bar。"""
    stale = datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc)
    dm = _FakeDM([_bar(stale)])

    def fake_fetch(symbol, exchange, interval_str):
        return [_bar(stale), _bar(stale + timedelta(days=1))]  # 一根旧的一根新的

    import quantmind.api.stock_download as sd
    monkeypatch.setattr(sd, "_fetch_stock_bars", fake_fetch)

    out = asyncio.run(_job_stock_auto_download(
        {"dm": dm}, symbols=["600000.SSE"], intervals=["1d"], manual=True))
    assert out["updated"] == 1
    assert out["failed"] == 0
    # 只落盘 1 根新 bar（旧 bar 被过滤）
    assert dm.disk_cache.save_calls == [1]
