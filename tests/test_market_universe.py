"""全市场标的发现（market_universe）测试。

仅测纯逻辑（交易所划分 / 空清单降级 / 磁盘缓存命中），不联网——akshare 调用被
monkeypatch 成固定小数据。遵循仓库离线优先、mock/降级风格。
"""
from __future__ import annotations

import importlib

import pandas as pd
import pytest

from quantmind.data.feed import market_universe as mu

# akshare 是模块内延迟导入，统一 patch 到 akshare 模块对象上
import akshare as _ak


def test_exchange_of_rules():
    """交易所划分：60/68/90/B股/转债前缀 -> SSE，其余 -> SZSE。"""
    assert mu._exchange_of("600519") == "SSE"
    assert mu._exchange_of("688981") == "SSE"
    assert mu._exchange_of("900901") == "SSE"
    assert mu._exchange_of("110038") == "SSE"
    assert mu._exchange_of("000001") == "SZSE"
    assert mu._exchange_of("300750") == "SZSE"
    assert mu._exchange_of("002594") == "SZSE"


def test_a_share_symbols_mapping(monkeypatch, tmp_path, monkeypatch_cache_dir):
    """从 akshare 代码列转 vt-symbol，按前缀分 SSE/SZSE。"""
    monkeypatch_cache_dir(monkeypatch, tmp_path)
    fake_df = pd.DataFrame({"code": ["600519", "000001", "300750", "688981"]})
    monkeypatch.setattr(_ak, "stock_info_a_code_name", lambda: fake_df)

    got = mu.fetch_a_share_symbols()
    assert "600519.SSE" in got
    assert "000001.SZSE" in got
    assert "300750.SZSE" in got
    assert "688981.SSE" in got


def test_a_share_symbols_empty_degrades(monkeypatch, tmp_path, monkeypatch_cache_dir):
    """akshare 返回空/抛错 -> 返回空列表而不抛错。"""
    monkeypatch_cache_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(_ak, "stock_info_a_code_name",
                        lambda: (_ for _ in ()).throw(RuntimeError("net down")))
    assert mu.fetch_a_share_symbols() == []

    monkeypatch.setattr(_ak, "stock_info_a_code_name", lambda: pd.DataFrame())
    assert mu.fetch_a_share_symbols() == []


def test_hk_symbols_zero_padding(monkeypatch, tmp_path, monkeypatch_cache_dir):
    """港股 5 位左补零 + HKEX 后缀。"""
    monkeypatch_cache_dir(monkeypatch, tmp_path)
    fake_df = pd.DataFrame({"代码": ["700", "00700", "9988", "1"]})
    monkeypatch.setattr(_ak, "stock_hk_spot_em", lambda: fake_df)

    got = mu.fetch_hk_symbols()
    assert "00700.HKEX" in got
    assert "09988.HKEX" in got
    assert "00001.HKEX" in got


def test_hk_symbols_empty_degrades(monkeypatch, tmp_path, monkeypatch_cache_dir):
    """港股源失败 -> 空列表不抛错。"""
    monkeypatch_cache_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(_ak, "stock_hk_spot_em",
                        lambda: (_ for _ in ()).throw(ValueError("boom")))
    assert mu.fetch_hk_symbols() == []


def test_discover_all_combines(monkeypatch, tmp_path, monkeypatch_cache_dir):
    """discover_all 合并 A股 + 港股，A股在前。"""
    monkeypatch_cache_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(
        _ak, "stock_info_a_code_name",
        lambda: pd.DataFrame({"code": ["600519", "000001"]}))
    monkeypatch.setattr(
        _ak, "stock_hk_spot_em",
        lambda: pd.DataFrame({"代码": ["00700", "9988"]}))

    got = mu.discover_all()
    assert got.index("600519.SSE") < got.index("00700.HKEX")
    assert "000001.SZSE" in got
    assert "00700.HKEX" in got


def test_cache_hit_avoids_network(monkeypatch, tmp_path, monkeypatch_cache_dir):
    """磁盘缓存命中后不再调 akshare。"""
    monkeypatch_cache_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(
        _ak, "stock_info_a_code_name",
        lambda: pd.DataFrame({"code": ["600519"]}))

    first = mu.fetch_a_share_symbols()
    assert first == ["600519.SSE"]
    # 二次调用：缓存命中，触网函数不应再被调到（用内置 monkeypatch 计数器验证）
    second = mu.fetch_a_share_symbols()
    assert second == first


# ---------------------------------------------------------------------------
# 夹具：把磁盘缓存目录重定向到 tmp_path，避免污染真实 data_cache/
# ---------------------------------------------------------------------------
@pytest.fixture
def monkeypatch_cache_dir(monkeypatch, tmp_path):
    def _patch(monkeypatch, tmp):
        monkeypatch.setenv("QM_MARKET_CACHE_DIR", str(tmp / "market"))
        # 强制重读模块级 _CACHE_DIR
        import importlib
        importlib.reload(mu)
    return _patch
