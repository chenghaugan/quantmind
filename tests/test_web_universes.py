"""Web 股票池/指数篮子解析（utils.universes + constants.resolve_basket_symbols）。

离线友好：不依赖网络/akshare；通过 monkeypatch 强制实时获取失败，验证离线兜底
与交易所后缀正确性。
"""
import sys
from pathlib import Path

import pytest

_WEB = Path(__file__).resolve().parents[1] / "quantmind" / "web"
sys.path.insert(0, str(_WEB))

import utils.universes as u  # noqa: E402
from utils.constants import (  # noqa: E402
    BASKET_CHOICES, INDEX_BASKETS, resolve_basket_symbols,
)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """强制实时获取失败：让所有解析走离线兜底（确定性、无网络）。"""
    monkeypatch.setattr(u, "_fetch_index_components",
                        lambda code: (_ for _ in ()).throw(RuntimeError("no net")))
    monkeypatch.setattr(u, "_fetch_all_a",
                        lambda **kw: (_ for _ in ()).throw(RuntimeError("no net")))
    u._mem_cache.clear()
    yield
    u._mem_cache.clear()


def _reset_disk():
    import shutil
    shutil.rmtree(u._CACHE_DIR, ignore_errors=True)


def test_index_baskets_in_choices():
    """四个常用指数/全A 股票池都出现在选项里。"""
    for name in ("全A市场", "沪深300", "中证500", "中证2000"):
        assert f"指数·{name}" in BASKET_CHOICES
    assert len(INDEX_BASKETS) == 4


def test_offline_fallback_resolves_with_exchange():
    """离线时，股票池回落内嵌成分，且每条都带 .SSE/.SZSE 后缀。"""
    for name in ("沪深300", "中证500", "中证2000", "全A市场"):
        symbols, exch = resolve_basket_symbols(f"指数·{name}", max_symbols=6)
        assert len(symbols) == 6
        assert exch == "SSE"
        assert all("." in s and s.rsplit(".", 1)[-1] in ("SSE", "SZSE") for s in symbols)


def test_max_symbols_truncates():
    """股票池按 max_symbols 截断。"""
    symbols, _ = resolve_basket_symbols("指数·沪深300", max_symbols=10)
    assert len(symbols) == 10


def test_offline_fallback_bundled_has_both_exchanges():
    """内嵌兜底涵盖沪深两市（不是单一交易所）。"""
    symbols, _ = resolve_basket_symbols("指数·沪深300")
    ex = {s.rsplit(".", 1)[-1] for s in symbols}
    assert "SSE" in ex and "SZSE" in ex


def test_static_basket_unchanged():
    """原有静态篮子行为不变（返回 (symbols, exchange)）。"""
    symbols, exch = resolve_basket_symbols("A股白马")
    assert exch == "SSE"
    assert symbols == ["600519", "601318", "600036", "600900", "601899"]


def test_exchange_mapping():
    """代码→交易所映射：6xx/68x→SSE，000/300→SZSE。"""
    assert u._exchange_of("600000") == "SSE"
    assert u._exchange_of("688981") == "SSE"
    assert u._exchange_of("000001") == "SZSE"
    assert u._exchange_of("300750") == "SZSE"
    assert u._exchange_of("002594") == "SZSE"
