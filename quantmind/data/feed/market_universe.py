"""全市场标的清单发现（A 股 / 港股），供调度器自动预热行情仓库使用。

与 ``web/utils/universes.py`` 的股票池解析解耦但共享同一套交易所划分：
- ``a_share_symbols``：沪深两市全部 A 股（akshare ``stock_info_a_code_name``，约 5000+）
- ``hk_symbols``：港股主板全部代码（akshare ``stock_hk_spot_em``，5 位左补零如 00700）

结果统一为 ``vt_symbol``（``{code}.{EXCHANGE}``）。带磁盘缓存（``data_cache/market/``，
TTL 24h），避免调度器每趟都打 akshare；失败返回空列表（由调用方跳过该趟），
与全库「mock/降级优先」风格一致。同步库用 ``asyncio.to_thread`` 包裹（由调用侧负责
或在本模块提供 async 便捷封装）。
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import List

_logger = logging.getLogger("quantmind.data.market_universe")

_CACHE_DIR = Path(os.environ.get(
    "QM_MARKET_CACHE_DIR",
    str(Path(__file__).resolve().parents[3] / "data_cache" / "market"),
))
_CACHE_TTL = 24 * 3600  # 标的清单一日一更足够


def _exchange_of(code: str) -> str:
    """A股代码 -> 交易所。6xx/68x(科创板)/9xx(B股)/11x → SSE；其余 → SZSE。"""
    c = code.strip()
    if c.startswith(("60", "68", "90", "11", "113", "110")):
        return "SSE"
    return "SZSE"


def _is_bjse(code: str) -> bool:
    """北交所/新三板代码（8xx/4xx/920）：当前无数据源支持，发现清单中排除。"""
    return code.startswith(("43", "83", "87", "88", "920")) \
        or (code.startswith("4") and len(code) == 6)


def _load_cache(name: str) -> List[str]:
    """读取磁盘缓存；未命中或在 TTL 内过期返回空列表。"""
    try:
        p = _CACHE_DIR / f"{name}.txt"
        if p.exists() and (time.time() - p.stat().st_mtime) < _CACHE_TTL:
            return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except Exception as exc:  # noqa: BLE001
        _logger.warning("读取标的清单缓存失败(%s): %s", name, exc)
    return []


def _write_cache(name: str, rows: List[str]) -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (_CACHE_DIR / f"{name}.txt").write_text("\n".join(rows), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        _logger.warning("写标的清单缓存失败(%s): %s", name, exc)


def fetch_a_share_symbols(cap: int = 6000) -> List[str]:
    """全A股列表 -> ``{code}.SSE/.SZSE``（去重、去空、可截断）。失败返回空列表。"""
    cached = _load_cache("a_share")
    if cached:
        return cached[:cap] if cap and cap > 0 else cached
    try:
        import akshare as ak

        df = ak.stock_info_a_code_name()
        if df is None or len(df) == 0:
            _logger.warning("akshare 未返回 A 股代码列表")
            return []
        seen: List[str] = []
        for raw in df["code"].tolist():
            code = str(raw).strip().zfill(6)
            if _is_bjse(code):
                continue  # 北交所：无数据源支持，不进预热/研究清单
            vt = f"{code}.{_exchange_of(code)}"
            if code and vt not in seen:
                seen.append(vt)
            if cap and cap > 0 and len(seen) >= cap:
                break
        _write_cache("a_share", seen)
        return seen
    except Exception as exc:  # noqa: BLE001
        _logger.warning("拉取全A股清单失败: %s", exc)
        return []


def fetch_hk_symbols(cap: int = 4000) -> List[str]:
    """港股全部代码 -> ``{code}.HKEX``（5 位左补零）。失败返回空列表。"""
    cached = _load_cache("hk")
    if cached:
        return cached[:cap] if cap and cap > 0 else cached
    try:
        import akshare as ak

        df = ak.stock_hk_spot_em()
        if df is None or len(df) == 0:
            _logger.warning("akshare 未返回港股代码列表")
            return []
        code_col = "代码" if "代码" in df.columns else df.columns[0]
        seen: List[str] = []
        for raw in df[code_col].tolist():
            code = str(raw).strip().zfill(5)
            vt = f"{code}.HKEX"
            if code and vt not in seen:
                seen.append(vt)
            if cap and cap > 0 and len(seen) >= cap:
                break
        _write_cache("hk", seen)
        return seen
    except Exception as exc:  # noqa: BLE001
        _logger.warning("拉取港股清单失败: %s", exc)
        return []


#: 市场标识 -> 所属交易所后缀集合（用于把市场与交易所解耦）
MARKET_EXCHANGES = {
    "A": {"SSE", "SZSE"},   # A股：沪深
    "HK": {"HKEX"},          # 港股
}


def market_exchanges(market: str) -> set:
    """返回某市场（'A' / 'HK'）覆盖的交易所后缀集合；未知市场返回空集。"""
    return MARKET_EXCHANGES.get(market, set())


def discover_market(market: str, cap: int = 6000) -> List[str]:
    """按市场返回全市场 vt-symbol 清单（带磁盘缓存）。

    - ``A`` ：A股（沪深两市，``.SSE`` / ``.SZSE``）
    - ``HK``：港股（``.HKEX``）

    失败返回空列表（由调用方跳过该市场）。
    """
    if market == "A":
        return fetch_a_share_symbols(cap=cap)
    if market == "HK":
        return fetch_hk_symbols(cap=cap)
    return []


def discover_all(cap_a: int = 6000, cap_hk: int = 4000) -> List[str]:
    """合并 A 股 + 港股全市场清单（顺序：A股在前）。任一失败不影响另一。"""
    out: List[str] = []
    out.extend(fetch_a_share_symbols(cap=cap_a))
    out.extend(fetch_hk_symbols(cap=cap_hk))
    return out
