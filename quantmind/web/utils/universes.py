"""A股指数/股票池（universe）篮子解析。

端到端/截面/搜索/挖掘流水线的「标的篮子」支持常用宽基指数与全市场股票池：

  - ``全A市场``  全部 A 股（沪深主板/创业板/科创板）
  - ``沪深300``  大盘蓝筹（akshare ``index_stock_cons("000300")``）
  - ``中证500``  中盘（``000905``）
  - ``中证2000`` 微盘（``932000``）

解析目标：把指数/全市场展开成 **带交易所后缀** 的 vt-symbol 列表（如
``600519.SSE`` / ``000001.SZSE``）。后端 ``_split_vt_symbol`` 已支持每标的自带
交易所，因此混合沪深两市的股票池可直接透传给现有 API，无需改动 data 层。

健壮性（离线优先，与全库 mock/降级风格一致）：
  - 有网（akshare 可用）→ 实时拉取成分股并做**磁盘缓存**（``data_cache/universes/``）+ 进程内缓存；
  - 无网 / akshare 接口变动 → 回落内嵌**代表性成分股**（保证离线也能跑通截面）；
  - 超大股票池默认**截断到 ``max_symbols``**（前端可控，避免数百~数千标的拖垮因子挖掘）。
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 统一交易所划分规则：优先复用后端模块（避免复制），在（测试/独立加载 utils 包时）
# 相对导入不可用则回落内联规则，保证两种加载方式都成立。
try:  # noqa: E402
    from ...data.feed.market_universe import _exchange_of
except Exception:  # noqa: BLE001 - 独立加载 utils 包（sys.path 到 web/）时相对导入越界
    def _exchange_of(code: str) -> str:
        c = code.strip()
        if c.startswith(("60", "68", "90", "11", "113", "110")):
            return "SSE"
        return "SZSE"

_logger = logging.getLogger("quantmind.web.universes")

# ---------------------------------------------------------------------------
# 内嵌代表性成分股（离线兜底）。数量刻意控小，保证 mock/无网也能跑通且不超时。
# 每组取若干有代表性、流动性较高的大/中/小盘样本，沪深两市各配若干。
# ---------------------------------------------------------------------------
_FALLBACK = {
    "沪深300": [
        "600519.SSE", "601318.SSE", "600036.SSE", "600900.SSE", "601899.SSE",
        "600030.SSE", "600276.SSE", "601166.SSE", "600887.SSE", "601012.SSE",
        "000001.SZSE", "000858.SZSE", "300750.SZSE", "000333.SZSE", "002594.SZSE",
        "000651.SZSE", "300059.SZSE", "002415.SZSE", "000725.SZSE", "002352.SZSE",
    ],
    "中证500": [
        "600157.SSE", "600392.SSE", "600711.SSE", "601698.SSE", "603019.SSE",
        "600153.SSE", "600588.SSE", "601138.SSE", "603501.SSE", "688981.SSE",
        "000883.SZSE", "002065.SZSE", "002460.SZSE", "002271.SZSE", "300014.SZSE",
        "000876.SZSE", "002595.SZSE", "002602.SZSE", "300347.SZSE", "002405.SZSE",
    ],
    "中证2000": [
        "600095.SSE", "600250.SSE", "600358.SSE", "600493.SSE", "600616.SSE",
        "601002.SSE", "603033.SSE", "603076.SSE", "603098.SSE", "603117.SSE",
        "000665.SZSE", "000780.SZSE", "000927.SZSE", "002166.SZSE", "002178.SZSE",
        "002240.SZSE", "002370.SZSE", "300155.SZSE", "300254.SZSE", "300314.SZSE",
    ],
    "全A市场": [
        "600519.SSE", "601318.SSE", "600036.SSE", "000001.SZSE", "000858.SZSE",
        "300750.SZSE", "000333.SZSE", "002594.SZSE", "600900.SSE", "601899.SSE",
        "600030.SSE", "000651.SZSE", "000725.SZSE", "002415.SZSE", "300059.SZSE",
        "600276.SSE", "601166.SSE", "688981.SSE", "000063.SZSE", "002352.SZSE",
    ],
}

#: 宽基指数 -> akshare symbol（东财指数代码：沪深300=000300，中证500=000905，
#: 中证2000=932000；全A无单一指数，走股票列表源）。
_INDEX_CODES: Dict[str, str] = {
    "沪深300": "000300",
    "中证500": "000905",
    "中证2000": "932000",
}

#: 全A/沪深300/中证500/中证2000 这些「指数/股票池」篮子名称
UNIVERSE_BASKETS = ("全A市场", "沪深300", "中证500", "中证2000")

# ---------------------------------------------------------------------------
# 进程内 + 磁盘缓存
# ---------------------------------------------------------------------------
_CACHE_DIR = Path(os.environ.get(
    "QM_WEB_CACHE_DIR",
    str(Path(__file__).resolve().parents[3] / "data_cache" / "universes"),
))
_CACHE_TTL = 24 * 3600  # 成分股一日一更足够
_mem_cache: Dict[str, Tuple[float, List[str]]] = {}
_MAX_INDEX_MS = 8_000  # 单次 akshare 调用超时上限（毫秒）


def _load_or_write_cache(name: str, rows: List[str]) -> List[str]:
    """写磁盘缓存（uid 注解进文件名）；命中且未过期则直接读回。"""
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        p = _CACHE_DIR / f"{name}.txt"
        if p.exists() and (time.time() - p.stat().st_mtime) < _CACHE_TTL:
            got = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if got:
                return got
        p.write_text("\n".join(rows), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        _logger.warning("universe 磁盘缓存读写失败(%s): %s", name, exc)
    return rows


def _fetch_index_components(index_code: str) -> List[str]:
    """用 akshare 拉指数成分股 -> ``code.EXCHANGE`` 列表（失败抛错由调用方兜底）。"""
    import akshare as ak

    df = ak.index_stock_cons(symbol=str(index_code))
    if df is None or len(df) == 0:
        raise ValueError(f"akshare 未返回 {index_code} 成分股")
    col = "品种代码"
    codes = [str(x).zfill(6) for x in df[col].tolist() if str(x).strip()]
    return [f"{c}.{_exchange_of(c)}" for c in codes]


def _fetch_all_a(raw_cap: int = 20_000) -> List[str]:
    """全A股票列表：akshare 实时列表，截断后缓存。失败抛错由调用方兜底。"""
    import akshare as ak

    df = ak.stock_info_a_code_name()
    if df is None or len(df) == 0:
        raise ValueError("akshare 未返回 A 股代码列表")
    col = "code"
    codes = [str(x).zfill(6) for x in df[col].tolist() if str(x).strip()]
    # 去重 + 截断（全A 约 5000+，截面挖掘默认只取前 max_symbols）
    seen: List[str] = []
    for c in codes:
        vt = f"{c}.{_exchange_of(c)}"
        if vt not in seen:
            seen.append(vt)
        if len(seen) >= raw_cap:
            break
    return seen


def resolve_universe(name: str, max_symbols: Optional[int] = None) -> List[str]:
    """解析一个指数/全A股票池为 ``code.EXCHANGE`` 列表。

    优先级：内存缓存 > 磁盘缓存 > akshare 实时 > 内嵌离线兜底。
    返回列表按 ``max_symbols`` 截断（不截断则取全量——注意全A/中证2000 很大，调用方应传上限）。
    """
    name = str(name).strip()
    fallback = _FALLBACK.get(name, [])

    # 1) 内存缓存
    hit = _mem_cache.get(name)
    if hit is not None and (time.time() - hit[0]) < _CACHE_TTL:
        rows = hit[1]
    else:
        rows = _fetch_once(name)
        _mem_cache[name] = (time.time(), rows)

    if max_symbols and max_symbols > 0:
        rows = rows[: max_symbols]
    return list(rows)


def _fetch_once(name: str) -> List[str]:
    """无内存命中时的真实获取（磁盘缓存 + 实时 + 兜底）。"""
    code = _INDEX_CODES.get(name)          # 宽基指数
    try:
        p = _CACHE_DIR / f"{name}.txt"
        if p.exists() and (time.time() - p.stat().st_mtime) < _CACHE_TTL:
            got = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if got:
                return got
        if code:
            rows = _fetch_index_components(code)
        elif name == "全A市场":
            rows = _fetch_all_a()
        else:
            raise ValueError(f"未知股票池: {name}")
        return _load_or_write_cache(name, rows)
    except Exception as exc:  # noqa: BLE001
        _logger.warning("股票池 %s 实时解析失败，回落离线兜底: %s", name, exc)
        return _FALLBACK.get(name, [])
