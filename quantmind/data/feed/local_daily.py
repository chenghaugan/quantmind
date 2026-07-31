"""通用本地日频 Parquet/CSV 适配器（A 股 / 港股 / 期权共用）。

把 astock-data-toolkit、a-stock-data、东方财富导出的港股/期权日频 K 线接进 feed 层，
使多资产回测/因子评估直接吃本地真实数据，不需依赖 mootdx / em / akshare 的限频与稳定性。

时间约定（与体系一致、且规避 UTC 日界坑）：
  - 文件时间为**交易日期**（日频、无具体时刻）时，保持交易日期不变；
  - 若文件带具体时刻（如盘后 15:00），则按北京时间减 8h 转 UTC。

路径探测（多布局自动，无需手工指定）：
  - {code}.{SUFFIX}.parquet / {code}.{SUFFIX}.csv
  - {code}.parquet / {code}.csv
  - {EXCHANGE}/ / {SUFFIX}/ / 自定义 extra_dirs 子目录
  - 递归兜底：匹配含 code 的行情文件
同时兼容 .csv/.txt 同名文件（离线调试友好）。

读取 .parquet 需 pyarrow 或 fastparquet；未安装时该源静默降级到下一数据源。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Set, Tuple

import pandas as pd

from .base import HistoryRequest
from .local_file import LocalFileFeed
from ...core.constant import Exchange

_logger = logging.getLogger("quantmind.data.local_daily")


class LocalDailyParquetFeed(LocalFileFeed):
    """通用本地日频 Parquet/CSV 适配器（A股/港股/期权共用基类）。

    :param exchanges: 允许接管的交易所集合；为空表示接管全部（不推荐，会抢走期货请求）。
    :param extra_dirs: 额外探测子目录前缀（相对 root），如 ["data/", "daily/", "stock/"]。
    :param suffix_map: 交易所枚举 -> 市场后缀（如 {SSE: "SH", SZSE: "SZ"}），用于补文件名后缀。
    :param tz_offset_hours: 默认 8（北京时间）。纯日期行不平移；带时刻行按此偏移转 UTC。
    """

    name = "local_daily_parquet"

    def __init__(
        self,
        root_dir: str,
        name: str = "local_daily_parquet",
        tz_offset_hours: int = 8,
        exchanges: Optional[Set[Exchange]] = None,
        extra_dirs: Optional[List[str]] = None,
        suffix_map: Optional[dict] = None,
    ) -> None:
        super().__init__(root_dir, name=name, tz_offset_hours=tz_offset_hours)
        self.exchanges = set(exchanges) if exchanges else set()
        self.extra_dirs = list(extra_dirs or [])
        self.suffix_map = suffix_map or {}

    # ---- 时间归一：纯日期保持交易日期，带时刻按北京时间转 UTC ----
    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        if "datetime" not in df.columns:
            raise ValueError("数据中缺少时间列（datetime/date/time 等）")
        dt = pd.to_datetime(df["datetime"], errors="coerce")
        if dt.isna().any():
            _logger.warning("存在无法解析的时间值，已置为 NaT")
        # 仅当日内含具体时刻（时/分/秒非零）才视为北京时间并减 8h；
        # 纯日期（日频，午夜 00:00）保持交易日期不变，避免 UTC 日界导致日期前移一天。
        timed = (dt.dt.hour != 0) | (dt.dt.minute != 0) | (dt.dt.second != 0)
        shifted = dt - pd.Timedelta(hours=self.tz_offset_hours)
        out = df.copy()
        out["datetime"] = dt.mask(timed, shifted)
        return out

    # ---- 路径解析（多布局自动探测） ----
    def _resolve_paths(self, req: HistoryRequest) -> Tuple[List[Path], bool]:
        if self.exchanges and req.exchange not in self.exchanges:
            return [], False

        sym = req.symbol.upper().strip()
        if "." in sym:
            code, suffix = sym.rsplit(".", 1)
        else:
            code = sym
            suffix = self.suffix_map.get(req.exchange, "")
        code = code.strip()
        suffix = suffix.upper()

        prefixes = ["", f"{req.exchange.value}/", f"{suffix}/"] + self.extra_dirs
        names = [f"{code}.{suffix}" if suffix else code, code]
        exts = [".parquet", ".csv", ".txt"]

        for prefix in prefixes:
            for name in names:
                for ext in exts:
                    found = sorted((self.root / prefix).glob(f"{name}{ext}"))
                    if found:
                        return [found[0]], False
        # 递归兜底：匹配含 code 的行情文件（避免布局差异漏配）
        rec = sorted(self.root.rglob(f"*{code}*{suffix}*"))
        rec = [p for p in rec if p.suffix.lower() in (".parquet", ".csv", ".txt")]
        if rec:
            return [rec[0]], False
        _logger.warning("本地源 %s 未找到 %s.%s", self.name, req.symbol, req.exchange.value)
        return [], False


# ----------------------------- 三类本地源 -----------------------------
class ChinaAStockParquetFeed(LocalDailyParquetFeed):
    """A 股（沪深）本地 Parquet/CSV 适配器。仅接管 SSE/SZSE。"""

    name = "china_astock_parquet"

    def __init__(self, root_dir: str, name: str = "china_astock_parquet", tz_offset_hours: int = 8) -> None:
        super().__init__(
            root_dir, name=name, tz_offset_hours=tz_offset_hours,
            exchanges={Exchange.SSE, Exchange.SZSE},
            extra_dirs=["data/", "daily/", "stock/", "stocks/"],
            suffix_map={Exchange.SSE: "SH", Exchange.SZSE: "SZ"},
        )


class ChinaHKAStockParquetFeed(LocalDailyParquetFeed):
    """港股本地 Parquet/CSV 适配器。仅接管 HKEX。"""

    name = "china_hk_parquet"

    def __init__(self, root_dir: str, name: str = "china_hk_parquet", tz_offset_hours: int = 8) -> None:
        super().__init__(
            root_dir, name=name, tz_offset_hours=tz_offset_hours,
            exchanges={Exchange.HKEX},
            extra_dirs=["hk/", "HK/", "data/", "daily/", "stock/", "stocks/"],
        )


class ChinaOptionParquetFeed(LocalDailyParquetFeed):
    """场内期权本地 Parquet/CSV 适配器。接管股指/ETF/商品期权所在交易所。"""

    name = "china_option_parquet"

    def __init__(self, root_dir: str, name: str = "china_option_parquet", tz_offset_hours: int = 8) -> None:
        super().__init__(
            root_dir, name=name, tz_offset_hours=tz_offset_hours,
            exchanges={Exchange.CFFEX, Exchange.SSE, Exchange.SZSE,
                       Exchange.DCE, Exchange.CZCE, Exchange.SHFE, Exchange.INE},
            extra_dirs=["option/", "options/", "data/", "daily/"],
        )
