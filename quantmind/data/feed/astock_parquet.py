"""A 股（沪深）本地 Parquet/CSV 适配器。

把 astock-data-toolkit、a-stock-data 等落地的 A 股日频 K 线接进 feed 层，
使 A 股回测 / 因子评估直接吃真实数据，而不依赖 mootdx / akshare 的限频与稳定性。

常见落地布局（自动探测，无需手工指定）：
  - {code}.{SH|SZ}.parquet        （如 600000.SH.parquet）
  - {code}.parquet
  - {SH|SZ|exchange}/{code}.parquet
  - data/{code}.parquet
  - daily/{code}.parquet
  - stock/ / stocks/ 子目录
同时兼容 .csv 同名文件（离线调试 / 部分工具导出 CSV）。

时间约定：文件时间为**交易日期**（日频，无具体时刻）时，保持交易日期不变；
若文件带具体时刻（如盘后 15:00），则按北京时间减 8h 转 UTC（与体系一致）。
A 股无主力连续概念，_resolve_paths 恒返回 is_main=False。

注意：读取 .parquet 需 pyarrow 或 fastparquet；未安装时该源静默降级到下一数据源。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from .base import HistoryRequest
from .local_file import LocalFileFeed, map_columns
from ...core.constant import Exchange

_logger = logging.getLogger("quantmind.data.astock")

# 交易所枚举 -> 市场后缀
_EXCH_SUFFIX = {Exchange.SSE: "SH", Exchange.SZSE: "SZ"}


class ChinaAStockParquetFeed(LocalFileFeed):
    """A 股（沪深）本地 Parquet/CSV 适配器。

    :param tz_offset_hours: 默认 8（北京时间）。纯日期行不平移；带时刻行按此偏移转 UTC。
    """

    name = "china_astock_parquet"

    def __init__(
        self,
        root_dir: str,
        name: str = "china_astock_parquet",
        tz_offset_hours: int = 8,
    ) -> None:
        super().__init__(root_dir, name=name, tz_offset_hours=tz_offset_hours)

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
        exch = req.exchange
        if exch not in (Exchange.SSE, Exchange.SZSE):
            return [], False

        sym = req.symbol.upper().strip()
        if "." in sym:
            code, suffix = sym.rsplit(".", 1)
        else:
            code = sym
            suffix = _EXCH_SUFFIX.get(exch, "")
        code = code.strip()
        suffix = suffix.upper()

        prefixes = [
            "",
            f"{exch.value}/",
            f"{suffix}/",
            "data/",
            "daily/",
            "stock/",
            "stocks/",
        ]
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
        _logger.warning("本地 A 股源 %s 未找到 %s.%s", self.name, req.symbol, exch.value)
        return [], False
