"""China Futures 5min 仓库（CC0）本地 CSV 适配器。

仓库结构：5min/<交易所>/<品种>/<品种>YYMM.csv
  - 交易所: CFFEX/CZCE/DCE/GFEX/INE/SHFE（与 Exchange 枚举值一致）
  - 品种:   IC/IF/IH/AP/RB/CU ...（大写）
  - 文件:   IC1505.csv（2015 年 5 月 IC 合约）

请求约定：
  - 具体交割合约: "IC1505.CFFEX"（symbol=IC1505）-> 直接读 5min/CFFEX/IC/IC1505.csv
  - 主连/连续:    "IC0.CFFEX" 或 "IC9999.CFFEX"（symbol=IC0/IC9999）
                 拼接该品种所有交割月文件成**连续主力合约**。

连续主力合约构造（continuous_method）：
  - "simple"（旧）: 按交割月月末窗口首尾衔接拼接，换月日价格可能跳变。
  - "back_adjusted"（默认，推荐）: 每个交易日按**持仓量(OIN)最大**的合约作为当日主力，
    并把历史价格做**向后复权**（最新价不变、历史价平移），消除换月跳变，使因子/回测
    吃到的连续序列在换月处平滑、收益连续。这是期货研究的业界标准做法。

LICENSE：CC0 公共领域，可自由使用。数据版权归交易所，仅供研究。
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from .base import HistoryRequest
from .local_file import LocalFileFeed, map_columns
from ...core.constant import Interval

_logger = logging.getLogger("quantmind.data.china_futures")


class ChinaFuturesCSVFeed(LocalFileFeed):
    """读取 china-futures-5min-2015-2025 仓库的 CSV 适配器。

    :param continuous_method: 主连构造方式，"back_adjusted"（默认，推荐）或 "simple"。
    """

    name = "china_futures_csv"

    def __init__(
        self,
        root_dir: str,
        name: str = "china_futures_csv",
        tz_offset_hours: int = 8,
        continuous_method: str = "back_adjusted",
    ) -> None:
        super().__init__(root_dir, name=name, tz_offset_hours=tz_offset_hours)
        if continuous_method not in ("back_adjusted", "simple"):
            raise ValueError(f"不支持的 continuous_method: {continuous_method}")
        self.continuous_method = continuous_method

    # ---- 路径解析（复用基类逻辑，仅增加主连识别） ----
    def _resolve_paths(self, req: HistoryRequest) -> Tuple[List[Path], bool]:
        exch = req.exchange.value.upper()
        symbol_u = req.symbol.upper().strip()
        base = self.root / "5min" / exch

        # 1) 主连识别
        if symbol_u.endswith("9999"):
            product = symbol_u[:-4]
            is_main = True
        elif symbol_u.endswith("0") and len(symbol_u) <= 4:
            product = symbol_u[:-1]
            is_main = True
        else:
            # 2) 具体交割合约：字母 + 4 位 YYMM
            m = re.fullmatch(r"([A-Z]+)(\d{4})", symbol_u)
            if m:
                product, yymm = m.group(1), m.group(2)
                fpath = base / product / f"{product}{yymm}.csv"
                return [fpath], False
            # 3) 其它：当作主连 product
            product = symbol_u
            is_main = True

        prod_dir = base / product
        if not prod_dir.exists():
            _logger.warning("本地期货目录不存在: %s", prod_dir)
            return [], True
        paths = sorted(prod_dir.glob(f"{product}*.csv"))
        if not paths:
            _logger.warning("本地期货品种目录无 CSV: %s", prod_dir)
        return paths, True

    @staticmethod
    def _contract_from_path(p: Path) -> str:
        m = re.search(r"([A-Z]+\d{4})", p.stem)
        return m.group(1) if m else p.stem

    # ---- 主连构造分发 ----
    async def fetch_bar_data(self, req: HistoryRequest) -> List["BarData"]:  # noqa: F821
        paths, is_main = self._resolve_paths(req)
        if not paths:
            _logger.warning("本地源 %s 未找到 %s.%s 的文件", self.name, req.symbol, req.exchange.value)
            return []
        records: List[Tuple[str, pd.DataFrame]] = []
        for p in paths:
            if not p.exists():
                continue
            try:
                raw = self._read_file(p)
            except Exception as exc:  # noqa: BLE001
                _logger.warning("读取 %s 失败: %s", p, exc)
                continue
            raw = map_columns(raw)
            raw = self._normalize(raw)
            records.append((self._contract_from_path(p), raw))
        if not records:
            return []
        if is_main and len(records) > 1:
            if self.continuous_method == "simple":
                pieces = [(self._expiry_end_day(Path(c)), df) for c, df in records]
                combined = super()._build_continuous(pieces)
            else:
                combined = self._build_continuous_backadjusted(records)
        else:
            combined = pd.concat([r[1] for r in records]).sort_values("datetime").reset_index(drop=True)
        # 频率：5m 原样返回（供分钟级研究），其他降采样到日频
        if req.interval != Interval.MINUTE_5:
            combined = self._resample_daily(combined)
        if req.start is not None:
            _s = pd.Timestamp(req.start)
            if _s.tzinfo is not None:
                _s = _s.tz_localize(None)  # 本表为 naive UTC，aware 比较会抛 TypeError
            combined = combined[combined["datetime"] >= _s]
        if req.end is not None:
            _e = pd.Timestamp(req.end)
            if _e.tzinfo is not None:
                _e = _e.tz_localize(None)
            combined = combined[combined["datetime"] <= _e]
        combined = combined.dropna(subset=["close"])
        if combined.empty:
            return []
        return self._to_bars(combined, req)

    # ---- 向后复权主力连续 ----
    def _build_continuous_backadjusted(self, records: List[Tuple[str, pd.DataFrame]]) -> pd.DataFrame:
        """按 OI 选主力 + 向后复权，产出平滑连续的日/分钟主力序列。

        步骤：
          1) 所有合约展平为长表（带 contract 标签）。
          2) 每个时间截面按持仓量最大选主力合约；OI 缺失时退化为按成交量。
          3) 取各时刻主力合约的 OHLC 作为原始连续序列。
          4) 向后复权：从最新向历史递推，换月处把历史整体平移以抵消跳变，
             最新价保持真实不变、历史收益连续。
        """
        frames = []
        for contract, df in records:
            d = df.copy()
            d["contract"] = contract
            frames.append(d)
        long = pd.concat(frames, ignore_index=True).sort_values("datetime")

        # 2) 主力选择（OI 优先，缺失退化到 volume）
        oi = long.pivot_table(index="datetime", columns="contract",
                              values="open_interest", aggfunc="last")
        if oi.notna().any().any():
            wide = oi
        else:
            wide = long.pivot_table(index="datetime", columns="contract",
                                    values="volume", aggfunc="last")
        main = wide.idxmax(axis=1)
        main = main.ffill().bfill()

        # 3) 原始连续序列：每个时刻取主力合约那一行
        main_df = pd.DataFrame({"datetime": wide.index, "main": main.values})
        merged = long.merge(main_df, on="datetime", how="inner")
        raw = (merged[merged["contract"] == merged["main"]]
               .sort_values("datetime")
               .drop_duplicates("datetime")
               .reset_index(drop=True))
        if raw.empty:
            return long.iloc[0:0]

        # 4) 向后复权（最新价不变，历史平移消除换月跳变）
        closes = raw["close"].to_numpy(dtype=float)
        mains = raw["main"].to_numpy()
        # 同时刻全合约收盘价：取换月时新旧主力的**同timestamp价差**（纯基差）
        close_wide = long.pivot_table(index="datetime", columns="contract",
                                      values="close", aggfunc="last")
        n = len(raw)
        adj = pd.Series(0.0, index=range(n))
        for t in range(n - 2, -1, -1):
            if mains[t] != mains[t + 1]:
                # 换月跳变 = 新旧主力的同刻价差（纯基差）。优先用 t+1 时刻旧主力报价；
                # 不重叠换月（旧主力在 t+1 无报价）时退回旧口径 closes[t+1]-closes[t]。
                # 不能无脑用 closes[t+1]-closes[t]：那把 t→t+1 的真实行情涨跌也一并“复权”掉，
                # 换月 bar 的收益被强制清零。
                jump = closes[t + 1] - closes[t]
                if mains[t] in close_wide.columns:
                    prev_close = close_wide.at[wide.index[t + 1], mains[t]]
                    if pd.notna(prev_close):
                        jump = closes[t + 1] - float(prev_close)
                adj[t] = adj[t + 1] - jump
            else:
                adj[t] = adj[t + 1]
        out = raw.copy()
        out["close"] = out["close"] - adj
        out["open"] = out["open"] - adj
        out["high"] = out["high"] - adj
        out["low"] = out["low"] - adj
        # volume/open_interest 取自当日主力合约（真实水平，不平移）
        return out.reset_index(drop=True)
