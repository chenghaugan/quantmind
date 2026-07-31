"""China Futures 5min 仓库（CC0）本地 CSV 适配器。

仓库结构：5min/<交易所>/<品种>/<品种>YYMM.csv
  - 交易所: CFFEX/CZCE/DCE/GFEX/INE/SHFE（与 Exchange 枚举值一致）
  - 品种:   IC/IF/IH/AP/RB/CU ...（大写）
  - 文件:   IC1505.csv（2015 年 5 月 IC 合约）

请求约定：
  - 具体交割合约: "IC1505.CFFEX"（symbol=IC1505）-> 直接读 5min/CFFEX/IC/IC1505.csv
  - 主连/连续:    "IC0.CFFEX" 或 "IC9999.CFFEX"（symbol=IC0/IC9999）
                 自动拼接该品种所有交割月文件成简单主力连续（按交割月窗口衔接）。

LICENSE：CC0 公共领域，可自由使用。数据版权归交易所，仅供研究。
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Tuple

from .base import HistoryRequest
from .local_file import LocalFileFeed

_logger = logging.getLogger("quantmind.data.china_futures")


class ChinaFuturesCSVFeed(LocalFileFeed):
    """读取 china-futures-5min-2015-2025 仓库的 CSV 适配器。"""

    name = "china_futures_csv"

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
