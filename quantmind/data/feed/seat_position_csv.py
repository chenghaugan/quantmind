"""期货席位持仓排名 CSV 适配器（对接 TradingAgents_for_Futures 仓库格式）。

数据布局（MIT 仓库，数据源自交易所每日持仓排名 / akshare，仅供学习研究）：
    <root>/<SYMBOL>/long_position_ranking.csv
    <root>/<SYMBOL>/short_position_ranking.csv
    <root>/<SYMBOL>/volume_ranking.csv

各文件表头（经核实）：
    排名,会员简称,持仓量,比上交易增减,date,contract,position_type,symbol

示例行：
    1,中信期货,226738,-16227.0,2025-08-18,rb2510,多单持仓,RB

本模块把「每日每席位 多/空/成交量排名」转换为 F1–F8 所需的**净持仓矩阵**
``seat_df``（行=交易日，列=会员简称，值=净持仓=多单-空单）与 ``total_oi``
（活跃合约当日多单总量，用作占比类因子分母）。

⚠️ 注意：该仓库自带 ``positioning_provider.py`` 读取的是 ``positioning_data.csv``，
但仓库实际文件名为 ``long/short/volume_ranking.csv``（代码已过期）。本适配器直接按
真实文件名解析，不依赖其 reader。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


def read_ranking_csv(path: Path) -> pd.DataFrame:
    """读取单个排名 CSV（long/short/volume 通用），并解析 date 列。"""
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


@dataclass
class SeatDataset:
    """一个品种的全部席位排名数据。"""

    symbol: str
    long: pd.DataFrame
    short: pd.DataFrame
    volume: pd.DataFrame

    @classmethod
    def load(cls, root: str, symbol: str) -> "SeatDataset":
        base = Path(root) / symbol.upper()
        files = {
            "long": base / "long_position_ranking.csv",
            "short": base / "short_position_ranking.csv",
            "volume": base / "volume_ranking.csv",
        }
        missing = [str(p) for p in files.values() if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"席位数据缺失（需 {base}/ 下 long/short/volume_position_ranking.csv）："
                + "; ".join(missing)
            )
        return cls(
            symbol=symbol.upper(),
            long=read_ranking_csv(files["long"]),
            short=read_ranking_csv(files["short"]),
            volume=read_ranking_csv(files["volume"]),
        )

    def to_net_position_matrix(self) -> Tuple[pd.DataFrame, pd.Series]:
        """构造 ``seat_df``（净持仓矩阵）与 ``total_oi``（活跃合约多单总量）。

        每个交易日选取**最活跃合约**（当日多单持仓量总和最大的合约），在该合约内
        按会员简称合并多头/空头持仓，净持仓 = 多单 - 空单。
        """
        lday_all = self.long
        sday_all = self.short
        dates = sorted(
            set(lday_all["date"].dropna().unique())
            | set(sday_all["date"].dropna().unique())
        )
        net_rows: dict = {}
        oi_rows: dict = {}
        for d in dates:
            lday = lday_all[lday_all["date"] == d]
            sday = sday_all[sday_all["date"] == d]
            if lday.empty:
                continue
            # 最活跃合约：当日多单持仓量总和最大
            tot = lday.groupby("contract")["持仓量"].sum()
            contract = tot.idxmax()
            lsub = lday[lday["contract"] == contract][["会员简称", "持仓量"]].rename(
                columns={"持仓量": "long_pos"}
            )
            if sday.empty:
                ssub = pd.DataFrame(columns=["会员简称", "short_pos"])
            else:
                ssub = sday[sday["contract"] == contract][["会员简称", "持仓量"]].rename(
                    columns={"持仓量": "short_pos"}
                )
            merged = lsub.merge(ssub, on="会员简称", how="outer").fillna(0)
            merged["net"] = merged["long_pos"] - merged["short_pos"]
            net_rows[d] = merged.set_index("会员简称")["net"]
            oi_rows[d] = float(tot.max())

        seat_df = pd.DataFrame(net_rows).T.sort_index()
        total_oi = pd.Series(oi_rows).reindex(seat_df.index).fillna(0.0)
        return seat_df, total_oi
