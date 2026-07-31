"""席位持仓排名 CSV 适配器测试（合成数据，离线可跑）。

模拟 TradingAgents_for_Futures 仓库的
long/short/volume_position_ranking.csv 格式：
    排名,会员简称,持仓量,比上交易增减,date,contract,position_type,symbol
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quantmind.data.feed.seat_position_csv import SeatDataset
from quantmind.research.factors.seat_futures import compute_seat_factors, seat_df_from_tradingagents

HEADER = "排名,会员简称,持仓量,比上交易增减,date,contract,position_type,symbol\n"


def _write_rankings(base: Path) -> None:
    base.mkdir(parents=True, exist_ok=True)
    # 两个交易日、两个合约(rb2509 活跃, rb2510 次活跃)；3 个席位
    # 2025-08-18: rb2509 多单总量 226738+115641+61117 = 403496 (活跃)
    long_csv = (
        HEADER +
        "1,中信期货,226738,-16227.0,2025-08-18,rb2509,多单持仓,RB\n"
        "2,国泰君安,115641,-2452.0,2025-08-18,rb2509,多单持仓,RB\n"
        "3,方正中期,61117,4840.0,2025-08-18,rb2509,多单持仓,RB\n"
        "1,中信期货,200000,1000.0,2025-08-18,rb2510,多单持仓,RB\n"
        "1,中信期货,230000,-1000.0,2025-08-19,rb2509,多单持仓,RB\n"
        "2,国泰君安,118000,2000.0,2025-08-19,rb2509,多单持仓,RB\n"
        "3,方正中期,60000,-1000.0,2025-08-19,rb2509,多单持仓,RB\n"
    )
    # 空头：中信期货 在 rb2509 上 80000，国泰君安 50000，方正 20000
    short_csv = (
        HEADER +
        "1,中信期货,80000,1000.0,2025-08-18,rb2509,空单持仓,RB\n"
        "2,国泰君安,50000,-500.0,2025-08-18,rb2509,空单持仓,RB\n"
        "3,方正中期,20000,300.0,2025-08-18,rb2509,空单持仓,RB\n"
        "1,中信期货,85000,2000.0,2025-08-19,rb2509,空单持仓,RB\n"
        "2,国泰君安,52000,1000.0,2025-08-19,rb2509,空单持仓,RB\n"
        "3,方正中期,21000,500.0,2025-08-19,rb2509,空单持仓,RB\n"
    )
    volume_csv = (
        HEADER +
        "1,中信期货,300000,5000.0,2025-08-18,rb2509,成交量,RB\n"
        "1,中信期货,310000,3000.0,2025-08-19,rb2509,成交量,RB\n"
    )
    (base / "long_position_ranking.csv").write_text(long_csv, encoding="utf-8")
    (base / "short_position_ranking.csv").write_text(short_csv, encoding="utf-8")
    (base / "volume_ranking.csv").write_text(volume_csv, encoding="utf-8")


def test_load_and_net_position(tmp_path: Path) -> None:
    base = tmp_path / "RB"
    _write_rankings(base)
    ds = SeatDataset.load(str(tmp_path), "RB")
    assert ds.symbol == "RB"
    assert "会员简称" in ds.long.columns
    assert "持仓量" in ds.short.columns

    seat_df, total_oi = ds.to_net_position_matrix()
    # 选了 rb2509 作为活跃合约（多单总量最大）
    assert len(seat_df) == 2
    # 2025-08-18 中信期货净持仓 = 226738 - 80000 = 146738
    d0 = pd.Timestamp("2025-08-18")
    assert seat_df.loc[d0, "中信期货"] == pytest.approx(146738.0)
    # 国泰君安 = 115641 - 50000 = 65641
    assert seat_df.loc[d0, "国泰君安"] == pytest.approx(65641.0)
    # total_oi = 活跃合约多单总量 = 403496
    assert total_oi.loc[d0] == pytest.approx(403496.0)


def test_seat_df_from_tradingagents_runs(tmp_path: Path) -> None:
    _write_rankings(tmp_path / "RB")
    seat_df, total_oi = seat_df_from_tradingagents(str(tmp_path), "RB")
    assert not seat_df.empty
    factors = compute_seat_factors(seat_df, total_oi, aggregate=True)
    assert len(factors) == 8
    for name, ser in factors.items():
        assert len(ser) == len(seat_df)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        SeatDataset.load(str(tmp_path), "RB")  # 目录不存在
