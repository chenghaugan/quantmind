"""Tier2 测试：期货席位因子 F1–F8。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantmind.research import (
    make_synthetic_seat_df, compute_seat_factors, SeatFactor,
)


def test_synthetic_seat_df_shape():
    df = make_synthetic_seat_df(n_days=120, n_seats=6, seed=1)
    assert df.shape == (120, 6)


def test_compute_seat_factors_aggregate():
    df = make_synthetic_seat_df(n_days=120, n_seats=6, seed=2)
    res = compute_seat_factors(df, aggregate=True)
    assert set(res.keys()) == {
        "F1_net_position", "F2_net_change", "F3_net_ratio", "F4_concentration",
        "F5_net_change_rate", "F6_net_accel", "F7_net_zscore", "F8_seat_sentiment",
    }
    for k, v in res.items():
        assert isinstance(v, pd.Series)
        assert len(v) == 120
        assert np.isfinite(v).all()


def test_seat_factor_requires_data():
    """未提供席位数据时 compute 应给出明确错误（而非静默 NaN）。"""
    from quantmind.core.object import BarData
    bars = [BarData(symbol="rb0", exchange=__import__("quantmind.core.constant", fromlist=["Exchange"]).Exchange.SHFE,
                    datetime=pd.Timestamp("2024-01-01"), interval=__import__("quantmind.core.constant", fromlist=["Interval"]).Interval.DAILY,
                    open_price=1, high_price=1, low_price=1, close_price=1, volume=1)]
    f = SeatFactor("F7_net_zscore")
    try:
        f.compute(bars)
        raised = False
    except RuntimeError:
        raised = True
    assert raised


def test_seat_factor_with_data():
    from quantmind.core.object import BarData
    from quantmind.core.constant import Exchange, Interval
    df = make_synthetic_seat_df(n_days=100, n_seats=5, seed=3)
    f = SeatFactor("F8_seat_sentiment").set_data(df)
    bars = [BarData(symbol="rb0", exchange=Exchange.SHFE, datetime=pd.Timestamp("2024-01-01"),
                    interval=Interval.DAILY, open_price=1, high_price=1, low_price=1,
                    close_price=1, volume=1) for _ in range(100)]
    s = f.compute(bars)
    assert len(s) == 100
    assert np.isfinite(s).all()
