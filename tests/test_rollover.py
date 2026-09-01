"""主力换月（rollover）成本估算测试（P3）。"""
from __future__ import annotations

import pytest

from quantmind.backtest.rollover import (
    round_trip_cost_bps, estimate_rollover_drag, summarize_rollovers,
)
from quantmind.backtest.cost import lookup_cost


def test_round_trip_cost_bps_positive_and_reasonable():
    """rb0（螺纹钢，平今免）单次往返 bps 应为正且在一个合理量级。"""
    cost = lookup_cost("rb0.SHFE")
    bps = round_trip_cost_bps(cost)
    assert bps > 0
    # 万1费率 → 开+平≈2个万分 → 2 bps上下（含最低手续费/滑点修正，给宽松区间）
    assert 0 < bps < 200


def test_round_trip_free_if_all_rates_zero():
    """全零费率+无滑点 → 单次往返成本为 0。"""
    from quantmind.backtest.cost import CostModel
    cost = CostModel(commission_rate=0.0, commission_per_lot=0.0,
                     min_commission=0.0, slippage_ticks=0.0, slippage_rate=0.0)
    assert round_trip_cost_bps(cost) == 0.0


def test_estimate_rollover_drag_scales_with_frequency():
    """年度拖累 = 单次 bps × 每年换月次数。"""
    cost = lookup_cost("IF.CFFEX")
    bps1, amt1 = estimate_rollover_drag(cost, notional_per_position=1_000_000, rollovers_per_year=4)
    bps2, amt2 = estimate_rollover_drag(cost, notional_per_position=1_000_000, rollovers_per_year=8)
    assert bps2 == pytest.approx(bps1 * 2)
    assert amt2 == pytest.approx(amt1 * 2)


def test_estimate_rollover_amount_matches_bps():
    """金额 = bps/1e4 × 名义金额（量纲校验）。"""
    cost = lookup_cost("rb0.SHFE")
    bps, amount = estimate_rollover_drag(cost, notional_per_position=500_000, rollovers_per_year=4)
    assert amount == pytest.approx(bps / 1e4 * 500_000)


def test_summarize_rollovers_shape():
    """汇总结构含关键字段。"""
    info = summarize_rollovers("rb0.SHFE", notional_per_position=100_000, rollovers_per_year=4)
    assert info["vt_symbol"] == "rb0.SHFE"
    for k in ("round_trip_cost_bps", "annual_rollover_drag_bps",
              "annual_rollover_drag_amount", "asset_class"):
        assert k in info
    assert info["asset_class"] == "future"
