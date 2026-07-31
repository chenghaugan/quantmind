"""结构化成本模型测试。"""
from __future__ import annotations

import math
from datetime import datetime

import pytest

from quantmind.backtest.cost import (
    CostModel, default_cost_table, lookup_cost, compute_commission, apply_slippage,
)
from quantmind.backtest import BacktestEngine
from quantmind.core.constant import Direction, Offset, Exchange
from quantmind.core.object import BarData
from quantmind.core.contracts import default_size
from quantmind.strategy.multifactor import MultiFactorStrategy
from tests.helpers import load_bars


# ---- 成本解析 ----
def test_lookup_cost_normalizes_symbol():
    # 连续/具体合约 -> 品种前缀
    assert lookup_cost("rb0.SHFE").commission_rate == 0.0001
    assert lookup_cost("IC2401.CFFEX").close_today_rate_multiplier == 0.0  # 股指平今免
    # 纯数字 -> A 股
    eq = lookup_cost("600519.SSE")
    assert eq.asset_class == "equity"
    assert eq.stamp_tax_rate == 0.001
    # 未知 -> 商品期货通用
    assert lookup_cost("ZZZ.COMEX").asset_class == "future"


def test_close_today_free_for_rb():
    cost = lookup_cost("rb0.SHFE")
    # 平今免收（倍率 0）
    fee_ct, _, _ = compute_commission(cost, 10, 3500, 10, Direction.SHORT, Offset.CLOSE, close_today_volume=10)
    assert fee_ct == 0.0
    # 开仓收费
    fee_open, _, _ = compute_commission(cost, 10, 3500, 10, Direction.LONG, Offset.OPEN, close_today_volume=0)
    assert fee_open == pytest.approx(3500 * 10 * 10 * 0.0001)


def test_min_commission_floor():
    eq = lookup_cost("600519.SSE")
    # 极小成交额：手续费低于 5 元 -> 取最低 5 元
    fee, _, _ = compute_commission(eq, 100, 1.0, 1, Direction.LONG, Offset.OPEN, close_today_volume=0)
    assert fee == pytest.approx(5.0)


def test_stamp_tax_only_on_sell():
    eq = lookup_cost("600519.SSE")
    _, tax_sell, _ = compute_commission(eq, 100, 10.0, 1, Direction.SHORT, Offset.CLOSE, close_today_volume=0)
    _, tax_buy, _ = compute_commission(eq, 100, 10.0, 1, Direction.LONG, Offset.OPEN, close_today_volume=0)
    assert tax_sell == pytest.approx(100 * 10.0 * 1 * 0.001)
    assert tax_buy == 0.0


def test_slippage_ticks_model():
    bar = BarData(symbol="rb0", exchange=Exchange.SHFE, datetime=datetime(2024, 1, 2),
                  open_price=3500.0)
    cost0 = lookup_cost("rb0.SHFE")            # 默认无滑点
    assert apply_slippage(cost0, bar, Direction.LONG) == 3500.0
    cost1 = CostModel(slippage_ticks=1, tick_size=1.0)  # 1 跳滑点
    assert apply_slippage(cost1, bar, Direction.LONG) == 3501.0
    assert apply_slippage(cost1, bar, Direction.SHORT) == 3499.0


def test_margin_for():
    cost = lookup_cost("rb0.SHFE")            # margin_rate 0.1, size 10
    assert cost.margin_for(10, 3500, 10) == pytest.approx(3500 * 10 * 10 * 0.1)


# ---- 引擎接入（直接驱动成交记账）----
def test_engine_close_today_accounting():
    """RB 平今免：同日开平，平仓那笔手续费应为 0。"""
    vt = "rb0.SHFE"
    eng = BacktestEngine({vt: []}, capital=1_000_000, cost_table=True, sizes={vt: 10})
    d = datetime(2024, 1, 2)
    eng._apply_fill(vt, Direction.LONG, 10, 3500.0, Offset.OPEN, d)
    eng._apply_fill(vt, Direction.SHORT, 10, 3500.0, Offset.CLOSE, d)  # 同日平今
    # 仅开仓手续费：3500*10*10*0.0001 = 35
    assert eng.total_commission == pytest.approx(35.0, abs=1e-6)
    assert eng.total_stamp == 0.0


def test_engine_equity_stamp_tax():
    """A 股卖出收印花税，开平均收佣金。"""
    vt = "600519.SSE"
    eng = BacktestEngine({vt: []}, capital=1_000_000, cost_table=True, sizes={vt: 100})
    d = datetime(2024, 1, 2)
    eng._apply_fill(vt, Direction.LONG, 100, 100.0, Offset.OPEN, d)
    eng._apply_fill(vt, Direction.SHORT, 100, 110.0, Offset.CLOSE, d)
    # 佣金: 100*100*100*0.00025=250 ; 110*100*100*0.00025=275 -> 525
    assert eng.total_commission == pytest.approx(250 + 275)
    assert eng.total_stamp == pytest.approx(110 * 100 * 100 * 0.001)
    assert eng.margin_used == pytest.approx(0.0)  # 股票 margin_rate=1 但平仓后净仓 0


def test_engine_legacy_commission_unchanged():
    """未启用成本表时，退回旧式单一费率，行为兼容。"""
    vt = "rb0.SHFE"
    eng = BacktestEngine({vt: []}, capital=1_000_000, commission=0.0002, sizes={vt: 10})
    eng._apply_fill(vt, Direction.LONG, 10, 3500.0, Offset.OPEN, datetime(2024, 1, 2))
    assert eng.total_commission == pytest.approx(3500 * 10 * 10 * 0.0002)


@pytest.mark.asyncio
async def test_cost_table_run_reports_costs():
    """启用成本表跑回测：成本字段存在、为正、且有限（无 NaN）。"""
    bars = await load_bars()
    vt = "rb0.SHFE"
    eng = BacktestEngine({vt: bars}, capital=1_000_000, sizes={vt: default_size(vt)}, cost_table=True)
    eng.add_strategy(MultiFactorStrategy, vt, {"size": default_size(vt), "max_pos": 1.0})
    rep = eng.run()
    assert rep.trade_count > 0
    assert rep.total_commission > 0
    assert rep.total_cost >= rep.total_commission
    assert math.isfinite(rep.total_cost)
    assert math.isfinite(rep.cost_ratio)
    # 报告可序列化
    d = rep.to_dict()
    assert "total_cost" in d and "margin_used" in d
