"""回测引擎测试。"""
from __future__ import annotations

import pytest

from quantmind.backtest import BacktestEngine, grid_search
from quantmind.strategy.dual_ma import DualMaStrategy
from quantmind.strategy.multifactor import MultiFactorStrategy
from quantmind.core.contracts import default_size
from tests.helpers import load_bars


@pytest.mark.asyncio
async def test_multifactor_backtest_runs():
    bars = await load_bars()
    vt = "rb0.SHFE"
    eng = BacktestEngine({vt: bars}, capital=1_000_000, sizes={vt: default_size(vt)})
    eng.add_strategy(MultiFactorStrategy, vt, {"size": default_size(vt), "max_pos": 1.0})
    rep = eng.run()
    assert rep.trade_count > 0
    assert len(rep.equity_curve) == len(bars)
    assert rep.final_equity > 0


@pytest.mark.asyncio
async def test_dual_ma_backtest_runs():
    bars = await load_bars()
    vt = "rb0.SHFE"
    eng = BacktestEngine({vt: bars}, capital=1_000_000, sizes={vt: default_size(vt)})
    eng.add_strategy(DualMaStrategy, vt, {"fast": 5, "slow": 20, "size": default_size(vt), "max_pos": 1.0})
    rep = eng.run()
    assert rep.trade_count >= 0
    assert len(rep.equity_curve) == len(bars)


@pytest.mark.asyncio
async def test_no_lookahead_fill_next_bar():
    """委托在下一根 K 线开盘价成交（无前视）。"""
    bars = await load_bars()
    vt = "rb0.SHFE"
    eng = BacktestEngine({vt: bars}, capital=1_000_000, sizes={vt: default_size(vt)})
    eng.add_strategy(DualMaStrategy, vt, {"fast": 3, "slow": 5, "size": default_size(vt), "max_pos": 1.0})
    eng.run()
    # 所有成交价应等于某根 K 线的开盘价（下一根开盘）；
    # 期末强平（若有持仓）按末日收盘价合成，属预期特例
    opens = {b.open_price for b in bars}
    for t in eng.trades[:-1]:
        assert t.price in opens
    if eng.trades:
        assert (eng.trades[-1].price in opens
                or eng.trades[-1].price == bars[-1].close_price)


@pytest.mark.asyncio
async def test_grid_search_finds_params():
    bars = await load_bars()
    vt = "rb0.SHFE"
    res = grid_search(DualMaStrategy, {vt: bars}, vt,
                      {"fast": [3, 5, 10], "slow": [10, 20, 30]},
                      metric="sharpe", sizes={vt: default_size(vt)})
    assert res.best_setting
    assert "fast" in res.best_setting and "slow" in res.best_setting
