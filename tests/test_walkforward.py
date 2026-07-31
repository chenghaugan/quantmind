"""Walk-forward 滚动验证测试。"""
from __future__ import annotations

import pytest

from quantmind.backtest import walk_forward
from quantmind.strategy.dual_ma import DualMaStrategy
from quantmind.core.contracts import default_size
from tests.helpers import load_bars


@pytest.mark.asyncio
async def test_walk_forward_produces_folds():
    bars = await load_bars()
    vt = "rb0.SHFE"
    res = walk_forward(bars, DualMaStrategy, {"fast": 5, "slow": 20, "size": default_size(vt), "max_pos": 1.0},
                       vt, train_window=60, test_window=30, step=30, sizes={vt: default_size(vt)})
    assert res.aggregate["n_folds"] >= 3
    assert len(res.folds) == res.aggregate["n_folds"]
    # 每折绩效字段齐全
    for f in res.folds:
        d = f.to_dict()
        assert "sharpe" in d and "total_return" in d
        assert d["fold"] >= 0
    assert isinstance(res.overfit_suspected, bool)
    assert "train_sharpe" in res.detail


@pytest.mark.asyncio
async def test_walk_forward_short_sample_raises():
    bars = await load_bars()
    vt = "rb0.SHFE"
    with pytest.raises(ValueError):
        walk_forward(bars, DualMaStrategy, {"fast": 5, "slow": 20, "size": default_size(vt), "max_pos": 1.0},
                     vt, train_window=300, test_window=200, sizes={vt: default_size(vt)})


@pytest.mark.asyncio
async def test_walk_forward_with_cost_model():
    bars = await load_bars()
    vt = "rb0.SHFE"
    from quantmind.backtest import default_cost_table
    res = walk_forward(bars, DualMaStrategy, {"fast": 5, "slow": 20, "size": default_size(vt), "max_pos": 1.0},
                       vt, train_window=60, test_window=30, step=30, sizes={vt: default_size(vt)},
                       cost=default_cost_table())
    assert res.aggregate["n_folds"] >= 3
