"""因子引擎测试。"""
from __future__ import annotations

import pytest

from quantmind.research import (
    MomentumFactor, MeanReversionFactor, FactorEvaluator,
    build_factor_registry, eval_factor_expression, FactorSpec, build_model_from_specs,
)
from tests.helpers import load_bars


@pytest.mark.asyncio
async def test_momentum_compute_aligned():
    bars = await load_bars()
    f = MomentumFactor(20)
    s = f.compute(bars)
    assert len(s) == len(bars)
    # 前 20 根应为 0（NaN 填充为 0）
    assert s.iloc[:19].sum() == 0


@pytest.mark.asyncio
async def test_evaluator_runs():
    bars = await load_bars()
    f = MomentumFactor(20)
    s = f.compute(bars); s.name = f.meta.name
    rep = FactorEvaluator().evaluate(s, bars)
    assert rep.n_samples == len(bars)
    # IC 应为有限数值（随机数据下接近 0，但必须可计算）
    assert rep.ic_mean == rep.ic_mean  # 非 NaN 检查（nan != nan）
    assert len(rep.ic_decay) == 5


@pytest.mark.asyncio
async def test_expression_dsl_safe():
    bars = await load_bars()
    from quantmind.research.factors.base import bars_to_df
    s = eval_factor_expression("(close/ref(close,20)-1)", bars_to_df(bars))
    assert len(s) == len(bars)


def test_expression_rejects_dangerous():
    from quantmind.research.factors.expression import eval_factor_expression, ExpressionError
    from quantmind.research.factors.base import bars_to_df
    import pandas as pd
    df = pd.DataFrame({"close": [1, 2, 3], "open": [1, 2, 3], "high": [1, 2, 3],
                       "low": [1, 2, 3], "volume": [1, 1, 1], "open_interest": [0, 0, 0]})
    with pytest.raises(ExpressionError):
        eval_factor_expression("__import__('os')", df)


def test_registry_lists_builtins():
    reg = build_factor_registry()
    names = [f["name"] for f in reg.list_factors()]
    assert "momentum_20" in names


@pytest.mark.asyncio
async def test_multifactor_target_nonempty():
    bars = await load_bars()
    specs = [FactorSpec(name="mom", kind="momentum", window=20, weight=1.0),
             FactorSpec(name="rev", kind="mean_reversion", window=60, weight=-0.5)]
    model = build_model_from_specs(specs, bars)
    target = model.target_position(bars, size=10, max_pos=1.0)
    assert len(target) == len(bars)
    assert int((target != 0).sum()) > 0
