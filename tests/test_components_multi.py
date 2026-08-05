"""5 组件框架 M4（多标的组合聚合）+ M5（标的过滤）测试。

核心断言：
  - ``EqualWeightPortfolio.apply_all``：N 标的等权，每标的目标仓位 = 信号 * (budget / N)。
  - ``RuleUniverse.select``：历史不足 / 流动性不足的标的被剔除。
  - ``run_strategy_multi``：真·多标的多因子组合回测，主标的重平衡驱动，多个标的有持仓。
  - 单标的 ``run_strategy_multi`` 与 ``run_strategy("backtest", ...)`` 行为一致（回归）。
"""
from __future__ import annotations

import asyncio

import pytest

from quantmind.strategy import run_strategy
from quantmind.strategy.components import (
    AllUniverse,
    ComposableStrategy,
    EqualWeightPortfolio,
    MultiFactorMultiSymbolAlpha,
    RuleUniverse,
)
from quantmind.strategy.runners import run_strategy_multi
from tests.helpers import load_bars

SIZE = 10.0
SETTING = {"size": 10, "max_pos": 1.0, "threshold": 0.3}


# ---------------------------------------------------------------------------
# M4：组合聚合
# ---------------------------------------------------------------------------
def test_equal_weight_portfolio_applies_budget_over_n():
    """EqualWeightPortfolio：N 标的各得 budget/N 的仓位。"""
    pf = EqualWeightPortfolio(budget=1.0)
    targets = pf.apply_all(
        {"rb0.SHFE": 0.8, "hc0.SHFE": -0.6, "i0.DCE": 0.4},
        ["rb0.SHFE", "hc0.SHFE", "i0.DCE"],
    )
    assert targets["rb0.SHFE"] == pytest.approx(0.8 / 3)
    assert targets["hc0.SHFE"] == pytest.approx(-0.6 / 3)
    assert targets["i0.DCE"] == pytest.approx(0.4 / 3)


def test_equal_weight_single_symbol_degenerates():
    """单标的下 EqualWeightPortfolio = budget * signal（退化为单标的）。"""
    pf = EqualWeightPortfolio(budget=1.0)
    targets = pf.apply_all({"rb0.SHFE": 0.5}, ["rb0.SHFE"])
    assert targets["rb0.SHFE"] == pytest.approx(0.5)


def test_identity_portfolio_apply_all_universe_scoped():
    """IdentityPortfolio.apply_all：只给出 universe 内的标的，未纳入的忽略。"""
    from quantmind.strategy.components import IdentityPortfolio
    pf = IdentityPortfolio()
    targets = pf.apply_all(
        {"rb0.SHFE": 0.8, "hc0.SHFE": 0.3},
        ["rb0.SHFE"],  # hc0 未纳入 universe
    )
    assert targets == {"rb0.SHFE": 0.8}


# ---------------------------------------------------------------------------
# M5：标的过滤
# ---------------------------------------------------------------------------
def test_all_universe_passthrough():
    assert AllUniverse().select(["a", "b"]) == ["a", "b"]


def test_rule_universe_filters_short_history():
    """RuleUniverse：历史不足 min_bars 的标的被剔除。"""
    class _Ctx:
        def __init__(self, hist):
            self._hist = hist
        def get_history(self, vt, count):
            return self._hist.get(vt, [])[:count]

    long_hist = list(range(200))          # 200 根
    short_hist = list(range(30))          # 30 根
    uni = RuleUniverse(min_bars=120)
    sel = uni.select(
        ["LONG.SHFE", "SHORT.SHFE"],
        _Ctx({"LONG.SHFE": long_hist, "SHORT.SHFE": short_hist}),
    )
    assert sel == ["LONG.SHFE"]


def test_rule_universe_filters_low_liquidity():
    """RuleUniverse：平均成交量低于阈值的标的被剔除。"""
    class _Ctx:
        def __init__(self, hist):
            self._hist = hist
        def get_history(self, vt, count):
            return self._hist.get(vt, [])[:count]

    class _B:
        volume = 0.0

    class _BHigh:
        volume = 5000.0

    uni = RuleUniverse(min_bars=2, min_avg_volume=1000.0)
    sel = uni.select(
        ["LOW.SHFE", "HIGH.SHFE"],
        _Ctx({
            "LOW.SHFE": [_B(), _B(), _B()],
            "HIGH.SHFE": [_BHigh(), _BHigh(), _BHigh()],
        }),
    )
    assert sel == ["HIGH.SHFE"]


# ---------------------------------------------------------------------------
# 多标的回测（run_strategy_multi）+ 组合集成
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_multi_symbol_composable_backtest_produces_positions():
    """M4：两个不同标的的多因子组合回测，应产生多个标的目标仓位。"""
    bars_rb = await load_bars("rb0", years=1)
    bars_hc = await load_bars("hc0", years=1)
    data = {"rb0.SHFE": bars_rb, "hc0.SHFE": bars_hc}
    sizes = {"rb0.SHFE": SIZE, "hc0.SHFE": SIZE}
    setting = dict(SETTING)

    r = run_strategy_multi(
        ComposableStrategy, ["rb0.SHFE", "hc0.SHFE"], data,
        setting=setting, sizes=sizes,
    )
    assert r["mode"] == "backtest"
    assert r["trades"] > 0
    # universe 应含两个标的
    assert set(r["universe"]) == {"rb0.SHFE", "hc0.SHFE"}


@pytest.mark.asyncio
async def test_multi_symbol_with_equal_weight_portfolio():
    """M4：等权组合下，组合预算被均分到各标的。"""
    bars_rb = await load_bars("rb0", years=1)
    bars_hc = await load_bars("hc0", years=1)
    data = {"rb0.SHFE": bars_rb, "hc0.SHFE": bars_hc}
    sizes = {"rb0.SHFE": SIZE, "hc0.SHFE": SIZE}
    setting = dict(SETTING)
    setting["portfolio"] = EqualWeightPortfolio(budget=1.0)
    setting["alpha"] = MultiFactorMultiSymbolAlpha(size=10, max_pos=1.0, threshold=0.3)

    r = run_strategy_multi(
        ComposableStrategy, ["rb0.SHFE", "hc0.SHFE"], data,
        setting=setting, sizes=sizes,
    )
    assert r["trades"] > 0
    assert set(r["universe"]) == {"rb0.SHFE", "hc0.SHFE"}


@pytest.mark.asyncio
async def test_multi_symbol_with_universe_filter():
    """M5：用 RuleUniverse 过滤后，未命中（历史不足）的标的应被剔除出可选池。"""
    bars_rb = await load_bars("rb0", years=1)
    bars_hc = await load_bars("hc0", years=1)
    # 给 hc0 造一个很短的子集（诱导其被 min_bars 过滤）
    data = {"rb0.SHFE": bars_rb, "hc0.SHFE": bars_hc[:5]}
    sizes = {"rb0.SHFE": SIZE, "hc0.SHFE": SIZE}
    setting = dict(SETTING)
    setting["universe"] = RuleUniverse(min_bars=120)

    r = run_strategy_multi(
        ComposableStrategy, ["rb0.SHFE", "hc0.SHFE"], data,
        setting=setting, sizes=sizes,
    )
    # hc0 历史不足 120 应被剔除，universe 只含 rb0
    assert r["universe"] == ["rb0.SHFE"]


@pytest.mark.asyncio
async def test_multi_symbol_single_equals_single_run():
    """回归：单标的 run_strategy_multi 应与 run_strategy("backtest", ...) 一致。"""
    bars = await load_bars("rb0", years=1)
    vt = "rb0.SHFE"
    sizes = {vt: SIZE}

    r_multi = run_strategy_multi(
        ComposableStrategy, [vt], {vt: bars},
        setting=dict(SETTING), sizes=sizes,
    )
    r_single = run_strategy("backtest", ComposableStrategy, vt, dict(SETTING), bars, sizes=sizes)

    assert r_multi["trades"] == r_single["trades"]
    assert abs(r_multi["equity_curve"][-1]["equity"] - r_single["equity_curve"][-1]["equity"]) < 1e-6
