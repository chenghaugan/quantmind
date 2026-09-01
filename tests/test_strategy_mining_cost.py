"""验证 LLM 策略挖掘自动回测默认接入差异化交易成本。"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from quantmind.strategy_mining.auto_backtest import AutoBacktestLoop
from quantmind.strategy_mining.schema import StrategySpec, RiskParams
from quantmind.paper.promotion import LifecycleManager


class _FakeStrategy:
    """伪造已编译策略：__dict__ 即 run_strategy 的 setting。"""
    def __init__(self):
        self.fast = 5
        self.slow = 20


def _make_spec() -> StrategySpec:
    return StrategySpec(
        name="t",
        symbol="rb0",
        exchange="SHFE",
        template="dual_ma",
        params={"fast": 5, "slow": 20},
        risk=RiskParams(max_position=1.0),
    )


def test_auto_backtest_cost_defaults_on():
    """默认 cost=True：挖掘回测按品种差异化计费，而非零成本/单一费率。"""
    loop = AutoBacktestLoop(lifecycle_manager=LifecycleManager())
    assert loop.cost is True


async def test_auto_backtest_forwards_cost_to_runner():
    """cost 参数应透传到 run_strategy(..., cost=...)。"""
    costs = []

    def _fake_compile(spec):
        return True, None, _FakeStrategy()

    def _fake_run_strategy(**kwargs):
        costs.append(kwargs.get("cost"))
        # 返回一份达标报告，让循环只跑一次即通过闸门
        return {"report": {"sharpe": 1.2, "max_drawdown": -0.05,
                           "annual_return": 0.1, "win_rate": 0.6,
                           "total_cost": 100.0, "cost_ratio": 0.1,
                           "trade_count": 5},
                "equity_curve": [], "trades": 5}

    loop = AutoBacktestLoop(
        lifecycle_manager=LifecycleManager(),
        max_iterations=1, min_sharpe=0.5, max_drawdown=-0.3,
        cost=True, max_cost_ratio=0.6, compare_zero_cost=True,
    )
    with patch("quantmind.strategy_mining.auto_backtest.compile_and_validate", _fake_compile), \
         patch("quantmind.strategy_mining.auto_backtest.run_strategy", _fake_run_strategy):
        result = await loop.run(_make_spec(), bars=[])

    # 第一次净回测传差异化成本表(True)，第二次零成本对照(False)
    assert costs == [True, False]
    assert result.passed is True          # 达标，进生命周期


def test_auto_backtest_cost_can_be_off():
    """cost=False 显式关闭，用于零成本对照。"""
    loop = AutoBacktestLoop(lifecycle_manager=LifecycleManager(), cost=False)
    assert loop.cost is False


async def test_cost_ratio_gate_rejects_high_turnover():
    """即使净 Sharpe/回撤达标，成本占比超上限也拒绝入库（高换手拦截）。"""
    calls = []

    def _fake_compile(spec):
        return True, None, _FakeStrategy()

    def _fake_run_strategy(**kwargs):
        calls.append(kwargs.get("cost"))
        if kwargs.get("cost"):
            # 净回测：Sharpe/回撤达标但成本占比过高
            return {"report": {"sharpe": 1.2, "max_drawdown": -0.1,
                               "annual_return": 0.2, "win_rate": 0.6,
                               "total_cost": 50000.0, "cost_ratio": 0.9,
                               "trade_count": 300},
                    "equity_curve": [], "trades": 300}
        # 零成本对照
        return {"report": {"sharpe": 2.5, "max_drawdown": -0.05,
                           "annual_return": 0.4, "win_rate": 0.6,
                           "total_cost": 0.0, "cost_ratio": 0.0,
                           "trade_count": 300},
                "equity_curve": [], "trades": 300}

    loop = AutoBacktestLoop(
        lifecycle_manager=LifecycleManager(),
        max_iterations=1, min_sharpe=0.5, max_drawdown=-0.3,
        cost=True, max_cost_ratio=0.6, compare_zero_cost=True,
    )
    with patch("quantmind.strategy_mining.auto_backtest.compile_and_validate", _fake_compile), \
         patch("quantmind.strategy_mining.auto_backtest.run_strategy", _fake_run_strategy):
        result = await loop.run(_make_spec(), bars=[])

    assert result.passed is False
    assert result.reject_reason
    assert "高换手" in result.reject_reason
    assert result.cost_ratio == pytest.approx(0.9)
    assert result.gross_sharpe == pytest.approx(2.5)     # 零成本对照
    assert result.cost_drag_sharpe == pytest.approx(2.5 - 1.2)
    assert calls == [True, False]                        # 净 + 零成本 各跑一次


async def test_cost_ratio_gate_accepts_low_turnover():
    """成本占比合理时正常通过；关闭拦截(=0)时不因成本拒绝。"""
    def _fake_compile(spec):
        return True, None, _FakeStrategy()

    def _fake_run_strategy(**kwargs):
        return {"report": {"sharpe": 1.2, "max_drawdown": -0.1,
                           "annual_return": 0.2, "win_rate": 0.6,
                           "total_cost": 10000.0, "cost_ratio": 0.2,
                           "trade_count": 20},
                "equity_curve": [], "trades": 20}

    loop = AutoBacktestLoop(
        lifecycle_manager=LifecycleManager(),
        max_iterations=1, min_sharpe=0.5, max_drawdown=-0.3,
        cost=True, max_cost_ratio=0.6, compare_zero_cost=False,
    )
    with patch("quantmind.strategy_mining.auto_backtest.compile_and_validate", _fake_compile), \
         patch("quantmind.strategy_mining.auto_backtest.run_strategy", _fake_run_strategy):
        result = await loop.run(_make_spec(), bars=[])

    assert result.passed is True
    assert result.reject_reason == ""
    assert result.cost_ratio == pytest.approx(0.2)
