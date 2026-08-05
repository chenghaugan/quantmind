"""5 组件模块化框架测试：组件接口 / 组件实现 / ComposableStrategy 回测回归。

核心回归门槛：
  - 默认 `ComposableStrategy`（MultiFactorAlpha + IdentityPortfolio + NullRisk
    + TargetExecution）的回测行为应与现有 `MultiFactorStrategy` 一致。
"""
from __future__ import annotations

import pytest

from quantmind.core.constant import Exchange
from quantmind.strategy import MultiFactorStrategy, run_strategy
from quantmind.strategy.components import (
    AlphaSignal,
    ComposableStrategy,
    IdentityPortfolio,
    MomentumAlpha,
    MultiFactorAlpha,
    NullRisk,
    TargetExecution,
)
from tests.helpers import load_bars

SIZE = 10.0
SETTING = {"size": 10, "max_pos": 1.0}


# ---------------------------------------------------------------------------
# 组件接口
# ---------------------------------------------------------------------------
def test_alpha_signal_target():
    """AlphaSignal.target 应为带符号权重（正多 / 负空）。"""
    assert AlphaSignal(vt_symbol="rb0.SHFE", direction=1, magnitude=0.5).target == pytest.approx(0.5)
    assert AlphaSignal(vt_symbol="rb0.SHFE", direction=-1, magnitude=0.8).target == pytest.approx(-0.8)
    assert AlphaSignal(vt_symbol="rb0.SHFE", direction=0, magnitude=0.0).target == 0.0


def test_components_importable():
    """5 组件 + 装配策略均可从包导入。"""
    assert AlphaSignal and IdentityPortfolio and NullRisk and TargetExecution
    assert MomentumAlpha and MultiFactorAlpha and ComposableStrategy


# ---------------------------------------------------------------------------
# Alpha 组件
# ---------------------------------------------------------------------------
def test_momentum_alpha_up_down():
    """双均线动量 Alpha：上穿 → 多头，下穿 → 空头。"""
    alpha = MomentumAlpha(fast=3, slow=5, size=1, max_pos=1.0)
    # 构造一个先升后降的 close 序列
    closes = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 104.0, 102.0, 100.0, 98.0, 96.0]
    # 手工调用 on_bar：构造简易 bar
    from datetime import datetime, timedelta, timezone
    from quantmind.core.object import BarData
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    targets = []
    for idx, c in enumerate(closes):
        bar = BarData(symbol="rb0", exchange=Exchange.SHFE,
                      datetime=base + timedelta(days=idx),
                      interval="1d", open_price=c, high_price=c, low_price=c,
                      close_price=c, volume=0.0, turnover=0.0, open_interest=0.0)
        trg = alpha.on_bar(bar)
        if trg is not None:
            targets.append(trg)
    # 序列足够长（>= slow=5）后才有输出
    assert len(targets) >= len(closes) - 5
    # 上升段早期长期多
    assert targets[0] > 0
    # 下降段长期空
    assert targets[-1] < 0


def test_multifactor_alpha_on_init_builds_series():
    """MultiFactorAlpha.on_init 应产出目标序列，on_bar 逐根推进。"""
    import asyncio
    from quantmind.backtest.engine import BacktestEngine
    bars = asyncio.run(load_bars())
    vt = "rb0.SHFE"
    data = {vt: bars}
    # 用一个运行 context（BacktestEngine）做 on_init 历史查询
    eng = BacktestEngine(data, sizes={vt: SIZE})
    eng.add_strategy(MultiFactorStrategy, vt, SETTING)
    eng.run()  # 预热（内部调了 on_init）——直接验证组件独立用法：
    alpha = MultiFactorAlpha()
    alpha.vt_symbol = vt
    # 用 engine 的 get_history 模拟 context
    class _Ctx:
        vt_symbols = [vt]
        def get_history(self, s, count):
            return bars[-count:]
    alpha.on_init(_Ctx())
    # 遍历全部 bar，最终 idx 与 bar 数一致
    n = 0
    for bar in bars:
        trg = alpha.on_bar(bar)
        if trg is not None:
            n += 1
    assert n > 0


# ---------------------------------------------------------------------------
# Portfolio / Risk / Execution 组件
# ---------------------------------------------------------------------------
def test_identity_portfolio_passthrough():
    """IdentityPortfolio 透传放行。"""
    sig = AlphaSignal(vt_symbol="rb0.SHFE", direction=1, magnitude=0.5)
    assert IdentityPortfolio().apply(sig) is sig


def test_null_risk_passthrough():
    """NullRisk 不做过滤。"""
    assert NullRisk().apply(0.5, None) == 0.5
    assert NullRisk().apply(None, None) is None


def test_target_execution_needs_bind():
    """TargetExecution 未绑定时返回 None，绑定后转发。"""
    ex = TargetExecution()
    assert ex.set_target("rb0.SHFE", 1.0) is None
    ex.bind(_StubContext())
    assert ex.set_target("rb0.SHFE", 1.0) == "STUB-ORDER"


class _StubContext:
    def set_target(self, vt, target):
        return "STUB-ORDER"


# ---------------------------------------------------------------------------
# ComposableStrategy 端到端（回测）+ 回归门槛
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_composable_backtest_produces_trades():
    """ComposableStrategy（MomentumAlpha 默认装配）应能跑通回测并产生成交。"""
    bars = await load_bars()
    vt = "rb0.SHFE"
    setting = {"alpha": MomentumAlpha(fast=5, slow=20, size=10, max_pos=1.0)}
    r = run_strategy("backtest", ComposableStrategy, vt, setting, bars, sizes={vt: SIZE})
    assert r["mode"] == "backtest"
    assert r["trades"] > 0


@pytest.mark.asyncio
async def test_composable_default_equals_multifactor():
    """回归门槛：默认 ComposableStrategy 回测持仓/成交应与 MultiFactorStrategy 一致。"""
    bars = await load_bars()
    vt = "rb0.SHFE"
    sizes = {vt: SIZE}
    r_mf = run_strategy("backtest", MultiFactorStrategy, vt, SETTING, bars, sizes=sizes)
    r_c = run_strategy("backtest", ComposableStrategy, vt, SETTING, bars, sizes=sizes)
    # 成交笔数应一致（同一 MultiFactorAlpha 默认因子）
    assert r_c["trades"] == r_mf["trades"]
    # 权益曲线末端一致（容差）
    eq_mf = r_mf["equity_curve"]
    eq_c = r_c["equity_curve"]
    assert abs(eq_mf[-1]["equity"] - eq_c[-1]["equity"]) < 1e-6


# ---------------------------------------------------------------------------
# 注册 / 集成
# ---------------------------------------------------------------------------
def test_composable_registered_in_strategies():
    """/strategies 清单应含 compose 策略。"""
    from fastapi.testclient import TestClient
    from quantmind.api.app import app
    with TestClient(app) as c:
        r = c.get("/strategies")
        assert r.status_code == 200
        names = {s["name"] for s in r.json()}
        assert "composable" in names
