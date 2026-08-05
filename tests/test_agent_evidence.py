"""AI 智能体闭环集成测试（research_with_evidence）。

验证：因子假设由真实面板截面 IC 证据验证（VERIFIED/REJECTED 依据阈值），
每条因子映射出可求值的面板 DSL 表达式，事实表并入真实评估 metrics，
且原 ``research()`` 方法保持向后兼容。
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from quantmind.ai.agent import AutoResearchAgent, HypothesisStatus
from quantmind.ai.expr_map import factor_spec_to_expression
from quantmind.ai.provider import MockProvider
from quantmind.research import panel_eval_expression
from quantmind.research.factors.alpha_cs import Panel


def _run(coro):
    return asyncio.run(coro)


def _utc(i: int):
    return datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=i)


def _make_panel(n_symbols: int = 8, n_dates: int = 150, seed: int = 3) -> Panel:
    rng = np.random.default_rng(seed)
    dates = [_utc(i) for i in range(n_dates)]
    cols = [f"S{i}" for i in range(n_symbols)]
    close = pd.DataFrame(np.abs(rng.normal(100, 10, (n_dates, n_symbols))), index=dates, columns=cols)
    open_ = close * 0.99
    high = close * 1.02
    low = close * 0.98
    volume = pd.DataFrame(np.abs(rng.normal(1000, 100, (n_dates, n_symbols))), index=dates, columns=cols)
    return Panel(close=close, open=open_, high=high, low=low, volume=volume)


def test_factor_spec_to_expression_maps_kinds():
    """kind → DSL 表达式映射可被面板求值器执行。"""
    from quantmind.research.target import FactorSpec
    panel = _make_panel()
    for kind in ("momentum", "mean_reversion", "volatility",
                 "volume_change", "open_interest_change", "term_structure"):
        spec = FactorSpec(name=f"{kind}_20", kind=kind, window=20)
        expr = factor_spec_to_expression(spec)
        assert expr
        out = panel_eval_expression(expr, panel)  # 不抛异常即为合法
        assert not out.empty


def test_factor_spec_expression_priority():
    """已提供 expression 时优先使用。"""
    from quantmind.research.target import FactorSpec
    spec = FactorSpec(name="x", kind="momentum", window=20, expression="rank(close)")
    assert factor_spec_to_expression(spec) == "rank(close)"


def test_research_with_evidence_verifies_by_real_ic():
    """每条因子假设由真实面板 IC 证据判定，且至少一条为 VERIFIED/REJECTED。"""
    panel = _make_panel()
    out = _run(AutoResearchAgent(provider=MockProvider()).research_with_evidence(
        "螺纹钢期货动量与均线", panel, use_cache=False))

    # 因子表项各自携带非空、可求值的表达式
    assert out.factors
    for f in out.factors:
        assert f.expression
        panel_eval_expression(f.expression, panel)  # 可求值

    # 每条因子假设状态由证据判定（不再是模棱两可的离线启发式）
    factor_hs = [h for h in out.hypotheses if h.id.startswith("H") and h.id not in ("H0",)]
    assert factor_hs
    for h in factor_hs:
        assert h.status in (HypothesisStatus.VERIFIED, HypothesisStatus.REJECTED)
        assert h.evidence, "证据不应为空"

    # 至少一条含真实 IC 特征（"ic_mean"）
    assert any("ic_mean" in h.evidence for h in factor_hs)


def test_research_with_evidence_fact_sheet_metrics():
    """事实表并入真实评估 metrics。"""
    panel = _make_panel()
    out = _run(AutoResearchAgent(provider=MockProvider()).research_with_evidence(
        "商品期货期限结构", panel, use_cache=False))
    assert out.fact_sheet.get("metrics"), "事实表 metrics 应非空"
    assert isinstance(out.fact_sheet["metrics"], dict)
    # metrics 中任一因子的指标应含 ic_mean 键
    has_ic = any("ic_mean" in v for v in out.fact_sheet["metrics"].values()
                 if isinstance(v, dict))
    assert has_ic


def test_research_with_evidence_backward_compat():
    """原 research() 仍正常工作（向后兼容）。"""
    out = _run(AutoResearchAgent(provider=MockProvider()).research("测试想法"))
    assert out.factors
    assert out.hypotheses
    assert len(out.log) >= 3
    assert isinstance(out.code_safe, bool)


def test_research_with_evidence_run_search():
    """run_search=True 短面板下不抛异常，并产生 CoT 搜索证据。"""
    panel = _make_panel(n_symbols=6, n_dates=80, seed=7)
    out = _run(AutoResearchAgent(provider=MockProvider()).research_with_evidence(
        "动量因子改进", panel, run_search=True, max_rounds=1, use_cache=False))
    # 必须成功返回（即使 CoT 无有效 IC 也只是标记 REJECTED）
    assert out.spec.idea
    assert out.hypotheses
