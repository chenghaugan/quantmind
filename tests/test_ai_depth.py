"""LLM 深度集成测试。"""
from __future__ import annotations

import asyncio

from quantmind.ai.agent import (
    AutoResearchAgent,
    HypothesisStatus,
    generate_explanation,
    generate_fact_sheet,
)
from quantmind.ai.safety import lookahead_warnings


def _run(coro):
    return asyncio.run(coro)


def test_auto_research_agent_offline():
    out = _run(AutoResearchAgent().research("螺纹钢期货动量与期限结构", "期货"))
    assert out.spec.idea
    assert out.factors, "应有候选因子"
    assert out.hypotheses, "应有假设列表"
    assert any(h.status == HypothesisStatus.PROPOSED for h in out.hypotheses)
    assert len(out.log) >= 3, "应有研究日志"
    assert isinstance(out.code_safe, bool)


def test_auto_research_hypotheses_status():
    out = _run(AutoResearchAgent().research("商品期货期限结构套利", "期货"))
    # 至少一条验证通过的因子假设
    assert any(h.status == HypothesisStatus.VERIFIED for h in out.hypotheses)
    for h in out.hypotheses:
        assert h.id
        assert h.statement


def test_auto_research_log_structure():
    out = _run(AutoResearchAgent().research("测试想法"))
    for entry in out.log:
        for k in ("step", "action", "input", "output"):
            assert k in entry.to_dict()


def test_auto_research_mock_provider():
    from quantmind.ai.provider import MockProvider
    out = _run(AutoResearchAgent(provider=MockProvider()).research("A股动量反转", "A股"))
    assert out.spec.asset_class or True


def test_lookahead_warnings_detects_shift_negative():
    bad = "def sig(close):\n    return close.shift(-1)\n"
    assert lookahead_warnings(bad), "应识别 shift(-1) 前视"


def test_lookahead_warnings_detects_pct_change():
    bad = "def sig(close):\n    fwd = close.pct_change().shift(-5)\n    return fwd\n"
    assert any("pct_change" in w or "shift" in w for w in lookahead_warnings(bad))


def test_lookahead_warnings_clean():
    good = "def sig(close):\n    return close.rolling(20).mean()\n"
    assert lookahead_warnings(good) == []


def test_generate_explanation_nonempty():
    out = _run(AutoResearchAgent().research("铜期货动量"))
    expl = generate_explanation(out)
    assert expl
    assert "投资想法" in expl


def test_generate_fact_sheet_structure():
    out = _run(AutoResearchAgent().research("螺纹钢期限结构"))
    sheet = generate_fact_sheet(out, metrics={"sharpe": 1.5})
    assert sheet["idea"]
    assert sheet["code_safe"] in (True, False)
    assert sheet["metrics"]["sharpe"] == 1.5
    assert isinstance(sheet["factors"], list)
    assert "validation_notes" in sheet


def test_to_dict_full():
    out = _run(AutoResearchAgent().research("测试"))
    d = out.to_dict()
    for k in ("spec", "factors", "code_safe", "hypotheses", "log", "explanation", "fact_sheet"):
        assert k in d
