"""AI 模块测试：研究智能体 + 安全沙箱。"""
from __future__ import annotations

import pytest

from quantmind.ai import ResearchAgent, validate_code, compile_strategy, build_provider
from quantmind.ai.codegen import _TEMPLATE


@pytest.mark.asyncio
async def test_research_agent_offline():
    agent = ResearchAgent(build_provider("mock"))
    out = await agent.research("螺纹钢期货的动量与期限结构因子组合策略", "期货")
    assert out.spec.asset_class
    assert len(out.factors) >= 1
    assert out.code_safe is True


def test_sandbox_rejects_dangerous():
    bad = "import os\nos.system('rm -rf /')\n"
    ok, errors = validate_code(bad)
    assert ok is False
    assert any("os" in e or "禁止" in e for e in errors)


def test_sandbox_rejects_exec():
    bad = "x = eval('1+1')\n"
    ok, errors = validate_code(bad)
    assert ok is False


def test_sandbox_accepts_generated():
    ok, errors = validate_code(_TEMPLATE.format(
        idea="x", specs="            FactorSpec(name='momentum_20', kind='momentum', window=20, weight=1.0),",
        threshold=0.3, size=1, max_pos=1.0))
    assert ok is True


def test_compile_strategy():
    ok, err, _ = compile_strategy(_TEMPLATE.format(
        idea="x", specs="            FactorSpec(name='momentum_20', kind='momentum', window=20, weight=1.0),",
        threshold=0.3, size=1, max_pos=1.0))
    assert ok is True
