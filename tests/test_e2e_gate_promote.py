# -*- coding: utf-8 -*-
"""端到端策略挖掘：门槛判定 + 自动入有效策略库（lifecycle）测试。

覆盖 ``SearchService._gate_judge_and_promote``（升级新增）：
  1. 复合 Sharpe 达标 → judge_strategy 判定 verified → promote=True 时自动写入
     lifecycle（state=BACKTEST，含 sharpe/mdd/symbols/code）；
  2. 不达标 → rejected，不写入；
  3. 达标但 promote=False → 仅判定，不写入；
  4. 失败闭合：异常只记录到返回 dict 的 gate.error，不抛出。

不跑完整 e2e（长跑），直接对判定/入库服务做单元级验证。
"""
from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

import pytest

from quantmind.ai.provider import MockProvider
from quantmind.api.services.search_service import (
    SearchService,
    _e2e_strategy_id,
    _hash_gate,
)
from quantmind.knowledge import KnowledgeStore


class _FakeReq:
    """最小请求桩（只含 _gate_judge_and_promote 需要的字段）。"""

    def __init__(self, idea: str, symbols: list, gate, promote: bool):
        self.idea = idea
        self.symbols = symbols
        self.gate = gate
        self.promote = promote


def _make_report(sharpe, mdd, fwd_ic=0.04) -> dict:
    """构造带复合 alpha 指标的 e2e report。"""
    return {
        "pipeline": {
            "composite": {
                "portfolio": {
                    "sharpe": sharpe,
                    "max_drawdown": mdd,
                    "total_return": 0.25,
                },
                "ic_report": {"ic_mean": fwd_ic},
            }
        },
        "strategy": {"code": "class GeneratedStrategy: pass", "code_safe": True},
    }


@pytest.fixture()
def svc(tmp_path: Path) -> SearchService:
    # 独立临时知识库，避免污染真实 db/knowledge.db
    os.environ["QM_KNOWLEDGE_DB"] = str(tmp_path / "kb_test.db")
    return SearchService(dm=None, provider=MockProvider())


def test_gate_verified_and_promoted(svc: SearchService) -> None:
    """达标 + promote → verified 且自动入有效策略库（lifecycle）。"""
    req = _FakeReq(
        "动量策略测试", ["IC0", "IF0"],
        {"min_sharpe": 1.0, "min_drawdown": -0.15}, promote=True)

    out = asyncio.run(svc._gate_judge_and_promote(_make_report(1.5, -0.05), req))

    assert out["status"] == "verified"
    assert out["promoted"] is True
    assert out["strategy_id"].startswith("e2e_")

    lc = KnowledgeStore().get_strategy_lifecycle(out["strategy_id"])
    assert lc is not None
    assert lc["state"] == "BACKTEST"
    assert lc["sharpe"] == pytest.approx(1.5)
    assert lc["max_drawdown"] == pytest.approx(-0.05)
    assert lc["symbols"] == ["IC0", "IF0"]


def test_gate_rejected_not_promoted(svc: SearchService) -> None:
    """不达标（低 Sharpe + 大回撤）→ rejected，不写入有效策略库。"""
    req = _FakeReq(
        "差策略", ["IC0", "IF0"],
        {"min_sharpe": 1.0, "min_drawdown": -0.15}, promote=True)

    out = asyncio.run(svc._gate_judge_and_promote(_make_report(0.3, -0.30), req))

    assert out["status"] == "rejected"
    assert out["promoted"] is False
    assert out["strategy_id"] == ""
    assert KnowledgeStore().get_strategy_lifecycle("e2e_whatever") is None


def test_gate_judge_only_without_promote(svc: SearchService) -> None:
    """达标但 promote=False → 仅判定，不入库。"""
    req = _FakeReq("只判定", ["IC0"], {"min_sharpe": 1.0}, promote=False)

    out = asyncio.run(svc._gate_judge_and_promote(_make_report(1.5, -0.05), req))

    assert out["status"] == "verified"
    assert out["promoted"] is False
    assert out["strategy_id"] == ""


def test_gate_failure_closed(svc: SearchService) -> None:
    """指标缺失（空报告）不抛异常，状态不误判为 verified。"""
    req = _FakeReq("空报告", ["IC0"], {"min_sharpe": 1.0}, promote=True)

    # portfolio 缺失 → sharpe 全 None → 规则判定应给出 rejected（无 OOS 数据）
    out = asyncio.run(svc._gate_judge_and_promote({"pipeline": {}}, req))

    assert out["enabled"] is True
    assert out["status"] in ("active", "rejected")
    assert out["promoted"] is False


def test_helpers() -> None:
    """辅助函数：gate 哈希稳定、strategy_id 可读且同日稳定。"""
    assert _hash_gate({"min_sharpe": 1.0, "min_drawdown": -0.15}) == \
        "min_drawdown=-0.15|min_sharpe=1.0"
    assert _hash_gate(None) == ""

    sid1 = _e2e_strategy_id("动量", ["IF0", "IC0"])
    sid2 = _e2e_strategy_id("动量", ["IC0", "IF0"])
    assert sid1 == sid2  # 标的顺序无关
    assert sid1.startswith("e2e_")
