"""策略级 AI 沉淀闭环 · API 编排层接线测试（不启动真实 FastAPI/网络）。

覆盖（直接测 service/编排层接线）：
- judge_strategy 集成：真实规则判读（verified / rejected / active）。
- _persist_backtest_lifecycle：把伪造 risk_xray 的真实夏普/回撤回填 + 判读落库
  （state=BACKTEST、status=verified/rejected）；无 sharpe 时安全跳过。
- _persist_paper_lifecycle：模拟盘判读 + 经验 brief 落库（status + brief）。
- _metrics_from_equity：从权益曲线算年化夏普/最大回撤/总收益。
- 注册路径「判空再 upsert」不覆盖已有回测指标（run_id 已存在时跳过 upsert）。

所有用例用 tmp_path 指向的独立 SQLite 文件，离线无网络。
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from quantmind.knowledge import KnowledgeStore
from quantmind.api.services.backtest_service import _metrics_from_equity
# 编排层接线函数（小助手，封装「落库 + 判读」）——注意用 importlib 取「模块」而非包导出
import importlib
app_module = importlib.import_module("quantmind.api.app")
from quantmind.api.app import _persist_backtest_lifecycle, _persist_paper_lifecycle
from quantmind.api.schemas import StrategyRegisterRequest


def _mock_provider():
    """Mock LLM provider —— _is_mock_provider 判定为 mock，判读走内置规则兜底。"""
    return SimpleNamespace(name="mock", chat=None)


def _wire_app_state(monkeypatch, provider=None, knowledge_store=None):
    """把 app.state 替换为轻量桩，供 _llm_provider / 临时 KnowledgeStore 注入。"""
    from types import SimpleNamespace as NS

    # 预留 KnowledgeStore 注入点：monkeypatch 掉 app 模块内的 KnowledgeStore 符号
    if knowledge_store is not None:
        monkeypatch.setattr(app_module, "KnowledgeStore", lambda: knowledge_store)

    fake_app = NS(state=NS(search_service=NS(provider=provider or _mock_provider())))
    monkeypatch.setattr(app_module, "app", fake_app)
    return fake_app


# ---------------------------------------------------------------------------
# (a) _persist_backtest_lifecycle —— 回测落库 + 判读
# ---------------------------------------------------------------------------
def test_persist_backtest_verified(tmp_path, monkeypatch):
    store = KnowledgeStore(db_path=str(tmp_path / "bt.db"))
    store.upsert_strategy_lifecycle("s1", idea="动量", state="RESEARCH", source="AI生成")
    _wire_app_state(monkeypatch, knowledge_store=store)

    risk_xray = {
        "return": {"sharpe": 1.2, "total": 0.35},
        "risk": {"max_drawdown": -0.10},
    }
    judged = asyncio.run(_persist_backtest_lifecycle("s1", risk_xray))
    assert judged["status"] == "verified"

    rec = store.get_strategy_lifecycle("s1")
    assert rec["state"] == "BACKTEST"
    assert rec["sharpe"] == 1.2
    assert rec["max_drawdown"] == -0.10
    assert rec["status"] == "verified"
    assert rec["reason"]  # 判读原因非空
    assert rec["idea"] == "动量"  # 未覆盖既有 idea


def test_persist_backtest_rejected_low_sharpe(tmp_path, monkeypatch):
    store = KnowledgeStore(db_path=str(tmp_path / "bt2.db"))
    store.upsert_strategy_lifecycle("s2", state="RESEARCH")
    _wire_app_state(monkeypatch, knowledge_store=store)

    risk_xray = {
        "return": {"sharpe": 0.2, "total": 0.05},
        "risk": {"max_drawdown": -0.05},
    }
    judged = asyncio.run(_persist_backtest_lifecycle("s2", risk_xray))
    assert judged["status"] == "rejected"

    rec = store.get_strategy_lifecycle("s2")
    assert rec["state"] == "BACKTEST"
    assert rec["sharpe"] == 0.2
    assert rec["status"] == "rejected"
    assert "夏普" in rec["reason"]


def test_persist_backtest_skips_without_sharpe(tmp_path, monkeypatch):
    """risk_xray 无 sharpe（或为空）时安全跳过，不落 state=BACKTEST。"""
    store = KnowledgeStore(db_path=str(tmp_path / "bt3.db"))
    store.upsert_strategy_lifecycle("s3", state="RESEARCH")
    _wire_app_state(monkeypatch, knowledge_store=store)

    judged = asyncio.run(_persist_backtest_lifecycle("s3", None))
    assert judged is None
    rec = store.get_strategy_lifecycle("s3")
    assert rec["state"] == "RESEARCH"  # 未被改动
    assert rec["sharpe"] is None


# ---------------------------------------------------------------------------
# (b) _persist_paper_lifecycle —— 模拟盘判读 + brief
# ---------------------------------------------------------------------------
def test_persist_paper_verified_with_brief(tmp_path, monkeypatch):
    store = KnowledgeStore(db_path=str(tmp_path / "pp.db"))
    store.upsert_strategy_lifecycle("p1", state="PAPER", source="AI生成")
    _wire_app_state(monkeypatch, knowledge_store=store)

    metrics = {"sharpe": 0.9, "max_drawdown": -0.12, "trade_count": 12}
    judged = asyncio.run(_persist_paper_lifecycle("p1", metrics, idea="螺纹钢动量"))
    assert judged["status"] == "verified"

    rec = store.get_strategy_lifecycle("p1")
    assert rec["status"] == "verified"
    assert rec["brief"]  # 经验 brief 非空（中文）
    assert rec["state"] == "PAPER"


# ---------------------------------------------------------------------------
# (c) judge_strategy 直接集成
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_judge_strategy_integration():
    from quantmind.research.knowledge_loop import judge_strategy

    provider = _mock_provider()
    gate = {"min_sharpe": 0.5, "min_drawdown": -0.30}

    v = await judge_strategy(
        provider,
        {"sharpe": 1.0, "max_drawdown": -0.10, "state": "BACKTEST"},
        gate=gate, fallback_rules=True,
    )
    assert v["status"] == "verified"

    r = await judge_strategy(
        provider,
        {"sharpe": 0.3, "max_drawdown": -0.10, "state": "BACKTEST"},
        gate=gate, fallback_rules=True,
    )
    assert r["status"] == "rejected"

    a = await judge_strategy(
        provider,
        {"state": "IDEA", "sharpe": None},
        gate=gate, fallback_rules=True,
    )
    assert a["status"] == "active"


# ---------------------------------------------------------------------------
# (d) _metrics_from_equity —— 权益曲线指标
# ---------------------------------------------------------------------------
def test_metrics_from_equity():
    import pandas as pd

    # 单调上涨权益 → 正夏普 / 无回撤
    up = _metrics_from_equity(pd.Series([1.0, 1.1, 1.21, 1.331]))
    assert up["sharpe"] > 0
    assert up["max_drawdown"] >= -1e-9
    assert up["total_return"] == pytest.approx(0.331, abs=1e-3)

    # 过短 → 空
    assert _metrics_from_equity(pd.Series([1.0])) == {}


def test_metrics_from_equity_dump_has_negative_sharpe():
    import pandas as pd

    # 先涨后深跌 → 负收益 / 明显回撤
    eq = pd.Series([1.0, 1.2, 1.4, 1.1, 0.8, 1.0, 0.7])
    out = _metrics_from_equity(eq)
    assert out["total_return"] < 0
    assert out["max_drawdown"] < 0


# ---------------------------------------------------------------------------
# (e) 注册路径「判空再 upsert」不覆盖已有回测指标
# ---------------------------------------------------------------------------
def test_register_upsert_does_not_overwrite_metrics(tmp_path):
    store = KnowledgeStore(db_path=str(tmp_path / "reg.db"))

    # 首次注册：写 run_id / idea / code
    store.upsert_strategy_lifecycle(
        "strat_a", run_id="run-1", idea="动量", state="RESEARCH",
        source="AI生成", code="def algo(): pass", code_safe=True,
    )
    # 回测后写入真实指标
    store.update_strategy_state("strat_a", state="BACKTEST", sharpe=1.5, max_drawdown=-0.08)

    # 二次注册：仅当「行为空或尚无 run_id」才 upsert —— 此处 run_id 已存在，应跳过
    existing = store.get_strategy_lifecycle("strat_a")
    assert existing is not None and existing.get("run_id")
    if existing is None or not (existing.get("run_id") or ""):
        store.upsert_strategy_lifecycle(
            strategy_id="strat_a", run_id="run-2", idea="新idea", state="RESEARCH",
            source="AI生成", code="def algo2(): pass", code_safe=True,
        )

    rec = store.get_strategy_lifecycle("strat_a")
    # 已有指标未被覆盖
    assert rec["sharpe"] == 1.5
    assert rec["max_drawdown"] == -0.08
    assert rec["run_id"] == "run-1"  # 既有 run_id 保留
    # 状态仍为回测后（未被注册流程降级回 RESEARCH）
    assert rec["state"] == "BACKTEST"


def test_register_upsert_on_fresh_row_persists_run_id(tmp_path):
    """首次注册（行为空）时 upsert 落 run_id/idea/code/source。"""
    store = KnowledgeStore(db_path=str(tmp_path / "reg2.db"))
    existing = store.get_strategy_lifecycle("strat_b")
    assert existing is None
    if existing is None or not (existing.get("run_id") or ""):
        store.upsert_strategy_lifecycle(
            strategy_id="strat_b", run_id="run-9", idea="趋势", state="RESEARCH",
            source="AI生成", code="def x(): pass", code_safe=True,
        )
    rec = store.get_strategy_lifecycle("strat_b")
    assert rec["run_id"] == "run-9"
    assert rec["idea"] == "趋势"
    assert rec["state"] == "RESEARCH"
    assert rec["metadata"]["strategy_id"] == "strat_b"
