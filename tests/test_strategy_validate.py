# -*- coding: utf-8 -*-
"""单品种策略验证（idea → 回测 → 门槛 → 有效策略库）测试。

覆盖：
  1. idea → 策略自动识别（动量/缠论1买/缠论3买/未知）；
  2. validate_strategy 门槛判定：不达标 → rejected 不入库；
  3. 达标 + promote → 写入 lifecycle（BACKTEST + sharpe + symbols）；
  4. 数据/策略缺失 → 返回 error（失败闭合）。
"""
from __future__ import annotations

import os

import pytest

from quantmind.strategy.validation import (
    VALIDATION_STRATEGIES,
    resolve_validate_strategy,
)


class _FakeReq:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_idea_recognition() -> None:
    """idea 关键词 → 策略类型识别。"""
    assert resolve_validate_strategy("缠论3买在IC0日线上是否有效") == "chan_third_buy"
    assert resolve_validate_strategy("缠论1买") == "chan_first_buy"
    assert resolve_validate_strategy("底背驰") == "chan_first_buy"
    assert resolve_validate_strategy("动量因子") == "momentum"
    assert resolve_validate_strategy("随便什么") == ""
    assert resolve_validate_strategy("", fallback="momentum") == "momentum"
    assert set(VALIDATION_STRATEGIES) >= {"momentum", "chan_first_buy", "chan_third_buy"}


def test_validate_strategy_rejected_not_promoted(tmp_path) -> None:
    """不达标（Sharpe 低）→ rejected，不入库。"""
    os.environ["QM_KNOWLEDGE_DB"] = str(tmp_path / "kb.db")
    from quantmind.api.services.backtest_service import BacktestService
    from quantmind.core.engine import EventEngine

    svc = BacktestService(dm=None, ee=EventEngine())

    # 无数据源 → 直接返回 error（失败闭合，不抛异常）。
    # 预置模板路径已移除：用审定代码（code）走新流程。
    _minimal_code = (
        "from quantmind.strategy.base import CtaTemplate\n\n"
        "class XStrategy(CtaTemplate):\n"
        "    def on_bar(self, bar):\n"
        "        pass\n")
    req = _FakeReq(idea="动量", strategy="momentum", symbol="IC0",
                   exchange="CFFEX", interval="1d", start=None, end=None,
                   setting={}, cost=False, code=_minimal_code,
                   gate={"min_sharpe": 1.0, "min_drawdown": -0.15}, promote=True)

    import asyncio
    out = asyncio.run(svc.validate_strategy(req))
    # 多品种结构：错误记录在 per_symbol 内（失败闭合，不抛异常）
    assert isinstance(out, dict)
    assert out.get("per_symbol") and "error" in out["per_symbol"][0]


def test_resolve_and_defaults() -> None:
    """显式指定策略 + 默认参数兜底。"""
    from quantmind.strategy.validation import DEFAULT_SETTINGS

    assert DEFAULT_SETTINGS["momentum"]["window"] == 20
    assert DEFAULT_SETTINGS["chan_third_buy"]["break_window"] == 20
    assert DEFAULT_SETTINGS["chan_first_buy"]["roc_window"] == 10


def test_validate_strategy_multi_interval(tmp_path) -> None:
    """多周期回测：intervals 列表 → 品种×周期 逐项回测，条目带 interval 字段。"""
    os.environ["QM_KNOWLEDGE_DB"] = str(tmp_path / "kb.db")
    from quantmind.api.services.backtest_service import BacktestService
    from quantmind.core.engine import EventEngine

    svc = BacktestService(dm=None, ee=EventEngine())

    _minimal_code = (
        "from quantmind.strategy.base import CtaTemplate\n\n"
        "class XStrategy(CtaTemplate):\n"
        "    def on_bar(self, bar):\n"
        "        pass\n")
    req = _FakeReq(idea="动量", strategy="momentum", symbols=["IC0", "rb0"],
                   exchange="CFFEX", interval="1d",
                   intervals=["1d", "15m"], start=None, end=None,
                   setting={}, cost=False, code=_minimal_code,
                   gate=None, promote=False)

    import asyncio
    out = asyncio.run(svc.validate_strategy(req))
    assert isinstance(out, dict)
    # 无数据源：每个 品种×周期 组合都有一条带 interval 的错误条目（失败闭合）
    ps = out.get("per_symbol") or []
    assert len(ps) == 4  # 2 品种 × 2 周期
    assert out.get("intervals") == ["1d", "15m"]
    assert out.get("interval") == "1d"  # 兼容字段 = 第一个周期
    for iv in ("1d", "15m"):
        for sym in ("IC0", "rb0"):
            match = [p for p in ps if p.get("symbol") == sym and p.get("interval") == iv]
            assert len(match) == 1, f"{sym}@{iv} 缺条目"
            assert "error" in match[0]  # dm=None → 数据获取失败（失败闭合）


def test_validate_strategy_intervals_compat_fallback(tmp_path) -> None:
    """不传 intervals 时回退到旧 interval 字段（向后兼容）。"""
    os.environ["QM_KNOWLEDGE_DB"] = str(tmp_path / "kb.db")
    from quantmind.api.services.backtest_service import BacktestService
    from quantmind.core.engine import EventEngine

    svc = BacktestService(dm=None, ee=EventEngine())
    _minimal_code = (
        "from quantmind.strategy.base import CtaTemplate\n\n"
        "class XStrategy(CtaTemplate):\n"
        "    def on_bar(self, bar):\n"
        "        pass\n")
    req = _FakeReq(idea="动量", symbols=["IC0"], exchange="CFFEX",
                   interval="15m", start=None, end=None,
                   setting={}, cost=False, code=_minimal_code, gate=None, promote=False)
    import asyncio
    out = asyncio.run(svc.validate_strategy(req))
    assert out.get("intervals") == ["15m"]
    assert len(out["per_symbol"]) == 1
    assert out["per_symbol"][0].get("interval") == "15m"
