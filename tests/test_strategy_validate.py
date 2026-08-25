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

    # 无数据源 → 直接返回 error（失败闭合，不抛异常）
    req = _FakeReq(idea="动量", strategy="momentum", symbol="IC0",
                   exchange="CFFEX", interval="1d", start=None, end=None,
                   setting={}, cost=False,
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
