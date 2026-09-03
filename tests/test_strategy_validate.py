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


# ================= 沙箱自修复循环（draft_strategy_code） =================

class _FakeProvider:
    """按序返回预设响应的假 Provider。"""

    name = "fake"
    last_fallback_reason = None

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def chat_messages(self, system, msgs):
        self.calls.append(msgs)
        return self._responses.pop(0)

    async def chat(self, system, msg):
        self.calls.append(msg)
        return self._responses.pop(0)


_MINIMAL = (
    "from quantmind.strategy.base import CtaTemplate\n\n"
    "class XStrategy(CtaTemplate):\n"
    "    def on_bar(self, bar):\n"
    "        pass\n")


def test_draft_self_repair_success(tmp_path) -> None:
    """首轮沙箱失败（getattr）→ 自修复轮通过 → sandbox_ok=True。"""
    import asyncio
    os.environ["QM_KNOWLEDGE_DB"] = str(tmp_path / "kb.db")
    from quantmind.api.services.backtest_service import BacktestService
    from quantmind.core.engine import EventEngine

    svc = BacktestService(dm=None, ee=EventEngine())
    bad = _MINIMAL.replace(
        "    def on_bar(self, bar):\n        pass\n",
        "    def on_bar(self, bar):\n        m = getattr(self, 'daily', None)\n")
    good = _MINIMAL.replace(
        "    def on_bar(self, bar):\n        pass\n",
        "    def on_bar(self, bar):\n"
        "        if bar.datetime.hour >= 6:\n"
        "            pass\n")
    provider = _FakeProvider([bad, good])
    out = asyncio.run(svc.draft_strategy_code(provider, "测试", interval="15m"))
    assert out["sandbox_ok"] is True
    assert out["repair_rounds"] == 1
    assert "getattr" not in out["code"]


def test_draft_self_repair_exhausted(tmp_path) -> None:
    """全部轮次失败 → 返回末版代码 + sandbox_ok=False + 轮次记录。"""
    import asyncio
    os.environ["QM_KNOWLEDGE_DB"] = str(tmp_path / "kb.db")
    from quantmind.api.services.backtest_service import BacktestService
    from quantmind.core.engine import EventEngine

    svc = BacktestService(dm=None, ee=EventEngine())
    bad = _MINIMAL.replace(
        "    def on_bar(self, bar):\n        pass\n",
        "    def on_bar(self, bar):\n        m = getattr(self, 'daily', None)\n")
    good = bad.replace(
        "        m = getattr(self, 'daily', None)\n",
        "        if bar.datetime.hour >= 6:\n            pass\n")
    provider = _FakeProvider([bad, bad, bad])
    out = asyncio.run(svc.draft_strategy_code(provider, "测试", interval="15m"))
    assert out["sandbox_ok"] is False
    assert out["repair_rounds"] == 2
    assert "getattr" in out["code"]  # 末版代码保留供人工修复


def test_generate_left_shift_1d_daily_ctx(tmp_path) -> None:
    """失败左移：1d 周期 + 代码引用 self.daily → 生成阶段直接报错。"""
    import asyncio
    os.environ["QM_KNOWLEDGE_DB"] = str(tmp_path / "kb.db")
    from quantmind.api.services.backtest_service import BacktestService
    from quantmind.core.engine import EventEngine

    svc = BacktestService(dm=None, ee=EventEngine())
    bad = _MINIMAL + "\n    # uses self.daily\n"
    bad = _MINIMAL.replace(
        "    def on_bar(self, bar):\n        pass\n",
        "    def on_bar(self, bar):\n        d = self.daily\n")
    provider = _FakeProvider([bad, bad, bad])
    code, err = asyncio.run(svc._llm_generate_strategy(provider, "测试", interval="1d"))
    assert code == ""
    assert "自修复" in err and ("不兼容" in err or "self.mtf" in err)


def test_draft_interval_hint_reaches_llm(tmp_path) -> None:
    """周期告知：draft 调用把【数据周期】hint 注入 LLM 消息。"""
    import asyncio
    os.environ["QM_KNOWLEDGE_DB"] = str(tmp_path / "kb.db")
    from quantmind.api.services.backtest_service import BacktestService
    from quantmind.core.engine import EventEngine

    svc = BacktestService(dm=None, ee=EventEngine())
    provider = _FakeProvider([_MINIMAL])
    asyncio.run(svc.draft_strategy_code(provider, "测试", interval="15m"))
    sent = provider.calls[0]
    flat = str(sent)
    assert "【数据周期】15m" in flat
    assert "self.mtf" in flat


def test_daily_close_times_no_pandas_normalize() -> None:
    """回归：BarData.datetime 是标准库 datetime，误用 pandas 的 normalize 会抛
    AttributeError 并被日线上下文的 except 吞掉，导致"日线数据不可用"。
    同时覆盖历史 16:00 UTC 旧约定 → 归一当日 00:00 再 +1 天。"""
    from datetime import datetime, timezone

    from quantmind.api.services.backtest_service import _daily_close_times
    from quantmind.core.object import BarData

    bars = [
        BarData(symbol="IC0", exchange="CFFEX",
                datetime=datetime(2026, 8, 25, 0, 0, tzinfo=timezone.utc),
                open_price=1, high_price=2, low_price=0.5, close_price=1.5,
                volume=10),
        BarData(symbol="IC0", exchange="CFFEX",
                datetime=datetime(2016, 1, 4, 16, 0, tzinfo=timezone.utc),
                open_price=1, high_price=2, low_price=0.5, close_price=1.5,
                volume=1),
    ]
    ct = _daily_close_times(bars)
    assert ct[0] == datetime(2026, 8, 26, 0, 0, tzinfo=timezone.utc)
    assert ct[1] == datetime(2016, 1, 5, 0, 0, tzinfo=timezone.utc)
