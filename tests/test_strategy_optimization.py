# -*- coding: utf-8 -*-
"""参数优化（防过拟合三防线）集成测试。

覆盖：
  1. 网格解析优先级：请求显式 > 策略类 PARAM_GRID > 预置模板内置网格；缺失 → error；
  2. 合成数据 e2e：IS/OOS 切分、试验数统计、DSR/高原字段、gate 用 OOS 判定；
  3. generate code 内 PARAM_GRID 提取与过滤（register_generated_strategy）；
  4. 失败闭合：数据不足/无网格 → per_symbol error 或顶层 error，不抛异常。
"""
from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta

import numpy as np
import pytest

from quantmind.core.constant import Exchange
from quantmind.core.object import BarData


# 最小可注册策略（无参数 → auto_param_grid 推导为空）
_MINIMAL_CODE = (
    "from quantmind.strategy.base import CtaTemplate\n\n"
    "class IdeaStrategy(CtaTemplate):\n"
    "    def on_bar(self, bar):\n"
    "        pass\n")

# 带数值参数的策略（auto_param_grid 可推导网格）
_GRID_CODE = (
    "from quantmind.strategy.base import CtaTemplate\n\n"
    "class IdeaStrategy(CtaTemplate):\n"
    "    parameters = ['window', 'threshold']\n\n"
    "    def __init__(self, context, setting=None):\n"
    "        self.window = 20\n"
    "        self.threshold = 0.03\n"
    "        super().__init__(context, setting)\n\n"
    "    def on_bar(self, bar):\n"
    "        pass\n")


class _FakeReq:
    def __init__(self, **kw):
        defaults = {
            "idea": "动量", "symbols": ["IC0"],
            "symbol": "", "exchange": "CFFEX", "interval": "1d",
            "start": None, "end": None, "setting": {}, "cost": False,
            "gate": None, "promote": False,
            "optimization": None, "code": _MINIMAL_CODE,
        }
        defaults.update(kw)
        self.__dict__.update(defaults)


def _synthetic_bars(n: int = 800, seed: int = 11) -> list:
    """带漂移的随机游走日线（保证有交易、有趋势）。"""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, 0.012, n)
    price = 4000 * np.exp(np.cumsum(rets))
    base = datetime(2022, 1, 1)
    bars = []
    for i in range(n):
        close = float(price[i])
        bars.append(BarData(
            gateway_name="test", symbol="IC0", exchange=Exchange.CFFEX,
            datetime=base + timedelta(days=i),
            open_price=close * 0.999, high_price=close * 1.005,
            low_price=close * 0.995, close_price=close,
            volume=1000.0, turnover=0.0, open_interest=0.0,
        ))
    return bars


class _FakeDM:
    """最小 DataManager 桩：只提供 get_bar_data。"""

    def __init__(self, bars: list):
        self._bars = bars

    async def get_bar_data(self, req) -> list:
        return self._bars


@pytest.fixture()
def kb_env(tmp_path, monkeypatch):
    monkeypatch.setenv("QM_KNOWLEDGE_DB", str(tmp_path / "kb.db"))
    yield


# ---------------------------------------------------------------- 网格解析
def test_grid_missing_returns_error(kb_env, monkeypatch) -> None:
    """预置网格清空且自动推导不可用 → 顶层 error（失败闭合）。"""
    import quantmind.strategy.validation as val_mod
    monkeypatch.setattr(val_mod, "DEFAULT_PARAM_GRIDS", {})
    from quantmind.api.services.backtest_service import BacktestService
    from quantmind.core.engine import EventEngine
    from quantmind.strategy import optimizer as _opt

    svc = BacktestService(dm=None, ee=EventEngine())
    req = _FakeReq(
                   optimization={"enabled": True, "param_grid": {}})
    # 同时屏蔽自动推导，验证真正的失败闭合路径
    monkeypatch.setattr(_opt, "auto_param_grid", lambda cls, **kw: {})
    out = asyncio.run(svc.validate_strategy(req))
    assert "error" in out
    assert "参数网格" in out["error"]


def test_request_grid_overrides_builtin(kb_env) -> None:
    """请求显式网格优先于预置表。"""
    from quantmind.api.services.backtest_service import BacktestService
    from quantmind.core.engine import EventEngine

    bars = _synthetic_bars()
    svc = BacktestService(dm=_FakeDM(bars), ee=EventEngine())
    grid = {"window": [5, 10]}
    req = _FakeReq(symbols=["IC0"], code=_GRID_CODE,
                   optimization={"enabled": True, "param_grid": grid,
                                 "min_trades": 0, "top_k": 2})
    out = asyncio.run(svc.validate_strategy(req))
    assert out.get("optim", {}).get("param_grid") == grid
    assert out["optim"]["grid_source"] == "request"
    assert out["optim"]["n_trials"] == 2


def test_default_builtin_grid_used(kb_env) -> None:
    """未显式给网格时 momentum 用**策略类自动推导**的邻域网格。"""
    from quantmind.api.services.backtest_service import BacktestService
    from quantmind.core.engine import EventEngine

    svc = BacktestService(dm=_FakeDM(_synthetic_bars()), ee=EventEngine())
    req = _FakeReq(symbols=["IC0"], code=_GRID_CODE,
                   optimization={"enabled": True, "min_trades": 0})
    out = asyncio.run(svc.validate_strategy(req))
    assert "error" not in out, out.get("error")
    assert out["optim"]["grid_source"] == "auto"
    assert out["optim"]["param_grid"] == {"window": [10, 20, 40],
                                          "threshold": [0.015, 0.03, 0.045]}
    assert out["optim"]["n_trials"] == 9


def test_builtin_grid_fallback_when_auto_empty(kb_env, monkeypatch) -> None:
    """自动推导不可用且无显式网格 → 顶层 error（失败闭合）。"""
    from quantmind.api.services.backtest_service import BacktestService
    from quantmind.core.engine import EventEngine
    from quantmind.strategy import optimizer as _opt

    monkeypatch.setattr(_opt, "auto_param_grid", lambda cls, **kw: {})
    svc = BacktestService(dm=_FakeDM(_synthetic_bars()), ee=EventEngine())
    req = _FakeReq(symbols=["IC0"],
                   optimization={"enabled": True, "min_trades": 0})
    out = asyncio.run(svc.validate_strategy(req))
    assert "error" in out
    assert "参数网格" in out["error"]


# ---------------------------------------------------------------- e2e 主链路
def test_e2e_optimization_flow(kb_env) -> None:
    """合成数据全链路：IS/OOS 切分 + 网格 + DSR + gate 判定。"""
    from quantmind.api.services.backtest_service import BacktestService
    from quantmind.core.engine import EventEngine

    bars = _synthetic_bars(1000)
    svc = BacktestService(dm=_FakeDM(bars), ee=EventEngine())
    req = _FakeReq(
        symbols=["IC0"], code=_GRID_CODE,
        optimization={"enabled": True,
                      "param_grid": {"window": [10, 20, 30],
                                     "threshold": [0.02, 0.03, 0.04]},
                      "min_trades": 0, "top_k": 3},
        gate={"min_sharpe": 0.1, "min_drawdown": -0.5},
        promote=True,
    )
    out = asyncio.run(svc.validate_strategy(req))
    assert "error" not in out, out.get("error")

    optim = out["optim"]
    # 试验数 = 网格组合数（IS 段穷举）
    assert optim["n_trials"] == 9
    assert optim["is_bars"] == 700
    assert optim["oos_bars"] == 300
    assert optim["use_dsr"] is True

    item = out["per_symbol"][0]
    assert "error" not in item
    detail = item["optim_detail"]
    # best_combo 必在网格内
    assert detail["best_combo"]["window"] in (10, 20, 30)
    # top-K 记录 ≤ 3
    assert 1 <= len(detail["top"]) <= 3
    # DSR 已计算且在 [0, 1]
    assert detail["dsr"] is not None
    assert 0.0 <= detail["dsr"] <= 1.0
    # 高原检验结果存在
    assert isinstance(detail["plateau"].get("ok"), bool)
    # OOS 段指标与 IS 最优不混用（OOS Sharpe 应来自 OOS 回测）
    assert item["report"]["sharpe"] == detail["oos_sharpe"]
    # gate 判定存在
    assert item["gate"]["enabled"] is True
    assert item["gate"]["status"] in ("verified", "rejected")
    # gate 指标里带 DSR
    assert item["gate"]["metrics"]["dsr"] == detail["dsr"]


def test_oos_only_used_for_gate_not_is_best(kb_env) -> None:
    """gate 判据使用 OOS 指标；即便 IS 最优 Sharpe 更高也不得分。"""
    from quantmind.api.services.backtest_service import BacktestService
    from quantmind.core.engine import EventEngine

    bars = _synthetic_bars(800, seed=23)
    svc = BacktestService(dm=_FakeDM(bars), ee=EventEngine())
    req = _FakeReq(
        symbols=["IC0"], code=_GRID_CODE,
        optimization={"enabled": True, "param_grid": {"window": [10, 20, 30]},
                      "min_trades": 0},
        gate={"min_sharpe": 99.0, "min_drawdown": -0.01},  # 理论上不可能达标
        promote=False,
    )
    out = asyncio.run(svc.validate_strategy(req))
    item = out["per_symbol"][0]
    if "error" in item:  # 数据太短等失败闭合情形直接通过
        return
    assert item["gate"]["status"] == "rejected"


# ---------------------------------------------------------------- PARAM_GRID 提取
def test_param_grid_extracted_from_generated_code(kb_env) -> None:
    """生成代码中的 PARAM_GRID 被提取；非法键/非数值被过滤。"""
    from quantmind.api.services.backtest_service import BacktestService
    from quantmind.core.engine import EventEngine

    svc = BacktestService(dm=None, ee=EventEngine())
    code = '''
from quantmind.strategy.base import CtaTemplate


class DemoStrategy(CtaTemplate):
    parameters = ["window", "threshold"]

    def __init__(self, context, setting=None):
        super().__init__(context, setting)


PARAM_GRID = {"window": [10, 20, 30], "threshold": [0.02, 0.03],
              "hacked": ["a"], "not_declared": [1, 2]}
'''
    ok, err, info = svc.register_generated_strategy("demo", code)
    assert ok, err
    assert set(info["param_grid"]) == {"window", "threshold"}
    assert info["param_grid"]["window"] == [10, 20, 30]


def test_fail_closed_short_history(kb_env) -> None:
    """历史过短（<200 根）→ per_symbol error，不抛异常。"""
    from quantmind.api.services.backtest_service import BacktestService
    from quantmind.core.engine import EventEngine

    svc = BacktestService(dm=_FakeDM(_synthetic_bars(150)), ee=EventEngine())
    req = _FakeReq(symbols=["IC0"], code=_GRID_CODE,
                   optimization={"enabled": True, "param_grid": {"window": [10]}})
    out = asyncio.run(svc.validate_strategy(req))
    assert out["per_symbol"][0].get("error")


_VALID_STRATEGY_CODE = '''
from quantmind.strategy.base import CtaTemplate
from quantmind.core.utility import ArrayManager


class DemoStrategy(CtaTemplate):
    parameters = ["window", "threshold"]

    def __init__(self, context, setting=None):
        super().__init__(context, setting)
        self.window = 20
        self.threshold = 0.03
        self.am = None
        self.last_target = 0.0

    def on_bar(self, bar):
        if self.am is None:
            self.am = ArrayManager(self.window + 5)
        self.am.update_bar(bar)
        if not self.am.inited:
            return
        closes = self.am.close
        mom = closes[-1] / closes[-self.window] - 1.0
        if mom > self.threshold:
            target = 1.0
        elif mom < -self.threshold:
            target = -1.0
        else:
            target = 0.0
        if target != self.last_target:
            self.set_target(bar.vt_symbol, target)
            self.last_target = target
            self.pos = target
'''


class _CodeProvider:
    """记录多轮消息的假 Provider。"""

    name = "fake"

    def __init__(self, code: str):
        self.code = code
        self.calls = []

    async def chat(self, system: str, user: str) -> str:
        self.calls.append([{"role": "user", "content": user}])
        return self.code

    async def chat_messages(self, system: str, messages) -> str:
        self.calls.append(list(messages))
        return self.code


def test_draft_multi_turn_history(kb_env) -> None:
    """对话式草稿：首轮带思想；多轮时完整历史原样传给 LLM。"""
    import asyncio

    from quantmind.api.services.backtest_service import BacktestService
    from quantmind.core.engine import EventEngine

    svc = BacktestService(dm=None, ee=EventEngine())
    provider = _CodeProvider(_VALID_STRATEGY_CODE)

    out = asyncio.run(svc.draft_strategy_code(provider, "动量策略，阈值 3%"))
    assert "error" not in out, out.get("error")
    assert out["sandbox_ok"] is True and out["code"]
    assert out["provider"] == "fake"
    # 首轮：单条 user 消息
    assert provider.calls[0] == [{"role": "user", "content": "动量策略，阈值 3%"}]

    # 第二轮：带完整历史（上轮代码 + 修改意见）
    hist = [{"role": "user", "content": "动量策略，阈值 3%"},
            {"role": "assistant", "content": _VALID_STRATEGY_CODE},
            {"role": "user", "content": "止损改成 2%"}]
    out2 = asyncio.run(svc.draft_strategy_code(provider, "", history=hist))
    assert "error" not in out2, out2.get("error")
    assert provider.calls[1] == hist


def test_draft_sandbox_fail_shown_not_hidden(kb_env) -> None:
    """草稿沙箱不过时不吞错：代码+错误详情一并返回，供界面审阅修复。"""
    import asyncio

    from quantmind.api.services.backtest_service import BacktestService
    from quantmind.core.engine import EventEngine

    svc = BacktestService(dm=None, ee=EventEngine())
    provider = _CodeProvider("def broken(:\n    pass")
    out = asyncio.run(svc.draft_strategy_code(provider, "任意"))
    assert out.get("sandbox_ok") is False
    assert out.get("sandbox_err")
    assert out.get("code")  # 代码仍然返回，供用户查看


def test_validate_with_approved_code(kb_env) -> None:
    """审定代码直传：跳过 LLM，直接注册回测；strategy_desc 标注来源。"""
    import asyncio

    from quantmind.api.services.backtest_service import BacktestService
    from quantmind.core.engine import EventEngine

    svc = BacktestService(dm=_FakeDM(_synthetic_bars(1000)), ee=EventEngine())
    req = _FakeReq(symbols=["IC0"], llm_code=True,
                   idea="动量", code=_VALID_STRATEGY_CODE)
    out = asyncio.run(svc.validate_strategy(req))
    assert "error" not in out, out.get("error")
    assert out["strategy_desc"] == "用户审定的 LLM 策略"
    assert out["strategy"] == "idea_strategy"
    item = out["per_symbol"][0]
    assert "error" not in item, item.get("error")
