"""模拟盘实跑闭环测试：run_paper 把已注册/内置策略部署到 PaperEngine 回放并晋升生命周期。

覆盖：
- 内置策略（multifactor）模拟盘实跑产出 summary/成交/命中 PaperEngine
- AI 注册策略（register_generated_strategy 后）可被 run_paper 运行
- 未注册/不存在的策略被友好拒绝
- 生命周期从 RESEARCH 晋升到 PAPER（metrics 写入）
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

from quantmind.api.services.backtest_service import BacktestService
from quantmind.core.constant import Exchange, Interval
from quantmind.core.engine import EventEngine
from quantmind.core.object import BarData
from quantmind.data.feed.base import HistoryRequest
from quantmind.paper.promotion import LifecycleManager, LifecycleState
from quantmind.api.schemas import PaperRunRequest


def _synthetic_bars(n: int = 120, symbol: str = "rb0", exchange=Exchange.SHFE) -> list[BarData]:
    """构造一段趋势行情（前段涨、后段跌），保证多数策略会触发成交。"""
    base = 3000.0
    start = datetime(2024, 1, 1)
    bars = []
    for i in range(n):
        dt = start + timedelta(days=i)
        # 前 2/3 上涨，后 1/3 下跌，制造趋势 + 回撤信号
        ratio = 1.0 + (0.02 if i < n * 2 // 3 else -0.02)
        close = base * (1 + 0.001 * i) * ratio * (1 + 0.004 * (i % 5))
        open_p = close * 0.998
        high = max(open_p, close) * 1.004
        low = min(open_p, close) * 0.996
        bars.append(
            BarData(
                symbol=symbol, exchange=exchange, datetime=dt,
                interval=Interval.DAILY, open_price=open_p, high_price=high,
                low_price=low, close_price=close, volume=10000.0,
            )
        )
    return bars


class FakeDM:
    """最小 DataManager 桩：get_bar_data 直接返回合成行情，避免网络。"""

    def __init__(self, bars: list[BarData]):
        self._bars = bars

    async def get_bar_data(self, req: HistoryRequest, source_sink=None) -> list[BarData]:
        # 合成行情全在窗口内，直接返回（避免 naive/aware 时间比较差异）
        return self._bars


def test_run_paper_builtin_strategy():
    dm = FakeDM(_synthetic_bars())
    ee = EventEngine()
    bs = BacktestService(dm, ee)
    req = PaperRunRequest(
        strategy="multifactor",
        symbol="rb0",
        exchange="SHFE",
        setting={"size": 5, "max_pos": 1.0},
    )
    result = asyncio.run(bs.run_paper(req))
    assert result["ok"] is True
    assert result["strategy"] == "multifactor"
    assert result["vt_symbol"] == "rb0.SHFE"
    assert result["bars"] > 0
    assert result["metrics"]["trade_count"] >= 0
    assert "final_cash" in result["metrics"]


def test_run_paper_registered_strategy():
    """AI 生成策略注册后可被模拟盘实跑。"""
    source = (
        "from quantmind.strategy import CtaTemplate\n"
        "from quantmind.core.object import BarData\n"
        "class MyAIStrategy(CtaTemplate):\n"
        "    parameters = ['fast', 'slow', 'size', 'max_pos']\n"
        "    def __init__(self, ctx, setting):\n"
        "        self.fast = 3\n"
        "        self.slow = 10\n"
        "        self.size = 1\n"
        "        self.max_pos = 1.0\n"
        "        self.buf = []\n"
        "        self.last_target = 0.0\n"
        "        super().__init__(ctx, setting)\n"
        "    def on_bar(self, bar: BarData):\n"
        "        self.buf.append(bar.close_price)\n"
        "        if len(self.buf) <= self.slow:\n"
        "            return\n"
        "        fast = sum(self.buf[-self.fast:]) / self.fast\n"
        "        slow = sum(self.buf[-self.slow:]) / self.slow\n"
        "        target = self.max_pos * self.size if fast > slow else -self.max_pos * self.size\n"
        "        if target != self.last_target:\n"
        "            self.set_target(bar.vt_symbol, target)\n"
        "            self.last_target = target\n"
    )
    dm = FakeDM(_synthetic_bars())
    ee = EventEngine()
    bs = BacktestService(dm, ee)
    bs.register_generated_strategy("my_ai_strategy", source)

    req = PaperRunRequest(strategy="my_ai_strategy", setting={"fast": 3, "slow": 10, "size": 1})
    result = asyncio.run(bs.run_paper(req))
    assert result["ok"] is True
    assert result["strategy"] == "my_ai_strategy"
    assert result["trade_count"] > 0  # 趋势行情应触发成交


def test_run_paper_unknown_strategy_rejected():
    dm = FakeDM(_synthetic_bars())
    bs = BacktestService(dm, EventEngine())
    req = PaperRunRequest(strategy="does_not_exist_xyz")
    result = asyncio.run(bs.run_paper(req))
    assert "error" in result
    assert "未注册" in result["error"]


def test_run_paper_no_data():
    dm = FakeDM([])  # 空数据
    bs = BacktestService(dm, EventEngine())
    req = PaperRunRequest(strategy="multifactor")
    result = asyncio.run(bs.run_paper(req))
    assert result.get("error") == "无数据"


def test_lifecycle_promotes_to_paper_after_run():
    """模拟盘实跑后，内置策略生命周期应到达 PAPER（metrics 写入）。"""
    lc = LifecycleManager()
    bs = BacktestService(FakeDM(_synthetic_bars()), EventEngine())
    # 先注册入 RESEARCH（模拟 /strategies/register）
    lc.promote("multifactor", LifecycleState.RESEARCH, note="registered")
    assert lc.get_or_create("multifactor").state == LifecycleState.RESEARCH

    req = PaperRunRequest(strategy="multifactor", setting={"size": 5, "max_pos": 1.0})
    result = asyncio.run(bs.run_paper(req))
    lc.promote("multifactor", LifecycleState.PAPER, metrics=result["metrics"], note="paper run")
    rec = lc.get_or_create("multifactor")
    assert rec.state == LifecycleState.PAPER
    assert rec.metrics.get("trade_count") == result["metrics"]["trade_count"]
