"""BacktestService 策略注册/装载：fence 剥离 + 挖掘策略入池/列表展示（耐重启）。

覆盖目标：端到端挖掘的策略持久化于 knowledge.db，应能被 list_strategies 展示、
可带 Markdown 围栏注册进运行池。
"""
from __future__ import annotations

import pytest

from quantmind.api.services.backtest_service import (
    BacktestService,
    _strip_code_fences,
    _strategy_class_name,
)

FENCED = (
    "```python\n"
    "from quantmind.strategy import CtaTemplate\n"
    "class FooStrategy(CtaTemplate):\n    pass\n"
    "```"
)


def test_strip_code_fences():
    clean = _strip_code_fences(FENCED)
    assert clean.startswith("from quantmind.strategy")
    assert "<" not in clean and "```" not in clean
    # 无围栏时原样保留
    assert _strip_code_fences("x = 1") == "x = 1"
    assert _strip_code_fences("") == ""


def test_strategy_class_name():
    assert _strategy_class_name(FENCED) == "FooStrategy"
    assert _strategy_class_name("class Bar:\n  pass") == "Bar"
    assert _strategy_class_name("") == ""


def test_register_generated_strategy_strips_fences():
    bs = BacktestService.__new__(BacktestService)
    bs._extra_strategies = {}
    ok, err, info = bs.register_generated_strategy("FooStrategy", FENCED)
    assert ok, err
    assert "FooStrategy" in bs._extra_strategies
    assert info["name"] == "FooStrategy"


def test_list_strategies_surfaces_mined_and_builtin_without_crash():
    bs = BacktestService.__new__(BacktestService)
    bs._extra_strategies = {"FooStrategy": object}
    lst = bs.list_strategies()
    names = {s.name for s in lst}
    # 内置策略仍在
    assert "multifactor" in names
    assert "dual_ma" in names
    # 运行池内的注册策略出现在列表中
    assert "FooStrategy" in names
    # knowledge.db 中挖掘的策略也被展示（非空时），且全程不抛异常
    for s in lst:
        assert s.name and hasattr(s, "description") and hasattr(s, "parameters")


def test_backtest_service_registers_mined_strategies():
    """BacktestService 启动即把规范适配策略注册进运行池，且按名可解析。"""
    from quantmind.api.services.backtest_service import BacktestService
    from quantmind.data import DataManager
    from quantmind.core.engine import EventEngine
    from quantmind.strategy.mined import MINED_STRATEGIES

    dm = DataManager.__new__(DataManager)
    ee = EventEngine()
    bs = BacktestService(dm, ee)
    pool = set(bs._extra_strategies)
    # 全部规范策略已注册
    for name in MINED_STRATEGIES:
        assert name in pool, f"{name} 未注册进运行池"
    # 按名解析命中
    for name in MINED_STRATEGIES:
        assert bs._resolve_strategy_class(name) is not None, f"resolve({name}) 失败"


@pytest.mark.asyncio
async def test_mined_strategies_backtest_run():
    """每个规范适配策略都能在回测引擎上真实运行（实例化 + on_bar 产出目标仓位）。"""
    from quantmind.backtest import BacktestEngine
    from quantmind.core.contracts import default_size
    from quantmind.strategy.mined import MINED_STRATEGIES
    from tests.helpers import load_bars

    bars = await load_bars()
    vt = "rb0.SHFE"
    for name, strat_cls in MINED_STRATEGIES.items():
        eng = BacktestEngine({vt: bars}, capital=1_000_000, sizes={vt: default_size(vt)})
        eng.add_strategy(strat_cls, vt, {"size": default_size(vt), "max_pos": 1.0})
        rep = eng.run()
        assert len(rep.equity_curve) == len(bars), f"{name} 回测长度不符"
        assert rep.final_equity > 0, f"{name} 权益异常"


def test_optimize_space_supports_mined_strategies():
    """参数优化后端为 AI 挖掘策略提供可用搜索空间并可解析（不抛未知策略）。"""
    from quantmind.api.services.optimize_service import OptimizeService
    from quantmind.strategy.mined import MINED_STRATEGIES

    svc = OptimizeService(dm=None)
    for name in MINED_STRATEGIES:
        space = svc.param_space_of(name)
        assert isinstance(space, dict) and space, f"{name} 应返回非空参数空间"
        # 真正解析出类（不再是恒真断言）
        assert svc._resolve(name) is not None  # noqa: SLF001
        assert set(space) <= set(svc._resolve(name).parameters)  # noqa: SLF001


def test_multifactor_param_space_targets_effective_params():
    """multifactor 网格应搜索真正影响结果的 size/max_pos，而非无效的 lookback/top_n。"""
    from quantmind.api.services.optimize_service import (
        OptimizeService, DEFAULT_PARAM_SPACE,
    )

    space = DEFAULT_PARAM_SPACE["multifactor"]
    assert "lookback" not in space and "top_n" not in space
    assert "size" in space and "max_pos" in space
    assert OptimizeService(dm=None).param_space_of("multifactor") == space


def test_optimize_uses_injected_resolver():
    """注入的解析器被 /optimize 路径采用（解析 knowledge.db 已沉淀策略的机制，①）。"""
    from quantmind.api.services.optimize_service import OptimizeService
    from quantmind.strategy.mined import MINED_STRATEGIES

    cls = next(iter(MINED_STRATEGIES.values()))
    svc = OptimizeService(dm=None, resolver=lambda n: cls)
    assert svc._resolve("任何名字") is cls            # noqa: SLF001 注入生效
    space = svc.param_space_of("任何名字")
    assert space and set(space) <= set(cls.parameters)
    # 未注入时默认解析器仍能解析规范挖掘类（回退安全）
    assert OptimizeService(dm=None)._resolve(next(iter(MINED_STRATEGIES))) is not None  # noqa: SLF001
