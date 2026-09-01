"""ComposableStrategy：把 5 个组件装配成一个 ``CtaTemplate`` 策略。

组装顺序（组件可选，缺省用对应默认实现）：

  Universe(every bar 缓存信号) → Alpha.on_bar(bar) → 目标仓位(float)
    → [主标的 when 触发] Portfolio.apply_all(signals, universe) → 组合目标权重(dict)
    → Risk.apply(target) → Execution.set_target(vt, target)

多标的行为（M4/M5）：
  - ``UniverseModel.select`` 从引擎候选标的中筛出可交易子集（M5）；
  - ``MultiFactorMultiSymbolAlpha`` 为每个标的独立建目标序列；
  - 在主标的的 bar 上触发一次组合重平衡：把最新信号经 ``PortfolioModel.apply_all``
    聚合成组合目标权重，再逐标的过 Risk 并执行（M4）。

单标的默认装配（AllUniverse + MultiFactorMultiSymbolAlpha + IdentityPortfolio
+ NullRisk + TargetExecution）逐 bar 重平衡，行为与现有 ``MultiFactorStrategy`` 一致
（回归门槛，见 tests/test_components.py）。

可以通过两种方式注入组件：
  1. 构造/类属性：``ComposableStrategy(alpha=..., portfolio=..., universe=...)``；
  2. setting 字典（供 backtest API / CLI 传参）：``{"alpha": ..., "universe": ...}``。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from ..base import CtaTemplate
from ..context import StrategyContext
from .alpha import MultiFactorMultiSymbolAlpha
from .base import AlphaSignal
from .execution import TargetExecution
from .portfolio import IdentityPortfolio
from .risk import NullRisk
from .universe import AllUniverse


class _ComponentContext:
    """给 Alpha/Universe/Portfolio 组件用的上下文适配器：暴露 vt_symbols 并委托给引擎 context。

    引擎 context（Backtest/Paper/LiveEngine）实现了 get_history/get_position，
    但没有 ``vt_symbols``（那是策略实例上的属性）。这里补一层适配。
    """

    def __init__(self, context, vt_symbols) -> None:
        self._context = context
        self.vt_symbols = list(vt_symbols)

    def get_history(self, vt_symbol, count):
        return self._context.get_history(vt_symbol, count)

    def get_position(self, vt_symbol):
        return self._context.get_position(vt_symbol)

    def write_log(self, msg, level=20):
        self._context.write_log(msg, level)


class ComposableStrategy(CtaTemplate):
    """5 组件可组合策略：把 Universe/Alpha/Portfolio/Risk/Execution 装配成模板。"""

    author = "QuantMind"
    parameters = ["alpha", "portfolio", "risk", "execution", "universe"]

    def __init__(self, context: StrategyContext, setting: Optional[Dict[str, Any]] = None) -> None:
        # 先给默认组件，再允许 setting 覆盖
        self.alpha = MultiFactorMultiSymbolAlpha()
        self.portfolio = IdentityPortfolio()
        self.risk = NullRisk()
        self.execution = TargetExecution()
        self.universe = AllUniverse()
        super().__init__(context, setting)
        # 兼容：允许通过类属性指定组件（如子类继承后改写 default_alpha）
        if getattr(self, "default_alpha", None) is not None:
            self.alpha = self.default_alpha
        if getattr(self, "default_risk", None) is not None:
            self.risk = self.default_risk
        if getattr(self, "default_universe", None) is not None:
            self.universe = self.default_universe
        # 把策略级参数（size/max_pos/threshold 等）同步进 Alpha 组件，
        # 使 ComposableStrategy 与单体模板用同一套 parameter/setting 语义。
        for _p in ("size", "max_pos", "threshold", "fast", "slow"):
            if hasattr(self, _p) and hasattr(self.alpha, _p):
                setattr(self.alpha, _p, getattr(self, _p))

    # ---- 内部状态 ----
    @property
    def universe_symbols(self) -> list:
        """当前选中的标的池（M5）。"""
        return getattr(self, "_universe", list(getattr(self, "vt_symbols", [])))

    def on_init(self) -> None:
        # 绑定执行上下文（引擎在 on_init 前已注入 context）
        if hasattr(self.execution, "bind"):
            self.execution.bind(self.context)
        # 组件上下文：暴露全部候选标的
        self._ctx = _ComponentContext(self.context, self.vt_symbols)
        # M5：从引擎候选标的中选出可交易子集
        candidates = list(self.vt_symbols) or []
        if hasattr(self.universe, "select"):
            self._universe = self.universe.select(candidates, self._ctx) or list(candidates)
        else:
            self._universe = list(candidates)
        # 主标的（重平衡触发点）：选中池第一个；为空则退化为候选第一
        self._primary = self._universe[0] if self._universe else (
            self.vt_symbols[0] if self.vt_symbols else "")
        # 逐标的信号缓存
        self._signals: Dict[str, float] = {}
        # Alpha：为每个标的独立建模型/目标序列
        if hasattr(self.alpha, "on_init"):
            self.alpha.on_init(self._ctx)
        self.context.write_log(
            f"组合策略就绪: universe={self._universe} primary={self._primary} "
            f"alpha={type(self.alpha).__name__} "
            f"portfolio={type(self.portfolio).__name__} "
            f"risk={type(self.risk).__name__}")
        self.inited = True

    def on_bar(self, bar) -> None:
        vt = bar.vt_symbol or f"{bar.symbol}.{bar.exchange.value}"

        # 1. Alpha：产出该标的目标仓位（None = 该标的本次不变）
        target = self.alpha.on_bar(bar)
        if target is not None:
            self._signals[vt] = target

        # 仅主标的的 bar 触发组合重平衡（多标的下每日期一次；单标的逐 bar）
        if not self._primary or vt != self._primary:
            return

        # 2. Portfolio：把最新信号聚合成组合目标权重（M4）
        targets = self.portfolio.apply_all(self._signals, self._universe, self.context)

        # 3&4. Risk 过滤 + Execution 逐标的执行
        primary_target = None
        for sym, tgt in targets.items():
            # 风控闸门按目标标的检查（非主标的传 vt_symbol，避免用主标的持仓/价格误判）
            vt_arg = sym if sym != (bar.vt_symbol or "") else None
            final_target = self.risk.apply(tgt, bar, self.context, vt_symbol=vt_arg)
            if final_target is None:
                continue
            self.execution.set_target(sym, float(final_target))
            if sym == self._primary:
                primary_target = float(final_target)

        # 主标的展示值仅在风控放行后更新（拒绝时不虚报仓位）
        if primary_target is not None:
            self.pos = primary_target

    # ---- 让 CtaTemplate.set_target 也可用（可选执行路径）----
    def _set_target_direct(self, vt, target):
        self.context.set_target(vt, target)


__all__ = ["ComposableStrategy"]
