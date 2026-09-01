"""历史 AI「挖掘策略」的真实可运行适配。

背景：knowledge.db 里沉淀的 6 条端到端挖掘策略由旧版（假 API）codegen 生成，
导入了不存在的模块（``quantmind.factor`` / ``quantmind.signal`` 等），无法 `exec` 运行。
本模块把它们的**意图**重写为当前真实框架的 `CtaTemplate`/`MultiFactorStrategy` 子类，
使「策略回测 / 模拟 / 实盘」能真正实例化并运行。

- 多因子类：沿用 ``MultiFactorStrategy``（FactorSpec 组合 → 目标仓位），仅覆盖因子组合。
- Chan 类：用 ``ArrayManager`` 实现突破/回踩近似。
类名与 DB 展示名一致，供 ``BacktestService._load_persisted_strategies`` /
``_resolve_strategy_class`` 按名解析并注册进运行池。
"""
from __future__ import annotations

from typing import Dict, Type

from ..core.object import BarData
from ..core.utility import ArrayManager
from .base import CtaTemplate
from .multifactor import MultiFactorStrategy
from ..research.target import FactorSpec


class RebarMomentumStrategy(MultiFactorStrategy):
    """AI 挖掘·适配：螺纹钢动量策略（回补短期动量）。"""

    author = "QuantMind（AI挖掘·适配）"
    parameters = ["threshold", "size", "max_pos"]

    def __init__(self, context, setting=None):
        setting = dict(setting or {})
        setting.setdefault("threshold", 0.5)
        super().__init__(context, setting)
        self.specs = [FactorSpec(name="momentum_20", kind="momentum", window=20, weight=1.0)]


class RebarMomentumTermStructureStrategy(MultiFactorStrategy):
    """AI 挖掘·适配：动量 + 期限结构（→动量 + 长短动量价差）。"""

    author = "QuantMind（AI挖掘·适配）"
    parameters = ["threshold", "size", "max_pos"]

    def __init__(self, context, setting=None):
        setting = dict(setting or {})
        setting.setdefault("threshold", 0.5)
        super().__init__(context, setting)
        self.specs = [
            FactorSpec(name="momentum_20", kind="momentum", window=20, weight=0.6),
            FactorSpec(name="term_10", kind="term_structure", window=10, weight=0.4),
        ]


class MomentumTermStructureStrategy(MultiFactorStrategy):
    """AI 挖掘·适配：动量 + 期限结构 + 波动率（→低波增强）。"""

    author = "QuantMind（AI挖掘·适配）"
    parameters = ["threshold", "size", "max_pos"]

    def __init__(self, context, setting=None):
        setting = dict(setting or {})
        setting.setdefault("threshold", 0.5)
        super().__init__(context, setting)
        self.specs = [
            FactorSpec(name="momentum_20", kind="momentum", window=20, weight=1.0),
            FactorSpec(name="term_10", kind="term_structure", window=10, weight=0.4),
            FactorSpec(name="vol_20", kind="volatility", window=20, weight=-0.3),
        ]


class SteelMomentumTermStructureStrategy(MultiFactorStrategy):
    """AI 挖掘·适配：钢材动量 + 期限结构 + 量能确认。"""

    author = "QuantMind（AI挖掘·适配）"
    parameters = ["threshold", "size", "max_pos"]

    def __init__(self, context, setting=None):
        setting = dict(setting or {})
        setting.setdefault("threshold", 0.5)
        super().__init__(context, setting)
        self.specs = [
            FactorSpec(name="momentum_20", kind="momentum", window=20, weight=1.0),
            FactorSpec(name="term_10", kind="term_structure", window=10, weight=0.4),
            FactorSpec(name="vol_chg_20", kind="volume_change", window=20, weight=0.3),
        ]


class ChanThirdBuyStrategy(CtaTemplate):
    """AI 挖掘·适配：缠论第三类买点（忠实实现）。

    以「中枢」为核心：近 ``break_window`` 高点上沿为中枢上沿 ``ZG``（回抽不破的界限）。
    规则：上行趋势（收盘 > 均线）且收盘站在中枢上沿 ZG 上方（房价回抽不重新进入中枢区间）
    → 持多；一旦收盘跌回 ZG 之下（中枢破坏）或趋势转弱 → 空仓。
    相较旧版（要求每次收盘都 >= 前高，回踩即出）更忠实「回抽不破 ZG」的延续性。
    """

    author = "QuantMind（AI挖掘·适配）"
    parameters = ["trend_window", "break_window", "size", "max_pos"]

    def __init__(self, context, setting=None):
        self.trend_window = 60
        self.break_window = 20
        self.size = 1
        self.max_pos = 1.0
        # ArrayManager 缓冲按**应用 settings 后**的窗口惰性构建（修复旧版按默认值定长导致
        # 用户调大窗口后 `len(closes)<window` 恒真、策略永不交易的静默 bug）。
        self.am = None
        self.last_target = 0.0
        super().__init__(context, setting)

    def on_bar(self, bar: BarData) -> None:
        if self.am is None:
            self.am = ArrayManager(max(self.trend_window, self.break_window) + 5)
        self.am.update_bar(bar)
        if not self.am.inited:
            return
        closes = self.am.close
        highs = self.am.high
        if len(closes) < self.break_window or len(closes) < self.trend_window:
            return
        # 中枢上沿 ZG：截至**上一根**的近 break_window 高点上沿（回抽不破的界限）。
        # 不能含当根 high：close<=high 恒成立，含当根会使 last>=zg 等价于 close==当根high，
        # 持有条件近乎不可满足。
        zg = max(highs[-self.break_window - 1:-1])
        trend = sum(closes[-self.trend_window:]) / self.trend_window
        last = closes[-1]
        # 三买持有：上行趋势 且 收盘站在 ZG 上方（回抽不重新进入中枢区间）。
        hold = last > trend and last >= zg
        target_vol = self.max_pos * self.size if hold else 0.0
        if target_vol != self.last_target:
            oid = self.set_target(bar.vt_symbol, target_vol)
            if oid == "":
                # 风控拒单：保留 last_target，下一根 bar 重试
                return
            self.last_target = target_vol
            self.pos = target_vol


# 按类名导出，供 BacktestService 运行池注册 / 惰性解析命中
MINED_STRATEGIES: Dict[str, Type[CtaTemplate]] = {
    cls.__name__: cls
    for cls in (
        RebarMomentumStrategy,
        RebarMomentumTermStructureStrategy,
        MomentumTermStructureStrategy,
        SteelMomentumTermStructureStrategy,
        ChanThirdBuyStrategy,
    )
}
