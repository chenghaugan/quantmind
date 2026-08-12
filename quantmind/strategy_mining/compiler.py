"""确定性编译：StrategySpec → 策略实例。"""
from __future__ import annotations

import logging
from typing import Any, Dict, Tuple, Type

from ..research.target import FactorSpec
from ..strategy.allweather import VolTargetStrategy
from ..strategy.base import CtaTemplate
from ..strategy.dual_ma import DualMaStrategy
from ..strategy.multifactor import MultiFactorStrategy
from ..strategy.pair import PairTradingStrategy
from .schema import StrategySpec, StrategyTemplateType, validate_spec

_logger = logging.getLogger("quantmind.strategy_mining")

# 模板注册表
TEMPLATE_REGISTRY: Dict[StrategyTemplateType, Type[CtaTemplate]] = {
    StrategyTemplateType.DUAL_MA: DualMaStrategy,
    StrategyTemplateType.MULTIFACTOR: MultiFactorStrategy,
    StrategyTemplateType.VOL_TARGET: VolTargetStrategy,
    StrategyTemplateType.PAIR_TRADING: PairTradingStrategy,
}


def compile_strategy(spec: StrategySpec) -> CtaTemplate:
    """将 StrategySpec 确定性编译为策略实例。

    这是关键桥接：JSON 规格 → 可执行策略。执行路径中无 LLM 生成代码。

    Args:
        spec: 策略规格

    Returns:
        策略实例（可直接用于回测）

    Raises:
        ValueError: 规格无效或模板不存在
    """
    # 验证规格
    is_valid, errors = validate_spec(spec)
    if not is_valid:
        raise ValueError(f"StrategySpec 验证失败：{'; '.join(errors)}")

    # 获取模板类
    template_cls = TEMPLATE_REGISTRY.get(spec.template)
    if template_cls is None:
        raise ValueError(f"未知模板：{spec.template}")

    # 构建 setting
    setting = _build_setting(spec)

    # 实例化策略（context=None，由 BacktestEngine 设置）
    strategy = template_cls(context=None, setting=setting)

    _logger.info(f"策略编译成功：{spec.name} ({spec.template.value})")
    return strategy


def compile_and_validate(spec: StrategySpec) -> Tuple[bool, str, CtaTemplate | None]:
    """编译策略并捕获异常。

    Returns:
        (success, error_message, strategy_instance)
    """
    try:
        strategy = compile_strategy(spec)
        return True, "", strategy
    except Exception as e:
        _logger.error(f"策略编译失败：{e}")
        return False, str(e), None


def _build_setting(spec: StrategySpec) -> Dict[str, Any]:
    """从 StrategySpec 构建策略 setting 字典。

    将 spec.params 映射为模板特定参数。
    """
    setting: Dict[str, Any] = {}

    if spec.template == StrategyTemplateType.DUAL_MA:
        setting["fast"] = spec.params.get("fast", 5)
        setting["slow"] = spec.params.get("slow", 20)
        setting["size"] = 1
        setting["max_pos"] = spec.risk.max_position

    elif spec.template == StrategyTemplateType.MULTIFACTOR:
        # 将 FactorInput 转换为 FactorSpec
        factor_specs = [
            FactorSpec(
                name=f.name,
                kind=f.kind,
                window=f.window,
                weight=f.weight,
                icir=f.icir,
                expression=f.expression,
            )
            for f in spec.factors
        ]
        setting["specs"] = factor_specs
        setting["threshold"] = spec.params.get("threshold", 0.3)
        setting["size"] = 1
        setting["max_pos"] = spec.risk.max_position

    elif spec.template == StrategyTemplateType.VOL_TARGET:
        setting["lookback"] = spec.params.get("lookback", 20)
        setting["target_vol"] = spec.params.get("target_vol", 0.20)
        setting["momentum_win"] = spec.params.get("momentum_win", 60)
        setting["size"] = 1
        setting["max_pos"] = spec.risk.max_position

    elif spec.template == StrategyTemplateType.PAIR_TRADING:
        setting["window"] = spec.params.get("window", 30)
        setting["entry_z"] = spec.params.get("entry_z", 1.5)
        setting["exit_z"] = spec.params.get("exit_z", 0.3)
        setting["size"] = 1
        setting["max_pos"] = spec.risk.max_position

    return setting
