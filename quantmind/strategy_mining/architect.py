"""策略架构师：因子 → StrategySpec。"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from ..ai.provider import LLMProvider
from .prompts import STRATEGY_ARCHITECT_SYSTEM, build_architect_prompt
from .schema import StrategySpec, StrategyTemplateType

_logger = logging.getLogger("quantmind.strategy_mining")


async def design_strategy(
    factors: List[Dict[str, Any]],
    constraint: Optional[str] = None,
    template_preference: Optional[str] = None,
    provider: Optional[LLMProvider] = None,
    symbol: str = "rb0",
    exchange: str = "SHFE",
) -> StrategySpec:
    """设计策略规格。

    Args:
        factors: 因子列表（含指标）
        constraint: 用户约束
        template_preference: 模板偏好
        provider: LLM 提供者（可选，无则用 mock）
        symbol: 交易标的
        exchange: 交易所

    Returns:
        StrategySpec 实例
    """
    # 尝试 LLM 设计
    if provider is not None:
        try:
            spec = await _llm_design(
                factors, constraint, template_preference, provider, symbol, exchange
            )
            return spec
        except Exception as e:
            _logger.warning(f"LLM 设计失败，回退到启发式：{e}")

    # 回退到启发式
    spec_dict = _mock_strategy_architect(factors, constraint, symbol, exchange)
    return StrategySpec.from_dict(spec_dict)


async def _llm_design(
    factors: List[Dict[str, Any]],
    constraint: Optional[str],
    template_preference: Optional[str],
    provider: LLMProvider,
    symbol: str,
    exchange: str,
) -> StrategySpec:
    """使用 LLM 设计策略。"""
    prompt = build_architect_prompt(factors, constraint, template_preference)
    response = await provider.chat(STRATEGY_ARCHITECT_SYSTEM, prompt)

    # 解析 JSON
    try:
        # 尝试提取 JSON（LLM 可能返回 markdown 包裹）
        json_str = response
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0]

        spec_dict = json.loads(json_str.strip())

        # 补充 symbol/exchange（如果 LLM 未提供）
        if "symbol" not in spec_dict:
            spec_dict["symbol"] = symbol
        if "exchange" not in spec_dict:
            spec_dict["exchange"] = exchange

        return StrategySpec.from_dict(spec_dict)
    except Exception as e:
        raise ValueError(f"无法解析 LLM 输出为 JSON: {e}")


def _mock_strategy_architect(
    factors: List[Dict[str, Any]],
    constraint: Optional[str],
    symbol: str,
    exchange: str,
) -> Dict[str, Any]:
    """启发式策略架构（Mock 实现）。

    基于因子类型和数量的规则匹配：
    - 单动量因子 → dual_ma
    - 多因子且 ICIR > 0.3 → multifactor（ICIR 加权）
    - 波动率 + 动量 → vol_target
    - 默认 → multifactor
    """
    if not factors:
        # 无因子，返回默认 dual_ma
        return {
            "name": "dual_ma_default",
            "template": "dual_ma",
            "description": "默认双均线策略",
            "factors": [],
            "params": {"fast": 5, "slow": 20},
            "risk": {"stop_loss": 0.05, "take_profit": 0.15, "max_position": 1.0},
            "symbol": symbol,
            "exchange": exchange,
            "capital": 1_000_000.0,
            "rationale": "无因子输入，使用默认双均线策略",
        }

    # 分析因子特征
    momentum_factors = [f for f in factors if f.get("kind") == "momentum"]
    reversion_factors = [f for f in factors if f.get("kind") == "mean_reversion"]
    vol_factors = [f for f in factors if f.get("kind") == "volatility"]

    # 决策逻辑
    if len(factors) == 1 and momentum_factors:
        # 单动量因子 → dual_ma
        f = momentum_factors[0]
        window = f.get("window", 20)
        return {
            "name": f"dual_ma_{window}",
            "template": "dual_ma",
            "description": f"基于 {window} 日动量的双均线策略",
            "factors": [],
            "params": {"fast": max(3, window // 4), "slow": window},
            "risk": {"stop_loss": 0.05, "take_profit": 0.15, "max_position": 1.0},
            "symbol": symbol,
            "exchange": exchange,
            "capital": 1_000_000.0,
            "rationale": f"单因子动量策略，使用 {window} 日窗口",
        }

    elif len(factors) >= 2 and all(f.get("icir", 0) > 0.3 for f in factors):
        # 多因子且 ICIR 都较好 → multifactor（ICIR 加权）
        factor_specs = [
            {
                "name": f["name"],
                "kind": f.get("kind", "momentum"),
                "window": f.get("window", 20),
                "weight": f.get("icir", 1.0),  # 按 ICIR 加权
                "icir": f.get("icir", 0.0),
            }
            for f in factors
        ]
        return {
            "name": "multifactor_combo",
            "template": "multifactor",
            "description": f"多因子组合策略（{len(factors)} 个因子）",
            "factors": factor_specs,
            "params": {"threshold": 0.3},
            "risk": {"stop_loss": 0.05, "take_profit": 0.15, "max_position": 1.0},
            "symbol": symbol,
            "exchange": exchange,
            "capital": 1_000_000.0,
            "rationale": "多因子 ICIR 加权组合，阈值 0.3",
        }

    elif vol_factors and momentum_factors:
        # 波动率 + 动量 → vol_target
        mom_window = momentum_factors[0].get("window", 60)
        return {
            "name": "vol_target_momentum",
            "template": "vol_target",
            "description": "波动率目标 + 动量过滤策略",
            "factors": [],
            "params": {
                "lookback": 20,
                "target_vol": 0.20,
                "momentum_win": mom_window,
            },
            "risk": {"stop_loss": 0.05, "take_profit": 0.15, "max_position": 1.0},
            "symbol": symbol,
            "exchange": exchange,
            "capital": 1_000_000.0,
            "rationale": f"波动率目标策略，动量窗口 {mom_window}",
        }

    else:
        # 默认 → multifactor
        factor_specs = [
            {
                "name": f["name"],
                "kind": f.get("kind", "momentum"),
                "window": f.get("window", 20),
                "weight": 1.0,
                "icir": f.get("icir", 0.0),
            }
            for f in factors
        ]
        return {
            "name": "multifactor_default",
            "template": "multifactor",
            "description": "多因子策略（默认配置）",
            "factors": factor_specs,
            "params": {"threshold": 0.3},
            "risk": {"stop_loss": 0.05, "take_profit": 0.15, "max_position": 1.0},
            "symbol": symbol,
            "exchange": exchange,
            "capital": 1_000_000.0,
            "rationale": "默认多因子策略",
        }
