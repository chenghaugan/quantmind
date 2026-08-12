"""策略挖掘数据模型：StrategySpec 及验证逻辑。"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class StrategyTemplateType(str, Enum):
    """支持的策略模板类型。"""

    DUAL_MA = "dual_ma"
    MULTIFACTOR = "multifactor"
    VOL_TARGET = "vol_target"
    PAIR_TRADING = "pair_trading"


@dataclass
class FactorInput:
    """从因子库选择的因子输入。"""

    name: str
    kind: str = "momentum"
    window: int = 20
    weight: float = 1.0
    icir: float = 0.0
    expression: Optional[str] = None


@dataclass
class RiskParams:
    """风险管理参数。"""

    stop_loss: float = 0.05
    take_profit: float = 0.15
    max_position: float = 1.0


@dataclass
class StrategySpec:
    """策略规格（LLM 输出 → 确定性编译）。

    这是 LLM 与执行引擎之间的契约：纯 JSON 可序列化规格，执行路径中无 LLM 生成代码。
    """

    name: str
    template: StrategyTemplateType
    description: str = ""
    factors: List[FactorInput] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    risk: RiskParams = field(default_factory=RiskParams)
    symbol: str = "rb0"
    exchange: str = "SHFE"
    capital: float = 1_000_000.0
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """序列化为 JSON 兼容字典。"""
        return {
            "name": self.name,
            "template": self.template.value,
            "description": self.description,
            "factors": [
                {
                    "name": f.name,
                    "kind": f.kind,
                    "window": f.window,
                    "weight": f.weight,
                    "icir": f.icir,
                    "expression": f.expression,
                }
                for f in self.factors
            ],
            "params": self.params,
            "risk": {
                "stop_loss": self.risk.stop_loss,
                "take_profit": self.risk.take_profit,
                "max_position": self.risk.max_position,
            },
            "symbol": self.symbol,
            "exchange": self.exchange,
            "capital": self.capital,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StrategySpec:
        """从字典反序列化。"""
        factors = [
            FactorInput(
                name=f["name"],
                kind=f.get("kind", "momentum"),
                window=f.get("window", 20),
                weight=f.get("weight", 1.0),
                icir=f.get("icir", 0.0),
                expression=f.get("expression"),
            )
            for f in data.get("factors", [])
        ]
        risk_data = data.get("risk", {})
        risk = RiskParams(
            stop_loss=risk_data.get("stop_loss", 0.05),
            take_profit=risk_data.get("take_profit", 0.15),
            max_position=risk_data.get("max_position", 1.0),
        )
        return cls(
            name=data["name"],
            template=StrategyTemplateType(data["template"]),
            description=data.get("description", ""),
            factors=factors,
            params=data.get("params", {}),
            risk=risk,
            symbol=data.get("symbol", "rb0"),
            exchange=data.get("exchange", "SHFE"),
            capital=data.get("capital", 1_000_000.0),
            rationale=data.get("rationale", ""),
        )


def validate_spec(spec: StrategySpec) -> tuple[bool, List[str]]:
    """验证 StrategySpec 是否符合模板要求。

    Returns:
        (is_valid, error_messages)
    """
    errors: List[str] = []

    # 按模板校验必填参数
    if spec.template == StrategyTemplateType.DUAL_MA:
        if "fast" not in spec.params or "slow" not in spec.params:
            errors.append("dual_ma 模板需要 'fast' 和 'slow' 参数")
        elif spec.params["fast"] >= spec.params["slow"]:
            errors.append("快线窗口必须小于慢线窗口")

    elif spec.template == StrategyTemplateType.MULTIFACTOR:
        if not spec.factors:
            errors.append("multifactor 模板需要至少一个因子")
        if "threshold" not in spec.params:
            errors.append("multifactor 模板需要 'threshold' 参数")

    elif spec.template == StrategyTemplateType.VOL_TARGET:
        required = ["lookback", "target_vol", "momentum_win"]
        for p in required:
            if p not in spec.params:
                errors.append(f"vol_target 模板需要 '{p}' 参数")

    elif spec.template == StrategyTemplateType.PAIR_TRADING:
        required = ["window", "entry_z", "exit_z"]
        for p in required:
            if p not in spec.params:
                errors.append(f"pair_trading 模板需要 '{p}' 参数")

    # 风险参数校验
    if spec.risk.stop_loss <= 0:
        errors.append("止损比例必须为正")
    if spec.risk.max_position <= 0 or spec.risk.max_position > 1:
        errors.append("最大仓位必须在 (0, 1] 范围内")

    return len(errors) == 0, errors
