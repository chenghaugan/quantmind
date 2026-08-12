"""投资者偏好画像（Profile）。

定义不同风险偏好的投资者配置，影响选股、风控、仓位等参数。
参考 aifa-quant 的 5 种 Profile 设计。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class InvestorProfile:
    """投资者偏好画像。"""

    id: str
    name: str                          # 中文名
    description: str = ""

    # 选股参数
    top_k: int = 20                    # 持仓数
    max_industry_pct: float = 0.25     # 单行业上限

    # 因子权重偏好 (factor_group → weight_multiplier)
    factor_weights: Dict[str, float] = field(default_factory=lambda: {
        "momentum": 1.0, "value": 1.0, "quality": 1.0,
        "alpha": 1.0, "low_volatility": 1.0, "volume": 1.0,
    })

    # 风控参数
    risk_profile: str = "default"      # default/conservative/unlimited
    regime_ma_threshold: float = 0.95  # 熊市过滤阈值（MA20/MA60）
    enable_regime_filter: bool = True

    # 仓位参数
    target_risk_pct: float = 0.02      # 组合目标风险
    max_position_pct: float = 0.10     # 单标的最大仓位

    # 回测参数
    benchmark: str = "000300.SH"       # 基准指数

    def to_dict(self) -> dict:
        """转为字典（用于 JSON 序列化）。"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "top_k": self.top_k,
            "max_industry_pct": self.max_industry_pct,
            "factor_weights": self.factor_weights,
            "risk_profile": self.risk_profile,
            "regime_ma_threshold": self.regime_ma_threshold,
            "enable_regime_filter": self.enable_regime_filter,
            "target_risk_pct": self.target_risk_pct,
            "max_position_pct": self.max_position_pct,
            "benchmark": self.benchmark,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "InvestorProfile":
        """从字典构造。"""
        return cls(**data)


# ============================================================
# 预定义 5 种 Profile
# ============================================================

PRESET_PROFILES: Dict[str, InvestorProfile] = {
    "aggressive": InvestorProfile(
        id="aggressive",
        name="激进型",
        description="高集中度，追求超额收益",
        top_k=15,
        max_industry_pct=0.35,
        factor_weights={
            "momentum": 1.5, "alpha": 1.3, "value": 0.5,
            "quality": 0.8, "low_volatility": 0.5, "volume": 1.2,
        },
        risk_profile="default",
        regime_ma_threshold=0.0,
        enable_regime_filter=False,
        target_risk_pct=0.03,
        max_position_pct=0.15,
        benchmark="000300.SH",
    ),
    "balanced": InvestorProfile(
        id="balanced",
        name="均衡型",
        description="攻守兼备，适合大多数人",
        top_k=20,
        max_industry_pct=0.25,
        factor_weights={
            "momentum": 1.0, "alpha": 1.0, "value": 1.0,
            "quality": 1.0, "low_volatility": 1.0, "volume": 1.0,
        },
        risk_profile="default",
        regime_ma_threshold=0.95,
        enable_regime_filter=True,
        target_risk_pct=0.02,
        max_position_pct=0.10,
        benchmark="000300.SH",
    ),
    "conservative": InvestorProfile(
        id="conservative",
        name="稳健型",
        description="充分分散，严控回撤",
        top_k=30,
        max_industry_pct=0.20,
        factor_weights={
            "momentum": 0.5, "alpha": 0.8, "value": 1.2,
            "quality": 1.3, "low_volatility": 1.5, "volume": 0.8,
        },
        risk_profile="conservative",
        regime_ma_threshold=0.97,
        enable_regime_filter=True,
        target_risk_pct=0.015,
        max_position_pct=0.05,
        benchmark="000300.SH",
    ),
    "growth": InvestorProfile(
        id="growth",
        name="成长型",
        description="聚焦高成长潜力股",
        top_k=20,
        max_industry_pct=0.30,
        factor_weights={
            "momentum": 1.2, "alpha": 1.2, "value": 0.6,
            "quality": 1.5, "low_volatility": 0.7, "volume": 1.0,
        },
        risk_profile="default",
        regime_ma_threshold=0.95,
        enable_regime_filter=True,
        target_risk_pct=0.025,
        max_position_pct=0.12,
        benchmark="000300.SH",
    ),
    "value": InvestorProfile(
        id="value",
        name="价值型",
        description="低估值选股，安全边际优先",
        top_k=25,
        max_industry_pct=0.25,
        factor_weights={
            "momentum": 0.5, "alpha": 0.8, "value": 1.8,
            "quality": 1.2, "low_volatility": 1.0, "volume": 0.6,
        },
        risk_profile="default",
        regime_ma_threshold=0.93,
        enable_regime_filter=True,
        target_risk_pct=0.02,
        max_position_pct=0.08,
        benchmark="000300.SH",
    ),
}


__all__ = ["InvestorProfile", "PRESET_PROFILES"]
