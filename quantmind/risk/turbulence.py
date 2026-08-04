"""Turbulence 市场状态检测（马氏距离，借鉴 FinRL）+ 风控适配。

思路：用**马氏距离**衡量当前点相对历史收益分布有多"极端"。turbulence 越高，
市场越偏离常态 → 建议降低敞口（HIGH）甚至清仓暂停（EXTREME）。

实现用纯 numpy（马氏距离需要协方差求逆，用 ``np.linalg.pinv`` 处理奇异），
不依赖 scipy。阈值默认用历史分布的经验分位数（可构造参数覆盖）。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from .limits import RiskCode, RiskDecision

_logger = logging.getLogger("quantmind.risk.turbulence")


class Regime(str, Enum):
    """市场状态。"""

    LOW = "LOW"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


@dataclass
class TurbulenceConfig:
    """Turbulence 检测配置。"""

    lookback: int = 100                  # 历史窗口（用于估计均值/协方差）
    window: int = 1                      # 当前评估窗口（通常 1 根）
    quantile_hi: float = 0.95            # 超过该分位 → HIGH
    quantile_extreme: float = 0.99       # 超过该分位 → EXTREME
    # 各状态下建议的敞口缩放系数
    scale_low: float = 1.0
    scale_high: float = 0.5
    scale_extreme: float = 0.0


class TurbulenceDetector:
    """Turbulence 检测器：输出当前 market regime 与建议缩放。"""

    def __init__(self, config: Optional[TurbulenceConfig] = None) -> None:
        self.config = config or TurbulenceConfig()
        self._history: Optional[np.ndarray] = None
        self._threshold_hi: Optional[float] = None
        self._threshold_extreme: Optional[float] = None

    def _fit_usage(self, hist: np.ndarray) -> None:
        """根据历史分布估计阈值（经验分位数）。"""
        q_hi = float(np.quantile(hist, self.config.quantile_hi))
        q_ext = float(np.quantile(hist, self.config.quantile_extreme))
        self._threshold_hi = max(q_hi, 1.0)   # 兜底：至少 1.0，避免全零序列误判
        self._threshold_extreme = max(q_ext, q_hi + 1e-9)

    def compute(self, returns: np.ndarray | pd.Series) -> float:
        """计算给定收益序列的 turbulence 时序（马氏距离）。

        :param returns: 收益序列（一维）。返回与输入等长的 turbulence 序列；
            前 ``lookback`` 个点因缺少历史无法估计 → NaN。
        """
        s = np.asarray(returns, dtype=float).ravel()
        lag = self.config.lookback
        n = len(s)
        out = np.full(n, np.nan)
        if n <= lag:
            return out

        # 用前 lookback 训练，逐点（自 lag 起）估计马氏距离
        train = s[:lag]
        mu = float(train.mean())
        cov = float(np.cov(train - mu)) if train.size > 1 else 1.0
        inv_var = 1.0 / cov if cov > 1e-12 else 1.0
        for t in range(lag, n):
            delta = s[t] - mu
            d = float(np.sqrt(delta * delta * inv_var))
            out[t] = d
        self._history = out[~np.isnan(out)]
        if len(self._history) > 0:
            self._fit_usage(self._history)
        return out

    def get_regime(self, turbulence: float) -> Regime:
        """依据 turbulence 判定市场状态。"""
        if not np.isfinite(turbulence):
            return Regime.LOW
        if self._threshold_extreme is not None and turbulence > self._threshold_extreme:
            return Regime.EXTREME
        if self._threshold_hi is not None and turbulence > self._threshold_hi:
            return Regime.HIGH
        return Regime.LOW

    def suggested_scale(self, turbulence: float) -> float:
        """当前状态建议的敞口缩放（1.0=满仓, 0.5=减半, 0.0=清仓/暂停）。"""
        regime = self.get_regime(turbulence)
        cfg = self.config
        return {Regime.LOW: cfg.scale_low,
                Regime.HIGH: cfg.scale_high,
                Regime.EXTREME: cfg.scale_extreme}[regime]

    def last(self) -> dict:
        return {
            "threshold_hi": self._threshold_hi,
            "threshold_extreme": self._threshold_extreme,
            "n_history": int(len(self._history)) if self._history is not None else 0,
        }


class TurbulenceRiskAdapter:
    """把 Turbulence 检测结果适配到风控风格：给出缩放建议与拒单判定。"""

    def __init__(self, detector: Optional[TurbulenceDetector] = None,
                 config: Optional[TurbulenceConfig] = None) -> None:
        self.detector = detector or TurbulenceDetector(config=config or TurbulenceConfig())

    def check_scale(self, returns: np.ndarray | pd.Series) -> float:
        """计算当前（最后一个点）应使用的敞口缩放系数。"""
        series = self.detector.compute(returns)
        if series is None or len(series) == 0:
            return self.detector.config.scale_low
        last = series[-1]
        if not np.isfinite(last):
            return self.detector.config.scale_low
        return self.detector.suggested_scale(last)

    def decision(self, returns: np.ndarray | pd.Series, vt_symbol: str = "") -> RiskDecision:
        """EXTREME 状态时返回拒开仓的 RiskDecision；否则放行。"""
        scale = self.check_scale(returns)
        if scale <= 0:
            return RiskDecision.reject(
                RiskCode.TURBULENCE_EXTREME,
                "市场处于极端波动（Turbulence EXTREME），暂停开仓",
                vt_symbol,
            )
        return RiskDecision.ok(vt_symbol)

    def to_dict(self) -> dict:
        return self.detector.last()


__all__ = [
    "Regime",
    "TurbulenceConfig",
    "TurbulenceDetector",
    "TurbulenceRiskAdapter",
]
