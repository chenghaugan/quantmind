"""多因子组合：把若干因子标准化后加权合成复合信号，并映射为目标仓位。

对应 5 组件框架的 ``Alpha -> Portfolio``：Alpha 产出原始 alpha 分数，
Portfolio 据此设定目标仓位（含风险缩放）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..core.object import BarData
from .factors.base import Factor, bars_to_df, expanding_zscore
from .factors.expression import eval_factor_expression
from .neutralize import orthogonalize_factors, winsorize


@dataclass
class FactorSpec:
    """因子规格（可由 AI/表达式生成）。"""

    name: str
    kind: str = "momentum"          # 内置因子类型（见 technical._FACTOR_CLASSES）
    window: int = 20
    expression: Optional[str] = None  # 若提供，优先用表达式 DSL 计算
    weight: float = 1.0
    icir: float = 0.0               # 信息比率（用于 ICIR 加权；<=0 视为无效）


def icir_weights(ic_means: List[float], ic_stds: List[float]) -> List[float]:
    """按 ICIR（mean/std）做非负加权；若全部无效则等权。"""
    icirs = []
    for m, s in zip(ic_means, ic_stds):
        if s and s > 0 and np.isfinite(m) and np.isfinite(s):
            icirs.append(max(m / s, 0.0))
        else:
            icirs.append(0.0)
    total = sum(icirs)
    if total <= 0:
        return [1.0 / len(ic_means)] * len(ic_means)
    return [w / total for w in icirs]


class MultiFactorModel:
    """多因子组合模型：z-score 标准化后加权合成。

    支持两种加权：
      - 固定权重（``weights`` 提供或在 FactorSpec 中给出）
      - ICIR 加权（``fit_weights_from_ics`` 用历史 IC/IR 估计）
    支持相关性去冗余（``dedup_correlated`` 删除高相关冗余因子；``combine(dedup=True)``
    在合成前对因子信号做正交化）。
    """

    def __init__(
        self,
        factors: List[Factor],
        weights: Optional[List[float]] = None,
        threshold: float = 0.5,
    ) -> None:
        self.factors = factors
        self.weights = weights or [1.0] * len(factors)
        self.threshold = threshold

    def compute_factor(self, factor: Factor, bars: List[BarData]) -> pd.Series:
        return factor.compute(bars)

    def combine(self, bars: List[BarData], dedup: bool = False) -> pd.Series:
        """返回与 bars 等长的复合信号（加权 z-score 和）。

        ``dedup=True`` 时对因子标准化信号先做正交化（去冗余）再加权。
        """
        if not self.factors:
            return pd.Series(dtype=float)
        raw_z = []
        for f in self.factors:
            raw = f.compute(bars)
            raw_z.append(expanding_zscore(raw).fillna(0.0))
        if dedup:
            names = [getattr(f, "meta", None) and f.meta.name or f"f{i}" for i, f in enumerate(self.factors)]
            raw_z = orthogonalize_factors(raw_z, names=names)
        composite = pd.Series(0.0, index=range(len(bars)))
        total_w = 0.0
        for z, w in zip(raw_z, self.weights):
            composite = composite + z * w
            total_w += abs(w)
        if total_w > 0:
            composite = composite / total_w
        return composite

    def fit_weights_from_ics(self, ic_records: Dict[str, Dict[str, float]]) -> List[float]:
        """用历史评估的 IC 统计估计权重（ICIR 加权）。

        ``ic_records``：{因子名: {"ic_mean": x, "ic_std": y}}。
        因子名与 ``self.factors`` 的 ``meta.name`` 匹配（匹配不上用等权）。
        """
        if not self.factors:
            return []
        ic_means, ic_stds = [], []
        for f in self.factors:
            name = getattr(f, "meta", None) and f.meta.name or ""
            rec = ic_records.get(name, {})
            ic_means.append(float(rec.get("ic_mean", 0.0) or 0.0))
            ic_stds.append(float(rec.get("ic_std", 0.0) or 0.0))
        self.weights = icir_weights(ic_means, ic_stds)
        return list(self.weights)

    def dedup_correlated(self, bars: List[BarData], threshold: float = 0.7) -> List[int]:
        """贪心去冗余：保留与已选因子相关性 < threshold 的因子（优先保留权重更大者）。

        直接修改 ``self.factors`` / ``self.weights``，返回保留的因子下标。
        """
        if len(self.factors) < 2:
            return list(range(len(self.factors)))
        # 计算各因子标准化信号
        sigs = []
        for f in self.factors:
            raw = f.compute(bars)
            sigs.append(expanding_zscore(raw).fillna(0.0))
        mat = pd.concat(sigs, axis=1).astype(float)
        mat.columns = list(range(len(self.factors)))
        corr = mat.corr().abs().fillna(0.0)

        order = sorted(range(len(self.factors)), key=lambda i: -abs(self.weights[i]))
        kept: List[int] = []
        for i in order:
            if all(corr.loc[i, k] < threshold for k in kept):
                kept.append(i)
        kept_sorted = sorted(kept)
        self.factors = [self.factors[i] for i in kept_sorted]
        self.weights = [self.weights[i] for i in kept_sorted]
        return kept_sorted

    def target_position(self, bars: List[BarData], size: int = 1, max_pos: float = 1.0, dedup: bool = False) -> pd.Series:
        """把复合信号映射为目标持仓（带阈值与上限）。

        信号 > threshold：满仓多头；< -threshold：满仓空头；否则空仓。
        ``size`` 为合约乘数（期货），``max_pos`` 为最大仓位比例（0~1）。
        """
        sig = self.combine(bars, dedup=dedup)
        pos = pd.Series(0.0, index=sig.index)
        pos[sig > self.threshold] = max_pos
        pos[sig < -self.threshold] = -max_pos
        return pos * size


def build_model_from_specs(specs: List[FactorSpec], bars: List[BarData]) -> MultiFactorModel:
    """由 FactorSpec 列表构建模型（支持表达式 DSL 与内置因子）。"""
    from .factors.technical import build_factor

    factors: List[Factor] = []
    weights: List[float] = []
    for spec in specs:
        if spec.expression:
            class _ExprFactor(Factor):
                def compute(self, b):
                    df = bars_to_df(b)
                    return eval_factor_expression(spec.expression, df)

            f = _ExprFactor()
            f.meta.name = spec.name
        else:
            f = build_factor(spec.kind, spec.window)
        factors.append(f)
        weights.append(spec.weight)
    return MultiFactorModel(factors, weights=weights)
