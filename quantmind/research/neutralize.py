"""因子中性化与正交化（去除风格/行业暴露、降低因子间冗余）。

  - ``winsorize``：截尾，去除极端值对回归/标准化的干扰。
  - ``cross_sectional_neutralize``：在每个交易日截面上，用 OLS 把因子对
    [1, log(市值), 风格因子..., 行业虚拟变量] 回归，取残差作为中性化因子。
    用于剥离市值/行业/风格暴露（A股/港股常用）。
  - ``orthogonalize_factors``：对一组（索引对齐的）因子序列做 Gram-Schmidt
    正交化，去除因子间的时间序列相关性（多因子组合去冗余）。

均为原生实现，不依赖第三方 GPL 代码。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def winsorize(series: pd.Series, p: float = 0.01) -> pd.Series:
    """对序列做双边截尾（默认 1% 分位）。"""
    s = series.astype(float)
    if s.notna().sum() == 0:
        return s
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lower=lo, upper=hi)


def _zscore(series: pd.Series) -> pd.Series:
    mu = series.mean()
    sd = series.std()
    if sd is None or pd.isna(sd) or sd == 0:
        return series - mu
    return (series - mu) / sd


def cross_sectional_neutralize(
    panel: pd.DataFrame,
    market_cap: Optional[pd.DataFrame] = None,
    industry: Optional[pd.DataFrame] = None,
    styles: Optional[Dict[str, pd.DataFrame]] = None,
    winsor: float = 0.01,
) -> pd.DataFrame:
    """截面中性化：日期 × 标的 的因子面板 -> 残差面板（剥离市值/行业/风格）。

    ``panel`` / ``market_cap`` / ``industry`` / ``styles[name]`` 均为
    ``date × symbol`` 的 DataFrame，索引与列对齐。逐交易日做 OLS 回归取残差。
    无任何暴露变量时退化为「去截面均值」（demean）。
    """
    if panel.empty:
        return panel.copy()
    out = panel.copy().astype(float)
    has_mc = market_cap is not None and not market_cap.empty
    has_ind = industry is not None and not industry.empty
    has_style = bool(styles)
    style_names = list(styles.keys()) if styles else []

    for d, row in panel.iterrows():
        y = row.dropna()
        if len(y) < 5:
            out.loc[d] = np.nan
            continue
        yw = winsorize(y, winsor)
        # 构造设计矩阵
        cols: List[pd.Series] = [pd.Series(1.0, index=yw.index, name="const")]
        valid_idx = yw.index
        if has_mc:
            mc = market_cap.loc[d].reindex(yw.index)
            mc = np.log(mc.replace(0, np.nan)).dropna()
            valid_idx = valid_idx.intersection(mc.index)
            yw = yw.loc[valid_idx]
            cols = [pd.Series(1.0, index=valid_idx, name="const"), mc.reindex(valid_idx).rename("mc")]
        if has_style:
            style_cols = []
            for nm in style_names:
                sp = styles[nm].loc[d].reindex(valid_idx).dropna()
                valid_idx = valid_idx.intersection(sp.index)
                style_cols.append(sp.rename(nm))
            if has_mc:
                # 市值变量不能被 style 分支的 cols 重建覆盖掉
                mc = np.log(market_cap.loc[d].reindex(valid_idx).replace(0, np.nan)).dropna()
                valid_idx = valid_idx.intersection(mc.index)
                style_cols.append(mc.rename("mc"))
            yw = yw.loc[valid_idx]
            cols = [pd.Series(1.0, index=valid_idx, name="const")] \
                + [s.reindex(valid_idx) for s in style_cols]
        if has_ind:
            ind = industry.loc[d].reindex(valid_idx)
            valid_idx = valid_idx.intersection(ind.dropna().index)
            yw = yw.loc[valid_idx]
            dummies = pd.get_dummies(ind.loc[valid_idx].astype("category"), drop_first=True)
            # 重新组装 X
            base = [pd.Series(1.0, index=valid_idx, name="const")]
            if has_mc:
                mc = np.log(market_cap.loc[d].reindex(valid_idx).replace(0, np.nan))
                base.append(mc.rename("mc"))
            for nm in style_names:
                base.append(styles[nm].loc[d].reindex(valid_idx).rename(nm))
            base.extend([dummies[c] for c in dummies.columns])
            X = pd.concat(base, axis=1).astype(float)
        else:
            X = pd.concat(cols, axis=1).astype(float)

        X = X.loc[valid_idx]
        yv = yw.loc[valid_idx].astype(float)
        if X.shape[0] < max(5, X.shape[1] + 3):
            out.loc[d] = np.nan
            continue
        try:
            beta, *_ = np.linalg.lstsq(X.values, yv.values, rcond=None)
            fitted = X.values @ beta
            resid = yv.values - fitted
        except Exception:  # noqa: BLE001
            out.loc[d] = np.nan
            continue
        res = pd.Series(resid, index=valid_idx)
        out.loc[d, valid_idx] = res

    return out


def orthogonalize_factors(
    factors: Sequence[pd.Series],
    names: Optional[List[str]] = None,
    winsor: float = 0.01,
) -> List[pd.Series]:
    """对一组索引对齐的因子序列做 Gram-Schmidt 正交化（去冗余）。

    返回与输入等长、彼此时间序列不相关的正交化因子（按输入顺序，第一个不变）。
    仅作用于时间序列相关（适用于单标的多因子去冗余）。
    """
    if not factors:
        return []
    # 对齐索引 + 截尾 + 标准化
    aligned = pd.concat([winsorize(f.astype(float), winsor) for f in factors], axis=1)
    aligned.columns = names or [f"f{i}" for i in range(len(factors))]
    z = aligned.apply(_zscore, axis=0)

    base = []
    ortho = []
    for i in range(z.shape[1]):
        v = z.iloc[:, i].copy()
        for b in base:
            denom = (b * b).sum()
            if denom <= 1e-12:
                # 零方差向量（常量因子）：跳过投影，避免 0/0=NaN 毒化后续所有因子
                continue
            proj = (v * b).sum() / denom
            v = v - proj * b
        base.append(v)
        ortho.append(v)
    # 还原为以原始因子名为 name 的 Series
    out = []
    for j, s in enumerate(ortho):
        s = s.rename(names[j] if names else f"f{j}")
        out.append(s)
    return out
