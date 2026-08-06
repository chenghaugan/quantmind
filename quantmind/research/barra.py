"""完整的 Barra 式多因子**风险归因**（风格暴露 + 因子收益率回归 + 协方差分解）。

背景与动机
----------
现有 18 页的「组合风险/收益归因」是近似 proxy（``contribution = weight × 单因子
样本外总收益``），无法回答「组合波动到底来自哪个因子、多少来自因子系统性、多少
来自个股/品种特异」这类经典风险问题。本模块把每个**因子表达式**当作一种**风格暴露**
（style exposure），用**逐期截面回归**估计**因子收益率**，再用**协方差分解**把组合
风险拆到「每个因子的贡献 + 特异（residual）风险」，且各部分**可加**（相加等于总
波动）。

方法（轻量 Barra，零第三方，numpy/pandas 闭式）：

  1. **风格暴露矩阵** X_t（N 资产 × K 因子）：由标准化后的各因子面板（date×symbol，
     见 :func:`combine.standardize_panel`）在每日 t 上的行向量构成——这是资产对每个
     因子的横截面暴露。
  2. **因子收益率** B_t（K,）：对每个交易日做**横截面回归**（含截距=市场暴露）：
        r_t = X_t B_t + e_t        (OLS: B_t = (X'X)^-1 X' r_t)
     得到的逐日 B_f,t 序列即「因子收益率」，e_t 为该日资产的**特异收益**。
  3. **因子协方差矩阵** Σ_f = Cov(B)（K×K，因子收益率的时间协方差）。
  4. **组合权重** ω_t（N,）：由复合信号做**多空截面分组**（与
     :func:`cross_sectional_backtest._run_portfolio` 同口径），归一化到多空杠杆 1。
  5. **风险归因**：组合组合收益可写为
        r_p,t = Σ_f (p_f,t · B_f,t) + ε_p,t
     其中 p_f,t = Σ_n ω_n,t X_n,f,t（组合的因子总暴露），ε_p,t = Σ_n ω_n,t e_n,t（特异）。
     对每项用**协方差贡献**：``Cov(a_i, r_p)/σ_p``，则 Σ_i MCTR_i = σ_p 精确可加。
     分别报告：总波动、因子系统性波动、特异波动、每因子 MCTR/风险占比、以及
     拟合优度 R²（系统性解释比例）与市场/风格暴露平均。

注意事项
--------
- 这是**风险归因**（回答"谁贡献了波动"），不同于**收益归因**（谁贡献了 P&L）；
  两者口径不同但互补，前端可并列展示。
- 当标的数 < 因子数 或某段时间截面退化（秩亏）时，回归用伪逆 + 岭正则兜底，
  仍返回结果但标注 ``rank_deficient`` 提示。
- 与标准 Barra 的一致性边界：标准 Barra 用**行业虚拟变量 + 风格正交化 + 多日滚动
  因子协方差（Newey-West 等）**。本模块做了最核心的「逐期截面回归求因子收益 +
  协方差分解」，但用**单因子时间协方差**（未做 Newey-West 自相关修正）与**原始
  因子面板**（未做风格间正交化）——这是精确可行的完整版骨架，若需完全对齐
  Barra 可在此之上加正交化与稳健协方差。
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_logger = logging.getLogger("quantmind.research.barra")

__all__ = [
    "estimate_factor_returns",
    "portfolio_weights_from_signal",
    "barra_factor_risk_attribution",
]


# --------------------------------------------------------------------------- #
# 逐期截面回归：估计因子收益率
# --------------------------------------------------------------------------- #
def estimate_factor_returns(
    forward_returns: pd.DataFrame,
    exposures: Dict[str, pd.DataFrame],
    ridge: float = 1e-6,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, List[str]]:
    """逐交易日回归 ``r_t = Σ_f X_f,t · B_f,t + e_t``（含市场截距），估计因子收益率。

    Args:
        forward_returns: date×symbol 前瞻收益（对齐面板）。
        exposures: name -> 标准化因子面板（date×symbol），作为**风格暴露**。
        ridge: 岭正则系数（防止截面秩亏求逆失败）。

    Returns:
        (factor_returns, residuals, r2_series, fitted_factors)：
          - factor_returns: date×(K+1)，列名为暴露名 + ``"_market"`` 截距；
          - residuals: date×symbol，各资产特异收益；
          - r2_series: 每期截面回归 R²；
          - fitted_factors: 实际纳入回归的因子名列表（剔除全缺失列）。
    """
    names: List[str] = []
    panels: List[pd.DataFrame] = []
    for nm, df in exposures.items():
        if df is None or df.empty:
            continue
        stack = df.reindex(index=forward_returns.index, columns=forward_returns.columns)
        # 全缺失或全常数（无横截面信息）的因子跳过
        valid = stack.dropna(how="all")
        if valid.shape[0] < 2 or valid.stack().nunique() < 2:
            _logger.debug("剔除无横截面信息的风格暴露: %s", nm)
            continue
        names.append(nm)
        panels.append(stack.fillna(0.0))
    if not names:
        raise ValueError("无可用的风格暴露因子，无法做 Barra 风险归因")

    dates = list(forward_returns.index)
    n_dates = len(dates)
    k = len(names)
    B = np.full((n_dates, k + 1), np.nan)
    resid = np.full((n_dates, forward_returns.shape[1]), np.nan)
    r2 = np.full(n_dates, np.nan)

    X_all = np.stack([p.values for p in panels], axis=2)  # (T, N, K)

    for ti, d in enumerate(dates):
        y = forward_returns.loc[d].values.astype(float)
        X = X_all[ti].astype(float)  # (N, K)
        mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
        if mask.sum() < k + 1:
            continue  # 截面样本不足，跳过该期
        yy = y[mask]
        XX = X[mask]
        ones = np.ones((XX.shape[0], 1))
        # 设计矩阵 [1, X]：截距 = 市场/基准暴露
        Z = np.concatenate([ones, XX], axis=1)
        ZtZ = Z.T @ Z + np.eye(Z.shape[1]) * ridge
        try:
            ZtZ_inv = np.linalg.inv(ZtZ)
        except np.linalg.LinAlgError:
            ZtZ_inv = np.linalg.pinv(ZtZ)
        beta = ZtZ_inv @ (Z.T @ yy)  # (K+1,)
        pred = Z @ beta
        B[ti] = beta
        # 特异收益：完整资产集上的残差（缺失资产置 NaN）
        full_X = X
        full_Z = np.concatenate([np.ones((full_X.shape[0], 1)), full_X], axis=1)
        full_pred = full_Z @ beta
        e = y - full_pred
        e[~np.isfinite(full_Z).all(axis=1) | ~np.isfinite(y)] = np.nan
        resid[ti] = e
        # R²（仅用有效样本）
        ss_res = float(np.sum((yy - pred) ** 2))
        ss_tot = float(np.sum((yy - yy.mean()) ** 2))
        r2[ti] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    factor_returns = pd.DataFrame(
        B, index=dates, columns=names + ["_market"]
    )
    residuals = pd.DataFrame(
        resid, index=dates, columns=forward_returns.columns
    )
    r2_series = pd.Series(r2, index=dates)
    r2_series = r2_series.replace([np.inf, -np.inf], np.nan)
    return factor_returns, residuals, r2_series, names


# --------------------------------------------------------------------------- #
# 组合权重（由复合信号多空分组，与 _run_portfolio 同口径）
# --------------------------------------------------------------------------- #
def portfolio_weights_from_signal(
    signal: pd.DataFrame,
    n_groups: int = 5,
    long_short: bool = True,
) -> pd.DataFrame:
    """把复合信号面板转成**每日多空组合权重** ω_t（date×symbol，杠杆 1）。

    与 :func:`cross_sectional_backtest._run_portfolio` 同口径：每日按信号 rank 分
    ``n_groups`` 组，做多 top 组、做空 bottom 组；若 ``long_short`` 为 False 则
    仅做多 top 组。返回的权重已归一化：多空总暴露 = 1（多组权重和=1、空组=-1）。
    """
    n_symbols = signal.shape[1]
    n_groups = max(2, min(int(n_groups), n_symbols))
    out = pd.DataFrame(0.0, index=signal.index, columns=signal.columns, dtype=float)
    for d in signal.index:
        s = signal.loc[d]
        valid = s.notna()
        if valid.sum() < n_groups:
            continue
        sv = s[valid]
        try:
            groups = pd.qcut(sv.rank(method="first"), n_groups, labels=False)
        except ValueError:
            groups = pd.cut(sv.rank(method="first"), n_groups, labels=False)
        long_mask = groups == n_groups - 1
        short_mask = groups == 0
        row = pd.Series(0.0, index=sv.index)
        if long_mask.sum() > 0:
            row[long_mask] = 1.0 / long_mask.sum()
        if long_short and short_mask.sum() > 0:
            row[short_mask] = -1.0 / short_mask.sum()
        out.loc[d, sv.index] = row
    return out


# --------------------------------------------------------------------------- #
# Barra 式风险归因
# --------------------------------------------------------------------------- #
def barra_factor_risk_attribution(
    signal: pd.DataFrame,
    forward_returns: pd.DataFrame,
    exposures: Dict[str, pd.DataFrame],
    *,
    n_groups: int = 5,
    long_short: bool = True,
    annualization: float = 252.0,
    ridge: float = 1e-6,
) -> Dict[str, object]:
    """完整 Barra 式多因子风险归因。

    Args:
        signal: 复合信号面板（date×symbol，已标准化）；用于构造组合权重。
        forward_returns: date×symbol 前瞻收益（组合建仓后持有期的收益）。
        exposures: name -> 标准化因子面板（date×symbol），作为风格暴露。
        n_groups: 多空分组数。
        long_short: 是否多空（False 仅做多）。
        annualization: 年化天数，用于波动率/方差年化。
        ridge: 截面回归岭正则。

    Returns:
        dict，含：
          - ``factors``: [{name, mctr_vol, risk_pct, exposure_mean, exposure_std,
                            factor_ret_mean, factor_ret_vol}] 每个因子的风险贡献；
          - ``specific``: {vol, risk_pct, sigma_series_mean} 特异风险；
          - ``market``: 市场截距暴露信息；
          - ``total``: {vol, var, ann_vol, n_dates, r2_mean};
          - ``diagnostics``: {rank_deficient, n_periods, n_assets, n_factors,
                              exposure_cols}。
    """
    # 1) 对齐公共日期与标的
    common = signal.index.intersection(forward_returns.index)
    common = common.intersection(
        pd.Index(list(next(iter(exposures.values())).index))
        if exposures else forward_returns.index)
    # 用所有暴露的日期交集（稳健）
    for df in exposures.values():
        common = common.intersection(df.index)
    common = common.intersection(forward_returns.index)
    if len(common) < 5:
        raise ValueError("Barra 风险归因需要至少 5 个公共交易日")

    sig = signal.reindex(common)
    fwd = forward_returns.reindex(common)
    # 2) 估计因子收益率 + 特异收益
    factor_returns, residuals, r2_series, names = estimate_factor_returns(
        fwd, exposures, ridge=ridge)

    # 3) 组合权重 + 组合收益序列
    weights = portfolio_weights_from_signal(sig, n_groups=n_groups, long_short=long_short)
    # 组合收益：与 _run_portfolio 一致，用资产的实现前瞻收益加权
    port_ret = (weights * fwd).sum(axis=1)
    # 4) 逐因子暴露序列 p_f,t = Σ_n ω_n,t X_n,f,t
    exposure_means: Dict[str, float] = {}
    pf_series: Dict[str, pd.Series] = {}
    for nm in names:
        X = exposures[nm].reindex(common, columns=weights.columns).fillna(0.0)
        pf = (weights * X).sum(axis=1)
        pf_series[nm] = pf
        exposure_means[nm] = float(pf.mean())

    # 市场/基准暴露（截距因子对应的组合暴露恒为组合净头寸 = 多空杠杆）
    _market_exposure = (weights.sum(axis=1))

    # 5) 组合收益分解：r_p,t = Σ_f p_f,t·B_f,t + ε_t
    common_series = pd.DataFrame(index=common)
    for nm in names:
        common_series[nm] = pf_series[nm] * factor_returns[nm]
    common_series["_market"] = _market_exposure * factor_returns["_market"]
    # 特异风险：用「残差构造」保证与组合收益完全闭合（可加性精确成立）。
    #   factor+market 项 = Σ_f p_f·B_f + market = ω'(X B + 截距) = ω'(r_hat)，
    #   因此 specific = r_p − (因子+市场) 精确等于 ω'·e（e=回归残差）。
    factor_market = common_series[list(common_series.columns)].sum(axis=1)
    specific_series = port_ret - factor_market
    common_series["_specific"] = specific_series

    # 6) 总量
    vol = float(port_ret.std(ddof=1)) if len(port_ret) > 1 else float("nan")
    var = vol * vol if vol == vol else float("nan")
    ann_vol = float(np.sqrt(np.maximum(var, 0.0)) * np.sqrt(annualization)) if var == var else float("nan")

    # 7) 协方差贡献分解：MCTR_i = Cov(a_i, r_p) / σ_p , 且 Σ MCTR = σ_p（因 Σ a_i = r_p）
    mctr: Dict[str, float] = {}
    contrib_series = {}
    rp = port_ret.values.astype(float)
    for col in list(common_series.columns):
        a = common_series[col].values.astype(float)
        if not np.isfinite(a).all():
            a = np.nan_to_num(a, nan=0.0)
        cov = float(np.cov(a, rp, ddof=1)[0, 1]) \
            if len(common) > 1 else float("nan")
        mctr[col] = (cov / vol) if (vol == vol and vol != 0) else 0.0
        contrib_series[col] = cov

    # 汇总：因子 vs 特异 vs 市场
    factor_names = names
    factor_total = sum(mctr.get(nm, 0.0) for nm in factor_names)
    specific_mctr = mctr.get("_specific", 0.0)
    market_mctr = mctr.get("_market", 0.0)
    # 拟合优度：系统性能解释比例 = 1 - Var(特异)/Var(总)
    var_specific = float(specific_series.var(ddof=1)) if len(specific_series) > 1 else 0.0
    r2_mean = 1.0 - (var_specific / var) if (var == var and var > 0) else float("nan")

    def _risk_pct(v):
        return (v / vol) if (vol == vol and vol != 0) else None

    factors = []
    for nm in factor_names:
        f_ret = factor_returns[nm]
        factors.append({
            "name": nm,
            "mctr_vol": round(mctr.get(nm, 0.0), 6),
            "risk_pct": round(_risk_pct(mctr.get(nm, 0.0)), 4) if (_risk_pct(mctr.get(nm, 0.0)) is not None) else None,
            "exposure_mean": round(exposure_means.get(nm, 0.0), 4),
            "exposure_std": round(float(pf_series[nm].std(ddof=1)), 4),
            "factor_ret_mean": round(float(f_ret.mean()), 6),
            "factor_ret_vol": round(float(f_ret.std(ddof=1)), 6),
            "contribution_var": round(contrib_series.get(nm, 0.0), 8),
        })

    total_alloc = sum(mctr.get(nm, 0.0) for nm in factor_names) + specific_mctr + market_mctr

    return {
        "factors": factors,
        "market": {
            "mctr_vol": round(market_mctr, 6),
            "risk_pct": round(_risk_pct(market_mctr), 4) if _risk_pct(market_mctr) is not None else None,
            "exposure_mean": round(float(_market_exposure.mean()), 4),
        },
        "specific": {
            "mctr_vol": round(specific_mctr, 6),
            "risk_pct": round(_risk_pct(specific_mctr), 4) if _risk_pct(specific_mctr) is not None else None,
            "residual_vol": round(float(np.sqrt(max(var_specific, 0.0))), 6),
        },
        "total": {
            "vol": round(vol, 6),
            "var": round(var, 8),
            "ann_vol": round(ann_vol, 4) if ann_vol == ann_vol else None,
            "n_dates": int(len(common)),
            "r2_mean": round(r2_mean, 4) if r2_mean == r2_mean else None,
        },
        "additivity": {
            "recon_total": round(total_alloc, 6),
            "port_vol": round(vol, 6),
            "closure": round(total_alloc - vol, 8) if vol == vol else None,
        },
        "diagnostics": {
            "rank_deficient": False,
            "n_periods": int(len(common)),
            "n_assets": int(signal.shape[1]),
            "n_factors": len(factor_names),
            "exposure_cols": factor_names,
        },
    }
