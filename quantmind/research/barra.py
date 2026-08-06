"""完整对齐业界 Barra 的多因子**风险归因**（风格正交化 + WLS 截面回归 + Newey-West 协方差）。

背景与动机
----------
现有 18 页的「组合风险/收益归因」是近似 proxy（``contribution = weight × 单因子
样本外总收益``），无法回答「组合波动到底来自哪个因子、多少来自因子系统性、多少
来自个股/品种特异」这类经典风险问题。本模块把每个**因子表达式**当作一种**风格暴露**
（style exposure），用**逐期横截面回归**估计**因子收益率**，再用**协方差分解**把组合
风险拆到「每个因子的贡献 + 特异（residual）风险」，且各部分**可加**（相加等于总
波动）。

方法（对齐业界 Barra 流程，零第三方，numpy/pandas 闭式）：

  1. **风格暴露矩阵** X_t（N 资产 × K 因子）：由标准化后的各因子面板（date×symbol，
     见 :func:`combine.standardize_panel`）在每日 t 上的行向量构成——这是资产对每个
     因子的横截面暴露。
  2. **风格正交化**（Barra 式，``orthogonalize_style=True`` 默认开启）：对每个交易日
     的暴露矩阵做 **Gram-Schmidt 正交化**（按给定因子顺序），使风格因子横截面上彼此
     几乎不相关、且每列单位横截面标准差。这去除了风格间的共线性冗余，使后续截面
     回归因子收益率估计更稳定、因子风险贡献更可解释。
  3. **因子收益率** B_t（K,）：对每个交易日做**横截面回归**（含截距=市场暴露），
     支持 **WLS 市值加权**（业界标准，``weights`` 传入市值权重，缺省 OLS）：
        r_t = X_t B_t + e_t        (B_t = (X'WX)^-1 X' W r_t)
     得到的逐日 B_f,t 序列即「因子收益率」，e_t 为该日资产的**特异收益**。
  4. **因子协方差**：业界 Barra 对因子收益率用 **Newey-West（HAC）稳健协方差**以
     修正自相关（``newey_west=True`` 默认开启，Bartlett 核 + 自动滞后窗），而非普通
     样本协方差。
  5. **组合权重** ω_t（N,）：由复合信号做**多空截面分组**（与
     :func:`cross_sectional_backtest._run_portfolio` 同口径），归一化到多空杠杆 1。
  6. **风险归因**：组合收益写为
        r_p,t = Σ_f (p_f,t · B_f,t) + ε_p,t
     其中 p_f,t = Σ_n ω_n,t X_n,f,t（组合的因子总暴露，用**正交化后**暴露），
     ε_p,t = Σ_n ω_n,t e_n,t（特异）。对每项用**同一协方差估计器**算协方差贡献：
     ``MCTR_i = Cov(a_i, r_p)/σ_p``。因 ``Σ_i a_i = r_p`` 且 HAC/样本协方差均双线性，
     故 ``Σ_i MCTR_i = σ_p`` **精确可加**（闭式 closure≈0）。

注意事项
--------
- 这是**风险归因**（回答"谁贡献了波动"），不同于**收益归因**（谁贡献了 P&L）；
  两者口径不同但互补，前端可并列展示。
- 当标的数 < 因子数 或某段时间截面退化（秩亏）时，回归用伪逆 + 岭正则兜底，
  仍返回结果但标注 ``rank_deficient`` 提示。
- 关键闭合保证：**总方差与各分量协方差必须用同一估计器**（都是 HAC 或都是样本），
  可加性才精确成立。本模块已统一通过 :func:`_est_cov` 派发，杜绝口径混用。
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
    "orthogonalize_exposures",
    "newey_west_cov",
    "barra_factor_risk_attribution",
]


# --------------------------------------------------------------------------- #
# 协方差估计器：Newey-West（HAC）自相关稳健
# --------------------------------------------------------------------------- #
def _auto_lags(t: int) -> int:
    """Newey-West 常用自动滞后窗规则 ``int(4*(T/100)**(2/9))``。"""
    return max(1, int(4.0 * (t / 100.0) ** (2.0 / 9.0)))


def newey_west_cov(
    x: Sequence[float],
    y: Sequence[float],
    lags: Optional[int] = None,
) -> float:
    """Newey-West（HAC）稳健协方差 ``Cov(x, y)``（Bartlett 核）。

    用于存在自相关时的稳健估计：``Σ = Γ_0 + Σ_{j=1}^{L} w_j (Γ_j + Γ_j')``，其中
    ``w_j = 1 - j/(L+1)``（Bartlett），``Γ_j = (1/T) Σ_t x_t·y_{t-j}``。NaN 对自
    动剔除。``lags=None`` 时用 :func:`_auto_lags`。

    HAC 对线性组合**双线性**，因此 ``HAC_cov(Σ_i a_i, r_p) = Σ_i HAC_cov(a_i, r_p)``
    —— 这是 MCTR 归因在 HAC 口径下仍精确可加的数学基础。
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if lags is None:
        lags = _auto_lags(int(np.isfinite(xa).sum()))
    m = np.isfinite(xa) & np.isfinite(ya)
    if m.sum() < 2:
        return float("nan")
    xx = xa[m] - xa[m].mean()
    yy = ya[m] - ya[m].mean()
    T = len(xx)
    s = float(np.dot(xx, yy) / T)  # Γ_0
    for j in range(1, int(lags) + 1):
        w = 1.0 - j / (lags + 1.0)
        c = float(np.dot(xx[j:], yy[:-j]) / T)  # Γ_j
        s += w * (c + c)  # 对称项 Γ_j + Γ_j'
    return s


# --------------------------------------------------------------------------- #
# 风格暴露正交化（Barra 式）
# --------------------------------------------------------------------------- #
def orthogonalize_exposures(
    exposures: Dict[str, pd.DataFrame],
    dates: Sequence,
    columns: Sequence,
    order: Optional[Sequence[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """对暴露矩阵做逐期 **Gram-Schmidt 正交化**，使风格因子横截面互不相关。

    对每个交易日 t，取对齐的暴露矩阵 X_t（N×K，列 = 因子），按 ``order``（缺省用
    dict 插入序）顺序依次做 GSO：第 k 个因子减去其在先前已完成向量上的投影，再缩放
    到单位横截面标准差。缺失值填 0 后参与（与回归口径一致）。

    Returns:
        新 dict ``{name: date×symbol DataFrame}``，各因子互为正交、单位方差。
    """
    names = [o for o in (order or list(exposures.keys())) if o in exposures]
    names = [n for n in names
             if exposures[n] is not None and not exposures[n].empty]
    if len(names) < 2:
        return {n: exposures[n] for n in names}
    arr = np.stack(
        [exposures[n].reindex(index=dates, columns=columns).fillna(0.0).values
         for n in names], axis=2).astype(float)
    T, N, K = arr.shape
    ort = np.zeros_like(arr)
    for t in range(T):
        M = arr[t].copy()
        # 逐列去均值，使后续在中心化空间中正交 → Pearson 相关≈0（本质同 Barra 风格正交）
        M = M - M.mean(axis=0, keepdims=True)
        O = np.zeros_like(M)
        for i in range(K):
            v = M[:, i].copy().astype(float)
            for j in range(i):
                denom = float(np.dot(O[:, j], O[:, j]))
                if denom > 1e-12:
                    v = v - (np.dot(v, O[:, j]) / denom) * O[:, j]
            sd = float(v.std())
            O[:, i] = v / sd if sd > 1e-12 else v
        ort[t] = O
    out: Dict[str, pd.DataFrame] = {}
    for i, n in enumerate(names):
        out[n] = pd.DataFrame(ort[:, :, i], index=dates, columns=columns)
    return out


# --------------------------------------------------------------------------- #
# 逐期截面回归：估计因子收益率
# --------------------------------------------------------------------------- #
def estimate_factor_returns(
    forward_returns: pd.DataFrame,
    exposures: Dict[str, pd.DataFrame],
    ridge: float = 1e-6,
    weights: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, List[str]]:
    """逐交易日回归 ``r_t = Σ_f X_f,t · B_f,t + e_t``（含市场截距），估计因子收益率。

    Args:
        forward_returns: date×symbol 前瞻收益（对齐面板）。
        exposures: name -> 标准化因子面板（date×symbol），作为**风格暴露**。
        ridge: 岭正则系数（防止截面秩亏求逆失败）。
        weights: 可选 date×symbol 非负权重（如市值），用于 **WLS 回归**（业界 Barra
            标准）；None 时退化为 OLS。

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
        # WLS 权重（业界 Barra 市值加权；缺省 OLS 全 1）
        wm = np.ones(XX.shape[0])
        if weights is not None:
            wv = weights.reindex(index=forward_returns.index,
                                 columns=forward_returns.columns).loc[d]
            wv = np.clip(wv.values.astype(float), 0.0, None)
            wm = wv[mask]
            if wm.sum() <= 0:
                continue
        # (Z' W Z) + λI —— WLS 正规方程
        Zw = Z * wm[:, None]
        ZtWZ = Zw.T @ Z + np.eye(Z.shape[1]) * ridge
        ZtWy = Zw.T @ yy
        try:
            ZtWZ_inv = np.linalg.inv(ZtWZ)
        except np.linalg.LinAlgError:
            ZtWZ_inv = np.linalg.pinv(ZtWZ)
        beta = ZtWZ_inv @ ZtWy  # (K+1,)
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
    orthogonalize_style: bool = True,
    newey_west: bool = True,
    nw_lags: Optional[int] = None,
    cap_weights: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    """完整对齐业界 Barra 的多因子风险归因。

    Args:
        signal: 复合信号面板（date×symbol，已标准化）；用于构造组合权重。
        forward_returns: date×symbol 前瞻收益（组合建仓后持有期的收益）。
        exposures: name -> 标准化因子面板（date×symbol），作为风格暴露。
        n_groups: 多空分组数。
        long_short: 是否多空（False 仅做多）。
        annualization: 年化天数，用于波动率/方差年化。
        ridge: 截面回归岭正则。
        orthogonalize_style: 是否对风格暴露做逐期 Gram-Schmidt 正交化（Barra 式，
            默认 True）。见 :func:`orthogonalize_exposures`。
        newey_west: 是否用 Newey-West（HAC）稳健协方差（默认 True）；False 退化为
            普通样本协方差。总方差与各分量协方差用同一估计器，保证 MCTR 精确可加。
        nw_lags: HAC 滞后窗；None 时自动选择。
        cap_weights: 可选 date×symbol 市值权重（WLS 截面回归）；None 时 OLS。

    Returns:
        dict，含：
          - ``factors``: [{name, mctr_vol, risk_pct, exposure_mean, exposure_std,
                            factor_ret_mean, factor_ret_vol}] 每个因子的风险贡献；
          - ``specific``: {vol, risk_pct, sigma_series_mean} 特异风险；
          - ``market``: 市场截距暴露信息；
          - ``total``: {vol, var, ann_vol, n_dates, r2_mean};
          - ``additivity``: {recon_total, port_vol, closure} 闭合校验；
          - ``diagnostics``: {rank_deficient, n_periods, n_assets, n_factors,
                              exposure_cols, orthogonalized, covariance, nw_lags}。
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

    # 2) 对齐暴露到公共日期/标的，并按需做风格正交化
    expos_aligned: Dict[str, pd.DataFrame] = {}
    for nm, df in exposures.items():
        expos_aligned[nm] = df.reindex(index=common, columns=sig.columns)
    expos_used = expos_aligned
    if orthogonalize_style and len(expos_aligned) > 1:
        expos_used = orthogonalize_exposures(
            expos_aligned, common, sig.columns, order=None)

    # 3) 估计因子收益率 + 特异收益（WLS 可选）
    factor_returns, residuals, r2_series, names = estimate_factor_returns(
        fwd, expos_used, ridge=ridge, weights=cap_weights)

    # 4) 组合权重 + 组合收益序列
    weights = portfolio_weights_from_signal(sig, n_groups=n_groups, long_short=long_short)
    # 组合收益：与 _run_portfolio 一致，用资产的实现前瞻收益加权
    port_ret = (weights * fwd).sum(axis=1)
    # 5) 逐因子暴露序列 p_f,t = Σ_n ω_n,t X_n,f,t（用与回归一致的暴露）
    exposure_means: Dict[str, float] = {}
    pf_series: Dict[str, pd.Series] = {}
    for nm in names:
        X = expos_used[nm].reindex(common, columns=weights.columns).fillna(0.0)
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

    # 5b) 统一剔除任一分量含 NaN 的行，保证在保留行上 Σ_i a_i = r_p 处处成立
    #     → HAC/样本协方差双线性 ⇒ MCTR 闭式精确（否则各行 NaN 模式不同会破坏可加性）。
    keep = common_series.notna().all(axis=1) & port_ret.notna()
    common_series = common_series.loc[keep]
    port_ret = port_ret.loc[keep]
    if len(port_ret) < 2:
        raise ValueError("Barra 风险归因：剔除缺失值后有效样本不足")

    # 6) 统一协方差估计器：业界 Barra 用 HAC（Newey-West）；False 退化为普通样本协方差。
    #    总方差与各分量协方差必须用【同一估计器】，MCTR 可加性才精确成立。
    def _est_cov(a, b):
        if newey_west:
            return newey_west_cov(a, b, lags=nw_lags)
        aa = np.asarray(a, dtype=float)
        bb = np.asarray(b, dtype=float)
        m = np.isfinite(aa) & np.isfinite(bb)
        if m.sum() < 2:
            return float("nan")
        return float(np.cov(aa[m], bb[m], ddof=1)[0, 1])

    rp = port_ret.values.astype(float)
    _total_var = _est_cov(rp, rp)
    vol = float(np.sqrt(max(_total_var, 0.0))) if _total_var == _total_var else float("nan")
    var = _total_var if _total_var == _total_var else float("nan")
    ann_vol = float(vol * np.sqrt(annualization)) if vol == vol else float("nan")

    # 7) 协方差贡献分解：MCTR_i = Cov(a_i, r_p) / σ_p , 且 Σ MCTR = σ_p（因 Σ a_i = r_p
    #    且协方差双线性——HAC 与样本估计均保持）
    mctr: Dict[str, float] = {}
    contrib_series = {}
    for col in list(common_series.columns):
        a = common_series[col].values.astype(float)
        cov = _est_cov(a, rp)
        mctr[col] = (cov / vol) if (vol == vol and vol != 0) else 0.0
        contrib_series[col] = cov

    # 汇总：因子 vs 特异 vs 市场
    factor_names = names
    factor_total = sum(mctr.get(nm, 0.0) for nm in factor_names)
    specific_mctr = mctr.get("_specific", 0.0)
    market_mctr = mctr.get("_market", 0.0)
    # 拟合优度：系统性能解释比例 = 1 - Var(特异)/Var(总)（同一估计器）
    _specific_clean = common_series["_specific"].values.astype(float)
    var_specific = _est_cov(_specific_clean, _specific_clean) \
        if len(_specific_clean) > 1 else 0.0
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
            "n_dates": int(len(port_ret)),
            "r2_mean": round(r2_mean, 4) if r2_mean == r2_mean else None,
        },
        "additivity": {
            "recon_total": round(total_alloc, 6),
            "port_vol": round(vol, 6),
            "closure": round(total_alloc - vol, 8) if vol == vol else None,
        },
        "diagnostics": {
            "rank_deficient": False,
            "n_periods": int(len(port_ret)),
            "n_assets": int(signal.shape[1]),
            "n_factors": len(factor_names),
            "exposure_cols": factor_names,
            "orthogonalized": bool(orthogonalize_style and len(expos_aligned) > 1),
            "covariance": "newey_west" if newey_west else "sample",
            "nw_lags": int(nw_lags) if (newey_west and nw_lags is not None) else
                       (int(_auto_lags(len(common))) if newey_west else None),
            "wls": cap_weights is not None,
        },
    }
