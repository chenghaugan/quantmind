"""多因子**组合构建与权重优化**（面板版，因子 → 可交易 alpha 组合）。

挖掘/去冗余产出一批代表性因子后，把它们**合成一个复合信号**（而非逐个单独
回测），是「因子研究」到「可交易的 alpha」的最后一步。本模块在**多标的面板**
（date × symbol）上：

  1. 对每个因子面板做**逐交易日截面标准化**（rank 归一或 z-score），使不同量纲
     的因子可比；
  2. 按给定**权重方案**优化因子权重（等权 / ICIR 加权 / 逆方差 / 最小方差）；
  3. 标准化信号按权重线性合成 → 复合 alpha 面板；
  4. （可选）把复合信号直接转成每日横截面多空组合回测，得到 Sharpe/回撤/IC。

**防泄漏（重要）**：权重必须仅在**样本内（train）**拟合，再在**样本外（test）**
用固定权重回测复合组合。``composite_backtest`` 因此接受两个面板：
``training_panel``（拟合 ICIR/方差权重）与 ``test_panel``（OOS 回测）。若只传一个，
则在同一面板上拟合并回测（仅作探索性验证，非严格 OOS）。

零第三方依赖：仅用 numpy/pandas 实现闭式最优权重，避免 scipy 的
``minimize``/``quadratic_program``。
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .factors.alpha_cs import Panel
from .factors.panel_expr import panel_eval_expression
from .eval import evaluate_expression
from .cross_sectional_backtest import _factor_scores, _run_portfolio, _portfolio_output
from .evaluator import FactorEvaluator
from .barra import barra_factor_risk_attribution

_logger = logging.getLogger("quantmind.research.combine")

__all__ = [
    "cs_rank_panel",
    "cs_zscore_panel",
    "standardize_panel",
    "equal_weights",
    "icir_weights",
    "inverse_variance_weights",
    "min_variance_weights",
    "combine_factor_panels",
    "optimize_weights",
    "composite_backtest",
]


# --------------------------------------------------------------------------- #
# 截面标准化
# --------------------------------------------------------------------------- #
def cs_rank_panel(factor_df: pd.DataFrame) -> pd.DataFrame:
    """逐交易日对因子面板做**截面 rank 归一**（每行 [0,1]）。

    把每个交易日上所有标的的因子值映射到 [0,1] 的横截面排名，使量纲无关、
    对极端值稳健，是合成多因子最常用的标准化（等价于每期 Spearman 信号的等分）。
    """
    return factor_df.rank(axis=1, pct=True)


def cs_zscore_panel(factor_df: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    """逐交易日对因子面板做**截面 z-score**（每行去均值/除标准差，可截尾）。

    对每个交易日：``z = (x - mean) / std``，不满 2 个有效标的的行置 NaN。
    可选 ``clip`` 截尾抑制极值（None 不截尾）。
    """
    out = factor_df.astype(float).copy()
    mu = factor_df.mean(axis=1)
    sd = factor_df.std(axis=1)
    valid = sd.replace(0.0, np.nan)
    out = factor_df.sub(mu, axis=0).div(valid, axis=0)
    out[factor_df.notna().sum(axis=1) < 2] = np.nan
    if clip is not None and clip > 0:
        out = out.clip(lower=-clip, upper=clip)
    return out


def standardize_panel(factor_df: pd.DataFrame, method: str = "zscore") -> pd.DataFrame:
    """统一入口：按 ``method`` 标准化因子面板（``zscore`` 或 ``rank``）。"""
    if method == "rank":
        return cs_rank_panel(factor_df)
    if method == "zscore":
        return cs_zscore_panel(factor_df)
    raise ValueError(f"未知标准化方法: {method!r}（可选 zscore/rank）")


# --------------------------------------------------------------------------- #
# 权重方案
# --------------------------------------------------------------------------- #
def equal_weights(n: int) -> np.ndarray:
    """等权（归一化到和为 1，非负）。"""
    if n <= 0:
        return np.array([], dtype=float)
    return np.full(n, 1.0 / n, dtype=float)


def icir_weights(
    ic_means: Sequence[float],
    ic_stds: Sequence[float],
    long_only: bool = True,
) -> np.ndarray:
    """ICIR（信息比率 mean/std）加权。

    ICIR 越高权重越大；负 ICIR 视为无效（若 ``long_only``，权重钳为 0，
    否则按符号保留方向）。全部无效则退化为等权（非负）。
    """
    n = len(list(ic_means))
    if n == 0:
        return np.array([], dtype=float)
    w = np.zeros(n, dtype=float)
    for i, (m, s) in enumerate(zip(ic_means, ic_stds)):
        if s and s > 0 and np.isfinite(m) and np.isfinite(s):
            icir = m / s
            w[i] = max(icir, 0.0) if long_only else icir
    tot = np.abs(w).sum()
    if tot <= 0:
        return equal_weights(n)
    return w / tot


def inverse_variance_weights(variances: Sequence[float]) -> np.ndarray:
    """逆方差加权：权重 ∝ 1/方差（波动越小权重越高），长仓归一。

    方差无效（<=0 或 NaN）的因子权重 0；全部无效则等权。
    """
    n = len(list(variances))
    if n == 0:
        return np.array([], dtype=float)
    w = np.zeros(n, dtype=float)
    for i, v in enumerate(variances):
        if v is not None and np.isfinite(v) and v > 0:
            w[i] = 1.0 / v
    tot = w.sum()
    if tot <= 0:
        return equal_weights(n)
    return w / tot


def min_variance_weights(
    corr: pd.DataFrame,
    diag_variances: Sequence[float] | None = None,
) -> np.ndarray:
    """最小方差权重（闭式，numpy-only，长仓归一）。

    在因子两两相关矩阵 ``corr``（index/columns=因子）上，用拉格朗日闭式解求
    「带对角方差尺度、且权重非负」的组合权重。思路：求解 min w'Σw 受 w 和=1
    （Σ=diag(var)^.5 · corr · diag(var)^.5），再把负权重钳为 0 并归一化。

    Returns:
        归一化到和为 1 的非负权重向量（按 corr 列序）。
    """
    names = list(corr.columns)
    n = len(names)
    if n == 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.array([1.0], dtype=float)
    C = corr.reindex(index=names, columns=names).astype(float).values
    # 对角方差尺度
    D = np.ones(n, dtype=float)
    if diag_variances is not None and len(list(diag_variances)) == n:
        for i, v in enumerate(diag_variances):
            if v is not None and np.isfinite(v) and v > 0:
                D[i] = v
    sig = np.sqrt(np.maximum(D, 1e-12))
    # 协方差 = diag(sig) @ corr @ diag(sig)
    S = (sig[:, None] * C) * sig[None, :]
    # 正则化保证可逆
    S = S + np.eye(n) * (1e-6 + 0.0)
    try:
        Sinv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        Sinv = np.linalg.pinv(S)
    ones = np.ones(n)
    num = Sinv @ ones
    den = ones @ num
    if not np.isfinite(den) or den <= 0:
        return equal_weights(n)
    w = num / den
    # 长仓钳位 + 归一化
    w = np.maximum(w, 0.0)
    tot = w.sum()
    if tot <= 0:
        return equal_weights(n)
    return w / tot


# --------------------------------------------------------------------------- #
# 组合合成
# --------------------------------------------------------------------------- #
def combine_factor_panels(
    factor_dfs: Dict[str, pd.DataFrame],
    weights: Sequence[float] | None = None,
    standardize: str = "zscore",
    orthogonalize: bool = False,
) -> pd.DataFrame:
    """把多个因子面板标准化后按权重合成为**复合 alpha 面板**。

    Args:
        factor_dfs: name -> date×symbol 因子面板。
        weights: 与 ``factor_dfs`` 同序的权重；None → 等权。
        standardize: ``zscore``（默认）或 ``rank`` 截面标准化。
        orthogonalize: True 时对标准化信号先做截面 Gram-Schmidt 正交化再去冗余
            （仅当权重等权/ICIR 场景谨慎使用；正交化会改变各因子解释度）。

    Returns:
        复合因子面板（index=日期，columns=标的），量纲一致、可直接回测。
    """
    if not factor_dfs:
        return pd.DataFrame()
    names = list(factor_dfs.keys())
    if weights is None:
        w = equal_weights(len(names))
    else:
        w = np.asarray(weights, dtype=float)
        if len(w) != len(names):
            raise ValueError(f"权重个数 {len(w)} 与因子数 {len(names)} 不符")
    stds = [standardize_panel(factor_dfs[nm], standardize) for nm in names]
    # 对齐公共行
    common_idx = stds[0].index
    for s in stds[1:]:
        common_idx = common_idx.intersection(s.index)
    stds = [s.reindex(common_idx) for s in stds]

    if orthogonalize and len(stds) > 1:
        # 逐交易日的截面正交化（去因子间截面冗余）
        ortho = [[] for _ in stds]
        for d in common_idx:
            vecs = [s.loc[d].fillna(0.0).values.astype(float) for s in stds]
            base: List[np.ndarray] = []
            for v in vecs:
                u = v.copy()
                for b in base:
                    proj = np.dot(u, b) / (np.dot(b, b) + 1e-12)
                    u = u - proj * b
                base.append(u)
            for j, u in enumerate(base):
                ortho[j].append(u)
        stds = [pd.DataFrame(o, index=common_idx, columns=stds[0].columns) for o in ortho]

    composite = pd.DataFrame(0.0, index=common_idx, columns=stds[0].columns)
    for s, wi in zip(stds, w):
        composite = composite + s.fillna(0.0) * wi
    return composite


# --------------------------------------------------------------------------- #
# 权重优化
# --------------------------------------------------------------------------- #
def _panel_variances(factor_dfs: Dict[str, pd.DataFrame], standardize: str) -> np.ndarray:
    """各标准化面板的（跨期合并）方差。"""
    stds = [standardize_panel(factor_dfs[k], standardize).stack().var() for k in factor_dfs]
    return np.asarray([v if v == v else np.nan for v in stds], dtype=float)


def _panel_corr(factor_dfs: Dict[str, pd.DataFrame], standardize: str) -> pd.DataFrame:
    """各标准化面板两两的截面相关矩阵（跨期 stack）。"""
    names = list(factor_dfs.keys())
    if len(names) < 2:
        return pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    mat = pd.DataFrame(index=names, columns=names, dtype=float)
    for i, a in enumerate(names):
        sa = standardize_panel(factor_dfs[a], standardize).stack()
        mat.loc[a, a] = 1.0
        for j in range(i + 1, len(names)):
            b = names[j]
            sb = standardize_panel(factor_dfs[b], standardize).stack()
            idx = sa.index.intersection(sb.index)
            r = sa.loc[idx].corr(sb.loc[idx]) if len(idx) >= 10 else float("nan")
            mat.loc[a, b] = r
            mat.loc[b, a] = r
    return mat


def optimize_weights(
    factor_dfs: Dict[str, pd.DataFrame],
    scheme: str = "icir",
    ic_reports: Optional[Dict[str, Dict[str, float]]] = None,
    standardize: str = "zscore",
) -> Dict[str, float]:
    """按 ``scheme`` 优化一批因子的权重，返回 name -> weight。

    Schemes:
      - ``equal``：等权。
      - ``icir``：ICIR 加权（需 ``ic_reports``：name -> {ic_mean, ic_std}；
        缺失则退化为逆方差/等权）。
      - ``inv_var``：逆方差（波动小者权重大）。
      - ``min_var``：最小方差（闭式，利用相关矩阵 + 方差）。

    Args:
        factor_dfs: name -> 因子面板。
        scheme: 权重方案。
        ic_reports: {name: {"ic_mean":..., "ic_std":...}}，供 ``icir``。
        standardize: 相关性/方差所用的截面标准化方法。

    Returns:
        ``{name: weight}``，权重和=1。
    """
    names = list(factor_dfs.keys())
    if not names:
        return {}
    n = len(names)
    if scheme == "equal":
        w = equal_weights(n)
    elif scheme == "icir":
        if ic_reports:
            means = [float(ic_reports.get(k, {}).get("ic_mean", 0.0) or 0.0) for k in names]
            stds = [float(ic_reports.get(k, {}).get("ic_std", 1.0) or 1.0) for k in names]
            w = icir_weights(means, stds, long_only=True)
        else:
            _logger.info("scheme=icir 但未提供 ic_reports，退化为逆方差")
            w = inverse_variance_weights(_panel_variances(factor_dfs, standardize))
    elif scheme == "inv_var":
        w = inverse_variance_weights(_panel_variances(factor_dfs, standardize))
    elif scheme == "min_var":
        w = min_variance_weights(
            _panel_corr(factor_dfs, standardize),
            _panel_variances(factor_dfs, standardize),
        )
    else:
        raise ValueError(f"未知权重方案: {scheme!r}（可选 equal/icir/inv_var/min_var）")
    return {k: float(wi) for k, wi in zip(names, w)}


def _daily_ic_ts(
    factor_dfs: Dict[str, pd.DataFrame],
    composite: pd.DataFrame,
    panel: "Panel",
    forward_periods: int,
    min_cross_section: int = 3,
) -> Dict[str, object]:
    """计算各因子 + 复合信号的**逐日截面 IC 时序**（JSON-safe）。

    对每个交易日取因子横截面值与前瞻收益的 **Spearman 秩相关**，得到逐日 IC 序列；
    返回 ``{"dates": [...], "factors": {name: [ic...]}, "composite": [ic...]}``，
    NaN → None 以利前端绘图。
    """
    fwd = panel.close.pct_change(forward_periods).shift(-forward_periods)
    dates = list(composite.index)
    factor_series: Dict[str, List[Optional[float]]] = {}
    comp_series: List[Optional[float]] = []
    dstr = [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
            for d in dates]
    for nm, fdf in factor_dfs.items():
        seq: List[Optional[float]] = []
        for d in dates:
            if d not in fdf.index or d not in fwd.index:
                seq.append(None)
                continue
            f = fdf.loc[d].dropna()
            r = fwd.loc[d].dropna()
            common = f.index.intersection(r.index)
            if len(common) < min_cross_section:
                seq.append(None)
                continue
            s = f[common].astype(float).rank()
            t = r[common].astype(float).rank()
            c = s.corr(t)
            seq.append(None if c != c else round(float(c), 4))
        factor_series[nm] = seq
    # 复合信号 IC
    for d in dates:
        if d not in composite.index or d not in fwd.index:
            comp_series.append(None)
            continue
        f = composite.loc[d].dropna()
        r = fwd.loc[d].dropna()
        common = f.index.intersection(r.index)
        if len(common) < min_cross_section:
            comp_series.append(None)
            continue
        s = f[common].astype(float).rank()
        t = r[common].astype(float).rank()
        c = s.corr(t)
        comp_series.append(None if c != c else round(float(c), 4))
    return {
        "dates": dstr,
        "factors": factor_series,
        "composite": comp_series,
    }


# --------------------------------------------------------------------------- #
# 高层：复合信号回测
# --------------------------------------------------------------------------- #
def composite_backtest(
    expressions: Sequence[str],
    panel: Panel,
    training_panel: Optional[Panel] = None,
    scheme: str = "icir",
    forward_periods: int = 1,
    n_groups: int = 5,
    long_short: bool = True,
    cost_rate: float = 0.0,
    standardize: str = "zscore",
    market: str = "",
    orthogonalize: bool = False,
    ic_reports: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, object]:
    """把一批表达式合成为复合 alpha 组合并回测。

    在 ``training_panel``（默认复用 ``panel``）上**求值 + 评估 IC + 优化权重**，
    再把各因子在 ``panel`` 上标准化合成为复合信号，转成每日横截面多空组合回测。

    Args:
        expressions: 代表性因子表达式列表（应已去冗余）。
        panel: 回测面板（不受权重拟合影响的前向一致性由调用方保证）。
        training_panel: 权重拟合面板；None → 用 ``panel``（探索性，非严格 OOS）。
        scheme: 权重方案（equal/icir/inv_var/min_var）。
        其余回测参数同 :func:`cross_sectional_backtest`。
        ic_reports: 可选的预计算 IC 报告 dict（name->{ic_mean,ic_std}）；
            None 时在 ``training_panel`` 上自行评估。

    Returns:
        ``{"scheme", "weights", "n_symbols", "n_dates", "ic_report",
          "factor_ics", "portfolio", "composite"}``。
    """
    exprs = [e for e in expressions if e and e.strip()]
    if not exprs:
        raise ValueError("无有效表达式")
    fit_panel = training_panel or panel

    # 1) 求值（权重拟合面板）→ 因子面板 dict
    factor_dfs: Dict[str, pd.DataFrame] = {}
    for e in exprs:
        try:
            factor_dfs[e] = panel_eval_expression(e, fit_panel)
        except Exception as exc:  # noqa: BLE001
            _logger.debug("表达式 %s 求值失败: %s", e, exc)
    if not factor_dfs:
        raise ValueError("全部表达式求值失败")
    exprs = list(factor_dfs.keys())

    # 2) 评估 IC（用于 icir 权重 + 因子贡献报告）
    ic_rep_map: Dict[str, Dict[str, float]] = {}
    evaluator = FactorEvaluator()
    for e in exprs:
        if ic_reports and e in ic_reports:
            ic_rep_map[e] = ic_reports[e]
            continue
        try:
            rep = evaluate_expression(e, fit_panel, forward_periods=forward_periods,
                                      market=market, use_cache=False)
            ic_rep_map[e] = {
                "ic_mean": float(rep.ic_mean) if rep.ic_mean == rep.ic_mean else 0.0,
                "ic_std": float(rep.ic_std) if rep.ic_std == rep.ic_std else 1.0,
                "ir": float(rep.ir) if rep.ir == rep.ir else 0.0,
            }
        except Exception as exc:  # noqa: BLE001
            _logger.debug("表达式 %s 评估失败: %s", e, exc)
            ic_rep_map[e] = {"ic_mean": 0.0, "ic_std": 1.0, "ir": 0.0}

    # 3) 优化权重（train）
    weights = optimize_weights(factor_dfs, scheme=scheme,
                               ic_reports=ic_rep_map, standardize=standardize)

    # 4) 在回测面板上重新求值因子（与权重拟合面板可能不同）
    back_factor_dfs: Dict[str, pd.DataFrame] = {}
    for e in exprs:
        try:
            back_factor_dfs[e] = panel_eval_expression(e, panel)
        except Exception as exc:  # noqa: BLE001
            _logger.debug("回测面板求值失败 %s: %s", e, exc)
    if not back_factor_dfs:
        raise ValueError("回测面板上全部表达式求值失败")
    wlist = [weights.get(e, 0.0) for e in exprs]
    composite = combine_factor_panels(
        back_factor_dfs, weights=wlist, standardize=standardize,
        orthogonalize=orthogonalize,
    )

    # 5) 复合信号截面多空回测
    gross_curve, net_curve, port_ret, g_perf, n_perf, turnover_list = _run_portfolio(
        panel, composite, forward_periods, n_groups, long_short, cost_rate)
    comp_ic = evaluator.evaluate_factor_panel(
        composite, panel, forward_periods=forward_periods,
        n_groups=n_groups, factor_name="composite")

    # 6) 因子两两相关矩阵（供前端热力图；标准化后跨期 stack 相关）
    corr_df = _panel_corr(back_factor_dfs, standardize)

    # 7) Barra 式多因子风险归因（完整版：风格暴露 + 截面回归因子收益 + 协方差分解）
    risk_attribution = None
    try:
        # 前瞻收益（与 _run_portfolio 同口径：pct_change(f).shift(-f)）
        fwd = panel.close.pct_change(forward_periods).shift(-forward_periods)
        # 风格暴露 = 各因子面板的截面标准化（clean 常用片段，剔除全缺失日期）
        exposures = {}
        for e in exprs:
            std = standardize_panel(back_factor_dfs[e], standardize)
            if std.empty:
                continue
            exposures[e] = std
        risk_attribution = barra_factor_risk_attribution(
            signal=composite,
            forward_returns=fwd,
            exposures=exposures,
            n_groups=n_groups,
            long_short=long_short,
        )
    except Exception as exc:  # noqa: BLE001 —— 归因失败不阻塞合成/回测
        _logger.warning("Barra 风险归因计算失败: %s", exc)
        risk_attribution = {"error": str(exc)[:200]}

    # 8) 因子/复合信号的逐日截面 IC 时序（供前端 C2 曲线）
    ic_ts = _daily_ic_ts(back_factor_dfs, composite, panel, forward_periods,
                         min_cross_section=min(6, max(3, len(panel.symbols) // 2)))

    return {
        "scheme": scheme,
        "weights": {k: round(float(v), 4) for k, v in weights.items()},
        "n_symbols": len(panel.symbols),
        "n_dates": len(gross_curve),
        "ic_report": comp_ic.to_dict(),
        "factor_ics": {k: round(float(v["ic_mean"]), 4) for k, v in ic_rep_map.items()},
        "ic_ts": ic_ts,
        "correlation": {
            "columns": list(corr_df.columns),
            "values": [[None if v != v else round(float(v), 4) for v in corr_df.loc[c]]
                       for c in corr_df.index],
        },
        "portfolio": _portfolio_output(g_perf, n_perf, port_ret, turnover_list),
        "gross_curve": gross_curve,
        "net_curve": net_curve,
        "risk_attribution": risk_attribution,
        "composite": composite,
    }
