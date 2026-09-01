"""因子去冗余 / 相关性聚类（对标 AlphaBench 中因子重叠与多因子的去共线）。

同一轮搜索（co/ea/tot）或批量评估会产出大量两两高度相关的因子（例如
``delta(close,20)`` 与 ``momentum_20`` 构成几乎相同）。直接全部进入组合会：
  1. 放大共线风险（组合有效敞口集中于同一隐性溢价）；
  2. 用掉组合容量、增加冗余换手；
  3. 让「因子数量」虚高，掩盖真实的信息维度。

本模块把一批因子（表达式或已算好的因子面板）按「截面 IC / 因子值 的 Pearson 相关
矩阵」做**贪婪聚类**：
  - 每次取未分配因子中 |IC| 最高的作为该簇代表；
  - 其余因子只要与代表相关 ≥ 阈值即并入该簇（去冗余）；
  - 迭代直到全部因子归簇。
每簇仅保留一个代表性因子，从而把「N 个高度相关的因子」压缩为「K 个信息维度的代表」。

无需 scipy：用确定性贪婪聚类替代层次聚类（对因子研究场景足够，且零依赖、可复现）。

用法::

    from quantmind.research import dedup_expressions, dedup_factor_panels

    # 1) 输入一批表达式 + 预计算的因子面板 dict
    kept = dedup_factor_panels(factor_dfs, correlation_threshold=0.7)
    # 返回 [{"expression": ..., "representative_of": [others], "n_removed": n}, ...]

    # 2) 直接对表达式列表去重（内部按需求值）
    kept = dedup_expressions(exprs, panel, correlation_threshold=0.7)
"""
from __future__ import annotations

import logging
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .factors.alpha_cs import Panel
from .factors.panel_expr import panel_eval_expression
from .eval import evaluate_expression

_logger = logging.getLogger("quantmind.research.dedup")

__all__ = [
    "factor_correlation_matrix",
    "greedy_cluster_dedup",
    "dedup_factor_panels",
    "dedup_expressions",
]


def factor_correlation_matrix(
    factors: Dict[str, "pd.DataFrame"],
) -> "pd.DataFrame":
    """计算一批因子面板两两的 Pearson 相关矩阵（index/columns = 因子名）。

    每个时间截面先对因子做 rank（稳健化，等同 Spearman），再逐列取有效交集后
    跨时间算相关。因子面板需 index（日期）一致。
    """
    import pandas as pd

    names = list(factors.keys())
    if len(names) < 2:
        cols = names or ["_dummy"]
        return pd.DataFrame(np.eye(len(names)), index=cols, columns=cols)
    # 对齐到公共时间轴（dropna 每因子各自保留；相关只取其交集）
    mat = pd.DataFrame(index=names, columns=names, dtype=float)
    for i, a in enumerate(names):
        fa = factors[a]
        ra = fa.rank(axis=1)  # 截面 rank 稳健化
        mat.loc[a, a] = 1.0
        for j in range(i + 1, len(names)):
            b = names[j]
            fb = factors[b]
            # 两因子共同有效样本（逐元素 nan 剔除后按行并集保留完整行）
            mask = fa.notna() & fb.notna()
            x = ra[mask].stack()
            y = fb[mask].rank(axis=1).stack()
            if len(x) < 10:
                r = float("nan")
            else:
                r = x.corr(y)
            mat.loc[a, b] = r
            mat.loc[b, a] = r
    return mat


def greedy_cluster_dedup(
    names: Sequence[str],
    corr_matrix: "pd.DataFrame",
    metric: Dict[str, float],
    correlation_threshold: float = 0.7,
    min_abs_metric: float = 0.0,
) -> List[Dict[str, object]]:
    """贪婪相关性聚类去冗余。

    按 ``metric``（如 |rank_ic|）从高到低取未分配因子为簇代表；与代表相关 ≥
    ``correlation_threshold`` 的未分配因子并入该簇并从候选移除。选代表的依据是
    「metric 最大」——默认给 |rank_ic|，即保留预测力最强的那一个。

    Args:
        names: 因子名（与 corr_matrix 对齐）。
        corr_matrix: 因子两两相关矩阵（DataFrame，index/columns=name）。
        metric: name -> 排序分值（越高越优先作代表）。推荐 |rank_ic|。
        correlation_threshold: 并簇相关阈值（[0,1]）。越高去重越保守。
        min_abs_metric: 代表因子 metric 的最小绝对值门槛（低于则视为噪声直接丢弃）。

    Returns:
        每个保留簇的信息：``{"name", "metric", "cluster", "representative"}``。
    """
    names = [n for n in names if n in corr_matrix.index]
    remaining = set(names)
    clusters: List[Dict[str, object]] = []

    def _score(n):
        v = metric.get(n)
        # 0.0 是合法 metric，不能因 falsy 被替换成 -inf（否则噪声丢弃逻辑失效）
        return float("-inf") if v is None else float(v)

    while remaining:
        # 取 metric 最高者为簇代表
        rep = max(remaining, key=_score)
        rep_score = _score(rep)
        if rep_score != rep_score or abs(rep_score) < min_abs_metric:
            # 剩余因子全部视为弱噪声：整体丢弃
            _logger.info("剩余 %d 个因子 |metric|<%s，视为噪声丢弃",
                         len(remaining), min_abs_metric)
            break
        cluster: List[str] = [rep]
        for other in list(remaining):
            if other == rep:
                continue
            r = float(corr_matrix.loc[rep, other])
            if r == r and r >= correlation_threshold:
                cluster.append(other)
        clusters.append({
            "name": rep,
            "metric": rep_score,
            "cluster": cluster,
            "representative": True,
        })
        remaining -= set(cluster)

    return clusters


def dedup_factor_panels(
    factor_dfs: Dict[str, "pd.DataFrame"],
    metric: Dict[str, float] | None = None,
    correlation_threshold: float = 0.7,
    min_abs_metric: float = 0.0,
) -> List[Dict[str, object]]:
    """对一批**已算好的因子面板**去冗余，返回每簇代表及其从属。

    Args:
        factor_dfs: name -> 因子面板 DataFrame（index=日期，columns=标的）。
        metric: name -> 排序分值（默认取面板 rank_ic 的近似：跨时间截面 rank 的
            波动代理不可得，故默认所有 1.0，代表按输入序取前者）。
        correlation_threshold: 并簇相关阈值。
        min_abs_metric: 代表 metric 最小绝对值门槛。

    Returns:
        ``[{"name", "metric", "cluster", "representative"}, ...]``，仅含代表簇
        （``representative=True``）。从属因子在 ``cluster`` 中列出。
    """
    names = list(factor_dfs.keys())
    if not names:
        return []
    corr = factor_correlation_matrix(factor_dfs)
    met = metric or {n: 1.0 for n in names}
    return greedy_cluster_dedup(names, corr, met, correlation_threshold,
                                min_abs_metric)


def dedup_expressions(
    expressions: Sequence[str],
    panel: Panel,
    correlation_threshold: float = 0.7,
    min_abs_metric: float = 0.0,
    forward_periods: int = 1,
    market: str = "",
    compute_ic: bool = True,
) -> List[Dict[str, object]]:
    """对一批**表达式**去冗余：默认各自求值 + 评估 IC，按 |rank_ic| 选代表。

    Args:
        expressions: 待去重的因子表达式列表。
        panel: 多标的面板（用于求值/评估）。
        correlation_threshold: 并簇相关阈值。
        min_abs_metric: |rank_ic| 门槛，低于则视为噪声丢弃。
        forward_periods: 前向周期。
        market: 市场标识（缓存键）。
        compute_ic: True 时用真实 rank_ic 作代表排序指标；False 时用表达式括号
            深度作复杂度代理（更快，适合超大候选池首筛）。

    Returns:
        ``[{"name(表达式)", "metric", "cluster", "representative"}, ...]``。
    """
    exprs = [e for e in expressions if e and e.strip()]
    if not exprs:
        return []
    uniq = list(dict.fromkeys(exprs))  # 去字面重复，保序
    factor_dfs: Dict[str, "pd.DataFrame"] = {}
    for e in uniq:
        try:
            factor_dfs[e] = panel_eval_expression(e, panel)
        except Exception as exc:  # noqa: BLE001
            _logger.debug("表达式 %s 求值失败，跳过去重: %s", e, exc)
    if not factor_dfs:
        return []

    # 先统一算 metric（|rank_ic| 或复杂度代理），两条路径都尊重 min_abs_metric
    met: Dict[str, float] = {}
    for e in factor_dfs:
        if not compute_ic:
            met[e] = float(e.count("("))  # 复杂度代理
            continue
        try:
            rep = evaluate_expression(e, panel, forward_periods=forward_periods,
                                      market=market, use_cache=False)
            ic = rep.ic_mean
            met[e] = abs(float(ic)) if ic == ic else 0.0
        except Exception as exc:  # noqa: BLE001
            _logger.debug("表达式 %s 评估失败，metric=0: %s", e, exc)
            met[e] = 0.0

    if len(factor_dfs) < 2:
        # 只有一个可求值因子：作为唯一代表（但需过 min_abs_metric 门槛）
        only = next(iter(factor_dfs))
        if abs(met.get(only, 0.0)) < min_abs_metric:
            _logger.info("仅有的因子 %s |metric|<%s，视为噪声丢弃", only, min_abs_metric)
            return []
        return [{"name": only, "metric": met.get(only, 0.0),
                 "cluster": [only], "representative": True}]

    corr = factor_correlation_matrix(factor_dfs)
    return greedy_cluster_dedup(list(factor_dfs.keys()), corr, met,
                                correlation_threshold, min_abs_metric)
