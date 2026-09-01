"""参数优化：网格枚举 + IS/OOS 切分 + 参数高原检验 + Deflated Sharpe。

为「LLM策略挖掘」提供带防过拟合防线的参数优化纯函数库（无 IO、不依赖回测引擎）：

1. **网格枚举**　`enumerate_grid` —— 参数候选值笛卡尔积，带组合数上限；
2. **邻域组合**　`neighbor_combos` —— 参数高原检验用（每维度取相邻候选值）；
3. **IS/OOS 切分**　`split_is_oos` —— 按时间顺序切分，OOS 段可带预热窗口
   （从 IS 尾部借 K 线供指标初始化，策略在预热期不产生交易）；
4. **Deflated Sharpe**　`deflated_sharpe` —— 按 Bailey & López de Prado (2014)
   对「N 次试验挑最大 Sharpe」的选择偏差做校正，返回概率值（0~1）；
5. **参数高原**　`plateau_check` —— 最优组合的邻域中位数 Sharpe / 最优 Sharpe
   达到阈值才视为稳健「高原」，否则判定为样本内噪声「尖峰」。

防过拟合三道防线（切分 / 高原 / DSR）互为冗余：任何一道不过，参数优化结果
都不应直接入库。
"""
from __future__ import annotations

import math
from statistics import NormalDist
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.object import BarData

_EULER_GAMMA = 0.5772156649015329  # Euler–Mascheroni 常数（López de Prado 论文用的 γ）


# ---------------------------------------------------------------------------
# 网格枚举
# ---------------------------------------------------------------------------

def enumerate_grid(param_grid: Dict[str, Sequence[float]],
                   max_combos: int = 60) -> List[Dict[str, float]]:
    """把 {参数名: 候选值列表} 枚举为参数组合列表（笛卡尔积）。

    :param param_grid: 参数名 → 候选值列表（每维 1~6 个值；单值=该维固定）。
    :param max_combos: 组合数硬上限，超过即抛 ValueError（前端应提示缩减网格）。
    :raises ValueError: 网格为空、候选值非法或组合数超上限。
    """
    if not param_grid:
        raise ValueError("param_grid 不能为空")
    for name, values in param_grid.items():
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(f"参数 {name} 的候选值必须是非空列表")
        if len(values) > 6:
            raise ValueError(f"参数 {name} 候选值过多（{len(values)} > 6），请缩减")
        for v in values:
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise ValueError(f"参数 {name} 含非数值候选值: {v!r}")

    names = list(param_grid.keys())
    combos: List[Dict[str, float]] = [{}]

    for name in names:
        values = param_grid[name]
        combos = [dict(c, **{name: v}) for c in combos for v in values]
        if len(combos) > max_combos:
            raise ValueError(
                f"网格组合数 {len(combos)} 超过上限 {max_combos}，请缩减候选值")

    if not combos:
        raise ValueError("网格枚举结果为空")
    return combos


def fit_grid(param_grid: Dict[str, Sequence[float]],
             max_combos: int = 60) -> Dict[str, List[float]]:
    """削减参数网格，使笛卡尔积 ≤ max_combos（多参数策略友好）。

    策略：轮流去掉候选值最多的维的**中间值**（保留端点，保住高原检验的邻域）；
    所有维都降到 2 个候选仍超限时，从最后一维开始整维删除。
    返回削减后的新网格（不改传入值）。
    """
    grid = {k: list(v) for k, v in param_grid.items()}

    def _product(g: Dict[str, list]) -> int:
        p = 1
        for v in g.values():
            p *= max(1, len(v))
        return p

    while _product(grid) > max_combos and grid:
        if len(grid) == 1:
            # 单维：保留前 max_combos 个（有序网格的前缀，通常代表从快到慢）
            only = next(iter(grid))
            grid[only] = grid[only][:max_combos]
            break
        name = max(grid, key=lambda k: len(grid[k]))
        if len(grid[name]) > 2:
            vals = grid[name]
            del vals[len(vals) // 2]
        else:
            del grid[list(grid.keys())[-1]]
    return grid


# 执行类参数：线性缩放仓位/资金，不改变信号序列，纳入网格纯属浪费试验数
_NON_SIGNAL_PARAMS = {"size", "max_pos", "capital", "inverse", "fixed_size"}


def auto_param_grid(strategy_class,
                    max_dims: int = 4, per_dim: int = 3,
                    source: Optional[str] = None) -> Dict[str, List[float]]:
    """从策略类源码自动推导参数搜索网格（**通用，不挑策略**）。

    AST 解析 ``__init__`` 中 ``self.<参数> = <数值字面量>`` 的默认值，
    生成默认值邻域：int 维 ``[默认÷2, 默认, ×2]``（下界 1），
    float 维 ``[0.5×, 默认, 1.5×]``——任意策略可用，含 LLM 生成的策略。

    跳过：未在 parameters 声明、默认值非字面量（如 ``self.x = compute()``）、
    非数值（bool/str/None）、0（无法猜测尺度）、以及 size/max_pos 等
    执行类参数（线性缩放不改变信号）。最多取前 max_dims 个参数，
    每维最多 per_dim 个候选值。无法推导时返回空 dict（调用方回退预置网格）。
    """
    import ast as _ast
    import inspect as _inspect

    declared = list(getattr(strategy_class, "parameters", []) or [])
    if not declared:
        return {}
    try:
        if source:
            tree = _ast.parse(source)
        else:
            tree = _ast.parse(_inspect.getsource(strategy_class))
    except (OSError, TypeError, SyntaxError, IndentationError):
        return {}

    def _is_self_attr(node) -> bool:
        return (isinstance(node, _ast.Attribute)
                and isinstance(node.value, _ast.Name) and node.value.id == "self")

    def _literal_number(expr):
        """ast 表达式 → 数值字面量（支持 -1 负号字面量）；非字面量返回 None。"""
        if isinstance(expr, _ast.Constant) and isinstance(expr.value, (int, float)) \
                and not isinstance(expr.value, bool):
            return expr.value
        if (isinstance(expr, _ast.UnaryOp) and isinstance(expr.op, _ast.USub)
                and isinstance(expr.operand, _ast.Constant)):
            v = expr.operand.value
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return -v
        return None

    defaults: Dict[str, object] = {}
    for node in _ast.walk(tree):
        if not (isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef))
                and node.name == "__init__"):
            continue
        for stmt in _ast.walk(node):
            target_attr = value_expr = None
            if (isinstance(stmt, _ast.Assign) and len(stmt.targets) == 1
                    and _is_self_attr(stmt.targets[0])):
                target_attr, value_expr = stmt.targets[0].attr, stmt.value
            elif isinstance(stmt, _ast.AnnAssign) and _is_self_attr(stmt.target):
                target_attr, value_expr = stmt.target.attr, stmt.value
            else:
                continue
            if target_attr in defaults:
                continue
            v = _literal_number(value_expr)
            if v is not None:
                defaults[target_attr] = v

    grid: Dict[str, List[float]] = {}
    for name in declared:
        if len(grid) >= max_dims:
            break
        if name in _NON_SIGNAL_PARAMS:
            continue
        v = defaults.get(name)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v == 0:
            continue
        if isinstance(v, int):
            cands = sorted({max(1, v // 2), v, v * 2})
        else:
            cands = sorted({round(v * 0.5, 6), v, round(v * 1.5, 6)})
        grid[name] = cands[:per_dim]
    return grid


def combo_key(combo: Dict[str, float]) -> Tuple:
    """把参数组合转为可哈希 key（用于结果缓存查找）。"""
    return tuple(sorted(combo.items()))


def neighbor_combos(param_grid: Dict[str, Sequence[float]],
                    combo: Dict[str, float]) -> List[Dict[str, float]]:
    """构造最优组合的**网格邻域**：每个参数维度取相邻候选值（其余维度不变）。

    例：grid={"w":[10,20,30], "k":[1.5,2.0]}，combo={"w":20,"k":2.0}
        → 邻居 = [{"w":10,"k":2.0}, {"w":30,"k":2.0}, {"w":20,"k":1.5}]
    """
    neighbors: List[Dict[str, float]] = []
    for name, values in param_grid.items():
        values = list(values)
        if len(values) < 2 or name not in combo:
            continue
        try:
            idx = values.index(combo[name])
        except ValueError:
            continue  # combo 不在该维网格内（理论上不应发生），跳过该维
        for j in (idx - 1, idx + 1):
            if 0 <= j < len(values):
                neighbor = dict(combo)
                neighbor[name] = values[j]
                neighbors.append(neighbor)
    return neighbors


# ---------------------------------------------------------------------------
# IS / OOS 切分
# ---------------------------------------------------------------------------

def split_is_oos(bars: List[BarData],
                 is_ratio: float = 0.7,
                 min_oos_bars: int = 60,
                 warmup_bars: int = 120,
                 ) -> Tuple[List[BarData], List[BarData], Dict]:
    """按时间顺序把 K 线切分为样本内（IS）与样本外（OOS）两段。

    OOS 段回测时会从 IS 尾部借 ``warmup_bars`` 根作指标预热（策略在预热期
    指标未就绪、不产生交易），避免 OOS 开头因 ArrayManager 未初始化而空跑。

    :param bars: 全量 K 线（按时间升序；未排序时内部会先排序）。
    :param is_ratio: IS 段占比（0~1，不含 OOS）。
    :param min_oos_bars: OOS 最少根数，不足则**降级**：不优化（oos_bars=0）。
    :param warmup_bars: OOS 回测预热根数（从 IS 尾部借）。
    :return: (is_bars, oos_bars_with_warmup, info)。
        降级时 oos_bars_with_warmup 为空列表、info["degraded"]=True。
    """
    if not 0.1 <= is_ratio <= 0.9:
        raise ValueError(f"is_ratio 必须在 [0.1, 0.9]，当前 {is_ratio}")
    if not bars:
        raise ValueError("bars 为空，无法切分")

    bars_sorted = sorted(bars, key=lambda b: b.datetime)
    n = len(bars_sorted)
    n_is = int(n * is_ratio)
    n_oos = n - n_is
    info = {"total": n, "is_bars": n_is, "oos_bars": 0,
            "warmup_bars": 0, "degraded": False}

    if n_oos < min_oos_bars or n_is < 50:
        info["degraded"] = True
        return bars_sorted, [], info

    is_bars = bars_sorted[:n_is]
    oos_bars = bars_sorted[n_is:]
    # 预热：从 IS 尾部借 K 线拼在 OOS 前面（策略预热期不交易，不污染 OOS 绩效）
    k = min(warmup_bars, max(n_is - 50, 0))  # IS 至少保留 50 根供自身回测
    oos_with_warmup = bars_sorted[n_is - k:] if k > 0 else oos_bars
    info.update({"oos_bars": len(oos_bars), "warmup_bars": k})
    return is_bars, oos_with_warmup, info


# ---------------------------------------------------------------------------
# Deflated Sharpe（Bailey & López de Prado, 2014）
# ---------------------------------------------------------------------------

def _skew_kurt(returns: np.ndarray) -> Tuple[float, float]:
    """日收益的偏度与峰度（Pearson，正态=3；总体矩，ddof=0）。"""
    m = returns.mean()
    d = returns - m
    m2 = float(np.mean(d ** 2))
    if m2 <= 0:
        return 0.0, 3.0
    skew = float(np.mean(d ** 3) / m2 ** 1.5)
    kurt = float(np.mean(d ** 4) / (m2 ** 2))
    return skew, kurt


def daily_returns_from_equity(equity_curve: List[Dict]) -> np.ndarray:
    """从回测输出的净值曲线（[{"date": ..., "equity": ...}]）重建日收益序列。"""
    if not equity_curve:
        return np.array([])
    equities = np.array([float(e["equity"]) for e in equity_curve], dtype=float)
    if len(equities) < 2:
        return np.array([])
    base = equities[:-1]
    base[base == 0] = np.nan
    rets = equities[1:] / base - 1.0
    return rets[~np.isnan(rets)]


def deflated_sharpe(sharpe_annual: float,
                    returns: np.ndarray,
                    n_trials: int,
                    trial_sharpes: Sequence[float] = (),
                    tdpy: int = 252) -> float:
    """Deflated Sharpe Ratio：把「N 次试验挑出的最大 Sharpe」折算为可信概率。

    :param sharpe_annual: 观测到的**年化** Sharpe（回测报告口径）。
    :param returns: 与该 Sharpe 对应的日收益序列（由 equity_curve 重建）。
    :param n_trials: 同一数据上尝试的参数组合总数（选择偏差来源）。
    :param trial_sharpes: 各次试验的年化 Sharpe 列表（用于估计跨试验方差；
        为空时用保守估计 sr_var = 1/tdpy，即日 Sharpe 波动 1/√tdpy）。
    :param tdpy: 每年交易日数（与回测引擎 PerformanceReport 口径一致，默认 252）。
    :return: DSR = P[SR* < SR观测]，0~1；越高越可信。入库建议阈值 ≥ 0.90。

    公式（JPM 2014）：
        SR₀ = √V[SRₙ] · ((1-γ)·Φ⁻¹(1-1/N) + γ·Φ⁻¹(1-1/(N·e)))
        DSR = Φ( (SR-SR₀)·√(T-1) / √(1-skew·SR+((kurt-1)/4)·SR²) )
    其中 SR 为**日频** Sharpe（年化值 ÷ √tdpy），N=1 时退化为 PSR（SR₀=0）。
    """
    T = len(returns)
    if T < 10 or n_trials < 1:
        return 0.0
    std = float(np.std(returns))
    if std <= 0:
        return 0.0  # 无波动（无交易）→ 无信息

    sr = float(sharpe_annual) / (tdpy ** 0.5)     # 年化 → 日频
    skew, kurt = _skew_kurt(returns)

    # 跨试验 Sharpe 的方差（同尺度：日频口径）
    if n_trials >= 2 and len(trial_sharpes) >= 2:
        trial_daily = np.asarray(trial_sharpes, dtype=float) / (tdpy ** 0.5)
        sr_var = float(np.var(trial_daily, ddof=1))
    else:
        sr_var = 1.0 / tdpy                        # 保守先验：日 Sharpe 波动 ≈ 1/√252

    # 期望最大 Sharpe（选择偏差基准）
    if n_trials > 1:
        z1 = NormalDist().inv_cdf(1 - 1.0 / n_trials)
        z2 = NormalDist().inv_cdf(1 - 1.0 / (n_trials * np.e))
        sr0 = float(np.sqrt(max(sr_var, 0.0))
                    * ((1 - _EULER_GAMMA) * z1 + _EULER_GAMMA * z2))
    else:
        sr0 = 0.0                                  # N=1 退化为 PSR

    denom = 1 - skew * sr + ((kurt - 1) / 4.0) * sr * sr
    if denom <= 0:
        return 0.0
    z = (sr - sr0) * np.sqrt(T - 1) / np.sqrt(denom)
    return float(NormalDist().cdf(z))


# ---------------------------------------------------------------------------
# 参数高原检验
# ---------------------------------------------------------------------------

def plateau_check(param_grid: Dict[str, Sequence[float]],
                  best_combo: Dict[str, float],
                  is_sharpes: Dict[Tuple, float],
                  ratio_threshold: float = 0.6) -> Dict:
    """参数高原检验：最优组合的邻域 IS Sharpe 中位数须 ≥ 阈值 × 最优值。

    尖峰（换个相邻参数就崩）通常是样本内噪声伪影；宽高原才可信。
    :param is_sharpes: {(组合key): IS 年化 Sharpe}，来自 IS 段穷举缓存。
    :return: {"ok": bool, "median_ratio": float|None, "neighbors": int, "reason": str}
    """
    best_key = combo_key(best_combo)
    best_sharpe = is_sharpes.get(best_key, 0.0)

    if best_sharpe <= 0:
        return {"ok": False, "median_ratio": None, "neighbors": 0,
                "reason": "最优组合 IS Sharpe ≤ 0，无高原可言"}

    neighbors = neighbor_combos(param_grid, best_combo)
    if not neighbors:
        # 退化为固定参数（无搜索）：不存在选择偏差，视为通过
        return {"ok": True, "median_ratio": None, "neighbors": 0,
                "reason": "网格无邻居可比较（参数未搜索），跳过高原检验"}
    found = [is_sharpes[combo_key(nb)] for nb in neighbors
             if combo_key(nb) in is_sharpes]
    if not found:
        return {"ok": True, "median_ratio": None, "neighbors": 0,
                "reason": "邻域结果缺失（不应发生），保守放行"}
    median = float(np.median(found))
    ratio = median / best_sharpe if best_sharpe > 0 else 0.0
    return {"ok": bool(ratio >= ratio_threshold),
            "median_ratio": round(ratio, 4),
            "neighbors": len(found),
            "reason": (f"邻域中位 Sharpe {median:.2f} / 最优 {best_sharpe:.2f}"
                       f" = {ratio:.0%}（阈值 {ratio_threshold:.0%}）")}
