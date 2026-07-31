"""因子有效性评估器（参考截面 IC / 信息比率 IR / IC 衰减 / 分位收益 / 换手 / 单调性 / 综合主分）。

单一标的（时序）场景：
  - 时序 IC：因子值 t 与 未来收益 t+horizon 的 Spearman（rank）相关（全样本）。
  - Pearson IC：同上但用原始值 Pearson 相关（与 rank IC 对照，捕捉线性/单调差异）。
  - IR：滚动窗口（默认 60）时序 IC 的均值/标准差。
  - IC 衰减：horizon=1..5 的 IC 序列；并拟合指数衰减求「半衰期」。
  - 分位收益：按因子值分 5/10 组，统计多头组与多空组未来平均收益，及分组单调性。
  - 分位组合：用扩张窗口分组（无前视）构造多空组合曲线，给出累计收益/Sharpe/最大回撤。
  - 换手率：因子标准化信号的年化换手（信号变化幅度 × 年化周期数）。
  - 综合主分：v2 归一加权（IC/Sharpe/收益/MDD/单调性/换手）。

多标的（截面）场景：
  - 每个时间截面上对所有标的计算因子值与未来收益的 Spearman/Pearson 相关，再对时间取均值/标准差。
  - 截面分位组合：每截面按因子分 5 组，多空组合曲线。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .factors.base import bars_to_df, expanding_zscore
from ..core.object import BarData


def _spearman(a: pd.Series, b: pd.Series) -> float:
    """Spearman 相关：对 a/b 先排序再取 Pearson（不依赖 scipy，兼容 pandas 3.x）。"""
    a = a.dropna()
    b = b.loc[a.index].dropna()
    common = a.index.intersection(b.index)
    if len(common) < 10:
        return float("nan")
    a = a.loc[common].rank()
    b = b.loc[common].rank()
    try:
        return float(a.corr(b))
    except Exception:  # noqa: BLE001
        return float("nan")


def _pearson(a: pd.Series, b: pd.Series) -> float:
    """Pearson 相关（原始值，不排序）。"""
    a = a.dropna()
    b = b.loc[a.index].dropna()
    common = a.index.intersection(b.index)
    if len(common) < 10:
        return float("nan")
    a = a.loc[common]
    b = b.loc[common]
    try:
        return float(a.corr(b))
    except Exception:  # noqa: BLE001
        return float("nan")


def _periods_per_year(bars: List[BarData]) -> int:
    """依据 K 线间隔推断年化周期数（日线≈252，分钟线更高）。"""
    if not bars or len(bars) < 2:
        return 252
    try:
        d0, d1 = bars[0].datetime, bars[1].datetime
        days = abs((d1 - d0).days)
    except Exception:  # noqa: BLE001
        days = 1
    if days <= 0:
        days = 1
    return max(1, round(365.0 / days))


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-2.0 * x))
    except OverflowError:
        return 1.0 if x > 0 else 0.0


@dataclass
class FactorReport:
    """因子评估报告。"""

    factor_name: str = ""
    ic_mean: float = float("nan")          # rank IC（全样本）
    ic_pearson: float = float("nan")       # pearson IC（对照）
    ic_std: float = float("nan")
    ir: float = float("nan")               # rank IC 的 IR
    ic_positive_ratio: float = float("nan")
    ic_decay: List[float] = field(default_factory=list)        # 1..5 期衰减
    ic_decay_half_life: float = float("nan")                   # 半衰期（周期数）
    ic_ci_low: float = float("nan")        # bootstrap 95% 下界
    ic_ci_high: float = float("nan")       # bootstrap 95% 上界
    top_quantile_return: float = float("nan")
    long_short_return: float = float("nan")                   # 组内多空均值差
    monotonicity_5: float = float("nan")                      # 5 组单调性
    monotonicity_10: float = float("nan")                     # 10 组单调性
    turnover_annual: float = float("nan")                     # 年化换手
    ls_portfolio_return: float = float("nan")                 # 多空组合累计收益
    ls_portfolio_sharpe: float = float("nan")
    ls_portfolio_mdd: float = float("nan")
    composite_score: float = float("nan")                     # v2 综合主分 [0,1]
    n_samples: int = 0
    note: str = ""

    def to_dict(self) -> dict:
        def r4(x):
            return round(float(x), 4) if pd.notna(x) else None

        def r6(x):
            return round(float(x), 6) if pd.notna(x) else None

        return {
            "factor_name": self.factor_name,
            "ic_mean": r4(self.ic_mean),
            "ic_pearson": r4(self.ic_pearson),
            "ic_std": r4(self.ic_std),
            "ir": r4(self.ir),
            "ic_positive_ratio": r4(self.ic_positive_ratio),
            "ic_decay": [r4(x) for x in self.ic_decay],
            "ic_decay_half_life": r4(self.ic_decay_half_life),
            "ic_ci_low": r4(self.ic_ci_low),
            "ic_ci_high": r4(self.ic_ci_high),
            "top_quantile_return": r6(self.top_quantile_return),
            "long_short_return": r6(self.long_short_return),
            "monotonicity_5": r4(self.monotonicity_5),
            "monotonicity_10": r4(self.monotonicity_10),
            "turnover_annual": r4(self.turnover_annual),
            "ls_portfolio_return": r6(self.ls_portfolio_return),
            "ls_portfolio_sharpe": r4(self.ls_portfolio_sharpe),
            "ls_portfolio_mdd": r4(self.ls_portfolio_mdd),
            "composite_score": r4(self.composite_score),
            "n_samples": self.n_samples,
            "note": self.note,
        }


class FactorEvaluator:
    """因子有效性评估。"""

    def evaluate(
        self,
        factor_values: pd.Series,
        bars: List[BarData],
        forward_periods: int = 1,
        roll_window: int = 60,
        n_groups: int = 5,
        periods_per_year: Optional[int] = None,
        bootstrap: bool = True,
        n_bootstrap: int = 500,
    ) -> FactorReport:
        """评估单一标的因子。

        ``factor_values`` 与 ``bars`` 等长（按时间升序）。返回 ``FactorReport``。
        """
        df = bars_to_df(bars)
        if df.empty or len(factor_values) != len(df):
            return FactorReport(
                factor_name=getattr(factor_values, "name", "factor"),
                note="数据为空或长度不匹配",
            )
        fv = factor_values.reset_index(drop=True)
        fwd_ret = df["close"].pct_change(forward_periods).shift(-forward_periods).reset_index(drop=True)
        factor_name = getattr(factor_values, "name", "factor")
        ppy = periods_per_year or _periods_per_year(bars)

        # 时序 IC（全样本）：rank + pearson
        ic_mean = _spearman(fv, fwd_ret)
        ic_pearson = _pearson(fv, fwd_ret)

        # 滚动 IC -> IR
        ic_series = pd.Series(index=range(len(df)), dtype=float)
        for i in range(roll_window, len(df)):
            sub_f = fv.iloc[max(0, i - roll_window): i]
            sub_r = fwd_ret.iloc[max(0, i - roll_window): i]
            ic_series.iloc[i] = _spearman(sub_f, sub_r)
        ic_valid = ic_series.dropna()
        ic_std = float(ic_valid.std()) if len(ic_valid) > 1 else float("nan")
        ir = (float(ic_valid.mean()) / ic_std) if (ic_std and pd.notna(ic_std) and ic_std > 0) else float("nan")
        ic_pos_ratio = float((ic_valid > 0).mean()) if len(ic_valid) else float("nan")

        # IC 衰减（1..5 期）
        decay = []
        for h in range(1, 6):
            r = df["close"].pct_change(h).shift(-h).reset_index(drop=True)
            decay.append(_spearman(fv, r))
        half_life = self._ic_decay_half_life(decay)

        # 分组收益 + 单调性
        top_ret, ls_ret = self._quantile_returns(fv, fwd_ret)
        mono5 = self._monotonicity(fv, fwd_ret, 5)
        mono10 = self._monotonicity(fv, fwd_ret, 10)

        # 多空组合曲线
        ls_ret_series = self._long_short_portfolio_single(fv, fwd_ret, n_groups, forward_periods)
        ls_total, ls_sharpe, ls_mdd = self._portfolio_stats(ls_ret_series, ppy)

        # 年化换手（标准化信号的变化幅度）
        z = expanding_zscore(fv)
        turn = float(z.diff().abs().mean()) if z.notna().any() else float("nan")
        turnover_annual = turn * ppy if pd.notna(turn) else float("nan")

        # Bootstrap 置信区间（IC 均值）
        ci_low, ci_high = (float("nan"), float("nan"))
        if bootstrap and len(fv) >= 30:
            ci_low, ci_high = self._bootstrap_ci(fv, fwd_ret, n_bootstrap)

        # 综合主分
        composite = self._composite(
            ic_mean=ic_mean, ls_return=ls_total, ls_sharpe=ls_sharpe,
            ls_mdd=ls_mdd, monotonicity=max(mono5, mono10), turnover=turnover_annual,
        )

        return FactorReport(
            factor_name=factor_name,
            ic_mean=ic_mean,
            ic_pearson=ic_pearson,
            ic_std=ic_std,
            ir=ir,
            ic_positive_ratio=ic_pos_ratio,
            ic_decay=decay,
            ic_decay_half_life=half_life,
            ic_ci_low=ci_low,
            ic_ci_high=ci_high,
            top_quantile_return=top_ret,
            long_short_return=ls_ret,
            monotonicity_5=mono5,
            monotonicity_10=mono10,
            turnover_annual=turnover_annual,
            ls_portfolio_return=ls_total,
            ls_portfolio_sharpe=ls_sharpe,
            ls_portfolio_mdd=ls_mdd,
            composite_score=composite,
            n_samples=len(df),
        )

    # ---- 分位收益 / 单调性 ----
    @staticmethod
    def _quantile_returns(fv: pd.Series, fwd_ret: pd.Series) -> tuple:
        df = pd.DataFrame({"f": fv, "r": fwd_ret}).dropna()
        if len(df) < 30:
            return float("nan"), float("nan")
        try:
            df["q"] = pd.qcut(df["f"].rank(method="first"), 5, labels=False)
        except Exception:  # noqa: BLE001
            return float("nan"), float("nan")
        grouped = df.groupby("q")["r"].mean()
        top = float(grouped.iloc[-1]) if len(grouped) == 5 else float("nan")
        bottom = float(grouped.iloc[0]) if len(grouped) == 5 else float("nan")
        return top, (top - bottom)

    def _monotonicity(self, fv: pd.Series, fwd_ret: pd.Series, n_groups: int) -> float:
        """分组平均收益是否随因子单调递增：用组号与组均收益的线性趋势衡量（[-1,1]）。

        用分组均值的相关系数（小样本直接用 Pearson，不依赖 _spearman 的 10 样本下限）。
        """
        df = pd.DataFrame({"f": fv, "r": fwd_ret}).dropna()
        if len(df) < n_groups * 6:
            return float("nan")
        try:
            df["q"] = pd.qcut(df["f"].rank(method="first"), n_groups, labels=False)
        except Exception:  # noqa: BLE001
            return float("nan")
        grp = df.groupby("q")["r"].mean().reset_index(drop=True)
        if len(grp) < 3:
            return float("nan")
        x = pd.Series(range(len(grp)), dtype=float)
        y = grp.astype(float)
        if y.std() == 0 or pd.isna(y.std()):
            return 0.0
        return float(x.corr(y))

    # ---- 多空组合（无前视：扩张窗口分组）----
    @staticmethod
    def _long_short_portfolio_single(fv: pd.Series, fwd_ret: pd.Series, n_groups: int, forward_periods: int) -> pd.Series:
        """每根 K 线用截至 t 的扩张窗口把因子分 n 组；处于最高组做多、最低组做空未来收益。"""
        out = []
        f = fv.reset_index(drop=True)
        r = fwd_ret.reset_index(drop=True)
        for t in range(len(f)):
            if pd.isna(r.iloc[t]):
                out.append(0.0)
                continue
            hist = f.iloc[: t + 1]
            if hist.notna().sum() < n_groups + 1:
                out.append(0.0)
                continue
            try:
                grp = pd.qcut(hist.rank(method="first"), n_groups, labels=False, duplicates="drop")
            except Exception:  # noqa: BLE001
                out.append(0.0)
                continue
            g = int(grp.iloc[-1])
            if g == n_groups - 1:
                out.append(float(r.iloc[t]))
            elif g == 0:
                out.append(-float(r.iloc[t]))
            else:
                out.append(0.0)
        return pd.Series(out)

    @staticmethod
    def _portfolio_stats(period_returns: pd.Series, periods_per_year: int) -> Tuple[float, float, float]:
        """由每期收益序列算累计收益 / 年化 Sharpe / 最大回撤。"""
        s = period_returns.dropna()
        if len(s) < 5:
            return float("nan"), float("nan"), float("nan")
        cum = (1.0 + s).cumprod()
        total = float(cum.iloc[-1] - 1.0)
        mean_r = float(s.mean())
        std_r = float(s.std())
        sharpe = (mean_r / std_r * (periods_per_year ** 0.5)) if (std_r and std_r > 0) else 0.0
        # 最大回撤（基于净值曲线）
        running_max = cum.cummax()
        dd = cum / running_max - 1.0
        mdd = float(dd.min())
        return total, sharpe, mdd

    # ---- IC 衰减半衰期 ----
    @staticmethod
    def _ic_decay_half_life(decay: List[float]) -> float:
        """对 decay[0..]（lag 1..5 的 IC）拟合 ln|IC| = a - b*lag，半衰期 = ln2 / b。"""
        xs = np.arange(1, len(decay) + 1, dtype=float)
        ys = np.array([abs(d) for d in decay], dtype=float)
        mask = np.isfinite(ys) & (ys > 1e-9)
        if mask.sum() < 3:
            return float("nan")
        x = xs[mask]
        y = np.log(ys[mask])
        try:
            slope = float(np.polyfit(x, y, 1)[0])
        except Exception:  # noqa: BLE001
            return float("nan")
        if slope >= 0:
            return float("inf") if slope == 0 else float("nan")
        return float(math.log(2.0) / (-slope))

    # ---- Bootstrap 置信区间 ----
    @staticmethod
    def _bootstrap_ci(fv: pd.Series, fwd_ret: pd.Series, n: int = 500, alpha: float = 0.05) -> Tuple[float, float]:
        """对 rank IC 均值做 Bootstrap 百分位置信区间。"""
        fv = fv.reset_index(drop=True)
        fwd_ret = fwd_ret.reset_index(drop=True)
        df = pd.DataFrame({"f": fv, "r": fwd_ret}).dropna()
        if len(df) < 30:
            return float("nan"), float("nan")
        rng = np.random.default_rng(42)
        means = np.empty(n, dtype=float)
        size = len(df)
        for i in range(n):
            idx = rng.integers(0, size, size)
            means[i] = _spearman(df["f"].iloc[idx].reset_index(drop=True),
                                 df["r"].iloc[idx].reset_index(drop=True))
        lo = float(np.nanpercentile(means, 100 * alpha / 2))
        hi = float(np.nanpercentile(means, 100 * (1 - alpha / 2)))
        return lo, hi

    # ---- 综合主分（v2 归一加权）----
    @staticmethod
    def _composite(ic_mean, ls_return, ls_sharpe, ls_mdd, monotonicity, turnover) -> float:
        """v2 权重：IC 0.2 + Sharpe 0.3 + 收益 0.3 + MDD 0.2 + 单调性 0.1 + 换手惩罚 0.1。

        各子分归一到 [0,1]（绝对值衡量预测力，方向由策略翻转处理）。
        """
        def num(x):
            return float(x) if pd.notna(x) else float("nan")

        ic = num(ic_mean)
        ret = num(ls_return)
        shp = num(ls_sharpe)
        mdd = num(ls_mdd)
        mon = num(monotonicity)
        to = num(turnover)

        ic_s = _sigmoid(ic / 0.05) if pd.notna(ic) else 0.5
        shp_s = _sigmoid(shp / 1.0) if pd.notna(shp) else 0.5
        ret_s = _sigmoid(ret / 0.30) if pd.notna(ret) else 0.5
        mdd_s = (1.0 - _sigmoid(abs(mdd) / 0.20)) if pd.notna(mdd) else 0.5
        mon_s = max(0.0, min(1.0, abs(mon))) if pd.notna(mon) else 0.5
        to_s = (1.0 - _sigmoid(to / 10.0)) if pd.notna(to) else 0.5

        return float(
            0.2 * ic_s + 0.3 * shp_s + 0.3 * ret_s + 0.2 * mdd_s + 0.1 * mon_s + 0.1 * to_s
        )

    # ---- 多标的截面评估 ----
    def evaluate_cross_sectional(
        self,
        factor_by_symbol: Dict[str, pd.Series],
        bars_by_symbol: Dict[str, List[BarData]],
        forward_periods: int = 1,
        n_groups: int = 5,
        periods_per_year: int = 252,
        bootstrap: bool = True,
    ) -> FactorReport:
        """多标的截面 IC：每个时间截面计算 Spearman/Pearson，再对时间取均值/标准差；
        并构造截面多空组合曲线。以各标的对齐到共同交易日（取交集）进行截面计算。"""
        fwd_ret_map = {}
        for sym, bars in bars_by_symbol.items():
            closes = pd.Series(
                [b.close_price for b in bars],
                index=[b.datetime for b in bars],
            )
            ret = closes.pct_change(forward_periods)
            fwd_ret_map[sym] = ret.shift(-forward_periods)
        base = list(factor_by_symbol.keys())[0]
        dates = list(factor_by_symbol[base].index)
        ic_list: List[float] = []
        ic_p_list: List[float] = []
        ls_ret_list: List[float] = []
        for d in dates:
            fvals, rvals = [], []
            ok = True
            for sym in factor_by_symbol:
                f = factor_by_symbol[sym]
                r = fwd_ret_map[sym]
                if d not in f.index or d not in r.index:
                    ok = False
                    break
                fv = f.loc[d]
                rv = r.loc[d]
                if pd.isna(fv) or pd.isna(rv):
                    ok = False
                    break
                fvals.append(fv)
                rvals.append(rv)
            if ok and len(fvals) >= 5:
                s = pd.Series(fvals).rank()
                t = pd.Series(rvals).rank()
                ic_list.append(float(s.corr(t)))
                ic_p_list.append(float(pd.Series(fvals).corr(pd.Series(rvals))))
                # 截面分组多空
                try:
                    grp = pd.qcut(pd.Series(fvals).rank(method="first"), n_groups, labels=False)
                except Exception:  # noqa: BLE001
                    continue
                rv = pd.Series(rvals)
                top = float(rv[grp == n_groups - 1].mean())
                bottom = float(rv[grp == 0].mean())
                if pd.notna(top) and pd.notna(bottom):
                    ls_ret_list.append(top - bottom)
        ic_valid = pd.Series(ic_list).dropna()
        if ic_valid.empty:
            return FactorReport(factor_name="cross_sectional", note="无足够截面样本")
        ic_mean = float(ic_valid.mean())
        ic_std = float(ic_valid.std())
        ir = ic_mean / ic_std if ic_std > 0 else float("nan")
        ic_pearson = float(pd.Series(ic_p_list).dropna().mean()) if ic_p_list else float("nan")

        ls_series = pd.Series(ls_ret_list)
        ls_total, ls_sharpe, ls_mdd = self._portfolio_stats(ls_series, periods_per_year)
        composite = self._composite(
            ic_mean=ic_mean, ls_return=ls_total, ls_sharpe=ls_sharpe,
            ls_mdd=ls_mdd, monotonicity=float("nan"), turnover=float("nan"),
        )
        return FactorReport(
            factor_name="cross_sectional",
            ic_mean=ic_mean,
            ic_pearson=ic_pearson,
            ic_std=ic_std,
            ir=ir,
            ic_positive_ratio=float((ic_valid > 0).mean()),
            ls_portfolio_return=ls_total,
            ls_portfolio_sharpe=ls_sharpe,
            ls_portfolio_mdd=ls_mdd,
            composite_score=composite,
            n_samples=len(ic_valid),
        )
