"""全库因子实证回测脚本。

对 ``FactorRegistry`` 中全部 94 个核心因子，在一段**带可观测结构**的合成日线
（趋势漂移 + 慢周期 + 噪声）上进行单标的评估：
  compute(bars) -> FactorEvaluator.evaluate -> FactorReport
聚合各因子的 IC / IR / composite，输出 markdown 报告。

用法：
    .\\venv\\Scripts\\python.exe scripts\\factor_empirical.py [--n 750] [--out docs/factor_empirical_report.md]
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from quantmind.core.constant import Exchange, Interval
from quantmind.data.feed.mock import MockFeed
from quantmind.research.evaluator import FactorEvaluator
from quantmind.research.factors.registry import FactorRegistry


def gen_bars(
    n: int = 750,
    seed: int = 42,
    drift: float = 0.0004,
    amp: float = 0.03,
    period: int = 90,
    vol: float = 0.012,
) -> List:
    """生成带结构（趋势 + 周期 + 噪声）的合成日线 K 线。

    用确定性 log-price 生成，保证同 seed 可复现；最后构造为 BarData 列表。
    """
    random.seed(seed)
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    log_price = drift * t + amp * np.sin(2 * np.pi * t / period)
    log_price += rng.normal(0, vol, n).cumsum()
    close = np.exp(log_price - log_price[0]) * 100.0

    start = datetime(2023, 1, 2)
    bars: List = []
    for i in range(n):
        dt = start + timedelta(days=i)
        c = float(close[i])
        prev = float(close[i - 1]) if i > 0 else c
        o = prev * (1 + float(rng.normal(0, vol / 4)))
        spread = abs(c - o) * 0.5 + c * 0.003
        h = max(o, c) + spread
        l = min(o, c) - spread
        v = float(rng.uniform(1000, 10000))
        bars.append(
            MockFeed._make_bar(
                symbol="RB", exchange=Exchange.SHFE, dt=dt, interval=Interval.DAILY,
                o=o, h=h, l=l, c=c, v=v, oi=float(rng.uniform(0, 5000)),
            )
        )
    return bars


def evaluate_all(n: int, seed: int) -> pd.DataFrame:
    bars = gen_bars(n=n, seed=seed)
    ev = FactorEvaluator()
    reg = FactorRegistry()
    factors = reg.list_factors()

    rows = []
    failed: List[str] = []
    for meta in factors:
        name = meta["name"]
        category = meta.get("category", "")
        try:
            factor = reg.get(name)
            fv = factor.compute(bars)
            if fv is None or len(fv) != len(bars):
                failed.append(f"{name}(len mismatch)")
                continue
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                # bootstrap=False: 499 次重采样 CI 对全库扫描无谓，且是主要耗时来源；
                # IC/IR/decay/composite 主指标均与 bootstrap 无关，不受影响。
                rep = ev.evaluate(fv, bars, bootstrap=False)
            rows.append({
                "name": name,
                "category": category,
                "ic": rep.ic_mean,
                "ir": rep.ir,
                "ic_std": rep.ic_std,
                "debias_ic_pos": rep.ic_positive_ratio,
                "decay_half_life": rep.ic_decay_half_life,
                "ls_ret": rep.long_short_return,
                "ls_sharpe": rep.ls_portfolio_sharpe,
                "mono": max(rep.monotonicity_5, rep.monotonicity_10),
                "composite": rep.composite_score,
            })
        except Exception as e:  # noqa: BLE001
            failed.append(f"{name}: {type(e).__name__}: {e}")

    df = pd.DataFrame(rows)
    return df, failed, len(bars)


def build_md(df: pd.DataFrame, failed: List[str], n: int, seed: int) -> str:
    n_ok = len(df)
    n_fail = len(failed)
    lines: List[str] = []
    lines.append("# 全库因子实证回测报告")
    lines.append("")
    lines.append(f"> 数据：{n} 根合成日线（seed={seed}，趋势+周期+噪声），单标的时间序列评估。")
    lines.append(f"> 评估：`FactorEvaluator`（IC=rank IC，IR=滚动IC均值/标准差，composite 综合主分）。")
    lines.append(f"> 因子：registry 全量 **{n_ok + n_fail}** 个 · 成功 **{n_ok}** · 失败 **{n_fail}**。")
    lines.append("")

    if failed:
        lines.append("## 失败因子")
        lines.append("")
        for f in failed:
            lines.append(f"- `{f}`")
        lines.append("")

    def fmt(v):
        return "—" if (v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))) else f"{v:.4f}"

    lines.append("## 按 IC 排序（TOP 30）")
    lines.append("")
    lines.append("| rank | 因子 | 类别 | IC | IR | IC+比例 | 半衰期 | 多空收益 | 多空Sharpe | 单调性 | 综合分 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    top = df.sort_values("ic", ascending=False).head(30)
    for i, r in enumerate(top.itertuples(), 1):
        lines.append(
            f"| {i} | `{r.name}` | {r.category} | {fmt(r.ic)} | {fmt(r.ir)} | "
            f"{fmt(r.debias_ic_pos)} | {fmt(r.decay_half_life)} | {fmt(r.ls_ret)} | "
            f"{fmt(r.ls_sharpe)} | {fmt(r.mono)} | {fmt(r.composite)} |"
        )
    lines.append("")

    lines.append("## 按综合分排序（TOP 30）")
    lines.append("")
    lines.append("| rank | 因子 | 类别 | 综合分 | IC | IR | 多空收益 | 多空Sharpe | 单调性 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    topc = df.sort_values("composite", ascending=False).head(30)
    for i, r in enumerate(topc.itertuples(), 1):
        lines.append(
            f"| {i} | `{r.name}` | {r.category} | {fmt(r.composite)} | {fmt(r.ic)} | {fmt(r.ir)} | "
            f"{fmt(r.ls_ret)} | {fmt(r.ls_sharpe)} | {fmt(r.mono)} |"
        )
    lines.append("")

    lines.append("## 类别汇总")
    lines.append("")
    lines.append("| 类别 | 因子数 | IC均值 | IR均值 | 多空Sharpe均值 | 综合分均值 |")
    lines.append("|---|---|---|---|---|---|")
    g = df.groupby("category").agg(
        n=("name", "count"),
        ic=("ic", "mean"),
        ir=("ir", "mean"),
        ls_sharpe=("ls_sharpe", "mean"),
        composite=("composite", "mean"),
    ).sort_values("composite", ascending=False)
    for cat, r in g.iterrows():
        lines.append(
            f"| {cat} | {int(r['n'])} | {fmt(r['ic'])} | {fmt(r['ir'])} | "
            f"{fmt(r['ls_sharpe'])} | {fmt(r['composite'])} |"
        )
    lines.append("")

    lines.append("## 生成信息")
    lines.append("")
    lines.append(f"- 脚本：`scripts/factor_empirical.py`")
    lines.append(f"- 数据根数：{n} · seed：{seed}")
    lines.append(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("- 说明：合成数据仅含趋势+周期+噪声，无真实市场结构；序用以验证因子可计算性与评估管线，非真实 alpha 证据。")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=750)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=str(PROJECT / "docs" / "factor_empirical_report.md"))
    args = ap.parse_args()

    df, failed, n = evaluate_all(args.n, args.seed)
    md = build_md(df, failed, n, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"[ok] 成功 {len(df)} 因子，失败 {len(failed)}。报告 → {out}")
    if failed:
        print("[warn] 失败明细:")
        for f in failed:
            print("  -", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
