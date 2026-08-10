"""真实数据多标的因子去冗余（去相关）脚本。

复用 ``factor_empirical_real`` 的真实期货日线拉取与共同交易日对齐，构建每个
成功因子的 **date × symbol 截面面板**，然后：

  1) ``factor_correlation_matrix`` 计算因子两两 Spearman 相关矩阵
  2) 以 |截面IC|（或 composite）为排序 metric，``greedy_cluster_dedup`` 在
     ``correlation_threshold`` 下做贪心聚类，识别**互相独立**的代表因子子集

输出 markdown 报告（docs/factor_dedup_real_report.md），作为组合构建/正交化输入。

用法：
    .\\venv\\Scripts\\python.exe scripts\\factor_dedup_real.py [--n_common 300] [--corr 0.7]
"""
from __future__ import annotations

import argparse
import asyncio
import math
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from quantmind.cli import _make_dm
from quantmind.core.constant import Exchange, Interval
from quantmind.data.feed.base import HistoryRequest
from quantmind.research import dedup
from quantmind.research.evaluator import FactorEvaluator
from quantmind.research.factors.registry import FactorRegistry

# 与 factor_empirical_real 相同的 20 个真实期货主力连续
from scripts.factor_empirical_real import SYMBOLS, clip_common


def fmt(v):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "—"
    return f"{v:.4f}"


async def fetch_bars(dm, days_back: int) -> Dict[str, List]:
    end = datetime.now()
    start = end - timedelta(days=days_back)
    out: Dict[str, List] = {}
    for sym, _name, exch in SYMBOLS:
        req = HistoryRequest(
            symbol=sym, exchange=exch, interval=Interval.DAILY,
            start=start, end=end,
        )
        bars = await dm.get_bar_data(req)
        if bars:
            out[sym] = bars
    return out


async def run() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=700, help="拉取回溯天数")
    ap.add_argument("--n_common", type=int, default=300, help="评估共同交易日数")
    ap.add_argument("--corr", type=float, default=0.7, help="并簇相关阈值")
    ap.add_argument("--out", type=str, default=str(PROJECT / "docs" / "factor_dedup_real_report.md"))
    args = ap.parse_args()

    dm = _make_dm()
    await dm.connect()
    try:
        print("[1/4] 拉取真实期货日线 ...", flush=True)
        bars_by_symbol = await fetch_bars(dm, args.days)
        if len(bars_by_symbol) < 5:
            print("[err] 真实标的不足（<5）。")
            return 2
        clip_bars, common_dates = clip_common(bars_by_symbol, args.n_common)
        n_dates = len(common_dates)
        print(f"      共同交易日: {n_dates} 个", flush=True)

        reg = FactorRegistry()
        factors = reg.list_factors()
        ev = FactorEvaluator()

        # 对每个成功因子构建 date×symbol 截面面板 + 截面 IC（作 metric）
        panels: Dict[str, pd.DataFrame] = {}
        metric: Dict[str, float] = {}
        ic_info: Dict[str, float] = {}
        failed: List[str] = []
        print(f"[2/4] 构建因子截面面板 + 截面IC（{len(factors)} 个）...", flush=True)
        t_start = time.time()
        for meta in factors:
            name = meta["name"]
            cat = meta.get("category", "")
            if cat == "futures_seat":
                continue
            try:
                factor = reg.get(name)
                factor_common: Dict[str, pd.Series] = {}
                ok = True
                for sym, bars in bars_by_symbol.items():
                    if len(bars) < 120:
                        ok = False
                        break
                    fv = factor.compute(bars)
                    if fv is None or len(fv) != len(bars):
                        ok = False
                        break
                    idx = pd.DatetimeIndex([b.datetime for b in bars], name=sym)
                    s = pd.Series(fv.values, index=idx, name=sym)
                    factor_common[sym] = s.reindex(common_dates)
                if not ok:
                    failed.append(f"{name}(len mismatch)")
                    continue
                # date×symbol 面板（列=symbol）
                panel_df = pd.DataFrame(factor_common)
                panels[name] = panel_df
                with np.errstate(all="ignore"):
                    rep = ev.evaluate_cross_sectional(factor_common, clip_bars)
                if rep.ic_mean is None or math.isnan(rep.ic_mean):
                    failed.append(f"{name}(无截面样本)")
                    panels.pop(name, None)
                    continue
                ic_info[name] = rep.ic_mean
                metric[name] = abs(rep.ic_mean)  # 用 |截面IC| 作排序 metric
            except Exception as e:  # noqa: BLE001
                failed.append(f"{name}: {type(e).__name__}: {str(e)[:50]}")
                panels.pop(name, None)
        print(f"      构建完成 {len(panels)} 因子，耗时 {time.time()-t_start:.0f}s", flush=True)
        if len(panels) < 3:
            print("[err] 成功因子不足。")
            return 2

        print("[3/4] 计算相关矩阵 + 贪心聚类去冗余 ...", flush=True)
        corr_mat = dedup.factor_correlation_matrix(panels)
        clusters = dedup.greedy_cluster_dedup(
            list(panels.keys()), corr_mat,
            metric=metric, correlation_threshold=args.corr, min_abs_metric=0.0,
        )

        # 统计冗余量
        n_total = len(panels)
        n_rep = len(clusters)
        n_dropped = n_total - n_rep

        print("[4/4] 生成报告 ...", flush=True)
        md = build_md(
            panels=panels, corr_mat=corr_mat, clusters=clusters,
            ic_info=ic_info, failed=failed, n_dates=n_dates,
            threshold=args.corr, symbols=list(clip_bars),
        )
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"[ok] 成功 {n_total} 因子 → 代表 {n_rep}，去冗余 {n_dropped}。报告 → {out}", flush=True)
        if failed:
            print("[warn] 失败/跳过:", flush=True)
            for f in failed:
                print("  -", f, flush=True)
        return 0
    finally:
        await dm.close()


def build_md(
    panels: Dict[str, pd.DataFrame],
    corr_mat: pd.DataFrame,
    clusters: List[Dict[str, object]],
    ic_info: Dict[str, float],
    failed: List[str],
    n_dates: int,
    threshold: float,
    symbols: List[str],
) -> str:
    n_total = len(panels)
    n_rep = len(clusters)
    lines: List[str] = []
    lines.append("# 真实数据多标的因子去冗余报告")
    lines.append("")
    lines.append(f"> 数据：**{len(symbols)}** 个真实期货主力连续，共同交易日 **{n_dates}** 个。")
    lines.append(f"> 因子：成功构建截面面板 **{n_total}** 个；Spearman 相关矩阵 + 贪心聚类"
                 f"（阈值 **{threshold}**，metric=|截面IC|），保留代表 **{n_rep}** · 去冗余 **{n_total - n_rep}**。")
    lines.append("")
    lines.append("> **说明**：代表因子为互相低相关（<阈值）的独立 alpha 源，适合作组合构建输入；")
    lines.append("  截面相关仍受期货主连池共性与样本期局限，非投资建议。")
    lines.append("")

    if failed:
        lines.append("## 未纳入（失败/跳过）")
        lines.append("")
        for f in failed:
            lines.append(f"- `{f}`")
        lines.append("")

    # 代表性优先度 = |截面IC| 降序
    lines.append("## 保留的代表因子（按 |截面IC| 降序）")
    lines.append("")
    lines.append("| # | 代表因子 | 截面IC | |IC| | 聚类内高度相关成员 |")
    lines.append("|---|---|---|---|---|")
    order = sorted(clusters, key=lambda c: -abs(float(c["metric"])))
    max_cluster = max((len(c["cluster"]) for c in clusters), default=1)
    for i, c in enumerate(order, 1):
        rep = c["name"]
        members = [m for m in c["cluster"] if m != rep]
        if len(members) > 4:
            shown = ", ".join(f"`{m}`" for m in members[:4]) + f" … (+{len(members)-4})"
        else:
            shown = ", ".join(f"`{m}`" for m in members) or "—"
        lines.append(f"| {i} | `{rep}` | {fmt(ic_info.get(rep))} | {fmt(abs(ic_info.get(rep, float('nan'))))} | {shown} |")
    lines.append("")

    lines.append("## 全部聚类（含被并入成员）")
    lines.append("")
    lines.append(f"| 簇代表 | 簇大小 | |截面IC| | 成员 |")
    lines.append("|---|---|---|---|")
    for c in sorted(clusters, key=lambda c: -abs(float(c["metric"]))):
        reps = ", ".join(f"`{m}`" for m in c["cluster"])
        lines.append(f"| `{c['name']}` | {len(c['cluster'])} | {fmt(abs(float(c['metric'])))} | {reps} |")
    lines.append("")

    lines.append("## 高相关对（|corr| ≥ 0.7，跨簇冗余热点）")
    lines.append("")
    pairs = []
    names = list(corr_mat.index)
    for i, a in enumerate(names):
        for j in range(i + 1, len(names)):
            b = names[j]
            r = float(corr_mat.loc[a, b])
            if r == r and abs(r) >= 0.7:
                pairs.append((abs(r), a, b, r))
    pairs.sort(reverse=True)
    if pairs:
        lines.append("| |corr| | 因子A | 因子B | corr |")
        lines.append("|---|---|---|---|")
        for _, a, b, r in pairs:
            lines.append(f"| {r:.3f} | `{a}` | `{b}` | {r:.3f} |")
    else:
        lines.append("无 |corr|≥0.7 的因子对。")
    lines.append("")

    lines.append("## 生成信息")
    lines.append("")
    lines.append("- 脚本：`scripts/factor_dedup_real.py`")
    lines.append(f"- 标的：{', '.join(symbols)}")
    lines.append(f"- 共同交易日：{n_dates} · 相关阈值：{threshold} · metric：|截面IC|")
    lines.append(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
