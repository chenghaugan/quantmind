"""真实数据多标的截面因子实证脚本。

对 ``FactorRegistry`` 中全部价量因子（94 个中可计算的 90 个，排除需席位数据的
futures_seat 4 个），在一组**真实商品/金融期货主力连续**日线上，做多标的截面实证：

  每个标的上 compute(因子) -> 赋 datetime index -> 对齐共同交易日
  -> FactorEvaluator.evaluate_cross_sectional(截面 IC / IR / 多空 / composite)

输出 markdown 报告（docs/factor_empirical_real_report.md）。

用法：
    .\\venv\\Scripts\\python.exe scripts\\factor_empirical_real.py
"""
from __future__ import annotations

import argparse
import asyncio
import math
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from quantmind.cli import _make_dm
from quantmind.core.constant import Exchange, Interval
from quantmind.data.feed.base import HistoryRequest
from quantmind.research.evaluator import FactorEvaluator
from quantmind.research.factors.registry import FactorRegistry

# 真实商品/金融期货主力连续（已验证可稳定通过 AKShare 拉到）。
# 覆盖黑色(螺纹/铁矿石/焦炭)、金融(沪深300)、有色(铜/铝/锌/金)、
# 农产(豆粕/豆一/白糖/棉花/豆油)、能化(PTA/甲醇/塑料/燃油/沥青/PP/乙二醇)。
SYMBOLS = [
    # 股指期货（CFFEX）— 四大品种全覆盖
    ("IF0", "沪深300股指", Exchange.CFFEX),
    ("IH0", "上证50股指", Exchange.CFFEX),
    ("IC0", "中证500股指", Exchange.CFFEX),
    ("IM0", "中证1000股指", Exchange.CFFEX),
    # 商品期货
    ("rb0", "螺纹钢", Exchange.SHFE),
    ("cu0", "沪铜", Exchange.SHFE),
    ("m0", "豆粕", Exchange.DCE),
    ("ta0", "PTA", Exchange.CZCE),
    ("i0", "铁矿石", Exchange.DCE),
    ("au0", "沪金", Exchange.SHFE),
    ("y0", "豆油", Exchange.DCE),
    ("j0", "焦炭", Exchange.DCE),
    ("al0", "沪铝", Exchange.SHFE),
    ("zn0", "沪锌", Exchange.SHFE),
    ("SR0", "白糖", Exchange.CZCE),
    ("CF0", "棉花", Exchange.CZCE),
    ("MA0", "甲醇", Exchange.CZCE),
    ("L0", "塑料", Exchange.DCE),
    ("FU0", "燃油", Exchange.SHFE),
    ("a0", "豆一", Exchange.DCE),
    ("bu0", "沥青", Exchange.SHFE),
    ("pp0", "聚丙烯", Exchange.DCE),
    ("eg0", "乙二醇", Exchange.DCE),
]


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


def clip_common(
    bars_by_symbol: Dict[str, List], n: int
) -> "tuple[Dict[str, List], List]":
    """取各标的共同交易日并裁剪到最近 ``n`` 个，重建对齐的 bounded bars + 日期列表。

    返回 (clipped_bars_by_symbol, common_dates)；每个标的的 BarData 只保留共同日期，
    且各标的都覆盖相同最近 n 个共同交易日，控制截面评估规模。
    """
    common = None
    for bars in bars_by_symbol.values():
        idx = set(b.datetime for b in bars)
        common = idx if common is None else (common & idx)
    common = sorted(common)[-n:]

    clipped: Dict[str, List] = {}
    for sym, bars in bars_by_symbol.items():
        by_dt = {b.datetime: b for b in bars}
        seq = [by_dt[d] for d in common if d in by_dt]
        clipped[sym] = seq
    return clipped, common


def fmt(v):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "—"
    return f"{v:.4f}"


def build_md(
    df: pd.DataFrame,
    failed: List[str],
    n_bars: int,
    n_dates: int,
    symbols: List[str],
) -> str:
    lines: List[str] = []
    lines.append("# 真实数据多标的截面因子实证报告")
    lines.append("")
    lines.append(f"> 数据：**{len(symbols)}** 个真实期货主力连续（{', '.join(symbols)}），")
    lines.append(f"  对齐到共同交易日 **{n_dates}** 个，日均 K 线样本约 **{n_bars}** 根。")
    lines.append(f"> 评估：多标的**截面 IC**（Spearman 秩相关，逐日截面取均值/标准差→IR），")
    lines.append("  收盘价 `pct_change(1)` 对齐截面前向收益；综合分同源 composite。")
    lines.append(f"> 因子：registry 价量因子（排除需席位数据的 futures_seat）**{len(df) + len(failed)}** 个 · "
                 f"成功 **{len(df)}** · 失败 **{len(failed)}**。")
    lines.append("")
    lines.append("> **说明**：本报告为**真实历史行情**上的因子截面有效性度量（非合成数据），")
    lines.append("  但仍受单一标的池（纯期货主连）与样本期局限，不构成投资建议。")
    lines.append("")

    if failed:
        lines.append("## 失败因子")
        lines.append("")
        for f in failed:
            lines.append(f"- `{f}`")
        lines.append("")

    lines.append("## 按截面 IC 排序（全因子）")
    lines.append("")
    lines.append("| rank | 因子 | 类别 | 截面IC | IR | IC+比例 | 多空收益 | 多空Sharpe | 综合分 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    top = df.sort_values("ic", ascending=False)
    for i, r in enumerate(top.itertuples(), 1):
        lines.append(
            f"| {i} | `{r.name}` | {r.category} | {fmt(r.ic)} | {fmt(r.ir)} | "
            f"{fmt(r.ic_pos)} | {fmt(r.ls_ret)} | {fmt(r.ls_sharpe)} | {fmt(r.composite)} |"
        )
    lines.append("")

    lines.append("## 按综合分排序（全因子）")
    lines.append("")
    lines.append("| rank | 因子 | 类别 | 综合分 | 截面IC | IR | 多空收益 | 多空Sharpe |")
    lines.append("|---|---|---|---|---|---|---|---|")
    topc = df.sort_values("composite", ascending=False)
    for i, r in enumerate(topc.itertuples(), 1):
        lines.append(
            f"| {i} | `{r.name}` | {r.category} | {fmt(r.composite)} | {fmt(r.ic)} | {fmt(r.ir)} | "
            f"{fmt(r.ls_ret)} | {fmt(r.ls_sharpe)} |"
        )
    lines.append("")

    lines.append("## 类别汇总")
    lines.append("")
    lines.append("| 类别 | 因子数 | 截面IC均值 | IR均值 | 多空Sharpe均值 | 综合分均值 |")
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
    lines.append("- 脚本：`scripts/factor_empirical_real.py`")
    lines.append(f"- 标的：{', '.join(symbols)}")
    lines.append(f"- 共同交易日：{n_dates} · 日均K线：{n_bars}")
    lines.append(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return "\n".join(lines)


async def run() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=700, help="拉取回溯天数")
    ap.add_argument("--n_common", type=int, default=300, help="用于截面评估的共同交易日数（取最近 N 个）")
    ap.add_argument("--out", type=str, default=str(PROJECT / "docs" / "factor_empirical_real_report.md"))
    args = ap.parse_args()

    dm = _make_dm()
    await dm.connect()
    try:
        print("[1/3] 拉取真实期货日线 ...", flush=True)
        bars_by_symbol = await fetch_bars(dm, args.days)
        print(f"      拉到 {len(bars_by_symbol)} 个标的: {list(bars_by_symbol)}", flush=True)
        if len(bars_by_symbol) < 5:
            print("[err] 真实标的不足（<5），无法做截面实证。")
            return 2

        # 裁剪到最近 N 个共同交易日：控制截面评估规模，保留更早历史供因子 warmup
        clip_bars, common_dates = clip_common(bars_by_symbol, args.n_common)
        n_dates = len(common_dates)
        print(f"      共同交易日（评估窗口）: {n_dates} 个 "
              f"({common_dates[0].date()} ~ {common_dates[-1].date()})", flush=True)

        reg = FactorRegistry()
        factors = reg.list_factors()
        ev = FactorEvaluator()

        rows: List[Dict] = []
        failed: List[str] = []
        print(f"[2/3] 逐因子截面评估（{len(factors)} 个）...", flush=True)
        t_start = time.time()
        for meta in factors:
            name = meta["name"]
            cat = meta.get("category", "")
            if cat == "futures_seat":
                continue  # 需席位数据，价量实证排除
            try:
                factor = reg.get(name)
                factor_common: Dict[str, pd.Series] = {}
                ok = True
                for sym, bars in bars_by_symbol.items():
                    if len(bars) < 120:
                        ok = False
                        break
                    fv = factor.compute(bars)  # RangeIndex, 全量长度
                    if fv is None or len(fv) != len(bars):
                        ok = False
                        break
                    # 赋 datetime index 并重采样到共同窗口
                    idx = pd.DatetimeIndex([b.datetime for b in bars], name=sym)
                    s = pd.Series(fv.values, index=idx, name=sym)
                    factor_common[sym] = s.reindex(common_dates)
                if not ok:
                    failed.append(f"{name}(len mismatch)")
                    continue
                with np.errstate(all="ignore"):
                    rep = ev.evaluate_cross_sectional(factor_common, clip_bars)
                if rep.ic_mean is None or math.isnan(rep.ic_mean):
                    failed.append(f"{name}(无截面样本)")
                    continue
                rows.append({
                    "name": name,
                    "category": cat,
                    "ic": rep.ic_mean,
                    "ir": rep.ir,
                    "ic_pos": rep.ic_positive_ratio,
                    "ls_ret": rep.ls_portfolio_return,
                    "ls_sharpe": rep.ls_portfolio_sharpe,
                    "composite": rep.composite_score,
                })
            except Exception as e:  # noqa: BLE001
                failed.append(f"{name}: {type(e).__name__}: {str(e)[:60]}")
        print(f"      完成 {len(rows)} 因子，耗时 {time.time()-t_start:.0f}s", flush=True)

        df = pd.DataFrame(rows)
        if df.empty:
            print("[err] 无成功因子。")
            return 2

        print("[3/3] 生成报告 ...", flush=True)
        n_bars = n_dates
        md = build_md(df, failed, n_bars, n_dates, list(clip_bars))
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"[ok] 成功 {len(df)} 因子，失败 {len(failed)}。报告 → {out}", flush=True)
        if failed:
            print("[warn] 失败明细:", flush=True)
            for f in failed:
                print("  -", f, flush=True)
        return 0
    finally:
        await dm.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
