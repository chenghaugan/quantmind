"""真实数据多标的因子组合构建 + 截面回测脚本。

在 ``factor_dedup_real`` 去冗余得到的 **54 个独立代表因子** 基础上：
  1) 在 20 个真实期货主连上重算各代表因子的 date×symbol 截面面板
  2) 用 ``combine_factor_panels`` 合成复合 alpha（等权 rank / ICIR 加权两种方案）
  3) 用 ``_run_portfolio`` 做截面多空回测（多空年化/Sharpe/回撤），
     ``evaluate_factor_panel`` 输出复合信号 IC/IR
  4) 单因子基线对比：验证合成组合相对最强单因子的分散化收益

输出 markdown 报告（docs/factor_combine_real_report.md）。

用法：
    .\\venv\\Scripts\\python.exe scripts\\factor_combine_real.py [--n_common 300] [--cost 0.0]
"""
from __future__ import annotations

import argparse
import asyncio
import math
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from quantmind.cli import _make_dm
from quantmind.core.constant import Exchange, Interval
from quantmind.data.feed.base import HistoryRequest
from quantmind.research.combine import combine_factor_panels, optimize_weights
from quantmind.research.cross_sectional_backtest import _run_portfolio
from quantmind.research.evaluator import FactorEvaluator
from quantmind.research.factors.alpha_cs import Panel
from quantmind.research.factors.registry import FactorRegistry

# 复用 factor_empirical_real 的 20 个真实期货主力连续 + 共同交易日对齐
from scripts.factor_empirical_real import SYMBOLS, clip_common

# 54 个独立代表因子（来自 factor_dedup_real_report.md，按 |截面IC| 降序）
REPRESENTATIVES: List[str] = [
    "alpha041", "acad_beta", "acad_bab", "open_interest_change_20", "gtja191_020",
    "alpha095", "gtja191_038", "mean_reversion_60", "gtja191_002", "alpha026",
    "alpha191_012", "alpha054", "alpha028", "alpha002", "alpha017",
    "gtja191_013", "alpha009", "qlib_atr_14", "alpha023", "momentum_60",
    "alpha027", "gtja191_012", "qlib_mom_10", "gtja191_026", "alpha101",
    "acad_value_proxy", "alpha012", "acad_downside_vol", "qlib_rsi_14", "volume_change_5",
    "gtja191_060", "acad_mom_12m_1m", "gtja191_006", "gtja191_007", "alpha075",
    "term_structure_20", "alpha006", "gtja191_015", "acad_skew_20", "alpha191_081",
    "alpha001", "alpha093", "gtja191_037", "gtja191_102", "alpha042",
    "qlib_macd_dea", "qlib_obv", "qlib_wr_14", "gtja191_096", "alpha191_042",
    "gtja191_033", "gtja191_001", "gtja191_045", "alpha191_056",
]


def fmt(v, nd: int = 4):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "—"
    return f"{v:.{nd}f}"


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


def _factor_single_backtest(reg, name, factor_common, panel, ev, fp, groups, cost):
    """单个因子的截面回测 + IC 报告。"""
    scores = pd.DataFrame(factor_common)
    try:
        rep = ev.evaluate_factor_panel(scores, panel, forward_periods=fp,
                                       n_groups=groups, factor_name=name)
    except Exception:
        rep = None
    try:
        _, port_ret, perf = _run_portfolio(panel, scores, fp, groups, True, cost)
    except Exception:
        perf, port_ret = None, []
    return rep, perf, port_ret


async def run() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=700, help="拉取回溯天数")
    ap.add_argument("--n_common", type=int, default=300, help="评估共同交易日数")
    ap.add_argument("--fp", type=int, default=1, help="前向收益周期（根）")
    ap.add_argument("--groups", type=int, default=5, help="多空分组数")
    ap.add_argument("--cost", type=float, default=0.0, help="每期双边成本")
    ap.add_argument("--out", type=str, default=str(PROJECT / "docs" / "factor_combine_real_report.md"))
    args = ap.parse_args()

    dm = _make_dm()
    await dm.connect()
    try:
        print("[1/5] 拉取真实期货日线 ...", flush=True)
        bars_by_symbol = await fetch_bars(dm, args.days)
        if len(bars_by_symbol) < 5:
            print("[err] 真实标的不足（<5）。")
            return 2
        clip_bars, common_dates = clip_common(bars_by_symbol, args.n_common)
        n_dates = len(common_dates)
        print(f"      共同交易日: {n_dates} 个 · 标的 {len(clip_bars)}", flush=True)

        panel = Panel.from_bars(clip_bars)
        reg = FactorRegistry()
        ev = FactorEvaluator()
        fp, groups, cost = args.fp, args.groups, args.cost

        # 对 54 个代表因子重算面板 + 单因子 IC（供 ICIR 加权）
        panels: Dict[str, pd.DataFrame] = {}
        ic_reports: Dict[str, Dict[str, float]] = {}
        single_results: Dict[str, dict] = {}
        failed: List[str] = []
        print(f"[2/5] 计算 {len(REPRESENTATIVES)} 个代表因子面板 + 单因子回测基线 ...", flush=True)
        t_start = time.time()
        for name in REPRESENTATIVES:
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
                    failed.append(f"{name}(len)")
                    continue
                panels[name] = pd.DataFrame(factor_common)
                rep, perf, port_ret = _factor_single_backtest(
                    reg, name, factor_common, panel, ev, fp, groups, cost)
                if rep is not None and rep.ic_mean is not None and not math.isnan(rep.ic_mean):
                    ic_reports[name] = {"ic_mean": float(rep.ic_mean),
                                        "ic_std": float(rep.ic_std or 0.0)}
                single_results[name] = {"ic": rep.ic_mean if rep else None,
                                        "ir": rep.ir if rep else None,
                                        "sharpe": perf.to_dict().get("sharpe") if perf else None,
                                        "annual": perf.to_dict().get("annual_return") if perf else None,
                                        "mdd": perf.to_dict().get("max_drawdown") if perf else None,
                                        "ls_ret": rep.ls_portfolio_return if rep else None}
            except Exception as e:  # noqa: BLE001
                failed.append(f"{name}: {type(e).__name__}: {str(e)[:50]}")
                panels.pop(name, None)
        print(f"      完成 {len(panels)} 个因子，耗时 {time.time()-t_start:.0f}s", flush=True)
        if len(panels) < 5:
            print("[err] 成功因子不足。")
            return 2

        # [3/5] 合成复合 alpha：等权 rank  vs  ICIR 加权
        print("[3/5] 合成复合 alpha（等权 rank + ICIR 加权）...", flush=True)
        composite_eq = combine_factor_panels(panels, standardize="rank")
        try:
            w = optimize_weights(panels, scheme="icir", ic_reports=ic_reports,
                                 standardize="rank")
        except Exception as e:  # noqa: BLE001
            print(f"      ICIR 优化失败（{type(e).__name__}），回退等权", flush=True)
            w = {k: 1.0 / len(panels) for k in panels}
        composite_icir = combine_factor_panels(panels, weights=list(w.values()),
                                               standardize="rank")

        # [4/5] 两种组合回测 + IC/IR
        print("[4/5] 截面回测等权/ICIR 复合 ...", flush=True)
        rep_eq = ev.evaluate_factor_panel(composite_eq, panel, forward_periods=fp,
                                          n_groups=groups, factor_name="composite_equal")
        _, port_ret_eq, perf_eq = _run_portfolio(panel, composite_eq, fp, groups, True, cost)
        rep_icir = ev.evaluate_factor_panel(composite_icir, panel, forward_periods=fp,
                                            n_groups=groups, factor_name="composite_icir")
        _, port_ret_icir, perf_icir = _run_portfolio(panel, composite_icir, fp, groups, True, cost)

        # [5/5] 报告
        print("[5/5] 生成报告 ...", flush=True)
        md = build_md(
            panels=panels, reps=REPRESENTATIVES, failed=failed,
            n_dates=n_dates, fp=fp, groups=groups, cost=cost,
            participant_names=list(panels.keys()),
            composites={
                "等权 rank": {"composite": composite_eq, "rep": rep_eq, "perf": perf_eq,
                               "daily": port_ret_eq, "w": {k: 1.0 / len(panels) for k in panels}},
                "ICIR 加权": {"composite": composite_icir, "rep": rep_icir, "perf": perf_icir,
                               "daily": port_ret_icir, "w": w},
            },
            single_results=single_results, ic_reports=ic_reports,
            symbols=list(clip_bars),
        )
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"[ok] 报告 → {out}", flush=True)
        if failed:
            print("[warn] 失败/跳过:", flush=True)
            for f in failed:
                print("  -", f, flush=True)
        return 0
    finally:
        await dm.close()


def _comp_stats(rep, perf, daily) -> dict:
    """从 FactorReport + PerformanceReport 提取可读指标。"""
    pd_ = perf.to_dict() if perf else {}
    return {
        "ic_mean": rep.ic_mean if rep else None,
        "ic_std": rep.ic_std if rep else None,
        "ir": rep.ir if rep else None,
        "ls_return": rep.ls_portfolio_return if rep else None,
        "ls_sharpe": rep.ls_portfolio_sharpe if rep else None,
        "ls_mdd": rep.ls_portfolio_mdd if rep else None,
        "bt_annual": pd_.get("annual_return"),
        "bt_sharpe": pd_.get("sharpe"),
        "bt_mdd": pd_.get("max_drawdown"),
        "bt_total": pd_.get("total_return"),
        "n_dates": len(daily) if daily else 0,
    }


def build_md(panels, reps, failed, n_dates, fp, groups, cost,
             participant_names, composites, single_results, ic_reports, symbols) -> str:
    lines: List[str] = []
    lines.append("# 真实数据多标的因子组合构建 + 截面回测报告")
    lines.append("")
    lines.append(f"> 数据：**{len(symbols)}** 个真实期货主力连续，共同交易日 **{n_dates}** 个。")
    lines.append(f"> 输入：去冗余得到的 **{len(participant_names)}** 个独立代表因子（跨 {len(panels)} 个成功计算）。")
    lines.append(f"> 回测：`forward_periods={fp}` · `n_groups={groups}` · `long_short=True` · `cost={cost}`。")
    lines.append("> 合成：`combine_factor_panels(standardize='rank')` — 等权 rank 与 ICIR 加权对比。")
    lines.append("")
    lines.append("> **说明**：截面因子池仍受期货主连池共性与样本期局限；多空组合未计资金占用/换手冲击，非投资建议。")
    lines.append("")

    # 1) 组合方案对比
    lines.append("## 组合方案对比（截面多空）")
    lines.append("")
    lines.append("| 方案 | 截面IC | IR | 多空收益 | 多空Sharpe | 回测年化 | 回测Sharpe | 回撤 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for label, cfg in composites.items():
        st = _comp_stats(cfg["rep"], cfg["perf"], cfg["daily"])
        lines.append(
            f"| {label} | {fmt(st['ic_mean'])} | {fmt(st['ir'])} | {fmt(st['ls_return'])} "
            f"| {fmt(st['ls_sharpe'])} | {fmt(st['bt_annual'])} | {fmt(st['bt_sharpe'])} "
            f"| {fmt(st['bt_mdd'])} |")
    # 单因子最佳基线
    best = None
    for nm, s in single_results.items():
        ir = s.get("ir") or -1e9
        if best is None or ir > best[1]:
            best = (nm, ir, s)
    if best and best[1] > -1e8:
        s = best[2]
        lines.append(
            f"| 单因子最佳(`{best[0]}`) | {fmt(s['ic'])} | {fmt(s['ir'])} | {fmt(s['ls_ret'])} "
            f"| — | {fmt(s['annual'])} | {fmt(s['sharpe'])} | {fmt(s['mdd'])} |")
    lines.append("")
    lines.append("*注：单因子回测用同一 `_run_portfolio` 多空设置，作分散化对比基线；"
                 "合成组合应通过低相关 alpha 源降低 IR 波动、提升稳健性。*")
    lines.append("")

    # 2) 等权 top 权重 / 参与因子数
    lines.append("## 合成方案实现")
    lines.append("")
    eq_cfg = composites["等权 rank"]
    lines.append(f"- **等权 rank**：{len(eq_cfg['w'])} 个因子各 1/N 权重，逐日横截面排名后取均值合成（天然截面对齐、去量纲）。")
    w = composites["ICIR 加权"]["w"]
    w_sorted = sorted(w.items(), key=lambda kv: -abs(kv[1])) if w else []
    top_w = ", ".join(f"`{k}`={v:.3f}" for k, v in w_sorted[:8]) or "等权"
    lines.append(f"- **ICIR 加权**：权重与 单因子IC/√IC方差 成正比，风险调整后聚焦强 IC；Top权重: {top_w}")
    lines.append("")

    # 3) 单因子基线表（全部参与因子，按 |截面IC| 降序）
    lines.append("## 参与因子单因子基线（按 |截面IC| 降序）")
    lines.append("")
    lines.append("| # | 因子 | 截面IC | IR | 多空收益 | 回测年化 | 回测Sharpe |")
    lines.append("|---|---|---|---|---|---|---|")
    order = sorted(single_results.items(), key=lambda kv: -abs(kv[1].get("ic") or 0.0))
    for i, (nm, s) in enumerate(order, 1):
        lines.append(
            f"| {i} | `{nm}` | {fmt(s['ic'])} | {fmt(s['ir'])} | {fmt(s['ls_ret'])} "
            f"| {fmt(s['annual'])} | {fmt(s['sharpe'])} |")
    lines.append("")

    if failed:
        lines.append("## 未纳入（失败/跳过）")
        lines.append("")
        for f in failed:
            lines.append(f"- `{f}`")
        lines.append("")

    lines.append("## 生成信息")
    lines.append("")
    lines.append("- 脚本：`scripts/factor_combine_real.py`")
    lines.append(f"- 标的：{', '.join(symbols)}")
    lines.append(f"- 共同交易日：{n_dates} · fp={fp} · groups={groups} · cost={cost}")
    lines.append(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
