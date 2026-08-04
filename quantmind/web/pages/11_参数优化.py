"""参数优化（网格搜索）：选择策略 / 标的 / 指标 / 参数空间，跑寻优并可视化。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from utils.theme import (  # noqa: E402
    setup_page, page_header, section, note, verdict, guard_error,
    kpi_row, fmt_num, fmt_pct, tone_of, badge,
)
from utils.api_client import APIClient  # noqa: E402
from utils.constants import STRATEGIES, EXCHANGES, EXCHANGE_NAMES, INTERVALS, INTERVAL_NAMES  # noqa: E402
from utils.charts import create_optimize_scatter  # noqa: E402

setup_page("参数优化", "🎛️")
page_header(
    "参数优化",
    "对策略参数做网格搜索，按选定绩效指标（Sharpe / 收益 / 回撤 / 胜率）寻优并可视化全空间。",
    "🎛️",
)

note(
    "网格搜索会穷举参数组合、逐一回测，按指标取最优。`max_combos` 上限保护，防止 Web 端跑爆。",
    "info",
)

# ----------------------------------------------------------------- 策略 & 空间
cl, cr = st.columns([1, 2], gap="medium")
with cl:
    strategy = st.selectbox("策略", list(STRATEGIES.keys()),
                            format_func=lambda k: f"{STRATEGIES[k]['name']} — {STRATEGIES[k]['desc']}")
    method = st.radio(
        "优化方法",
        ["网格搜索", "Optuna 贝叶斯"],
        horizontal=True,
        help="网格搜索穷举所有组合；Optuna 用贝叶斯在参数区间内智能采样（更省算力）。",
    )
    asset_class = st.selectbox("资产类别", list(EXCHANGES.keys()))
    exchange = st.selectbox("交易所", EXCHANGES[asset_class],
                            format_func=lambda x: f"{x} · {EXCHANGE_NAMES.get(x, '')}")
    symbol = st.text_input("合约代码", "rb0")
    interval = st.selectbox("周期", INTERVALS, index=INTERVALS.index("1d"),
                            format_func=lambda x: INTERVAL_NAMES.get(x, x))

with cr:
    st.markdown("**参数空间**")
    space_res = APIClient.optimize_space(strategy)
    if guard_error(space_res, "参数空间"):
        st.stop()
    default_space = space_res.get("param_space", {}) or {}
    metrics = space_res.get("metrics", []) or []
    if not default_space:
        st.warning("该策略暂未配置推荐参数空间，请手动指定参数。")

    chosen: dict = {}
    param_ranges: dict = {}
    if method == "网格搜索":
        for name, vals in default_space.items():
            picked = st.multiselect(
                f"参数 {name}", vals, default=list(vals),
                key=f"ps_{strategy}_{name}",
                help="选择要参与搜索的取值（至少 1 个）",
            )
            if picked:
                chosen[name] = picked
    else:
        st.caption("为每个参数设置搜索区间 [low, high, step]（整数）。")
        for name, vals in default_space.items():
            lo = min(vals) if vals else 1
            hi = max(vals) if vals else lo + 1
            r1, r2, r3 = st.columns(3)
            with r1:
                low = st.number_input(f"{name} 下限", value=int(lo), step=1, key=f"lo_{name}")
            with r2:
                high = st.number_input(f"{name} 上限", value=int(hi), step=1, key=f"hi_{name}")
            with r3:
                lo2 = st.number_input(f"{name} 步长", value=1, min_value=1, step=1, key=f"st_{name}")
            param_ranges[name] = [int(low), int(high), int(lo2)]

    metric = st.selectbox("优化指标",
                          [m["key"] for m in metrics] if metrics else ["sharpe"],
                          format_func=lambda k: dict((m["key"], m["label"]) for m in metrics).get(
                              k, k) if metrics else k,
                          index=0)
    capital = st.number_input("初始资金", value=1_000_000, step=100_000, format="%d")
    if method == "Optuna 贝叶斯":
        n_trials = st.number_input("试验次数", value=30, min_value=1, step=10,
                                   help="Optuna 最多采样的参数组合数")
        n_trials = int(n_trials)
    else:
        n_trials = 0
        max_combos = st.number_input("组合数上限", value=200, min_value=1, step=10)

    run_btn = st.button("🎛️ 开始寻优", type="primary", width="stretch")

# ----------------------------------------------------------------- 运行
if not run_btn:
    note("选择参数空间后点击「开始寻优」。结果会展示最优参数、散点分布与全组合明细表。",
         "info")
    st.stop()

if method == "网格搜索":
    if not chosen:
        note("请至少为一个参数选择一个取值。", "warning")
        st.stop()
    combos = 1
    for v in chosen.values():
        combos *= len(v)
    if combos > max_combos:
        note(f"组合数 {combos} 超过上限 {max_combos}，请缩减参数空间或调高上限。", "error")
        st.stop()
else:
    if not param_ranges:
        note("请为至少一个参数设置搜索区间。", "warning")
        st.stop()
    combos = n_trials

payload = {
    "strategy": strategy, "symbol": symbol, "exchange": exchange,
    "interval": interval, "metric": metric, "capital": capital,
}
if method == "Optuna 贝叶斯":
    payload["method"] = "optuna"
    payload["param_ranges"] = param_ranges
    payload["n_trials"] = n_trials
    payload["max_combos"] = n_trials
    api_call = APIClient.optimize_optuna
    spinner_txt = f"正在 Optuna 贝叶斯搜索 {combos} 次试验（{STRATEGIES[strategy]['name']}）…"
else:
    payload["param_space"] = chosen
    payload["max_combos"] = int(max_combos)
    api_call = APIClient.optimize
    spinner_txt = f"正在网格搜索 {combos} 个组合（{STRATEGIES[strategy]['name']}）…"

with st.spinner(spinner_txt):
    res = api_call(payload)

if guard_error(res, "参数优化"):
    st.stop()

best_setting = res.get("best_setting", {}) or {}
best_metric = res.get("best_metric")
results = res.get("results", []) or []
param_names = res.get("param_names", [])

# ----------------------------------------------------------------- 概览
section("寻优概览")
kpi_row([
    {"label": "评估组合数", "value": res.get("combos", combos), "tone": "accent"},
    {"label": "使用 K 线数", "value": res.get("bars", 0), "tone": "accent"},
    {"label": f"最优 {metric}",
     "value": fmt_num(best_metric, 4) if best_metric is not None else "—",
     "tone": tone_of(best_metric)},
    {"label": "最优参数", "value": " · ".join(f"{k}={v}" for k, v in best_setting.items()),
     "tone": "neutral", "hint": res.get("vt_symbol", "")},
])

if best_metric is None:
    verdict("所有组合的指标均为 NaN（数据不足或参数无效），请检查标的历史区间。", "warn", icon="⚠️")
else:
    verdict(f"已找到使 [{metric}] 最优的参数组合："
            + " · ".join(f"{k}={v}" for k, v in best_setting.items()), "ok", icon="✅")

# ----------------------------------------------------------------- 可视化
if results:
    section("参数空间可视化")
    st.plotly_chart(create_optimize_scatter(results, metric, "参数寻优结果", height=420),
                    width="stretch", key="opt_scatter")

    section("全组合明细")
    rows = []
    for r in results:
        base = {"setting": " · ".join(f"{k}={v}" for k, v in (r.get("setting") or {}).items()),
                "metric": r.get("metric")}
        base.update({k: v for k, v in r.items() if k not in ("setting", "metric")})
        rows.append(base)
    df = pd.DataFrame(rows)
    if "metric" in df.columns:
        df = df.sort_values("metric", ascending=False, na_position="last").reset_index(drop=True)
    show_cols = ["setting", "metric"] + [c for c in [
        "total_return", "sharpe", "max_drawdown", "calmar", "win_rate",
        "trade_count"] if c in df.columns]
    st.dataframe(df[[c for c in show_cols if c in df.columns]],
                 width="stretch", height=400, hide_index=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 下载寻优结果 CSV", data=csv,
                      file_name=f"optimize_{symbol}_{strategy}.csv", mime="text/csv")

with st.expander("🔎 原始返回", expanded=False):
    st.json(res)

st.caption("💡 把最优参数抄到「策略回测」做带真实成本的确认，再上「Walk-Forward」验证稳健性。")
