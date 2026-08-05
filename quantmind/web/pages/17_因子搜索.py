"""因子搜索页面：seed → co（链式精炼）/ ea（进化）/ tot（树状）迭代搜索 → best + 轨迹。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st  # noqa: E402

from utils.theme import (  # noqa: E402
    setup_page, page_header, section, note, verdict, guard_error,
    kpi_row, fmt_num, fmt_pct, badge,
)
from utils.api_client import APIClient  # noqa: E402
from utils.constants import CS_BASKETS, ALL_EXCHANGES, EXCHANGE_NAMES  # noqa: E402

setup_page("因子搜索", "🔎")
page_header(
    "AI 因子搜索",
    "从 seed 因子出发，用 co（链式精炼）/ ea（进化）/ tot（树状）三种算法迭代挖掘更强的因子（对标 AlphaBench T3）。",
    "🔎",
)

note(
    "**搜索原理**：在标的面板上用截面 Rank-IC 当裁判，每一轮让 LLM（或离线变异器）"
    "根据历史轨迹变异/交叉/分支出改进候选，保留更优者。搜索结果仅用搜索期数据决策，"
    "可选独立验证期做防泄漏评估。<br>"
    "离线时（无 LLM key）自动回落为确定性变异器，流程可跑通。",
    "info",
)

# ----------------------------------------------------------------- 输入区
l, r = st.columns([2, 1], gap="medium")
with l:
    seed = st.text_input(
        "🧬 初始因子表达式（seed）", "mean(close, 20)",
        help="可用的面板 DSL 变量: close/open/high/low/volume/amount；算子: mean/std/sum/corr/ts_zscore/rank/delta/slope…",
    )
    algo = st.selectbox(
        "搜索算法", ["co", "ea", "tot"],
        format_func=lambda x: {
            "co": "链式精炼（Chain-of-Thought / CoE）",
            "ea": "进化算法（Evolutionary，种群变异+选择）",
            "tot": "树状思维（Tree-of-Thought，分支+剪枝）",
        }[x],
    )
    rounds = st.slider("迭代深度（co=rounds / ea=generations / tot=depth）", 1, 10, 3)
with r:
    basket = st.selectbox("标的篮子", list(CS_BASKETS.keys()), format_func=lambda x: str(x))
    symbols, exch = CS_BASKETS[basket]
    st.caption("篮子：" + " · ".join(symbols[:5]) + ("…" if len(symbols) > 5 else ""))
    custom = st.text_input("自定义标的（逗号分隔，覆盖篮子）", "")
    exchange = st.selectbox("交易所", ALL_EXCHANGES, index=ALL_EXCHANGES.index(exch),
                            format_func=lambda x: f"{x} · {EXCHANGE_NAMES.get(x, '')}")
    forward_periods = st.slider("前向期数", 1, 20, 1)
    use_val = st.checkbox("用独立验证期防泄漏", value=False)
    run_btn = st.button("🔎 运行因子搜索", type="primary", width="stretch")

if not run_btn:
    note("输入 seed 表达式、选算法后点击运行。结果展示 best 表达式、Rank-IC 改进与完整搜索轨迹。", "info")
    st.stop()

final_symbols = [s.strip() for s in custom.split(",") if s.strip()] or list(symbols)
if len(final_symbols) < 2:
    note("因子搜索至少需要 2 个标的。", "warning")
    st.stop()

payload = {
    "seed": seed,
    "symbols": final_symbols,
    "exchange": exchange,
    "interval": "1d",
    "algo": algo,
    "rounds": rounds,
    "forward_periods": forward_periods,
}
if use_val:
    payload["val_symbols"] = final_symbols

with st.spinner(f"正在运行 {algo.upper()} 因子搜索（{rounds} 轮）…"):
    result = APIClient.search_factor(payload)

if guard_error(result, "因子搜索"):
    st.stop()

# ----------------------------------------------------------------- 结论
improved = result.get("improved")
seed_ric = result.get("seed_rank_ic")
best_ric = result.get("best_rank_ic")
if improved:
    verdict(f"发现更优因子：Rank-IC 从 {fmt_num(seed_ric,4)} 提升到 {fmt_num(best_ric,4)}。", "ok", icon="✅")
else:
    verdict(f"未能显著改进（seed Rank-IC={fmt_num(seed_ric,4)}，best={fmt_num(best_ric,4)}）。", "warn", icon="🔁")

# ----------------------------------------------------------------- 关键指标
section("结果概览")
kpi_row([
    {"label": "算法", "value": result.get("algo", algo).upper(), "tone": "accent"},
    {"label": "Seed Rank-IC", "value": fmt_num(seed_ric, 4), "tone": "neutral"},
    {"label": "Best Rank-IC", "value": fmt_num(best_ric, 4), "tone": tone_of_improve(best_ric, seed_ric)},
    {"label": "迭代轮数", "value": str(result.get("rounds", 0)), "tone": "accent"},
    {"label": "标的数", "value": str(result.get("n_symbols", 0)), "tone": "accent"},
])

# ----------------------------------------------------------------- Best
section("最优因子")
st.code(result.get("best_expression", ""), language="text")
row = {
    "best_rank_ic": result.get("best_rank_ic"),
    "best_ic": result.get("best_ic"),
    "val_rank_ic": result.get("val_rank_ic"),
    "val_ic": result.get("val_ic"),
}
st.dataframe(
    [{"Best Rank-IC": fmt_num(row["best_rank_ic"], 4),
      "Best IC": fmt_num(row["best_ic"], 4),
      "验证期 Rank-IC（防泄漏）": fmt_num(row["val_rank_ic"], 4),
      "验证期 IC": fmt_num(row["val_ic"], 4)}],
    width="stretch", hide_index=True,
)

# ----------------------------------------------------------------- 轨迹
section("搜索轨迹")
history = result.get("history", []) or []
if history:
    rows = []
    for h in history:
        rows.append({
            "轮次": h.get("round"),
            "表达式": h.get("expression", ""),
            "Rank-IC": fmt_num(h.get("rank_ic"), 4),
            "IC": fmt_num(h.get("ic"), 4),
            "最优": "★" if h.get("is_best") else "",
        })
    st.dataframe(rows, width="stretch", hide_index=True, height=min(60 + 35 * len(rows), 400))
else:
    st.caption("无搜索轨迹（种子评估失败或未产生候选）。")

# ----------------------------------------------------------------- 下一步
section("下一步")
l2, r2 = st.columns(2)
with l2:
    st.markdown("把找到的最优表达式拿到「截面研究」或「因子研究」做完整 IC/IR/多空验证。")
with r2:
    try:
        st.page_link("pages/12_截面研究.py", label="🧬 截面研究 — 完整 IC/多空验证")
    except Exception:
        st.markdown("**🧬 截面研究** → `pages/12_截面研究.py`")

with st.expander("🔎 原始返回", expanded=False):
    st.json(result)

st.caption("提示：想对比不同算法可切换上方算法重复运行；接入真实 LLM 后搜索质量会显著提升。")


def tone_of_improve(best, seed):
    try:
        b, s = float(best), float(seed)
        return "accent" if b > s else "neutral"
    except (TypeError, ValueError):
        return "neutral"
