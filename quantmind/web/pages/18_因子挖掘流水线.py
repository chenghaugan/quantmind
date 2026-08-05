"""因子挖掘流水线页面：挖掘 → 去冗余 → 逐因子OOS回测 → 复合 alpha 组合可视化。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from utils.theme import (  # noqa: E402
    setup_page, page_header, section, note, verdict, guard_error,
    kpi_row, fmt_num, fmt_pct, badge,
)
from utils.api_client import APIClient  # noqa: E402
from utils.constants import CS_BASKETS, ALL_EXCHANGES, EXCHANGE_NAMES  # noqa: E402

setup_page("因子挖掘流水线", "🧬")
page_header(
    "端到端因子挖掘流水线",
    "把整条研究链一次跑通：多 seed → LLM/变异迭代挖掘（co/ea/tot）→ 相关性去冗余 → "
    "防泄漏切分 → 逐代表因子的样本外多空回测 → 组合权重优化成复合 alpha 并回测。",
    "🧬",
)

note(
    "**链路**：seed 池 → 迭代搜索（LLM 或离线变异）→ train 期去冗余选代表 → "
    "train/val/test 防泄漏切分 → 每代表做 test 期 OOS 多空回测 →（可选）按 "
    "ICIR/最小方差 等方案合成复合 alpha 并回测。<br>"
    "离线时（无 LLM key）自动回落为确定性变异器，流程可跑通；接入真实 LLM 后 "
    "搜索质量显著提升。",
    "info",
)

# ------------------------------------------------------------- 输入区
seeds_default = ["delta(close,5)", "ts_zscore(close,20)", "rank(close,10)",
                 "ts_rank(close,20)", "corr(close,volume,10)"]
l, r = st.columns([2, 1], gap="medium")
with l:
    seeds_txt = st.text_area(
        "🧬 因子种子（每行一个 DSL 表达式）",
        "\n".join(seeds_default),
        height=150,
        help="可用的面板变量 close/open/high/low/volume/amount；算子 mean/std/sum/rank/ts_zscore/ts_rank/corr/delta/slope…",
    )
    algo = st.selectbox(
        "搜索算法", ["co", "ea", "tot"],
        format_func=lambda x: {
            "co": "链式精炼 (CoT)",
            "ea": "进化算法 (EA，种群变异+选择)",
            "tot": "树状思维 (ToT，分支+剪枝)",
        }[x],
    )
    rounds = st.slider("每 seed 迭代深度（co=rounds / ea=generations / tot=depth）", 1, 8, 3)
    c1, c2 = st.columns(2)
    with c1:
        dedup_th = st.slider("去冗余相关阈值", 0.5, 0.95, 0.7, 0.05,
                             help="两因子相关 ≥ 此值视为冗余，每簇仅保留 |IC| 最高者")
    with c2:
        composite_scheme = st.selectbox(
            "复合权重方案", ["icir", "min_var", "equal", "inv_var"],
            format_func=lambda x: {
                "icir": "ICIR 加权（信息比率）",
                "min_var": "最小方差（闭式）",
                "equal": "等权",
                "inv_var": "逆方差",
            }[x],
        )
    run_composite = st.checkbox("启用复合 alpha 组合回测", value=True)
with r:
    basket = st.selectbox("标的篮子", list(CS_BASKETS.keys()), format_func=lambda x: str(x))
    symbols, exch = CS_BASKETS[basket]
    st.caption("篮子：" + " · ".join(symbols[:5]) + ("…" if len(symbols) > 5 else ""))
    custom = st.text_input("自定义标的（逗号分隔，覆盖篮子）", "")
    exchange = st.selectbox("交易所", ALL_EXCHANGES, index=ALL_EXCHANGES.index(exch),
                            format_func=lambda x: f"{x} · {EXCHANGE_NAMES.get(x, '')}")
    forward_periods = st.slider("前向期数", 1, 20, 1)
    train_frac = st.slider("训练期占比", 0.3, 0.9, 0.6, 0.05)
    val_frac = st.slider("验证期占比", 0.1, 0.4, 0.2, 0.05)
    run_btn = st.button("🧬 运行因子挖掘流水线", type="primary", width="stretch")

if not run_btn:
    note("填好种子与参数后点击运行，结果会展示：<b>流水线汇总</b> / <b>代表因子表</b> / "
         "<b>复合 alpha 组合</b>（净值曲线 + 权重 + 绩效）。", "info")
    st.stop()

final_symbols = [s.strip() for s in custom.split(",") if s.strip()] or list(symbols)
seeds = [s.strip() for s in seeds_txt.splitlines() if s.strip()]
if len(final_symbols) < 2:
    note("因子挖掘至少需要 2 个标的。", "warning")
    st.stop()
if not seeds:
    note("至少需要 1 个 seed 表达式。", "warning")
    st.stop()

payload = {
    "seeds": seeds,
    "symbols": final_symbols,
    "exchange": exchange,
    "interval": "1d",
    "algo": algo,
    "rounds": rounds,
    "forward_periods": forward_periods,
    "dedup_threshold": dedup_th,
    "train_frac": train_frac,
    "val_frac": val_frac,
    "run_composite": run_composite,
    "composite_scheme": composite_scheme,
}

with st.spinner(f"正在运行端到端因子挖掘流水线（{algo.upper()} × {len(seeds)} seeds，{rounds} 轮）…"):
    result = APIClient.factor_pipeline(payload)

if guard_error(result, "因子挖掘流水线"):
    st.stop()

summary = result.get("summary") or {}
steps = result.get("steps") or []
composite = result.get("composite") or {}

# ------------------------------------------------------------- 数据源标识 + 缓存状态
_data_src = result.get("data_sources") or {}
_src_label = {k: v for k, v in _data_src.items()}
if result.get("is_real"):
    st.markdown(
        badge(f"✅ 真实行情 · {', '.join(sorted(set(_src_label.values())))}", "success"),
        unsafe_allow_html=True,
    )
    st.caption("本次因子挖掘基于真实行情数据（akshare/本地源）。")
else:
    st.markdown(badge("🧪 Mock 合成数据（离线）", "warning"), unsafe_allow_html=True)
    st.caption("未取到真实行情，已回退 Mock 合成数据。配置真实数据源后可跑真实标的。")

_cache = result.get("cache") or {}
if _cache.get("enabled"):
    _last = _cache.get("last_datetime")
    _cache_note = f"本地行情仓库已启用 · {_cache.get('files', 0)} 文件 / {_cache.get('rows', 0)} 根 K 线"
    if isinstance(_cache.get("last_datetime"), str):
        _cache_note += f" · 最新数据 {_cache['last_datetime'][:10]}"
    else:
        _cache_note += f" · 最新数据 {_last}"
    st.caption(f"⚡ {_cache_note}（真实源结果落盘，二次运行秒级返回，无需重复联网拉取）")

# ------------------------------------------------------------- 结论
mean_oos = summary.get("mean_test_sharpe")
if mean_oos is not None and mean_oos > 0:
    verdict(f"流水线产出 {len(steps)} 个代表因子，复合 alpha 平均 OOS Sharpe="
            f"{fmt_num(mean_oos, 2)}（>0）。", "ok", icon="✅")
else:
    verdict("流水线已跑通。样本外（OOS）无明显正 Sharpe——这是合成/回测数据上"
            "常见的诚实结果，提示依赖样本外验证而非训练集 IC。", "warn", icon="🔁")

# ------------------------------------------------------------- KPI
kpi_row([
    {"label": "算法", "value": (result.get("algo") or algo).upper(), "tone": "accent"},
    {"label": "候选因子", "value": str(summary.get("candidate_count", 0)), "tone": "accent"},
    {"label": "代表因子", "value": str(summary.get("representative_count", 0)), "tone": "accent"},
    {"label": "回测数", "value": str(summary.get("backtested_count", 0)), "tone": "accent"},
    {"label": "标的", "value": str(result.get("n_symbols", 0)), "tone": "neutral"},
])
kpi_row([
    {"label": "Train IC", "value": fmt_num(summary.get("mean_train_ic"), 4),
     "tone": "neutral"},
    {"label": "Val IC", "value": fmt_num(summary.get("mean_val_ic"), 4), "tone": "neutral"},
    {"label": "Test IC", "value": fmt_num(summary.get("mean_test_ic"), 4), "tone": "neutral"},
    {"label": "OOS Sharpe", "value": fmt_num(summary.get("mean_test_sharpe"), 2),
     "tone": "accent" if (summary.get("mean_test_sharpe") or 0) > 0 else "neutral"},
])

# ------------------------------------------------------------- 代表因子
section("代表因子（去冗余后，test 期 OOS 回测）")
if steps:
    steps_df = pd.DataFrame([{
        "表达式": s.get("expression", ""),
        "Train IC": fmt_num(s.get("train_ic"), 4),
        "Val IC": fmt_num(s.get("val_ic"), 4),
        "Test IC": fmt_num(s.get("test_ic"), 4),
        "OOS Sharpe": fmt_num(s.get("test_sharpe"), 2),
        "OOS 收益": fmt_pct(s.get("test_return")),
        "OOS 回撤": fmt_pct(s.get("test_mdd")),
        "吸收冗余": f"{len(s.get('removed_redundant') or [])}" if s.get("removed_redundant") else "0",
    } for s in steps])
    st.dataframe(steps_df, width="stretch", hide_index=True,
                 height=min(60 + 35 * len(steps_df), 420))
    with st.expander("🔎 代表因子详情", expanded=False):
        for s in steps:
            st.markdown(f"**{s.get('expression', '')}**")
            cols = st.columns(4)
            cols[0].metric("Train IC", fmt_num(s.get("train_ic"), 4))
            cols[1].metric("Val IC", fmt_num(s.get("val_ic"), 4))
            cols[2].metric("Test IC", fmt_num(s.get("test_ic"), 4))
            cols[3].metric("OOS Sharpe", fmt_num(s.get("test_sharpe"), 2))
            if s.get("removed_redundant"):
                st.caption("吸收冗余: " + " · ".join(s["removed_redundant"]))
            st.divider()

    # 各因子 Train/Val/Test IC 分组柱状图
    ic_rows = []
    for s in steps:
        ic_rows.append({
            "因子": s.get("expression", "")[:24] + ("…" if len(s.get("expression", "")) > 24 else ""),
            "Train IC": s.get("train_ic"),
            "Val IC": s.get("val_ic"),
            "Test IC": s.get("test_ic"),
        })
    if len(ic_rows) >= 1:
        icdf = pd.DataFrame(ic_rows).set_index("因子")
        # 仅保留至少有一个非空 IC 的行
        icdf = icdf.dropna(how="all")
        if not icdf.empty:
            import plotly.graph_objects as go_ic
            figic = go_ic.Figure()
            colors = {"Train IC": "#60a5fa", "Val IC": "#a78bfa", "Test IC": "#22d3ee"}
            for col in ["Train IC", "Val IC", "Test IC"]:
                ys = [None if pd.isna(v) else v for v in icdf[col]]
                figic.add_trace(go_ic.Bar(name=col, x=icdf.index, y=ys,
                                         marker_color=colors[col],
                                         marker_line_width=0))
            figic.update_layout(barmode="group", height=340,
                                title="各代表因子 IC（Train / Val / Test）",
                                margin=dict(t=44, b=30))
            figic.add_hline(y=0, line=dict(color="rgba(148,163,184,.4)", dash="dash"))
            st.plotly_chart(figic, use_container_width=True, config={"displayModeBar": False})
else:
    st.caption("无代表因子产出（可能全部被 min_abs_ic / 去冗余过滤，或搜索失败）。")

# ------------------------------------------------------------- 复合 alpha 组合
if composite:
    section("复合 alpha 组合（权重优化 + OOS 回测）")
    pf = composite.get("portfolio") or {}
    weights = composite.get("weights") or {}
    ckpi = {
        "Sharpe": fmt_num(pf.get("sharpe"), 2),
        "总收益": fmt_pct(pf.get("total_return")),
        "最大回撤": fmt_pct(pf.get("max_drawdown")),
        "前向 IC": fmt_num((composite.get("ic_report") or {}).get("ic_mean"), 4),
    }
    kpi_row([{"label": k, "value": v, "tone": "accent" if k == "Sharpe" and (pf.get("sharpe") or 0) > 0 else "neutral"}
             for k, v in ckpi.items()])

    col_eq, col_w = st.columns([2, 1], gap="medium")
    with col_eq:
        daily = pf.get("daily_returns") or []
        if daily:
            # 由日收益重建净值曲线
            import math
            nav = []
            cur = 1.0
            for i, r in enumerate(daily):
                cur *= (1.0 + float(r))
                nav.append({"date": f"T+{i}", "equity": cur})
            fig = px.line(pd.DataFrame(nav), x="date", y="equity",
                          title="复合 alpha 净值（OOS）")
            fig.update_layout(xaxis_title="", yaxis_title="净值", height=360,
                              margin=dict(t=44, b=30))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("无日收益数据，无法绘制净值曲线。")
    with col_w:
        if weights:
            wdf = pd.DataFrame([
                {"因子": k if len(k) <= 28 else k[:27] + "…", "权重": v}
                for k, v in weights.items()
            ]).sort_values("权重", ascending=True)
            figw = px.bar(wdf, x="权重", y="因子", orientation="h", title="组合权重")
            figw.update_layout(height=360, margin=dict(t=44, l=8, b=8),
                               xaxis_title="权重", yaxis_title="")
            figw.update_yaxes(automargin=True)
            st.plotly_chart(figw, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("无权重数据。")

    with st.expander("🔎 各因子 IC 贡献", expanded=False):
        facs = composite.get("factor_ics") or {}
        if facs:
            fdf = pd.DataFrame(sorted(facs.items(), key=lambda kv: -kv[1]),
                               columns=["因子", "IC"])
            st.dataframe(fdf, width="stretch", hide_index=True)
        else:
            st.caption("无因子 IC 数据。")

    # 因子相关矩阵热力图（去冗余后代表间的残差相关性）
    corr = composite.get("correlation")
    if corr and corr.get("columns") and corr.get("values"):
        import plotly.graph_objects as go_hm
        cols = corr["columns"]
        z = [[(float(v) if v is not None else float("nan"))
              for v in row] for row in corr["values"]]
        fig_hm = go_hm.Figure(go_hm.Heatmap(
            z=z, x=cols, y=cols, zmin=-1, zmax=1,
            colorscale=[[0, "#22d3ee"], [0.5, "#1e293b"], [1, "#f43f5e"]],
            zmid=0, texttemplate="%{z:.2f}", textfont=dict(size=9),
            colorbar=dict(title="r", thickness=10),
        ))
        fig_hm.update_layout(height=420, title="去冗余后代表因子相关性热力图",
                             margin=dict(t=44, b=30, l=8, r=8),
                             xaxis=dict(side="bottom", tickangle=-30),
                             yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_hm, use_container_width=True, config={"displayModeBar": False})
else:
    if run_composite:
        note("未启用复合组合，或面板过小无法分组。可调大训练期占比后重试。", "info")

st.caption(f"日期范围：{result.get('date_range') or '—'}　标的数：{result.get('n_symbols', 0)}")

with st.expander("🔎 原始返回", expanded=False):
    st.json(result)

st.caption("提示：想对比算法/权重方案可切换上方参数重复运行；接入真实 LLM 后挖掘质量与因子多样性会显著提升。")
