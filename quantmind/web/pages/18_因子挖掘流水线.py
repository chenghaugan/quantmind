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

    # ---- 逐因子样本外可视化：净值叠加 + 回撤水下 + IC 时序 ----
    import math

    _nav_fig = go.Figure()
    _dd_fig = go.Figure()
    _ic_fig = go.Figure()
    _palette = ["#60a5fa", "#f472b6", "#34d399", "#fbbf24", "#a78bfa",
                "#fb7185", "#22d3ee", "#a3e635", "#f97316", "#38bdf8"]
    for _i, s in enumerate(steps):
        _dret = s.get("daily_returns") or []
        _expr_label = s.get("expression", "")[:20] + ("…" if len(s.get("expression", "")) > 20 else "")
        if _dret:
            _nav = []
            _cur = 1.0
            _peak = 1.0
            _dd = []
            for _r in _dret:
                _cur *= (1.0 + float(_r))
                _peak = max(_peak, _cur)
                _dd.append((_cur - _peak) / _peak)
                _nav.append({"t": f"T+{len(_nav)}", "nav": _cur})
            _ndf = pd.DataFrame(_nav)
            _nav_fig.add_trace(go.Scatter(
                x=_ndf["t"], y=_ndf["nav"], name=_expr_label,
                line=dict(width=1.6, color=_palette[_i % len(_palette)])))
            _dd_fig.add_trace(go.Scatter(
                x=_ndf["t"], y=_dd, name=_expr_label,
                fill="tozeroy", mode="lines", line=dict(width=1.2, color=_palette[_i % len(_palette)]),
                fillcolor=f"rgba(99,102,241,0.08)"))
        _ics = s.get("ic_series") or []
        if _ics:
            # 滚动 10 期均值平滑，更直观
            _ic = [_ics[0]] if _ics else []
            for _j in range(1, len(_ics)):
                _win = [x for x in _ics[max(0, _j - 9): _j + 1] if x is not None]
                _ic.append(sum(_win) / len(_win) if _win else None)
            _ic_fig.add_trace(go.Scatter(
                x=[f"T+{k}" for k in range(len(_ics))], y=_ic,
                name=_expr_label, line=dict(width=1.4, color=_palette[_i % len(_palette)])))
    if len(_nav_fig.data):
        _nav_fig.update_layout(height=360, title="各代表因子 OOS 净值叠加（多空，由日收益重建）",
                               margin=dict(t=44, b=30), legend=dict(font=dict(size=10)),
                               hovermode="x unified")
        _nav_fig.add_hline(y=1.0, line=dict(color="rgba(148,163,184,.4)", dash="dash"))
        st.plotly_chart(_nav_fig, use_container_width=True, config={"displayModeBar": False})
    if len(_dd_fig.data):
        _dd_fig.update_layout(height=300, title="回撤水下曲线（各代表因子）",
                              margin=dict(t=44, b=30),
                              legend=dict(font=dict(size=10)), hovermode="x unified")
        st.plotly_chart(_dd_fig, use_container_width=True, config={"displayModeBar": False})
    if len(_ic_fig.data):
        _ic_fig.update_layout(height=320, title="样本外截面 IC 时序（10 期滚动均值）",
                              margin=dict(t=44, b=30),
                              legend=dict(font=dict(size=10)), hovermode="x unified")
        _ic_fig.add_hline(y=0, line=dict(color="rgba(148,163,184,.4)", dash="dash"))
        st.plotly_chart(_ic_fig, use_container_width=True, config={"displayModeBar": False})
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

    # 组合风险/收益归因（近似）：weight × 成分 OOS 收益的贡献分解
    _contrib = composite.get("contribution") or []
    if _contrib:
        st.subheader("组合风险归因（近似贡献分解）")
        st.caption("`contribution = 权重 × 成分样本外收益`（横截面组合非严格可加，此为近似影响）；"
                   "`abs_pct` 为相对贡献占比。")
        cdf = pd.DataFrame([{
            "因子": r.get("expression", "")[:28],
            "权重": fmt_num(r.get("weight"), 3),
            "OOS IC": fmt_num(r.get("test_ic"), 3),
            "OOS Sharpe": fmt_num(r.get("test_sharpe"), 2),
            "OOS 收益": fmt_pct(r.get("test_return")),
            "贡献": fmt_num(r.get("contribution"), 4),
            "贡献占比": fmt_pct(r.get("abs_pct")),
        } for r in _contrib])
        st.dataframe(cdf, width="stretch", hide_index=True)
        # 贡献柱状图（按 |贡献| 排序）
        _contrib_valid = [r for r in _contrib if r.get("contribution") is not None]
        if _contrib_valid:
            _cdf = pd.DataFrame([{
                "因子": r.get("expression", "")[:24],
                "贡献": r.get("contribution"),
            } for r in _contrib_valid]).sort_values("贡献")
            _figc = px.bar(_cdf, x="贡献", y="因子", orientation="h",
                           color="贡献", color_continuous_scale="RdBu_r",
                           title="组合收益贡献分解（weight × OOS 收益）")
            _figc.update_layout(height=280, margin=dict(t=44, l=8, b=8),
                                xaxis_title="贡献", yaxis_title="")
            _figc.add_vline(x=0, line=dict(color="rgba(148,163,184,.5)", dash="dash"))
            st.plotly_chart(_figc, use_container_width=True, config={"displayModeBar": False})

    # 完整 Barra 式多因子风险归因（风格暴露 + 截面回归因子收益 + 协方差分解）
    _ra = composite.get("risk_attribution")
    if isinstance(_ra, dict) and "error" in _ra:
        st.caption(f"⚠️ Barra 风险归因不可用: {_ra.get('error', '')[:120]}")
    elif isinstance(_ra, dict) and _ra.get("factors"):
        st.subheader("多因子风险归因（Barra 式）")
        st.caption("**方法**：把每个因子看作一种风格暴露，做**风格正交化**（因子横截面互不相关）+ "
                   "逐交易日**横截面回归**（含市场截距）估计因子收益率，再用**协方差分解**"
                   "（MCTR=Cov(成分,组合收益)/σ）把组合波动拆到各因子+特异风险，分解**可加**"
                   "（各项之和=总波动）。")
        # 总量 KPI
        _tot = _ra.get("total") or {}
        _spec = _ra.get("specific") or {}
        _mkt = _ra.get("market") or {}
        kpi_row([
            {"label": "组合年化波动", "value": fmt_num(_tot.get("ann_vol"), 3),
             "tone": "accent"},
            {"label": "日波动 σ", "value": fmt_num(_tot.get("vol"), 4), "tone": "neutral"},
            {"label": "系统解释 R²", "value": fmt_pct(_tot.get("r2_mean")),
             "tone": ("success" if (_tot.get("r2_mean") or 0) > 0.5 else "neutral")},
            {"label": "特异风险占比", "value": fmt_pct(_spec.get("risk_pct")),
             "tone": "neutral"},
            {"label": "市场因子", "value": fmt_num(_mkt.get("mctr_vol"), 4), "tone": "neutral"},
        ])
        # 因子风险贡献表
        _frows = [{
            "因子": f.get("name", "")[:28],
            "MCTR 波动": fmt_num(f.get("mctr_vol"), 4),
            "风险占比": fmt_pct(f.get("risk_pct")),
            "因子收益(均)": fmt_num(f.get("factor_ret_mean"), 5),
            "因子收益(波动)": fmt_num(f.get("factor_ret_vol"), 4),
            "平均暴露": fmt_num(f.get("exposure_mean"), 3),
        } for f in _ra.get("factors") or []]
        st.dataframe(pd.DataFrame(_frows), width="stretch", hide_index=True)
        # 风险贡献条形图（因子 + 特异 + 市场）
        _bar_rows = ([{"项": f.get("name", "")[:20], "MCTR": f.get("mctr_vol"),
                       "类别": "因子"}
                      for f in _ra.get("factors") or []]
                     + [{"项": "_特异", "MCTR": _spec.get("mctr_vol"), "类别": "特异"},
                        {"项": "_市场", "MCTR": _mkt.get("mctr_vol"), "类别": "市场"}])
        _bdf = pd.DataFrame(_bar_rows).sort_values("MCTR")
        _figr = px.bar(_bdf, x="MCTR", y="项", orientation="h", color="类别",
                       color_discrete_map={"因子": "#38bdf8", "特异": "#a78bfa",
                                           "市场": "#fbbf24"},
                       title="组合风险贡献分解（Barra 协方差，各部分可加=总波动 σ）")
        _figr.update_layout(height=320, margin=dict(t=44, l=8, b=8),
                            xaxis_title="风险贡献（波动 σ）", yaxis_title="",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                        xanchor="right", x=1))
        _figr.add_vline(x=0, line=dict(color="rgba(148,163,184,.5)", dash="dash"))
        st.plotly_chart(_figr, use_container_width=True, config={"displayModeBar": False})
        _add = _ra.get("additivity") or {}
        _diag = _ra.get("diagnostics") or {}
        _cov = _diag.get("covariance", "n/a")
        _orth = "·正交化" if _diag.get("orthogonalized") else ""
        _nwl = f"·NW滞后{_diag.get('nw_lags')}" if _cov == "newey_west" else ""
        st.caption(f"闭合校验：Σ 因子+特异+市场 = {fmt_num(_add.get('recon_total'), 4)} "
                   f"≈ 组合 σ = {fmt_num(_tot.get('vol'), 4)}"
                   f"（残差 {fmt_num(_add.get('closure'), 6)}）　·　"
                   f"协方差={_cov}{_orth}{_nwl}")

        # ============ 前端展示增量（A1/A2/A3 + B1/B2 + C1/C2） ============
        st.markdown("---")

        # A1) 因子收益率累计净值曲线（风格因子收益路径）
        _fr = _ra.get("factor_returns_ts") or {}
        if _fr.get("series"):
            _fdr = _fr.get("dates") or []
            _fdf = pd.DataFrame(_fr["series"], index=_fdr).fillna(0.0)
            _fdf_cum = (1 + _fdf).cumprod()
            st.subheader("因子收益率累计净值（风格因子收益路径）")
            _fig_a1 = px.line(
                _fdf_cum, x=_fdf_cum.index, y=_fdf_cum.columns,
                title="因子累计净值（cumprod(1+r_f)）",
                color_discrete_sequence=px.colors.qualitative.Set2)
            _fig_a1.update_layout(height=320, margin=dict(t=44, l=8, b=8),
                                  xaxis_title="交易日", yaxis_title="累计净值",
                                  legend=dict(orientation="h", yanchor="bottom",
                                              y=1.02, xanchor="right", x=1))
            _fig_a1.add_hline(y=1.0, line=dict(color="rgba(148,163,184,.5)",
                                               dash="dash"))
            st.plotly_chart(_fig_a1, use_container_width=True,
                            config={"displayModeBar": False})

        # A2) 逐日截面 R² 时序（系统性解释的时变）
        _r2 = _ra.get("r2_ts") or {}
        if _r2.get("r2"):
            _r2df = pd.DataFrame({"r2": _r2.get("r2")}, index=_r2.get("dates"))
            _r2df["r2_roll"] = _r2df["r2"].rolling(20, min_periods=1).mean()
            st.subheader("逐日截面回归 R²（系统性解释比例）")
            _fig_a2 = go.Figure()
            _fig_a2.add_trace(go.Scatter(
                x=_r2df.index, y=_r2df["r2"], name="逐日 R²", mode="lines",
                line=dict(color="rgba(56,189,248,.35)", width=1)))
            _fig_a2.add_trace(go.Scatter(
                x=_r2df.index, y=_r2df["r2_roll"], name="滚动均值(20)",
                line=dict(color="#38bdf8", width=2.2)))
            _fig_a2.update_layout(height=300, margin=dict(t=44, l=8, b=8),
                                  xaxis_title="交易日", yaxis_title="R²",
                                  yaxis=dict(range=[0, 1]),
                                  legend=dict(orientation="h", yanchor="bottom",
                                              y=1.02, xanchor="left", x=0))
            _fig_a2.add_hline(y=float(_tot.get("r2_mean") or 0),
                              line=dict(color="rgba(148,163,184,.5)", dash="dash"))
            st.plotly_chart(_fig_a2, use_container_width=True,
                            config={"displayModeBar": False})

        # A3) 组合逐因子风格暴露时变（组合风格漂移）
        _ex = _ra.get("exposure_ts") or {}
        if _ex.get("series"):
            _exdf = pd.DataFrame(_ex["series"], index=_ex.get("dates")).fillna(0.0)
            st.subheader("组合风格暴露时变（因子暴露漂移）")
            _fig_a3 = px.line(
                _exdf, x=_exdf.index, y=_exdf.columns,
                title="组合因子总暴露 p_f,t = Σ ω·X_f",
                color_discrete_sequence=px.colors.qualitative.Set2)
            _fig_a3.update_layout(height=300, margin=dict(t=44, l=8, b=8),
                                  xaxis_title="交易日", yaxis_title="暴露",
                                  legend=dict(orientation="h", yanchor="bottom",
                                              y=1.02, xanchor="right", x=1))
            _fig_a3.add_hline(y=0.0, line=dict(color="rgba(148,163,184,.5)",
                                               dash="dash"))
            st.plotly_chart(_fig_a3, use_container_width=True,
                            config={"displayModeBar": False})

        # B1) 正交化前后因子收益率相关对比热力图
        _frr = _ra.get("factor_returns_raw_ts") or {}
        if _frr.get("series") and _fr.get("series"):
            import numpy as _np  # noqa: PLC0415

            def _corr_heat_matrix(df_vals, cols):
                _z = df_vals[cols].values.astype(float)
                _m = _np.corrcoef(_z.T)
                _m = [[(None if v != v else round(float(v), 3)) for v in row]
                      for row in _m]
                _f = go.Figure(go.Heatmap(
                    z=_m, x=cols, y=cols, zmin=-1, zmax=1,
                    colorscale=[[0, "#22d3ee"], [0.5, "#1e293b"], [1, "#f43f5e"]],
                    zmid=0, texttemplate="%{z:.2f}", textfont=dict(size=9),
                    colorbar=dict(title="r", thickness=10)))
                return _f

            _cols_b1 = [c for c in _fr["series"].keys()]
            _raw_b1 = pd.DataFrame(_frr["series"], index=_frr["dates"]).fillna(0.0)
            _oth_b1 = pd.DataFrame(_fr["series"], index=_fr["dates"]).fillna(0.0)
            st.subheader("因子收益率相关：正交化前 vs 后（风格去相关验证）")
            c_a, c_b = st.columns(2)
            with c_a:
                _fig_b1a = _corr_heat_matrix(_raw_b1, _cols_b1)
                _fig_b1a.update_layout(height=340, title="正交化前（原始暴露因子收益）",
                                       margin=dict(t=44, b=30, l=8, r=8),
                                       xaxis=dict(side="bottom", tickangle=-30),
                                       yaxis=dict(autorange="reversed"))
                st.plotly_chart(_fig_b1a, use_container_width=True,
                                config={"displayModeBar": False})
            with c_b:
                _fig_b1b = _corr_heat_matrix(_oth_b1, _cols_b1)
                _fig_b1b.update_layout(height=340, title="正交化后（因子收益率）",
                                       margin=dict(t=44, b=30, l=8, r=8),
                                       xaxis=dict(side="bottom", tickangle=-30),
                                       yaxis=dict(autorange="reversed"))
                st.plotly_chart(_fig_b1b, use_container_width=True,
                                config={"displayModeBar": False})

        # B2) 滚动风险分解（特异 vs 系统性 vs 各因子 MCTR 时间切片）
        _rr = _ra.get("rolling_risk") or {}
        if _rr.get("dates"):
            _rrdf = pd.DataFrame(index=_rr.get("dates"))
            for _c, _vals in (_rr.get("factors") or {}).items():
                _rrdf[_c] = _vals
            if _rr.get("portfolio_vol"):
                _rrdf["_组合σ"] = _rr.get("portfolio_vol")
            st.subheader(f"滚动风险分解（窗口 {_rr.get('window', '—')} 交易日）")
            _fig_b2 = go.Figure()
            for _c, _vals in (_rr.get("factors") or {}).items():
                _fig_b2.add_trace(go.Scatter(
                    x=_rrdf.index, y=_rrdf[_c], name=_c[:20], mode="lines",
                    stackgroup="risk", line=dict(width=1)))
            _fig_b2.update_layout(height=340, margin=dict(t=44, l=8, b=8),
                                  xaxis_title="交易日", yaxis_title="MCTR（波动贡献）",
                                  legend=dict(orientation="h", yanchor="bottom",
                                              y=1.02, xanchor="right", x=1))
            if _rr.get("portfolio_vol"):
                _fig_b2.add_trace(go.Scatter(
                    x=_rrdf.index, y=_rrdf["_组合σ"], name="组合σ",
                    mode="lines", line=dict(color="#fbbf24", width=2)))
            st.plotly_chart(_fig_b2, use_container_width=True,
                            config={"displayModeBar": False})

        # C1) 收益归因：因子/特异/市场对组合累计收益的贡献
        _rattr = _ra.get("return_attribution") or {}
        if _rattr.get("ts"):
            _ra_ts = _rattr["ts"]
            _ra_cols = [c for c in _ra_ts.keys()]
            _ra_df = pd.DataFrame({c: _ra_ts[c] for c in _ra_cols})
            st.subheader("收益归因：累计贡献（因子/特异/市场）")
            _fig_c1 = px.line(
                _ra_df, y=_ra_df.columns,
                title="组合累计收益贡献（Σ p_f·B_f 累计）",
                color_discrete_map=dict(
                    **{f.get("name", ""): "#38bdf8" for f in _ra.get("factors") or []},
                    **{"_specific": "#a78bfa", "_market": "#fbbf24"}))
            _fig_c1.update_layout(height=320, margin=dict(t=44, l=8, b=8),
                                  xaxis_title="交易日", yaxis_title="累计收益",
                                  legend=dict(orientation="h", yanchor="bottom",
                                              y=1.02, xanchor="right", x=1))
            st.plotly_chart(_fig_c1, use_container_width=True,
                            config={"displayModeBar": False})
            # 收益归因汇总表
            _ra_rows = ([{"项": f.get("name", "")[:24],
                          "累计收益贡献": fmt_num(_rattr["factors"].get(f.get("name"), 0), 4),
                          "类别": "因子"}
                         for f in _ra.get("factors") or []]
                        + [{"项": "_特异", "累计收益贡献": fmt_num(_rattr.get("specific"), 4),
                            "类别": "特异"},
                           {"项": "_市场", "累计收益贡献": fmt_num(_rattr.get("market"), 4),
                            "类别": "市场"}])
            st.dataframe(pd.DataFrame(_ra_rows), width="stretch", hide_index=True)

        # C2) 因子/复合信号的逐日截面 IC 时序
        _icts = composite.get("ic_ts") or {}
        if _icts.get("dates") and (_icts.get("factors") or _icts.get("composite")):
            _icdf = pd.DataFrame(index=_icts.get("dates"))
            for _nm, _seq in (_icts.get("factors") or {}).items():
                _icdf[_nm] = _seq
            if _icts.get("composite"):
                _icdf["_复合α"] = _icts.get("composite")
            st.subheader("逐日截面 IC 时序（因子 & 复合 α）")
            _fig_c2 = px.line(
                _icdf, x=_icdf.index, y=_icdf.columns,
                title="每日 IC（横截面秩相关）",
                color_discrete_map=dict(
                    **{n: "#38bdf8" for n in (_icts.get("factors") or {})},
                    **{"_复合α": "#fbbf24"}))
            _fig_c2.update_layout(height=320, margin=dict(t=44, l=8, b=8),
                                  xaxis_title="交易日", yaxis_title="IC",
                                  legend=dict(orientation="h", yanchor="bottom",
                                              y=1.02, xanchor="right", x=1))
            _fig_c2.add_hline(y=0.0, line=dict(color="rgba(148,163,184,.5)",
                                               dash="dash"))
            st.plotly_chart(_fig_c2, use_container_width=True,
                            config={"displayModeBar": False})

        st.markdown("---")

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
