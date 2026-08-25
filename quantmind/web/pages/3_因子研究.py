"""因子研究：Alpha 因子有效性评估（单标的时序 / 多标的截面）

- 单标的模式：因子在单一标的上随时间的预测力（时序 IC / IR / 衰减 / 置信区间 / 多空）
- 多标的模式：因子在多个标的上横向的选标能力（截面 IC / IR / 置信区间 / 单调性 / 截面多空），
  支持从内置篮子多选（可混选商品/指数）、也可手动加自定义标的。

两类 IC 视角不同（时序=择时、截面=选标的），分开展示、不互相污染。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from utils.theme import (  # noqa: E402
    setup_page, page_header, section, kpi_row, note, verdict, guard_error,
    fmt_num, fmt_pct, tone_of, badge,
)
from utils.api_client import APIClient  # noqa: E402
from utils.charts import (create_ic_chart, create_factor_radar, create_gauge,  # noqa: E402
                           create_equity_curve, create_quantile_bar)
from utils.constants import (  # noqa: E402
    EXCHANGES, EXCHANGE_NAMES, BASKET_CHOICES, CS_BASKETS,
    resolve_basket_symbols,
)

setup_page("因子研究", "🔬")
page_header(
    "因子研究",
    "Alpha 因子有效性评估：单标的时序 IC / 多标的截面 IC · IC/IR/衰减/置信区间/分位单调性/多空组合",
    "🔬",
)


@st.cache_data(ttl=600, show_spinner=False)
def load_factors():
    res = APIClient.factors(timeout=10)
    return res.get("factors", []) if isinstance(res, dict) else []


registry = load_factors()
names = [f.get("name") for f in registry if f.get("name")]
desc_map = {f.get("name"): f.get("description", "") for f in registry}
cat_map = {f.get("name"): f.get("category", "") for f in registry}


@st.cache_data(ttl=600, show_spinner=False)
def load_cs_factors():
    """多标（截面）模式：WorldQuant 截面 Alpha 因子清单。"""
    res = APIClient.cs_factors(timeout=10)
    return res.get("factors", []) if isinstance(res, dict) else []

cs_names = load_cs_factors()


@st.cache_data(ttl=600, show_spinner=False)
def load_seat_factors():
    """席位因子 F1-F8 清单。"""
    res = APIClient.seat_factors(timeout=10)
    return res.get("factors", {}) if isinstance(res, dict) else {}


@st.cache_data(ttl=600, show_spinner=False)
def load_data_roots():
    return APIClient.data_roots(timeout=10)


factor_desc = load_seat_factors()
seat_roots = load_data_roots()

# ============================ 评估设置 ============================
section("评估设置")

# 标的模式（form 外，radio 变化自动重跑脚本）
mode_target = st.radio(
    "标的模式", ["单标的（时序）", "多标的（截面）"], horizontal=True,
    help="单标的=在一个标的上按时间评估（择时）；多标的=在多个标的上按横向排序评估（选标的）。两者 IC 含义不同，分开展示。",
)

# 因子来源（form 外，全局可见；席位因子=商品期货持仓因子）
mode = st.radio("因子来源", ["内置因子库", "自定义表达式", "席位因子"], horizontal=True)

with st.form("factor_form"):
    left, right = st.columns(2, gap="medium")

    with left:
        st.markdown("**标的**")
        if mode == "席位因子":
            _root = seat_roots.get("seat_data_root", "") if isinstance(seat_roots, dict) else ""
            seat_root = st.text_input(
                "席位数据根目录", _root,
                help="指向 qihuo/database/positioning，含 <品种>/long/short/volume_ranking.csv")
            seat_symbol = st.text_input("品种代码", "RB", help="如 RB / CU / AG")
            symbol = seat_symbol
            exchange = "SHFE"
            _multi_symbols = None
        elif mode_target == "单标的（时序）":
            symbol = st.text_input("合约代码", "IF0")
            all_ex = [e for exs in EXCHANGES.values() for e in exs]
            exchange = st.selectbox("交易所", all_ex, index=1,
                                    format_func=lambda x: f"{x} · {EXCHANGE_NAMES.get(x, '')}")
            _multi_symbols = None
        else:
            # 多标的：多选篮子（可混选商品期货 / A股 / 指数）+ 自定义标的
            st.caption("可多选篮子（商品/股指/股票/指数，允许混选），另可手动加自定义标的。")
            baskets = st.multiselect(
                "标的篮子（可多选）", BASKET_CHOICES, default=["股指期货（CFFEX）"],
                format_func=lambda x: str(x), max_selections=8,
            )
            custom = st.text_input(
                "自定义标的（逗号分隔，与篮子合并）", "",                help="例：rb0,cu0,IF0 或 600519.SSE。与上方篮子取并集。",
            )
            max_sym = st.number_input("指数篮子标的数上限", 5, 300, 100,
                                      help="对 全A/沪深300/中证500/中证2000 等股票池生效，避免大截面拖慢计算")
            # 汇总标的清单
            _merged: list[str] = []
            _exch_default = None
            for b in baskets:
                syms, exc = resolve_basket_symbols(b, max_symbols=int(max_sym))
                _merged.extend(syms)
                if _exch_default is None:
                    _exch_default = exc
            if custom.strip():
                for s in custom.split(","):
                    s = s.strip()
                    if s:
                        _merged.append(s)
            # 去重保序
            _multi_symbols = list(dict.fromkeys(_merged))
            exchange = _exch_default or "SHFE"
            st.caption(f"共 {len(_multi_symbols)} 个标的：{_multi_symbols[:6]}" + ("…" if len(_multi_symbols) > 6 else ""))

        # 日期范围
        st.markdown("**数据区间**（可选，留空使用全部数据）")
        dc1, dc2 = st.columns(2)
        with dc1:
            start_date = st.date_input("开始日期", value=None, format="YYYY-MM-DD")
        with dc2:
            end_date = st.date_input("结束日期", value=None, format="YYYY-MM-DD")

    with right:
        if mode == "内置因子库":
            if mode_target == "多标的（截面）":
                # 多标的模式：用 WorldQuant 截面 Alpha 因子清单
                pool = cs_names or names
                default_idx = pool.index("alpha002") if "alpha002" in pool else 0
                factor_name = st.selectbox(
                    "截面因子", pool, index=default_idx,
                    help="多标模式使用 WorldQuant 截面 Alpha（共 %d 个）" % len(pool))
                expression = None
                st.caption("截面 Alpha 因子（对应 /factor/cs-factors 清单）")
            else:
                cats = sorted({c for c in cat_map.values() if c})
                cat = st.selectbox("分类", ["全部"] + cats)
                pool = [n for n in names if cat == "全部" or cat_map.get(n) == cat] or names
                default_idx = pool.index("momentum_20") if "momentum_20" in pool else 0
                factor_name = st.selectbox("因子", pool, index=default_idx,
                                           help="共 %d 个内置因子" % len(names))
                expression = None
                if desc_map.get(factor_name):
                    st.caption(f"📖 {desc_map[factor_name]}")
        elif mode == "自定义表达式":
            factor_name = "custom"
            expression = st.text_area(
                "因子表达式", "(close/ref(close,20)-1)", height=88,
                help="单标：ref/delta/corr/ts_rank…；多标（截面）用 delay/mean/rank… "
                     "（ref 会自动映射为 delay）",
            )

        if mode_target == "多标的（截面）" and mode != "席位因子":
            gc1, gc2, gc3 = st.columns(3)
            n_groups = gc1.slider("分组数", 2, 10, 5)
            cost_rate = gc2.number_input("单边成本率", value=0.0, step=0.0001,
                                         format="%.4f", help="如 0.0005=单边万五")
            long_short = gc3.checkbox("多空组合", value=True)
        else:
            n_groups, cost_rate = 5, 0.0
        model_col, _ = st.columns([1, 1])
        forward = model_col.slider("前向期数", 1, 20, 1, help="IC 使用未来 N 期收益")

    # 单标的专属参数（时序窗口）；多标的时窗口不影响截面
    if mode_target == "单标的（时序）":
        window = st.slider("滚动窗口", 5, 120, 20)
    else:
        window = 20
        st.caption("多标的模式：截面 IC 直接在每个标的上计算因子值后横向评估（滚动窗口仅对时序模式生效）。")

    submitted = st.form_submit_button("🔬 评估因子", type="primary", width="stretch")

if not registry:
    note("未能从后端加载因子清单，请确认 API 已启动。", "warning")

# ============================ 缓冲与执行 ============================
@st.cache_data(ttl=3600, show_spinner=False)
def evaluate(symbol, exchange, factor, expression, window, forward, start, end):
    return APIClient.factor(symbol=symbol, exchange=exchange, factor=factor,
                            expression=expression, window=window, forward_periods=forward,
                            start=start, end=end)


@st.cache_data(ttl=3600, show_spinner=False)
def evaluate_cs(symbols, exchange, factor, expression, forward, long_short,
                n_groups, cost_rate, start, end):
    payload = {
        "symbols": symbols, "exchange": exchange, "factor": factor,
        "expression": expression, "forward_periods": forward,
        "long_short": bool(long_short), "n_groups": int(n_groups),
        "cost_rate": float(cost_rate), "backtest": True,
    }
    if start:
        payload["start"] = start
    if end:
        payload["end"] = end
    return APIClient.cross_section(payload, timeout=600)


@st.cache_data(ttl=3600, show_spinner=False)
def evaluate_seat(seat_root, symbol, exchange, factor, forward, n_groups):
    return APIClient.seat_factor({
        "symbol": symbol, "exchange": exchange, "interval": "1d",
        "seat_data_root": seat_root, "factor": factor, "aggregate": True,
        "forward_periods": int(forward), "n_groups": int(n_groups), "long_short": True,
    }, timeout=600)


if not submitted:
    note(
        "选择因子与标的（单标的或多标的）后点击评估。结果会缓存 1 小时（后端 + 前端双层）。\n"
        "多标的模式指标更全（含 Bootstrap 置信区间、5/10 分位单调性、换手、截面多空组合）。",
        "info",
    )
    st.stop()

start_str = start_date.isoformat() if start_date else None
end_str = end_date.isoformat() if end_date else None

with st.spinner("正在计算因子并评估有效性…"):
    if mode == "席位因子":
        r = evaluate_seat(seat_root, symbol, exchange, factor_name, forward, n_groups)
        if guard_error(r, "席位因子评估"):
            st.stop()
        if r.get("error"):
            note(f"<b>计算失败</b>：{r['error']}", "error")
            st.stop()
        # 席位专属展示（复用 theme/charts）
        ic = r.get("ic_mean") or 0.0
        ir = r.get("ir") or 0.0
        n = r.get("n_samples") or 0
        abs_ic = abs(ic)
        if n < 60:
            verdict(f"样本量仅 {n}，统计结论不可靠 —— 请确认席位数据与价格数据对齐。", "warn", icon="⚠️")
        elif abs_ic >= 0.05 and abs(ir) >= 0.5:
            verdict("席位因子预测力较强：IC 与 IR 双达标，可作为商品期货 alpha 信号。", "ok")
        elif abs_ic >= 0.03:
            verdict("因子有一定预测力但强度中等，建议与其他因子合成使用。", "warn")
        else:
            verdict("因子未通过有效性检验：IC 偏弱，建议换因子或检查席位数据完整性。", "bad")
        section("核心指标")
        kpi_row([
            {"label": "IC 均值（Pearson）", "value": fmt_num(ic, 4),
             "tone": tone_of(abs_ic - 0.03), "hint": "|IC|>0.05 优秀 · >0.03 可用"},
            {"label": "IR（信息比）", "value": fmt_num(ir, 4), "tone": tone_of(abs(ir) - 0.5)},
            {"label": "IC 正比率", "value": fmt_pct(r.get("ic_positive_ratio"), 1)},
            {"label": "有效样本", "value": f"{n:,}", "tone": "accent" if n >= 250 else "neutral"},
            {"label": "席位数 / 日期数", "value": f"{r.get('n_seats', 0)} / {r.get('n_dates', 0)}",
             "tone": "accent"},
            {"label": "综合评分", "value": fmt_num(r.get("composite_score"), 3), "tone": "accent"},
        ])
        st.write("")
        kpi_row([
            {"label": "Top 分位收益", "value": fmt_pct(r.get("top_quantile_return")),
             "tone": tone_of(r.get("top_quantile_return"))},
            {"label": "多空收益", "value": fmt_pct(r.get("long_short_return")),
             "tone": tone_of(r.get("long_short_return"))},
        ])
        if r.get("long_short_return") is not None:
            section("因子画像")
            c1, c2 = st.columns([1, 1], gap="medium")
            with c1:
                st.plotly_chart(
                    create_gauge(ic, "IC 均值（|IC|>0.03 可用）", vmin=-0.1, vmax=0.1,
                                 good=0.03, height=260),
                    width="stretch", key="seat_gauge",
                )
            with c2:
                st.plotly_chart(
                    create_quantile_bar({"Top": r.get("top_quantile_return"), "Bottom": 0.0,
                                         "多空": r.get("long_short_return")}, "分位收益"),
                    width="stretch", key="seat_bar",
                )
        with st.expander("🔎 原始返回", expanded=False):
            st.json(r)
        st.caption("下一步：把验证有效的席位因子通过「策略回测」接入自定义多因子策略。")
        st.stop()
    elif mode_target == "多标的（截面）":
        if len(_multi_symbols) < 2:
            note("多标的模式至少需要 2 个标的。", "warning")
            st.stop()
        r = evaluate_cs(_multi_symbols, exchange, factor_name, expression,
                        forward, long_short, n_groups, cost_rate, start_str, end_str)
        if guard_error(r, "因子研究"):
            st.stop()
        ic_rep = r.get("ic_report") or {}
        portfolio = r.get("portfolio") or {}
        symbol_label = f"{len(_multi_symbols)} 个标的 · 截面"
        # 从 ic_report + portfolio 组装统一指标视图
        view = dict(ic_rep)
        for k in ("ls_portfolio_return", "ls_portfolio_sharpe", "ls_portfolio_mdd"):
            if k not in view and portfolio.get(k) is not None:
                view[k] = portfolio.get(k)
        _cs_portfolio_daily = portfolio.get("daily_returns") or []
    else:
        r = evaluate(symbol, exchange, factor_name, expression, window, forward,
                     start_str, end_str)
        if guard_error(r, "因子研究"):
            st.stop()
        view = r
        symbol_label = f"{symbol}.{exchange} · 时序"
        _cs_portfolio_daily = None

# ============================ 统一展示 ============================
ic = view.get("ic_mean") or 0.0
ir = view.get("ir") or 0.0
n = view.get("n_samples") or 0
ci_low, ci_high = view.get("ic_ci_low"), view.get("ic_ci_high")
mono5 = view.get("monotonicity_5")
score = view.get("composite_score")

abs_ic = abs(ic)
ci_sig = (ci_low is not None and ci_high is not None and ci_low * ci_high > 0)

# ---------------------------------------------------------------- 结论
section("结论")
if n < 60:
    verdict(f"样本量仅 {n}，统计结论不可靠 —— 请拉长历史区间后再判断。", "warn")
elif abs_ic >= 0.05 and abs(ir) >= 0.5 and ci_sig:
    verdict(f"因子有效性较强：IC 与 IR 双双达标，且置信区间不跨零，可进入组合构建。"
            f"（{symbol_label}）", "ok")
elif abs_ic >= 0.03 and ci_sig:
    verdict(f"因子具备一定预测力，但强度中等，建议与其他因子合成后使用。"
            f"（{symbol_label}）", "warn")
else:
    verdict(f"因子未通过有效性检验：IC 偏弱或置信区间跨零，不建议单独使用。"
            f"（{symbol_label}）", "bad")

# ---------------------------------------------------------------- 核心指标
section("核心指标")
if mode_target == "多标的（截面）":
    st.caption("以下为**截面视角**（rank-IC：同一时点因子值跨标的排序 vs 未来收益）；"
               "回答『买哪个/空哪个』的选标问题。")
else:
    st.caption("以下为**时序视角**（因子在该标的上随时间的预测力）；回答『何时买卖』的择时问题。")

kpi_row([
    {"label": "IC 均值（Spearman）", "value": fmt_num(ic, 4), "tone": tone_of(abs_ic - 0.03),
     "hint": "|IC|>0.05 优秀 · >0.03 可用"},
    {"label": "IR（信息比）", "value": fmt_num(ir, 4), "tone": tone_of(abs(ir) - 0.5),
     "hint": "|IR|>0.5 稳定"},
    {"label": "IC 正比率", "value": fmt_pct(view.get("ic_positive_ratio"), 1),
     "hint": "偏离 50% 越多越好"},
    {"label": "样本数", "value": f"{n:,}", "tone": "accent" if n >= 250 else "neutral",
     "hint": "建议 ≥ 250"},
    {"label": "综合评分", "value": fmt_num(score, 1) if score is not None and score == score else "—",
     "tone": "accent"},
])
st.write("")
kpi_row([
    {"label": "Top 分位收益", "value": fmt_pct(view.get("top_quantile_return")),
     "tone": tone_of(view.get("top_quantile_return"))},
    {"label": "多空收益", "value": fmt_pct(view.get("long_short_return")),
     "tone": tone_of(view.get("long_short_return"))},
    {"label": "多空夏普", "value": fmt_num(view.get("ls_portfolio_sharpe"), 2),
     "tone": tone_of(view.get("ls_portfolio_sharpe"))},
    {"label": "多空最大回撤", "value": fmt_pct(view.get("ls_portfolio_mdd")),
     "tone": "down" if (view.get("ls_portfolio_mdd") or 0) < -0.2 else "neutral"},
    {"label": "年化换手", "value": fmt_num(view.get("turnover_annual"), 1),
     "hint": "越高交易成本越重"},
])

# ---------------------------------------------------------------- 图表
section("IC 衰减与画像")
c1, c2 = st.columns([1.5, 1], gap="medium")
with c1:
    st.plotly_chart(
        create_ic_chart(view.get("ic_decay") or [], title="IC 随前向期数衰减", height=320),
        width="stretch", key="f_ic",
    )
    hl = view.get("ic_decay_half_life")
    if hl and hl == hl:
        st.caption(f"IC 半衰期约 **{hl:.1f}** 期 —— 建议调仓周期不超过该值。")
with c2:
    radar = {
        "IC 强度": min(abs_ic / 0.08, 1.0),
        "稳定性(IR)": min(abs(ir) / 1.0, 1.0),
        "正比率": abs((view.get("ic_positive_ratio") or 0.5) - 0.5) * 2,
        "单调性": abs(mono5 or 0.0),
        "低换手": max(0.0, 1 - min((view.get("turnover_annual") or 0) / 250, 1.0)),
    }
    st.plotly_chart(create_factor_radar(radar, title="因子画像（归一化）", height=320),
                    width="stretch", key="f_radar")

# ---------------------------------------------------------------- 截面多空净值（多标的）
if _cs_portfolio_daily:
    import numpy as np
    section("多空组合回测（截面）")
    rets = np.array([float(x) for x in _cs_portfolio_daily], dtype=float)
    equity = np.cumprod(1 + rets)
    eq_curve = [{"date": i, "equity": float(e)} for i, e in enumerate(equity)]
    st.plotly_chart(
        create_equity_curve(eq_curve, "每日横截面多空组合净值", height=320),
        width="stretch", key="f_cs_eq",
    )

# ---------------------------------------------------------------- 统计显著性
section("统计显著性")
c1, c2 = st.columns([1, 1.5], gap="medium")
with c1:
    st.plotly_chart(
        create_gauge(ic, "IC 均值（|IC|>0.03 可用）", vmin=-0.1, vmax=0.1,
                     good=0.03, height=250),
        width="stretch", key="f_gauge",
    )
with c2:
    if ci_low is not None and ci_high is not None:
        tag = badge("显著（不跨零）", "success") if ci_sig else badge("不显著（跨零）", "danger")
        st.markdown(
            f"**Bootstrap 95% 置信区间**　{tag}<br>"
            f"<span style='font-family:monospace;font-size:1.05rem;color:#93c5fd'>"
            f"[{float(ci_low):.4f} , {float(ci_high):.4f}]</span>",
            unsafe_allow_html=True,
        )
    else:
        st.caption("后端未返回置信区间。")
    st.write("")
    rows = [
        ("IC 均值（Spearman）", fmt_num(ic, 4)),
        ("IC 均值（Pearson）", fmt_num(view.get("ic_pearson"), 4)),
        ("IC 标准差", fmt_num(view.get("ic_std"), 4)),
        ("IC 半衰期", fmt_num(view.get("ic_decay_half_life"), 2)),
        ("5 分位单调性", fmt_num(mono5, 3)),
        ("10 分位单调性", fmt_num(view.get("monotonicity_10"), 3)),
    ]
    st.dataframe([{"指标": a, "数值": b} for a, b in rows],
                 width="stretch", hide_index=True, height=250)

if r.get("note"):
    note(f"📝 {r['note']}", "info")

if mode_target == "多标的（截面）" and r.get("missing"):
    note(f"缺失标的（未纳入）：{', '.join(r['missing'])}", "warning")

with st.expander("🔎 原始返回", expanded=False):
    st.json(r)

# ---------------------------------------------------------------- 导出 HTML 研报
def _factor_html_report() -> str:
    import plotly.io as pio
    fig = create_gauge(ic, "IC 均值（|IC|>0.03 可用）", vmin=-0.1, vmax=0.1, good=0.03, height=300)
    gauge_chart = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")
    metrics = [
        ("因子", factor_name), ("标的", symbol_label),
        ("IC 均值", fmt_num(ic, 4)), ("IR", fmt_num(ir, 4)),
        ("IC 正比率", fmt_pct(view.get('ic_positive_ratio'), 1)),
        ("样本数", n), ("Top 分位收益", fmt_pct(view.get('top_quantile_return'))),
        ("多空收益", fmt_pct(view.get('long_short_return'))),
        ("综合评分", fmt_num(view.get('composite_score'), 2)),
    ]
    rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in metrics)
    return f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>QuantMind 因子研报 · {factor_name}</title><style>
body{{font-family:Segoe UI,Microsoft YaHei,sans-serif;margin:32px auto;max-width:900px;color:#1f2937}}
h1{{font-size:22px}}.sub{{color:#6b7280;font-size:14px}}
table{{border-collapse:collapse;width:100%;margin-top:16px}}
th,td{{border:1px solid #e5e7eb;padding:7px 10px;text-align:left;font-size:13px}}
th{{background:#f3f4f6}}.sec{{margin-top:24px;font-weight:700;font-size:16px;border-left:4px solid #8b5cf6;padding-left:8px}}
</style></head><body>
<h1>QuantMind 因子有效性研报</h1>
<div class="sub">因子：{factor_name} · 标的：{symbol_label} · 生成：{pd.Timestamp.now():%Y-%m-%d %H:%M}</div>
<div class="sec">① IC 画像</div>{gauge_chart}
<div class="sec">② 核心指标</div><table><thead><tr><th>指标</th><th>值</th></tr></thead><tbody>{rows}</tbody></table>
</body></html>"""


st.download_button(
    "📄 导出 HTML 研报",
    _factor_html_report().encode("utf-8"),
    file_name=f"factor_report_{factor_name}.html", mime="text/html",
    help="生成含 IC 画像与核心指标的可分享 HTML 研报",
)

st.caption("下一步：把因子接入策略请前往「策略回测」；完整策略规则测试请前往「策略挖掘」。")
