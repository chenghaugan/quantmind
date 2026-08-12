"""因子研究：单标的 Alpha 因子有效性评估"""
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
from utils.charts import create_ic_chart, create_factor_radar, create_gauge  # noqa: E402
from utils.constants import EXCHANGES, EXCHANGE_NAMES  # noqa: E402

setup_page("因子研究", "🔬")
page_header(
    "因子研究",
    "单标的因子有效性评估：IC / IR / 衰减 / Bootstrap 置信区间 / 分位单调性 / 多空组合",
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

section("评估设置")
with st.form("factor_form"):
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        symbol = st.text_input("合约代码", "IF0")
        all_ex = [e for exs in EXCHANGES.values() for e in exs]
        exchange = st.selectbox("交易所", all_ex, index=1,
                                format_func=lambda x: f"{x} · {EXCHANGE_NAMES.get(x, '')}")
        mode = st.radio("因子来源", ["内置因子库", "自定义表达式"], horizontal=True)
    with c2:
        if mode == "内置因子库":
            cats = sorted({c for c in cat_map.values() if c})
            cat = st.selectbox("分类", ["全部"] + cats)
            pool = [n for n in names if cat == "全部" or cat_map.get(n) == cat] or names
            default_idx = pool.index("momentum_20") if "momentum_20" in pool else 0
            factor_name = st.selectbox("因子", pool, index=default_idx,
                                       help="共 %d 个内置因子" % len(names))
            expression = None
            if desc_map.get(factor_name):
                st.caption(f"📖 {desc_map[factor_name]}")
        else:
            factor_name = "custom"
            expression = st.text_area(
                "因子表达式", "(close/ref(close,20)-1)", height=88,
                help="可用变量 close/open/high/low/volume；函数 ref/delta/corr/ts_rank/ts_max/ts_min/std/sma …",
            )
        w1, w2 = st.columns(2)
        window = w1.slider("滚动窗口", 5, 120, 20)
        forward = w2.slider("前向期数", 1, 20, 1, help="IC 使用未来 N 期收益")

    submitted = st.form_submit_button("🔬 评估因子", type="primary", width="stretch")

if not registry:
    note("未能从后端加载因子清单，请确认 API 已启动。", "warning")


@st.cache_data(ttl=3600, show_spinner=False)
def evaluate(symbol, exchange, factor, expression, window, forward):
    return APIClient.factor(symbol=symbol, exchange=exchange, factor=factor,
                            expression=expression, window=window, forward_periods=forward)


if not submitted:
    note(
        "选择一个因子后点击评估。评估结果会缓存 1 小时（后端 + 前端双层），"
        "重复查看同一组合不会重新计算。",
        "info",
    )
    st.stop()

with st.spinner("正在计算因子并评估有效性…"):
    r = evaluate(symbol, exchange, factor_name, expression, window, forward)

if guard_error(r, "因子评估"):
    st.stop()

ic = r.get("ic_mean") or 0.0
ir = r.get("ir") or 0.0
n = r.get("n_samples") or 0
ci_low, ci_high = r.get("ic_ci_low"), r.get("ic_ci_high")
mono5 = r.get("monotonicity_5")
score = r.get("composite_score")

# ---------------------------------------------------------------- 结论
abs_ic = abs(ic)
ci_sig = (ci_low is not None and ci_high is not None and ci_low * ci_high > 0)
if n < 60:
    verdict(f"样本量仅 {n}，统计结论不可靠 —— 请拉长历史区间后再判断。", "warn")
elif abs_ic >= 0.05 and abs(ir) >= 0.5 and ci_sig:
    verdict("因子有效性较强：IC 与 IR 双双达标，且置信区间不跨零，可进入组合构建。", "ok")
elif abs_ic >= 0.03 and ci_sig:
    verdict("因子具备一定预测力，但强度中等，建议与其他因子合成后使用。", "warn")
else:
    verdict("因子未通过有效性检验：IC 偏弱或置信区间跨零，不建议单独使用。", "bad")

# ---------------------------------------------------------------- 核心指标
section("核心指标")
kpi_row([
    {"label": "IC 均值（Spearman）", "value": fmt_num(ic, 4), "tone": tone_of(abs_ic - 0.03),
     "hint": "|IC|>0.05 优秀 · >0.03 可用"},
    {"label": "IR（信息比）", "value": fmt_num(ir, 4), "tone": tone_of(abs(ir) - 0.5),
     "hint": "|IR|>0.5 稳定"},
    {"label": "IC 正比率", "value": fmt_pct(r.get("ic_positive_ratio"), 1),
     "hint": "偏离 50% 越多越好"},
    {"label": "样本数", "value": f"{n:,}", "tone": "accent" if n >= 250 else "neutral",
     "hint": "建议 ≥ 250"},
    {"label": "综合评分", "value": fmt_num(score, 1) if score is not None else "—",
     "tone": "accent"},
])

st.write("")
kpi_row([
    {"label": "Top 分位收益", "value": fmt_pct(r.get("top_quantile_return")),
     "tone": tone_of(r.get("top_quantile_return"))},
    {"label": "多空收益", "value": fmt_pct(r.get("long_short_return")),
     "tone": tone_of(r.get("long_short_return"))},
    {"label": "多空夏普", "value": fmt_num(r.get("ls_portfolio_sharpe"), 2),
     "tone": tone_of(r.get("ls_portfolio_sharpe"))},
    {"label": "多空最大回撤", "value": fmt_pct(r.get("ls_portfolio_mdd")),
     "tone": "down" if (r.get("ls_portfolio_mdd") or 0) < -0.2 else "neutral"},
    {"label": "年化换手", "value": fmt_num(r.get("turnover_annual"), 1),
     "hint": "越高交易成本越重"},
])

# ---------------------------------------------------------------- 图表
section("IC 衰减与画像")
c1, c2 = st.columns([1.5, 1], gap="medium")
with c1:
    st.plotly_chart(
        create_ic_chart(r.get("ic_decay") or [], title="IC 随前向期数衰减", height=320),
        width="stretch", key="f_ic",
    )
    hl = r.get("ic_decay_half_life")
    if hl:
        st.caption(f"IC 半衰期约 **{hl:.1f}** 期 —— 建议调仓周期不超过该值。")
with c2:
    radar = {
        "IC 强度": min(abs_ic / 0.08, 1.0),
        "稳定性(IR)": min(abs(ir) / 1.0, 1.0),
        "正比率": abs((r.get("ic_positive_ratio") or 0.5) - 0.5) * 2,
        "单调性": abs(mono5 or 0.0),
        "低换手": max(0.0, 1 - min((r.get("turnover_annual") or 0) / 250, 1.0)),
    }
    st.plotly_chart(create_factor_radar(radar, title="因子画像（归一化）", height=320),
                    width="stretch", key="f_radar")

# ---------------------------------------------------------------- 置信区间
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
            f"[{ci_low:.4f} , {ci_high:.4f}]</span>",
            unsafe_allow_html=True,
        )
    else:
        st.caption("后端未返回置信区间。")
    st.write("")
    rows = [
        ("IC 均值（Spearman）", fmt_num(ic, 4)),
        ("IC 均值（Pearson）", fmt_num(r.get("ic_pearson"), 4)),
        ("IC 标准差", fmt_num(r.get("ic_std"), 4)),
        ("IC 半衰期", fmt_num(r.get("ic_decay_half_life"), 2)),
        ("5 分位单调性", fmt_num(mono5, 3)),
        ("10 分位单调性", fmt_num(r.get("monotonicity_10"), 3)),
    ]
    st.dataframe([{"指标": a, "数值": b} for a, b in rows],
                 width="stretch", hide_index=True, height=250)

if r.get("note"):
    note(f"📝 {r['note']}", "info")

with st.expander("🔎 原始返回", expanded=False):
    st.json(r)

# ---------------------------------------------------------------- 导出 HTML 研报
def _factor_html_report() -> str:
    import plotly.io as pio
    fig = create_gauge(ic, "IC 均值（|IC|>0.03 可用）", vmin=-0.1, vmax=0.1, good=0.03, height=300)
    gauge_chart = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")
    metrics = [
        ("因子", factor_name), ("标的", f"{symbol}.{exchange}"),
        ("IC 均值", fmt_num(ic, 4)), ("IR", fmt_num(ir, 4)),
        ("IC 正比率", fmt_pct(r.get('ic_positive_ratio'), 1)),
        ("样本数", n), ("Top 分位收益", fmt_pct(r.get('top_quantile_return'))),
        ("多空收益", fmt_pct(r.get('long_short_return'))),
        ("综合评分", fmt_num(r.get('composite_score'), 2)),
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
<div class="sub">因子：{factor_name} · 标的：{symbol}.{exchange} · 生成：{pd.Timestamp.now():%Y-%m-%d %H:%M}</div>
<div class="sec">① IC 画像</div>{gauge_chart}
<div class="sec">② 核心指标</div><table><thead><tr><th>指标</th><th>值</th></tr></thead><tbody>{rows}</tbody></table>
</body></html>"""


st.download_button(
    "📄 导出 HTML 研报",
    _factor_html_report().encode("utf-8"),
    file_name=f"factor_report_{factor_name}.html", mime="text/html",
    help="生成含 IC 画像与核心指标的可分享 HTML 研报",
)

st.caption("下一步：多标的横截面表现请前往「截面研究」；把因子接入策略请前往「策略回测」。")
