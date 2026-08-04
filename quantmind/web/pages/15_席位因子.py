"""席位因子研究：商品期货独有持仓因子 F1-F8 的有效性评估。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st  # noqa: E402

from utils.theme import (  # noqa: E402
    setup_page, page_header, section, note, verdict, guard_error,
    kpi_row, fmt_num, fmt_pct, tone_of, badge,
)
from utils.api_client import APIClient  # noqa: E402
from utils.charts import create_gauge, create_quantile_bar  # noqa: E402
from utils.constants import EXCHANGES, EXCHANGE_NAMES  # noqa: E402

setup_page("席位因子", "🪑")
page_header(
    "席位因子研究",
    "商品期货独有：基于交易所会员持仓排名（席位净持仓）的 F1–F8 因子，评估其对次日收益的预测力（IC / IR / 分组收益）。",
    "🪑",
)

note(
    "<b>数据依赖</b>：需先把 TradingAgents_for_Futures 仓库的 <code>qihuo/database/positioning</code> "
    "目录映射到「本地数据」的<code>席位数据</code>路径（可在「设置」页配置），格式为 "
    "<code>&lt;品种&gt;/long/short/volume_ranking.csv</code>。否则无法计算真实 IC。",
    "warning",
)

# ---------------------------------------------------------------- 因子清单
@st.cache_data(ttl=600, show_spinner=False)
def load_seat_factors():
    res = APIClient.seat_factors(timeout=10)
    return res.get("factors", {}) if isinstance(res, dict) else {}


@st.cache_data(ttl=600, show_spinner=False)
def load_data_roots():
    return APIClient.data_roots(timeout=10)


factor_desc = load_seat_factors()
roots = load_data_roots()

section("评估设置")
with st.form("seat_form"):
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        default_root = roots.get("seat_data_root", "") if isinstance(roots, dict) else ""
        seat_root = st.text_input(
            "席位数据根目录", default_root,
            help="指向 qihuo/database/positioning，含 <品种>/long/short/volume_ranking.csv",
        )
        symbol = st.text_input("品种代码", "RB", help="如 RB / CU / AG")
        all_ex = [e for exs in EXCHANGES.values() for e in exs]
        exchange = st.selectbox("交易所", all_ex, index=0,
                                format_func=lambda x: f"{x} · {EXCHANGE_NAMES.get(x, '')}")
    with c2:
        names = list(factor_desc.keys()) if factor_desc else []
        idx = names.index("F7_net_zscore") if "F7_net_zscore" in names else 0
        factor = st.selectbox("席位因子", names, index=idx) if names else st.text_input("席位因子", "F7_net_zscore")
        if factor_desc:
            st.caption(f"📖 {factor_desc.get(factor, '')}")
        w1, w2 = st.columns(2)
        forward = w1.slider("前向期数", 1, 10, 1, help="IC 使用未来 N 期收益")
        n_groups = w2.slider("分组数", 3, 10, 5, help="按因子值分位分组")

    submitted = st.form_submit_button("🪑 评估席位因子", type="primary", width="stretch")

if not factor_desc:
    note("未能从后端加载席位因子清单，请确认 API 已启动。", "warning")

if not submitted:
    note("选择席位数据根目录与因子后点击评估：计算 IC / IR / IC 正比率 / Top 与多空分位收益。", "info")
    st.stop()

payload = {
    "symbol": symbol, "exchange": exchange, "interval": "1d",
    "seat_data_root": seat_root, "factor": factor, "aggregate": True,
    "forward_periods": int(forward), "n_groups": int(n_groups), "long_short": True,
}
with st.spinner("正在加载席位 CSV 并计算因子…"):
    r = APIClient.seat_factor(payload)

if guard_error(r, "席位因子评估"):
    st.stop()

if r.get("error"):
    note(f"<b>计算失败</b>：{r['error']}", "error")
    st.stop()

# ---------------------------------------------------------------- 结论
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

# ---------------------------------------------------------------- 核心指标
section("核心指标")
kpi_row([
    {"label": "IC 均值（Pearson）", "value": fmt_num(ic, 4), "tone": tone_of(abs_ic - 0.03),
     "hint": "|IC|>0.05 优秀 · >0.03 可用"},
    {"label": "IR（信息比）", "value": fmt_num(ir, 4), "tone": tone_of(abs(ir) - 0.5)},
    {"label": "IC 正比率", "value": fmt_pct(r.get("ic_positive_ratio"), 1),
     "hint": "偏离 50% 越多越好"},
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

# ---------------------------------------------------------------- 图表
if r.get("long_short_return") is not None:
    section("因子画像")
    c1, c2 = st.columns([1, 1], gap="medium")
    with c1:
        st.plotly_chart(
            create_gauge(ic, "IC 均值（|IC|>0.03 可用）", vmin=-0.1, vmax=0.1, good=0.03, height=260),
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
