"""数据质量评分：缺口 / 异常值 / 换月跳变 / 时效 综合打分。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from utils.theme import (  # noqa: E402
    setup_page, page_header, section, note, verdict, guard_error,
    kpi_row, fmt_num, badge,
)
from utils.api_client import APIClient  # noqa: E402
from utils.constants import EXCHANGES, EXCHANGE_NAMES, INTERVALS, INTERVAL_NAMES, SYMBOL_PRESETS  # noqa: E402
from utils.charts import create_gauge  # noqa: E402

setup_page("数据质量", "🧪")
page_header(
    "数据质量",
    "对行情做缺口 / 异常值 / 换月跳变 / 时效性诊断，输出 0~100 综合健康分与问题清单。",
    "🧪",
)

note(
    "评分逻辑：基础 100 分，依次扣减缺口（≤35）、异常值（≤35）、换月跳变（≤15）、过期（20）。"
    "分数越低代表该标的行情越不可信，回测前建议校验。",
    "info",
)

# ----------------------------------------------------------------- 输入
cl, cr = st.columns([1, 2], gap="medium")
with cl:
    asset_class = st.selectbox("资产类别", list(EXCHANGES.keys()))
    exchange = st.selectbox("交易所", EXCHANGES[asset_class],
                            format_func=lambda x: f"{x} · {EXCHANGE_NAMES.get(x, '')}")
    presets = SYMBOL_PRESETS.get(asset_class, [])
    preset_sym = st.selectbox("速选标的", ["" ] + [f"{s}（{n}）" for s, _, n in presets])
    symbol = st.text_input("合约代码", "IF0")
    if preset_sym:
        symbol = preset_sym.split("（")[0]
        exchange = next((e for s, e, _ in presets if s == preset_sym.split("（")[0]), exchange)
    interval = st.selectbox("周期", INTERVALS, index=INTERVALS.index("1d"),
                            format_func=lambda x: INTERVAL_NAMES.get(x, x))
    freshness_days = st.number_input("时效阈值（天）", value=5, min_value=0, step=1,
                                     help="超过该天数无新数据判定为过期；0 不校验")
    run_btn = st.button("🧪 评估数据质量", type="primary", width="stretch")

with cr:
    st.markdown("**诊断维度**")
    st.markdown(
        "- **缺口（gaps）**：交易日序列中的缺失根数。\n"
        "- **异常值（outliers）**：涨跌幅 / 价格突破 N 倍中位绝对偏差的点。\n"
        "- **换月跳变（rollover_jumps）**：主力换月造成的非交易性跳空。\n"
        "- **时效性（stale）**：距最新一根 K 线是否超过时效阈值。\n\n"
        "以上综合为 `score ∈ [0,100]` 的健康分。"
    )

# ----------------------------------------------------------------- 运行
if not run_btn:
    note("填写标的与时间区间后点击评估。结果会给出健康分、问题清单与明细统计。", "info")
    st.stop()

with st.spinner(f"正在评估 {symbol}.{exchange} 数据质量…"):
    res = APIClient.data_quality(symbol, exchange, interval,
                                 start=None, end=None, freshness_days=freshness_days or None)

if guard_error(res, "数据质量"):
    st.stop()

score = res.get("score", 0.0)
total = res.get("total", 0)
gaps = res.get("gaps", 0)
outliers = res.get("outliers", 0)
rollover = res.get("rollover_jumps", 0)
stale = res.get("stale", False)
last_ts = res.get("last_ts")
issues = res.get("issues", []) or []

# ----------------------------------------------------------------- 结论
if total == 0:
    verdict("空数据：该标的在所选区间没有任何行情，无法评估。", "bad", icon="⛔")
elif score >= 90:
    verdict(f"数据健康（{score:.0f} 分）：缺口 / 异常 / 时效均在可接受范围。", "ok", icon="✅")
elif score >= 70:
    verdict(f"数据基本可用（{score:.0f} 分）：存在少量问题，回测前建议留意。", "warn", icon="⚠️")
else:
    verdict(f"数据质量偏差（{score:.0f} 分）：问题较多，建议换源或清洗后再回测。",
            "bad", icon="⛔")

# ----------------------------------------------------------------- 健康分
section("综合健康分")
gc1, gc2 = st.columns([1, 2], gap="medium")
with gc1:
    # create_gauge 的 good 是数值阈值：>=0.7 视为健康（绿），否则关注（琥珀）
    good_th = 0.9 if score >= 90 else 0.7
    st.plotly_chart(create_gauge(score / 100.0, "数据健康分", vmin=0, vmax=1,
                                 good=good_th, height=240),
                    width="stretch", key="dq_gauge")
with gc2:
    kpi_row([
        {"label": "总根数", "value": f"{total:,}", "tone": "accent"},
        {"label": "缺口", "value": gaps, "tone": "warn" if gaps else "neutral"},
        {"label": "异常值", "value": outliers, "tone": "warn" if outliers else "neutral"},
        {"label": "换月跳变", "value": rollover, "tone": "warn" if rollover else "neutral"},
        {"label": "是否过期", "value": "是" if stale else "否",
         "tone": "bad" if stale else "ok"},
    ])
    if last_ts:
        st.caption(f"最新一根 K 线：{last_ts}")

# ----------------------------------------------------------------- 问题清单
section("问题清单")
if issues:
    for iss in issues:
        st.markdown(f"- {iss}")
else:
    st.markdown(badge("无问题", "success"), unsafe_allow_html=True)
    st.caption("未检测到明显质量缺陷。")

with st.expander("🔎 原始返回", expanded=False):
    st.json(res)

st.caption("💡 数据质量差的标的会直接污染回测结论，建议在上「策略回测」前先在此核查。")
