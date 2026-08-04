"""截面研究：多标的面板因子 IC/IR 评估 + 多空组合回测。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from utils.theme import (  # noqa: E402
    setup_page, page_header, section, note, verdict, guard_error,
    kpi_row, fmt_num, fmt_pct, tone_of, badge,
)
from utils.api_client import APIClient  # noqa: E402
from utils.constants import CS_BASKETS, EXCHANGE_NAMES  # noqa: E402
from utils.charts import create_ic_chart, create_equity_curve, create_drawdown_chart, create_gauge  # noqa: E402

setup_page("截面研究", "🧬")
page_header(
    "截面研究",
    "在多个标的构成的面板上评估 Alpha 截因子的 IC / IR / 单调性，并构建每日横截面多空组合回测。",
    "🧬",
)

note(
    "截面研究至少需要 2 个标的。因子来自 WorldQuant Alpha 系列（alpha002..alpha101 / alpha191_*）。"
    "结果给出全样本 rank-IC 与多空组合净值。",
    "info",
)

# ----------------------------------------------------------------- 因子列表
factors_res = APIClient.cs_factors()
factors = factors_res if isinstance(factors_res, list) else (factors_res.get("factors") if isinstance(factors_res, dict) else [])
if not factors:
    note("未能加载截面因子清单（/cross-section/factors）。请确认后端已启动。", "error")
    st.stop()

# ----------------------------------------------------------------- 输入
cl, cr = st.columns([1, 2], gap="medium")
with cl:
    basket = st.selectbox("标的篮子", list(CS_BASKETS.keys()),
                          format_func=lambda x: f"{x}")
    symbols, exch = CS_BASKETS[basket]
    st.caption("篮子标的：" + " · ".join(symbols))
    custom = st.text_input("自定义标的（逗号分隔，覆盖篮子）", "")
    exchange = st.selectbox("交易所", [exch] + list(EXCHANGE_NAMES.keys()),
                            format_func=lambda x: f"{x} · {EXCHANGE_NAMES.get(x, '')}")
    factor = st.selectbox("截面因子", factors)

    st.markdown("**回测参数**")
    forward_periods = st.slider("前向期数", 1, 20, 1)
    n_groups = st.slider("分组数", 2, 10, 5)
    long_short = st.checkbox("多空组合（头组多 / 尾组空）", value=True)
    cost_rate = st.number_input("单边成本率", value=0.0, step=0.0001, format="%.4f",
                                help="如 0.0005 表示单边万五")
    run_btn = st.button("🧬 运行截面研究", type="primary", width="stretch")

with cr:
    st.markdown("**说明**")
    st.markdown(
        "- **IC（rank-IC）**：因子值与未来收益的截面秩相关，越大越好。\n"
        "- **IR**：IC 的均值 / 标准差，衡量稳定性（|IR|>0.5 较稳）。\n"
        "- **单调性**：头组到尾组收益是否单调，验证因子区分度。\n"
        "- **多空组合**：每日按因子值排名，头组做多、尾组做空，等权，回测净值。\n"
        "- 可用标的内置篮子：黑色系 / 有色 / 贵金属能化 / 农产品 / A股白马。"
    )

# ----------------------------------------------------------------- 运行
if not run_btn:
    note("选择篮子与因子后点击运行。结果包含 IC 衰减、因子画像与多空组合净值。", "info")
    st.stop()

final_symbols = [s.strip() for s in custom.split(",") if s.strip()] or list(symbols)
if len(final_symbols) < 2:
    note("截面研究至少需要 2 个标的。", "warning")
    st.stop()

payload = {
    "symbols": final_symbols, "exchange": exchange, "factor": factor,
    "forward_periods": forward_periods, "n_groups": n_groups,
    "long_short": long_short, "cost_rate": cost_rate, "backtest": True,
}
with st.spinner(f"正在截面研究（{factor}，{len(final_symbols)} 个标的）…"):
    res = APIClient.cross_section(payload)

if guard_error(res, "截面研究"):
    st.stop()

ic_rep = res.get("ic_report") or {}
portfolio = res.get("portfolio") or {}
missing = res.get("missing", []) or []
date_range = res.get("date_range") or [None, None]

ic_mean = ic_rep.get("ic_mean")
ir = ic_rep.get("ir")
ic_pos = ic_rep.get("ic_positive_ratio")
mono5 = ic_rep.get("monotonicity_5")
mono10 = ic_rep.get("monotonicity_10")

# ----------------------------------------------------------------- 结论
abs_ic = abs(ic_mean) if ic_mean is not None and ic_mean == ic_mean else 0.0
if ic_mean is None:
    verdict("未得到有效 IC（数据不足或因子无效）。", "warn", icon="⚠️")
elif abs_ic >= 0.05 and abs(ir or 0) >= 0.5:
    verdict(f"截面因子有效：IC={fmt_num(ic_mean, 4)}、IR={fmt_num(ir, 3)}，"
            "可进入多空组合构建。", "ok", icon="✅")
elif abs_ic >= 0.03:
    verdict(f"因子具备一定截面预测力（IC={fmt_num(ic_mean, 4)}），建议谨慎使用。",
            "warn", icon="⚠️")
else:
    verdict(f"因子截面预测力弱（IC={fmt_num(ic_mean, 4)}），不建议单独使用。",
            "bad", icon="⛔")

# ----------------------------------------------------------------- 概览
section("研究概览")
kpi_row([
    {"label": "标的数", "value": res.get("n_symbols", 0), "tone": "accent"},
    {"label": "样本日数", "value": res.get("n_dates", 0), "tone": "accent"},
    {"label": "IC 均值", "value": fmt_num(ic_mean, 4), "tone": tone_of(abs_ic - 0.03)},
    {"label": "IR", "value": fmt_num(ir, 3), "tone": tone_of(abs(ir or 0) - 0.5)},
    {"label": "IC 正比率", "value": fmt_pct(ic_pos, 1), "tone": "neutral"},
    {"label": "缺失标的", "value": len(missing), "tone": "warn" if missing else "neutral"},
])

if date_range[0] and date_range[1]:
    st.caption(f"样本区间：{date_range[0]} ～ {date_range[1]}"
               + (f"　缺失：{', '.join(missing)}" if missing else ""))

# ----------------------------------------------------------------- 多空组合
if portfolio and portfolio.get("daily_returns"):
    section("多空组合回测")
    rets = np.array([float(x) for x in portfolio["daily_returns"]], dtype=float)
    equity = np.cumprod(1 + rets)
    eq_curve = [{"date": i, "equity": float(e)} for i, e in enumerate(equity)]
    total_ret = float(equity[-1] - 1)
    sharpe = float(np.mean(rets) / np.std(rets) * np.sqrt(252)) if np.std(rets) > 0 else 0.0
    peak = np.maximum.accumulate(equity)
    mdd = float((equity / peak - 1).min())
    win = float((rets > 0).mean())

    kpi_row([
        {"label": "多空总收益", "value": fmt_pct(total_ret), "tone": tone_of(total_ret)},
        {"label": "多空夏普", "value": fmt_num(sharpe, 2), "tone": tone_of(sharpe)},
        {"label": "最大回撤", "value": fmt_pct(mdd), "tone": "down" if mdd < -0.2 else "neutral"},
        {"label": "日胜率", "value": fmt_pct(win, 1), "tone": tone_of(win - 0.5)},
    ])

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.plotly_chart(create_equity_curve(eq_curve, "多空组合净值", height=340),
                        width="stretch", key="cs_eq")
    with c2:
        st.plotly_chart(create_drawdown_chart(eq_curve, "多空组合回撤", height=340),
                        width="stretch", key="cs_dd")

# ----------------------------------------------------------------- IC 画像
section("IC 与因子画像")
cc1, cc2 = st.columns([1.5, 1], gap="medium")
with cc1:
    # IC 衰减构造（单点也可视化）
    ic_decay = []
    if ic_mean is not None:
        ic_decay = [ic_mean] + [None] * min(forward_periods, 5)
    st.plotly_chart(create_ic_chart(ic_decay, f"IC（前向 {forward_periods} 期）", height=320),
                    width="stretch", key="cs_ic")
with cc2:
    radar = {
        "IC 强度": min(abs_ic / 0.08, 1.0),
        "稳定性(IR)": min(abs(ir or 0) / 1.0, 1.0),
        "正比率": abs((ic_pos or 0.5) - 0.5) * 2,
        "单调性5": abs(mono5 or 0.0),
        "单调性10": abs(mono10 or 0.0),
    }
    from utils.charts import create_factor_radar
    st.plotly_chart(create_factor_radar(radar, "因子截面画像", height=320),
                    width="stretch", key="cs_radar")

# ----------------------------------------------------------------- 明细表
section("IC 明细")
rows = [
    ("IC 均值", fmt_num(ic_mean, 4)),
    ("IC 标准差", fmt_num(ic_rep.get("ic_std"), 4)),
    ("IR", fmt_num(ir, 3)),
    ("IC 正比率", fmt_pct(ic_pos, 1)),
    ("5 组单调性", fmt_num(mono5, 3)),
    ("10 组单调性", fmt_num(mono10, 3)),
]
st.dataframe(pd.DataFrame(rows, columns=["指标", "数值"]),
             width="stretch", hide_index=True, height=220)

with st.expander("🔎 原始返回", expanded=False):
    st.json(res)

st.caption("💡 截面有效的因子可接入「策略回测」的多因子打分；单一标的验证参见「因子研究」。")
