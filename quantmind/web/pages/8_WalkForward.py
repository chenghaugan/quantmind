"""Walk-Forward 滚动样本外验证页面。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from utils.theme import (  # noqa: E402
    setup_page, page_header, section, note, verdict, guard_error,
    kpi_row, fmt_pct, fmt_num, tone_of, badge,
)
from utils.api_client import APIClient  # noqa: E402
from utils.constants import EXCHANGES, STRATEGIES  # noqa: E402
from utils.charts import create_fold_chart, create_returns_histogram  # noqa: E402

setup_page("Walk-Forward", "🔁")
page_header(
    "Walk-Forward 滚动样本外验证",
    "把历史切成「训练窗 + 测试窗」折，逐折在样本外回测，对比全样本与样本外均值以诊断过拟合。",
    "🔁",
)

note(
    "**滚动窗口验证**是严谨回测的标配：单段历史的偶然好成绩 ≠ 稳定 alpha。"
    "通过对比「全样本(in-sample)」与「各折样本外均值」，识别过拟合。",
    "info",
)

# ----------------------------------------------------------------- 输入区
cl, cr = st.columns([1, 2], gap="medium")
with cl:
    st.markdown("**策略配置**")
    strategy_key = st.selectbox(
        "策略", list(STRATEGIES.keys()),
        format_func=lambda k: f"{STRATEGIES[k]['name']} — {STRATEGIES[k]['desc']}",
    )
    asset_class = st.selectbox("资产类别", list(EXCHANGES.keys()))
    exchange = st.selectbox("交易所", EXCHANGES[asset_class])
    symbol = st.text_input("合约代码", "rb0")

    st.markdown("**窗口参数**")
    train_window = st.number_input("训练窗口（根）", value=250, min_value=50, step=10)
    test_window = st.number_input("测试窗口（根）", value=60, min_value=10, step=10)
    step = st.number_input("滚动步长（根）", value=60, min_value=1, step=10)

    st.markdown("**资金与成本**")
    capital = st.number_input("初始资金", value=1_000_000, step=100_000, format="%d")
    cost_model = st.checkbox("启用结构化成本模型", value=False,
                             help="按品种差异化费率（含最低手续费、平今仓倍率、A 股印花税）")
    run_btn = st.button("▶️ 运行 Walk-Forward 验证", type="primary", width="stretch")

with cr:
    st.markdown("**参数说明**")
    st.markdown(
        "**训练窗口 (train_window)**：每折用于预热的历史长度，策略用它计算指标（均线 / 波动率），**不参与绩效统计**。\n\n"
        "**测试窗口 (test_window)**：每折用于样本外验证的长度，策略在测试窗产生信号并计算绩效。\n\n"
        "**滚动步长 (step)**：相邻折之间的滚动步长，默认等于测试窗口（不重叠切分）。\n\n"
        "**过拟合检测**：若全样本 Sharpe 显著高于样本外均值，或样本内盈利而样本外亏损，则判定为疑似过拟合。"
    )

# ----------------------------------------------------------------- 运行
if not run_btn:
    st.stop()

payload = {
    "strategy": strategy_key, "symbol": symbol, "exchange": exchange,
    "train_window": train_window, "test_window": test_window, "step": step,
    "capital": capital, "cost": cost_model,
}
with st.spinner(f"正在运行 Walk-Forward 验证（{STRATEGIES[strategy_key]['name']}）…"):
    result = APIClient.post("/walkforward", json=payload, timeout=180)

if guard_error(result, "Walk-Forward"):
    st.stop()

aggregate = result.get("aggregate", {}) or {}
detail = result.get("detail", {}) or {}
overfit = result.get("overfit_suspected", False)
folds = result.get("folds", []) or []

# ----------------------------------------------------------------- 概览
section("验证概览")
kpi_row([
    {"label": "总折数", "value": aggregate.get("n_folds", 0), "tone": "accent"},
    {"label": "平均 Sharpe", "value": fmt_num(aggregate.get("mean_sharpe", 0), 3),
     "tone": tone_of(aggregate.get("mean_sharpe", 0) - 0.5)},
    {"label": "平均收益", "value": fmt_pct(aggregate.get("mean_total_return", 0)),
     "tone": tone_of(aggregate.get("mean_total_return", 0))},
    {"label": "收益标准差", "value": fmt_pct(aggregate.get("std_total_return", 0)),
     "tone": "neutral"},
    {"label": "盈利折占比", "value": fmt_pct(aggregate.get("positive_rate", 0), 1),
     "tone": "accent"},
])

# ----------------------------------------------------------------- 过拟合结论
if overfit:
    verdict(
        f"疑似过拟合：全样本 Sharpe {fmt_num(detail.get('train_sharpe', 0), 3)} "
        f"vs 样本外 {fmt_num(detail.get('test_sharpe', 0), 3)}；样本外显著衰减，建议简化策略或加正则。",
        "bad", icon="⛔",
    )
else:
    verdict(
        f"过拟合检测通过：全样本 Sharpe {fmt_num(detail.get('train_sharpe', 0), 3)} "
        f"vs 样本外 {fmt_num(detail.get('test_sharpe', 0), 3)}；内外一致。",
        "ok", icon="✅",
    )

# ----------------------------------------------------------------- 各折明细
if folds:
    section("各折绩效明细")
    df = pd.DataFrame(folds)
    df["fold"] = df["fold"] + 1
    df["start"] = pd.to_datetime(df["start"]).dt.strftime("%Y-%m-%d")
    df["end"] = pd.to_datetime(df["end"]).dt.strftime("%Y-%m-%d")
    display = {
        "fold": "折号", "start": "开始", "end": "结束",
        "total_return": "总收益", "sharpe": "Sharpe", "sortino": "Sortino",
        "max_drawdown": "最大回撤", "calmar": "Calmar",
        "win_rate": "胜率", "trade_count": "交易次数",
    }
    st.dataframe(df[list(display.keys())].rename(columns=display),
                 width="stretch", height=400, hide_index=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 下载各折绩效 CSV",
                      data=csv, file_name=f"walkforward_{symbol}_{strategy_key}.csv",
                      mime="text/csv")

    # ----------------------------------------------------------------- 可视化
    section("可视化分析")
    t1, t2 = st.tabs(["Sharpe 折线", "收益分布"])
    with t1:
        st.plotly_chart(create_fold_chart(folds, "sharpe", "各折 Sharpe", height=380),
                        width="stretch", key="wf_sharpe")
    with t2:
        st.plotly_chart(create_returns_histogram(
            [f["total_return"] for f in folds], "样本外收益分布", height=380),
            width="stretch", key="wf_hist")

# ----------------------------------------------------------------- 结论
section("验证结论")
positive_rate = aggregate.get("positive_rate", 0)
mean_sharpe = aggregate.get("mean_sharpe", 0)
if positive_rate > 0.6 and mean_sharpe > 0.5 and not overfit:
    verdict("策略表现稳健：盈利折占比 > 60%、平均 Sharpe > 0.5，样本外与样本内一致，建议进入模拟交易。",
            "ok", icon="✅")
elif positive_rate > 0.5 and mean_sharpe > 0:
    verdict("策略表现一般：盈利折占比或平均 Sharpe 偏低，建议优化参数或换因子组合。", "warn", icon="⚠️")
else:
    verdict("策略表现不佳：盈利折占比 < 50% 或平均 Sharpe < 0，样本外不稳定，不建议进入下一阶段。",
            "bad", icon="⛔")

st.caption("💡 建议在进入模拟交易前至少运行一次 Walk-Forward，确认策略不是历史偶然。")
