"""策略生命周期管理：IDEA → RESEARCH → BACKTEST → PAPER → APPROVED → LIVE 晋升闸门。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from utils.theme import (  # noqa: E402
    setup_page, page_header, section, note, verdict, guard_error, badge, kpi_row,
)
from utils.api_client import APIClient  # noqa: E402
from utils.constants import LIFECYCLE_STATES, LIFECYCLE_DESC  # noqa: E402

setup_page("生命周期", "🔄")
page_header(
    "策略生命周期",
    "策略从研发到实盘需通过六道闸门，每道都有严格的绩效指标门槛，确保质量后再放量。",
    "🔄",
)

note(
    "流程：**IDEA → RESEARCH → BACKTEST → PAPER → APPROVED → LIVE**。"
    "晋升只能按顺序、不可跳级；到达 LIVE 前需满足 Sharpe、回撤、模拟天数与风险审查等闸门。",
    "info",
)

# ----------------------------------------------------------------- 闸门说明
with st.expander("📋 晋升闸门要求", expanded=False):
    st.markdown(
        "**从 PAPER / APPROVED 晋升到 LIVE 的核心要求**：\n"
        "- ✅ Sharpe Ratio ≥ 0.5\n"
        "- ✅ 最大回撤 ≤ 30%（即 `max_drawdown ≥ -0.30`）\n"
        "- ✅ 模拟交易天数 ≥ 1 天\n"
        "- ✅ 风险审查标记（备注需包含 `risk_reviewed`）\n\n"
        "其他阶段由 `PromotionGate` 配置控制。"
    )

# ----------------------------------------------------------------- 输入区
col_left, col_right = st.columns([1, 2], gap="medium")

with col_left:
    st.markdown("**策略信息**")
    strategy_id = st.text_input("策略 ID", "strat-001", help="唯一标识符")
    st.markdown("**目标状态**")
    to_state = st.selectbox(
        "晋升到", LIFECYCLE_STATES[1:], index=3,
        help="只能按顺序晋升，不可跳级",
    )

with col_right:
    st.markdown("**绩效指标**")
    c1, c2 = st.columns(2)
    with c1:
        sharpe = st.number_input("Sharpe Ratio", value=0.8, step=0.1, format="%.2f")
        max_dd = st.number_input("最大回撤", value=-0.15, step=0.01, format="%.2f",
                                 help="负数，如 -0.15 表示 15%")
    with c2:
        paper_days = st.number_input("模拟交易天数", value=30, min_value=0, step=1)
        risk_reviewed = st.checkbox("已完成风险审查", value=True)

    st.markdown("**备注**")
    note_text = st.text_area(
        "备注", "策略表现稳定，Sharpe > 1.0，回撤可控",
        height=100,
        help="若目标状态是 LIVE，必须包含 'risk_reviewed'",
    )

# ----------------------------------------------------------------- 执行
if st.button("🚀 执行晋升", type="primary", width="stretch"):
    metrics = {"sharpe": sharpe, "max_drawdown": max_dd, "paper_days": paper_days}
    full_note = note_text
    if risk_reviewed and "risk_reviewed" not in note_text.lower():
        full_note = "risk_reviewed\n" + note_text if note_text else "risk_reviewed"

    with st.spinner(f"正在晋升 {strategy_id} → {to_state}…"):
        result = APIClient.lifecycle(strategy_id, to_state, metrics, full_note)

    if guard_error(result, "晋升"):
        st.stop()

    if result.get("ok"):
        state = result.get("state")
        verdict(f"晋升成功！当前状态：**{state}**", "ok", icon="✅")
        history = result.get("history", []) or []
        if history:
            section("晋升历史")
            df = pd.DataFrame(history)
            if "at" in df.columns:
                df["at"] = pd.to_datetime(df["at"])
            st.dataframe(df, width="stretch", hide_index=True)
    else:
        reasons = result.get("reasons", []) or []
        verdict("晋升被闸门拦截：" + (reasons[0] if reasons else "未满足晋升条件"),
                "bad", icon="⛔")
        for r in reasons:
            st.markdown(f"- {r}")

# ----------------------------------------------------------------- 状态定义
section("状态定义")
rows = [
    {"状态": s, "说明": LIFECYCLE_DESC.get(s, ""),
     "徽章": badge(s, "success" if s == "LIVE" else ("info" if s in ("BACKTEST", "APPROVED") else "muted"))}
    for s in LIFECYCLE_STATES
]
st.dataframe(
    pd.DataFrame(rows)[["状态", "说明", "徽章"]],
    width="stretch", hide_index=True,
    column_config={"徽章": st.column_config.Column(help="阶段徽章", width="small")},
)

st.caption("💡 晋升后前往「策略回测」用真实成本复跑一遍，再进入「实时监控」做模拟验证。")
