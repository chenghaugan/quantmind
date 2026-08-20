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


def _kb_conn_note():
    """页面顶部显示当前知识库连接状态：真实 DB 路径 + 各类条数。

    直读本地 KnowledgeStore。任何异常都降级为温和提示，绝不抛异常阻断页面；
    某类目读取慢/异常时仅该类显示 '—'。
    """
    try:
        from quantmind.knowledge import KnowledgeStore
        ks = KnowledgeStore()
        dbp = str(getattr(ks, "db_path", "") or "—")
    except Exception:  # noqa: BLE001 包/库不可用则降级提示
        st.caption("📚 知识库：后端/本地库不可用（页面仍可正常使用，从项目根启动可连到 db/knowledge.db）")
        return
    counts = {}
    for kind in ("factor", "strategy", "methodology", "research_log"):
        try:
            counts[kind] = len(ks.list_items(kind=kind, limit=500))
        except Exception:  # noqa: BLE001 单类读取失败仅该类降级
            counts[kind] = "—"
    try:
        counts["lifecycle"] = len(ks.list_strategy_lifecycles())
    except Exception:  # noqa: BLE001
        counts["lifecycle"] = "—"
    st.caption(
        f"📚 知识库：<code>{dbp}</code> ｜ 因子 {counts['factor']} · "
        f"策略 {counts['strategy']} · 方法论 {counts['methodology']} · "
        f"研究日志 {counts['research_log']} · 生命周期 {counts['lifecycle']}",
        unsafe_allow_html=True,
    )


setup_page("生命周期", "🔄")
page_header(
    "策略生命周期",
    "策略从研发到实盘需通过六道闸门，每道都有严格的绩效指标门槛，确保质量后再放量。",
    "🔄",
)

_kb_conn_note()

note(
    "流程：**IDEA → RESEARCH → BACKTEST → PAPER → APPROVED → LIVE**。"
    "晋升只能按顺序、不可跳级；到达 LIVE 前需满足 Sharpe、回撤、模拟天数与风险审查等闸门。",
    "info",
)

# ----------------------------------------------------------------- 生命周期一览（已落库）
section("📊 生命周期一览（已落库）")
try:
    from quantmind.knowledge import KnowledgeStore  # 本地包，随页加载，降级不阻断
except Exception:  # noqa: BLE001 包不可用则降级
    KnowledgeStore = None
if KnowledgeStore is not None:
    try:
        _lc_recs = KnowledgeStore().list_strategy_lifecycles(limit=200)
    except Exception:  # noqa: BLE001 库未初始化/不可读则降级
        _lc_recs = []
else:
    _lc_recs = []
if not _lc_recs:
    note("暂无已落库的策略生命周期记录（注册/回测/模拟盘后自动入库，本地库 `db/knowledge.db`）。", "info")
else:
    _lc_rows = []
    for _rec in _lc_recs:
        _st = _rec.get("status") or ""
        _st_badge = badge(
            _st or "—",
            "success" if _st in ("verified", "paper", "backtested", "approved", "live")
            else ("danger" if _st == "rejected" else "warning"),
        )
        _sharpe = _rec.get("sharpe")
        _mdd = _rec.get("max_drawdown")
        _lc_rows.append({
            "策略ID": _rec.get("strategy_id") or "—",
            "阶段": badge(_rec.get("state") or "—", "info"),
            "判读": _st_badge,
            "来源run_id": _rec.get("run_id") or "—",
            "想法": _rec.get("idea") or "—",
            "Sharpe": (f"{_sharpe:.3f}" if _sharpe is not None else "—"),
            "MaxDD": (f"{_mdd:.3f}" if _mdd is not None else "—"),
            "判读理由": _rec.get("reason") or "—",
        })
    st.dataframe(pd.DataFrame(_lc_rows), width="stretch", hide_index=True)
    st.caption("💡 晋升前可先查看 AI 判读历史，避免对已 rejected 的策略重复晋升。")

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
