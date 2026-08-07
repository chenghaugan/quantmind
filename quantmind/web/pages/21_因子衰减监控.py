"""因子衰减监控：对已沉淀因子定期扫描 IC / Sharpe 的衰减，状态机 ACTIVE→MONITORING→DECAYED→DISABLED。

借鉴 Vibe-Trading 的 strategy-dev-manager 开发治理思路：防止历史表现优异的因子
随市场结构性变化而过拟合失效，指标衰减时先在 MONITORING 观察，持续失效则降级。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st  # noqa: E402

from utils.theme import (  # noqa: E402
    setup_page, page_header, section, note, verdict, guard_error,
    kpi_row, fmt_num, fmt_pct, badge,
)
from utils.api_client import APIClient  # noqa: E402

setup_page("因子衰减监控", "🩻")
page_header(
    "因子衰减监控",
    "对已沉淀因子扫描 IC / Sharpe 指标的衰减程度，驱动 ACTIVE → MONITORING → "
    "DECAYED → DISABLED 状态机，防止过拟合因子进入线上（对标 Vibe-Trading 开发治理）。",
    "🩻",
)

note(
    "**原理**：每个因子维护其历史 IC / Sharpe 时序，扫描时比对「近期窗口」与「历史窗口」的均值，"
    "算出衰减比（recent/history）。比值显著低于阈值即判定因子开始衰减，进入 MONITORING 观察；"
    "持续衰减则 DECAYED，最终 DISABLED 停用。",
    "info",
)

# ----------------------------------------------------------------- 数据加载
col_l, col_r = st.columns([2, 1], gap="medium")
with col_l:
    note("点击「全量扫描」对知识库中所有 active 因子跑一次衰减检测；仅查看则用「刷新状态」。", "info")
with col_r:
    st.write("")
    b1, b2 = st.columns(2, gap="small")
    with b1:
        refresh_clicked = st.button("🔄 刷新状态", width="stretch")
    with b2:
        scan_clicked = st.button("🔎 全量扫描", type="primary", width="stretch")

if scan_clicked:
    with st.spinner("正在扫描全部因子衰减…"):
        data = APIClient.factor_decay_scan(timeout=120)
    if guard_error(data, "衰减扫描"):
        st.stop()
    scanned = data.get("scanned", 0)
    verdict(f"扫描完成：共检测 {scanned} 个因子。", "ok", icon="✅")
else:
    refresh_clicked = True  # 首次进入自动拉取现状
    data = APIClient.factor_decay(timeout=10)
    if guard_error(data, "衰减状态"):
        st.stop()

factors = data.get("factors") or []

# ----------------------------------------------------------------- 概览
section("状态概览")
if not factors:
    note("暂无因子衰减记录，点击「全量扫描」开始监控。", "info")
else:
    from collections import Counter
    counts = Counter(f.get("state", "ACTIVE") for f in factors)
    STATE_META = {
        "ACTIVE": ("active", "success"),
        "MONITORING": ("monitoring", "warning"),
        "DECAYED": ("decayed", "danger"),
        "DISABLED": ("disabled", "muted"),
    }
    kpi_row([
        {"label": "ACTIVE 存活", "value": counts.get("ACTIVE", 0),
         "tone": "up", "hint": "运行正常"},
        {"label": "MONITORING 观察", "value": counts.get("MONITORING", 0),
         "tone": "accent", "hint": "开始衰减"},
        {"label": "DECAYED 失效", "value": counts.get("DECAYED", 0),
         "tone": "down", "hint": "显著衰减"},
        {"label": "DISABLED 停用", "value": counts.get("DISABLED", 0),
         "tone": "neutral", "hint": "下线"},
    ])

# ----------------------------------------------------------------- 明细
    section("因子明细")
    rows = []
    for f in factors:
        state = f.get("state", "ACTIVE")
        meta = STATE_META.get(state, ("", "info"))
        note_text = "；".join(f.get("notes") or [])
        rows.append({
            "因子ID": f.get("factor_id", ""),
            "状态": f"{meta[0]} · {state}",
            "近期IC均值": fmt_num(f.get("ic_mean_recent"), 4),
            "历史IC均值": fmt_num(f.get("ic_mean_history"), 4),
            "IC衰减比": fmt_num(f.get("ic_decay_ratio"), 3),
            "近期Sharpe": fmt_num(f.get("sharpe_recent"), 2),
            "历史Sharpe": fmt_num(f.get("sharpe_history"), 2),
            "Sharpe衰减比": fmt_num(f.get("sharpe_decay_ratio"), 3),
            "最后扫描": str(f.get("last_scan_at", ""))[:19],
            "备注": note_text,
        })
    st.dataframe(rows, width="stretch", hide_index=True)

    section("状态机说明")
    st.markdown(
        f"{badge('ACTIVE', 'success')} → {badge('MONITORING', 'warning')} → "
        f"{badge('DECAYED', 'danger')} → {badge('DISABLED', 'muted')}　"
        "—— 指标衰减比低于阈值即进入下一档，最终停用过拟合因子。",
        unsafe_allow_html=True,
    )

st.caption("下一步：衰减中的因子可回到「因子搜索 / 因子挖掘」迭代出新版本，替换失效因子。")
