"""行情仓库总览页：本地 Parquet 写缓存（DiskBarCache）的逐标的覆盖区间可视化+运维。

数据集市：真实行情落盘后秒级复用；本页展示缓存里每个标的的行数、起止区间、最新交易日，
并支持手动预热（从真实源拉取落盘）与清空重建。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from utils.theme import (  # noqa: E402
    setup_page, page_header, section, note, guard_error,
    kpi_row, fmt_num, badge,
)
from utils.api_client import APIClient  # noqa: E402
from utils.constants import ALL_EXCHANGES, EXCHANGE_NAMES  # noqa: E402

setup_page("行情仓库总览", "🗄️")
page_header(
    "本地行情仓库（Parquet 写缓存）",
    "真实行情落盘后秒级复用，不再每次联网拉取。此处查看每个标的的覆盖纵深、最新交易日，"
    "并支持手动预热与清空重建。",
    "🗄️",
)

note(
    "**取数链路**：请求 → ① 本地行情仓库（秒级） → ② 持久库 → ③ 真实源（akshare 等）→ 回写。<br>"
    "只有**真实源**数据会落盘入库；mock 合成数据不入库（避免污染）。"
    "盘中新数据可点「预热」从真实源自动追新。",
    "info",
)

EXCH_MAP = {e: EXCHANGE_NAMES.get(e, e) for e in ALL_EXCHANGES}


# ------------------------------------------------------------- 加载仓库状态
def _load_stats():
    return APIClient.cache_stats(timeout=30)


if "qm_cache_stats" not in st.session_state:
    st.session_state.qm_cache_stats = _load_stats()
stats = st.session_state.qm_cache_stats

if guard_error(stats, "行情仓库"):
    st.stop()

if not stats.get("enabled"):
    st.warning("本地行情仓库未启用（未挂载 DiskBarCache）。")
    st.stop()

if "qm_cache_history" not in st.session_state:
    _h_res = APIClient.cache_history(timeout=15)
    st.session_state.qm_cache_history = (_h_res.get("history") if isinstance(_h_res, dict)
                                         else None) or []
_symbols = stats.get("symbols") or []
_hist = st.session_state.qm_cache_history or []

# ---- 新鲜度统计 ----
_up = [s for s in _symbols if s.get("up_to_date")]
_stale = [s for s in _symbols if s.get("up_to_date") is False]
_stale_days = [s.get("staleness_days") for s in _symbols if s.get("staleness_days") is not None]
_worst = max(_stale_days) if _stale_days else 0
_last_refresh = None
if _hist:
    _last_refresh = _hist[0].get("ts")

# ---- 顶部 KPI ----
kpi_row([
    {"label": "标的数", "value": fmt_num(len(_symbols), 0), "tone": "accent"},
    {"label": "K 线总数", "value": fmt_num(stats.get("rows", 0), 0), "tone": "neutral"},
    {"label": "最新交易日", "value": (stats.get("last_datetime") or "—")[:10],
     "tone": "neutral"},
    {"label": "落后标的", "value": f"{len(_stale)} / {len(_symbols)} (最差 {_worst}d)",
     "tone": "danger" if _stale else "success"},
])

st.caption("仓库路径: " + str(stats.get("root", ""))
           + ("  ·  上次刷新: " + (_last_refresh or "—")[:19] if _last_refresh else ""))

# ------------------------------------------------------------- 逐标的覆盖明细表格
section("逐标的覆盖明细", "行数 / 覆盖区间 / 最新交易日 / 新鲜度")
if _symbols:
    rows = []
    for s in _symbols:
        _sd = s.get("staleness_days")
        _f = s.get("up_to_date")
        _freshtxt = "✅ 最新" if _f else (f"⚠️ 落后 {_sd}d" if _sd is not None else "—")
        rows.append({
            "标的": f"{s.get('symbol')}.{s.get('exchange')}",
            "周期": s.get("interval"),
            "K线数": fmt_num(s.get("rows", 0), 0),
            "起点": (s.get("start") or "—")[:10],
            "终点": (s.get("end") or "—")[:10],
            "最新": (s.get("last") or "—")[:10],
            "新鲜度": _freshtxt,
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True, height=min(50 + 35 * len(df), 460))
else:
    st.info("仓库为空：可点下方「预热」从真实源拉取标的数据。")

# ------------------------------------------------------------- 覆盖区间条形图
if _symbols:
    section("各标的覆盖区间（时间纵深）")
    fig = px.timeline(
        pd.DataFrame([{
            "标的": f"{s.get('symbol')}.{s.get('exchange')}",
            "起点": pd.to_datetime(s.get("start")) if s.get("start") else None,
            "终点": pd.to_datetime(s.get("end")) if s.get("end") else None,
            "K线数": s.get("rows", 0),
        } for s in _symbols]),
        x_start="起点", x_end="终点", y="标的",
        color="K线数", color_continuous_scale="Blues",
    )
    fig.update_layout(height=280, margin=dict(t=24, b=24),
                      xaxis_title="", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ------------------------------------------------------------- 刷新历史
section("刷新历史", "自动(交易日17:00) + 手动触发")
if _hist:
    _hrows = []
    for h in _hist[:20]:
        _tag = badge("成功", "success") if h.get("status") == "ok" else (
            badge("空", "warning") if h.get("status") == "empty" else badge("失败", "danger"))
        _hrows.append({
            "时间": (h.get("ts") or "—")[:19],
            "标的": f"{h.get('symbol')}.{h.get('exchange')}",
            "状态": h.get("status", "—"),
            "写入": fmt_num(h.get("rows", 0), 0),
            "最新K线": (h.get("latest") or "—")[:10],
        })
    st.dataframe(pd.DataFrame(_hrows), width="stretch", hide_index=True,
                 height=min(50 + 30 * len(_hrows), 360))
    _sched_note = "距下次自动刷新: 每个交易日 17:00（cron `0 17 * * 1-5`）"
    st.caption(_sched_note)
else:
    st.info("暂无刷新记录。触发一次「全量刷新」或「预热」后这里会展示历史。")

# ------------------------------------------------------------- 运维操作
section("运维操作", "全量刷新（追新）/ 预热 / 清空重建")
c1, c2 = st.columns([2, 1], gap="medium")
with c1:
    _warm_txt = st.text_input(
        "预热标的（逗号分隔，如 rb0,hc0,bu0,i0）",
        value="rb0,hc0,bu0,i0", help="从真实源（akshare 等）拉取并落盘；已有缓存也会增量追新",
    )
    _exch = st.selectbox("交易所", list(EXCH_MAP.keys()),
                         format_func=lambda e: EXCH_MAP[e], index=0)
with c2:
    st.write("")
    st.write("")
    _refresh_btn = st.button("🔄 全量刷新（追新）", type="primary", width="stretch")
    _warm_btn = st.button("🔥 预热", width="stretch")
    _purge_btn = st.button("🗑️ 清空重建", width="stretch")

_refresh_triggered = False
if _refresh_btn:
    with st.spinner("正在对仓库全部标的做增量追新（真实源→落盘）..."):
        res = APIClient.cache_refresh(timeout=600)
    _refresh_triggered = True

if _warm_btn:
    syms = [s.strip() for s in _warm_txt.split(",") if s and s.strip()]
    if not syms:
        st.error("请输入至少一个标的。")
    else:
        with st.spinner(f"正在从真实源拉取并落盘: {', '.join(syms)}（首次较慢）..."):
            res = APIClient.cache_warm(syms, exchange=_exch, timeout=600)
        _refresh_triggered = True

if _purge_btn:
    with st.spinner("正在清空本地行情仓库..."):
        res = APIClient.cache_purge(timeout=30)
    if guard_error(res, "清空"):
        st.stop()
    st.success(f"已清空 {res.get('removed', 0)} 个文件。下次请求将自动从真实源重建。")
    st.session_state.qm_cache_stats = _load_stats()
    st.session_state.qm_cache_history = APIClient.cache_history(timeout=15).get("history") or []
    st.rerun()

if _refresh_triggered:
    if guard_error(res, "刷新"):
        st.stop()
    st.success(f"刷新完成: 成功 {res.get('refreshed', 0)} / 失败 {res.get('failed', 0)}")
    for r in res.get("results") or []:
        _k = r.get("key") or {}
        _tag = badge(f"{r.get('n', 0)} 根 · 源 {r.get('source', '?')}", "success") \
            if r.get("status") == "ok" else (
                badge(f"失败: {r.get('error', '?')[:40]}", "danger")
                if r.get("status") == "error" else badge("空数据", "warning"))
        st.markdown(f"- **{_k.get('symbol', '?')}** {_tag}", unsafe_allow_html=True)
    st.session_state.qm_cache_stats = _load_stats()
    st.session_state.qm_cache_history = APIClient.cache_history(timeout=15).get("history") or []
    st.rerun()
