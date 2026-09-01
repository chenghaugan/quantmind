"""行情仓库总览页：本地 Parquet 写缓存（DiskBarCache）的**聚合**覆盖概览 + 按需下钻。

几千只标的下不再逐标的铺开渲染：默认只展示 总览 KPI + 聚合桶（交易所/周期/新鲜度 +
Top-N），逐标的明细走「按需下钻」（搜索/筛选/分页），从后端轻接口取，避免前端一次
渲染几万行。

数据集市：真实行情落盘后秒级复用。支持手动预热（从真实源拉取落盘）与清空重建。
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
    "真实行情落盘后秒级复用，不再每次联网拉取。此处按 市场/交易所/周期 聚合总览覆盖纵深，"
    "逐标的明细按需下钻（搜索 / 筛选 / 分页），支持手动预热与清空重建。",
    "🗄️",
)

note(
    "**取数链路**：请求 → ① 本地行情仓库（秒级） → ② 持久库 → ③ 真实源（akshare 等）→ 回写。<br>"
    "只有**真实源**数据会落盘入库；mock 合成数据不入库（避免污染）。"
    "盘中新数据可点「预热」从真实源自动追新。",
    "info",
)

EXCH_MAP = {e: EXCHANGE_NAMES.get(e, e) for e in ALL_EXCHANGES}


# ------------------------------------------------------------- 加载聚合总览
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

_agg = stats.get("agg") or {}
_hist = st.session_state.qm_cache_history or []

# ---- 顶部 KPI（来自聚合，不再逐标的）----
_files = stats.get("files", 0)
_rows = stats.get("rows", 0)
_fr = _agg.get("freshness") or {}
_n_stale = _fr.get("stale_1_3d", 0) + _fr.get("stale_gt3d", 0)
_stale_top = _agg.get("stale_top") or []
_worst = max((s.get("staleness_days") or 0) for s in _stale_top) if _stale_top else 0
_last_refresh = None
if _hist:
    _last_refresh = _hist[0].get("ts")

kpi_row([
    {"label": "标的数", "value": fmt_num(_files, 0), "tone": "accent",
     "hint": "文件数（标的×周期）"},
    {"label": "K 线总数", "value": fmt_num(_rows, 0), "tone": "neutral"},
    {"label": "最新交易日", "value": (stats.get("last_datetime") or "—")[:10],
     "tone": "neutral"},
    {"label": "落后标的", "value": f"{_n_stale} / {_files} (最差 {_worst}d)",
     "tone": "danger" if _n_stale else "success"},
])

st.caption("仓库路径: " + str(stats.get("root", ""))
           + ("  ·  上次刷新: " + (_last_refresh or "—")[:19] if _last_refresh else ""))

# ------------------------------------------------------------- 聚合汇总区
section("按市场 / 交易所聚合", "各市场/交易所的标的数与 K 线数（默认总览，不是逐标的）")
_by_exch = _agg.get("by_exchange") or []
if _by_exch:
    market_rows = []
    _mkt_agg = {}
    for b in _by_exch:
        m = b.get("market") or "其他"
        mm = _mkt_agg.setdefault(m, {"市场": m, "标的数": 0, "K线数": 0})
        mm["标的数"] += b.get("symbols", 0)
        mm["K线数"] += b.get("rows", 0)
        market_rows.append({
            "交易所": f"{b.get('exchange')} ({m})",
            "标的数": fmt_num(b.get("symbols", 0), 0),
            "K线数": fmt_num(b.get("rows", 0), 0),
        })
    c_mkt, c_exch = st.columns([1, 2])
    with c_mkt:
        st.markdown("**市场**")
        st.dataframe(pd.DataFrame(list(_mkt_agg.values())),
                     width="stretch", hide_index=True)
    with c_exch:
        st.markdown("**交易所**")
        st.dataframe(pd.DataFrame(market_rows), width="stretch", hide_index=True,
                     height=min(50 + 30 * len(market_rows), 360))
else:
    st.info("仓库暂无语义聚合（可能为空或需刷新）。")

# 周期与新鲜度
c_int, c_fr = st.columns([1, 1])
with c_int:
    section("按周期聚合")
    _by_int = _agg.get("by_interval") or []
    if _by_int:
        st.dataframe(pd.DataFrame([{
            "周期": b.get("interval"),
            "标的数": b.get("symbols", 0),
            "K线数": fmt_num(b.get("rows", 0), 0),
        } for b in _by_int]), width="stretch", hide_index=True)
with c_fr:
    section("新鲜度")
    st.dataframe(pd.DataFrame([{
        "桶": "最新", "数量": _fr.get("fresh", 0)},
        {"桶": "落后 1-3 天", "数量": _fr.get("stale_1_3d", 0)},
        {"桶": "落后 >3 天", "数量": _fr.get("stale_gt3d", 0)},
    ]), width="stretch", hide_index=True)

# ------------------------------------------------------------- 覆盖纵深（聚合版，按交易所）
if _by_exch:
    section("覆盖区间（按交易所聚合，不再每标的一根）")
    try:
        _cov = pd.DataFrame([{
            "交易所": f"{b.get('exchange')} ({b.get('market','')})",
            "起点": pd.to_datetime(b.get("coverage_start")) if b.get("coverage_start") else None,
            "终点": pd.to_datetime(b.get("coverage_end")) if b.get("coverage_end") else None,
            "K线数": b.get("rows", 0),
            "标的数": b.get("symbols", 0),
        } for b in _by_exch])
        fig = px.timeline(_cov, x_start="起点", x_end="终点", y="交易所",
                          color="K线数", color_continuous_scale="Blues",
                          hover_data={"标的数": True})
        fig.update_layout(height=240, margin=dict(t=24, b=24),
                          xaxis_title="", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.caption(f"共 {len(_by_exch)} 个交易所聚合区间（行情数据的实际覆盖纵深）。")
    except Exception as exc:  # noqa: BLE001
        st.info(f"覆盖区间可视化暂不可用：{exc}")

# ------------------------------------------------------------- Top-N（只看大头，不铺全量）
_top = _agg.get("top_rows") or []
if _top:
    section("数据量 Top 50", "只看体积最大的标的，避免逐标的铺开")
    st.dataframe(pd.DataFrame([{
        "标的": f"{s.get('symbol')}.{s.get('exchange')}",
        "周期": s.get("interval"),
        "K线数": fmt_num(s.get("rows", 0), 0),
        "最新": (s.get("last") or "—")[:10],
    } for s in _top]), width="stretch", hide_index=True, height=360)

# ------------------------------------------------------------- 按需下钻（默认折叠）
with st.expander("🔎 逐标的明细下钻（搜索 / 筛选 / 分页）", expanded=False):
    if "qm_sym_page" not in st.session_state:
        st.session_state.qm_sym_page = 1
    _dr1, _dr2, _dr3, _dr4 = st.columns(4)
    with _dr1:
        _mkt = st.selectbox("市场", ["全部"] + list(_agg.get("markets") or []),
                            key="dr_market")
    with _dr2:
        _exch = st.selectbox("交易所", ["全部"] + list(_agg.get("exchanges") or []),
                             key="dr_exch", format_func=lambda e: EXCH_MAP.get(e, e))
    with _dr3:
        _intv = st.selectbox("周期", ["全部"] + list(_agg.get("intervals") or []),
                             key="dr_interval")
    with _dr4:
        _frz = st.selectbox("新鲜度", ["全部", "最新", "落后"], key="dr_fresh")
    _q = st.text_input("标的搜索（如 600519 / rb0）", key="dr_q")

    # 筛选条件变化时重置页码，否则旧页码超过新结果页数会显示空页
    _filter_sig = (_exch, _mkt, _intv, _frz, _q.strip())
    if st.session_state.get("_dr_filter_sig") != _filter_sig:
        st.session_state["_dr_filter_sig"] = _filter_sig
        st.session_state.qm_sym_page = 1
    _page = st.session_state.qm_sym_page
    _sym_res = APIClient.cache_symbols(
        exchange="" if _exch == "全部" else _exch,
        market="" if _mkt == "全部" else _mkt,
        interval="" if _intv == "全部" else _intv,
        freshness={"最新": "fresh", "落后": "stale"}.get(_frz, ""),
        q=_q.strip(), page=_page, page_size=50,
    )
    if isinstance(_sym_res, dict) and "symbols" in _sym_res:
        _total = _sym_res.get("total", 0)
        st.caption(f"命中 {_total} 条（第 {_page} 页 / 每页 50）")
        _rows = pd.DataFrame([{
            "标的": f"{s.get('symbol')}.{s.get('exchange')}",
            "周期": s.get("interval"),
            "K线数": fmt_num(s.get("rows", 0), 0),
            "起点": (s.get("start") or "—")[:10],
            "终点": (s.get("last") or "—")[:10],
            "最新": (s.get("last") or "—")[:10],
            "新鲜度": ("✅ 最新" if s.get("up_to_date") is True else
                       (f"⚠️ 落后 {s.get('staleness_days')}d"
                        if s.get("up_to_date") is False else "—")),
        } for s in _sym_res["symbols"]])
        st.dataframe(_rows, width="stretch", hide_index=True, height=400)
        _n_pages = max(1, -(-_total // 50))
        _pg1, _pg2, _pg3 = st.columns([1, 2, 1])
        with _pg1:
            if st.button("⬅️ 上一页", disabled=(_page <= 1)):
                st.session_state.qm_sym_page = max(1, _page - 1)
                st.rerun()
        with _pg3:
            if st.button("下一页 ➡️", disabled=(_page >= _n_pages)):
                st.session_state.qm_sym_page = _page + 1
                st.rerun()
        with _pg2:
            st.markdown(f"<div style='text-align:center'>共 {_n_pages} 页</div>",
                        unsafe_allow_html=True)
    else:
        st.info("暂无匹配标的。")

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
else:
    st.info("暂无刷新记录。触发一次「全量刷新」或「预热」后这里会展示历史。")

# ------------------------------------------------------------- 运维操作
section("运维操作", "全量刷新（追新）/ 预热 / 清空重建")
c1, c2 = st.columns([2, 1], gap="medium")
with c1:
    _warm_txt = st.text_input(
        "预热标的（逗号分隔，如 rb0,hc0,bu0,i0）",
        value="IF0,IH0,IC0,IM0,rb0,hc0,bu0,i0", help="从真实源（akshare 等）拉取并落盘；已有缓存也会增量追新",
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
