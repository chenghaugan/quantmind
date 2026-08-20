"""知识库浏览页：按 kind 分类浏览方法论 / 因子 / 策略 / 研究日志 + 语义检索。

复用后端 GET /knowledge 与 POST /knowledge/search（kind 支持 methodology）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from utils.theme import (  # noqa: E402
    setup_page, page_header, section, note, kpi_row, badge, guard_error,
)
from utils.api_client import APIClient  # noqa: E402


def _kb_conn_note():
    """页面顶部显示当前知识库连接状态：真实 DB 路径 + 各类条数。

    直读本地 KnowledgeStore（库文件路径、各类目条数）。任何异常都降级为温和
    提示，绝不抛异常阻断页面；某类目读取慢/异常时仅该类显示 '—'。
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


setup_page("知识库", "🗂️")
page_header(
    "知识库",
    "按类型浏览交易理论与研究沉淀，并支持关键词语义检索。",
    "🗂️",
)

_kb_conn_note()

note(
    "**知识库浏览**：方法论（缠论3买、威科夫、海龟等）、因子、策略与研究日志分层归档；"
    "方法论条目可展开查看完整正文。没有后端时页面会自然降级，不影响加载。",
    "info",
)

#: kind → 展示名 / 图标
KINDS = [
    ("methodology", "方法论", "📖"),
    ("factor", "因子", "🔬"),
    ("strategy", "策略", "🧠"),
    ("research_log", "研究日志", "📝"),
]


def _extract_items(resp: dict):
    """统一抽取 knowledge_list 返回值中的条目列表。"""
    if isinstance(resp, dict):
        return resp.get("items") or []
    if isinstance(resp, list):
        return resp
    return []


def _render_common(item: dict) -> dict:
    """非方法论条目：折叠为通用列表行（kind + text 截断）。"""
    return {
        "ID": item.get("kb_id", ""),
        "类型": item.get("kind", ""),
        "内容": (item.get("text") or item.get("summary") or "")[:120],
    }


def _load_lifecycle_records():
    """尽力读取本地已落库的策略生命周期记录；后端/本地库不可用时降级为空列表。

    优先用 APIClient 返回的条目本身携带的 lifecycle 字段；否则直接读
    ``quantmind/db/knowledge.db`` 的 lifecycle 表（``list_strategy_lifecycles``）。
    两者都不可用时返回空列表，页面自然降级，不影响加载。
    """
    try:
        from quantmind.knowledge import KnowledgeStore  # 本地包，随页加载
    except Exception:  # noqa: BLE001 包不可用则降级
        return []
    try:
        return KnowledgeStore().list_strategy_lifecycles(limit=200)
    except Exception:  # noqa: BLE001 数据库不可读/未初始化则降级
        return []


def _lifecycle_lookup(records, item: dict) -> dict:
    """把知识条目与生命周期记录按 strategy_id / code 最佳匹配（无匹配返回空 dict）。"""
    if not records:
        return {}
    id_key = item.get("strategy_id") or item.get("kb_id") or ""
    code_key = ((item.get("metadata") or {}).get("code") or "") if isinstance(item, dict) else ""
    keys = {k for k in (id_key, code_key) if k}
    for rec in records:
        rec_keys = {k for k in ((rec.get("strategy_id") or ""), (rec.get("code") or "")) if k}
        if keys and rec_keys & keys:
            return rec
    return {}


def _render_strategy(items: list) -> None:
    """策略条目：通用列 + 生命周期 state/判读 status 列 + 详情 expander。

    数据来源：优先用 ``APIClient.knowledge_list(kind="strategy")`` 返回的条目；
    若条目本身不含生命周期字段，则用 ``KnowledgeStore().list_strategy_lifecycles()``
    读取本地库合并展示；两者都没有时自然降级为普通策略列表。
    """
    if not items:
        note("暂无策略条目。", "info")
        return
    records = _load_lifecycle_records()
    rows = []
    for it in items:
        rec = _lifecycle_lookup(records, it)
        state = rec.get("state") or (it.get("metadata") or {}).get("state") or ""
        status = rec.get("status") or (it.get("metadata") or {}).get("status") or ""
        state_badge = badge(state or "—", "info") if state else "—"
        status_badge = "—"
        if status:
            status_badge = badge(status, "success" if status in ("verified", "paper", "backtested", "approved", "live") else ("danger" if status in ("rejected",) else "warning"))
        rows.append({
            "ID": it.get("kb_id", ""),
            "类型": "strategy",
            "内容": (it.get("text") or it.get("summary") or "")[:120],
            "阶段": state_badge,
            "判读": status_badge,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    for it in items:
        rec = _lifecycle_lookup(records, it)
        if not rec:
            continue
        meta = it.get("metadata") or {}
        title = it.get("kb_id") or rec.get("strategy_id") or "策略"
        with st.expander(f"🧠 {title}"):
            st.markdown(
                badge(rec.get("state") or "—", "info")
                + " "
                + badge(
                    rec.get("status") or "—",
                    "success" if rec.get("status") in ("verified", "paper", "backtested", "approved", "live") else ("danger" if rec.get("status") == "rejected" else "warning"),
                ),
                unsafe_allow_html=True,
            )
            if rec.get("run_id"):
                st.markdown(f"**来源 run_id**：`{rec['run_id']}`")
            if rec.get("idea") or meta.get("idea"):
                st.markdown(f"**想法 idea**：{rec.get('idea') or meta.get('idea')}")
            if rec.get("reason"):
                st.markdown(f"**判读理由 reason**：{rec['reason']}")
            if rec.get("brief"):
                st.markdown("**策略经验 brief**")
                st.write(rec["brief"])
            metrics = []
            if rec.get("sharpe") is not None:
                metrics.append(f"Sharpe = {rec['sharpe']:.3f}")
            if rec.get("max_drawdown") is not None:
                metrics.append(f"MaxDD = {rec['max_drawdown']:.3f}")
            if metrics:
                st.markdown("**真实指标**：" + "，".join(metrics))


def _render_methodology(items: list) -> None:
    """方法论：标题/概念/摘要/来源/标签 + 全文 expander。"""
    if not items:
        note("暂无方法论条目。", "info")
        return
    rows = []
    for it in items:
        title = it.get("title") or it.get("concept") or it.get("text") or it.get("kb_id", "")
        rows.append({
            "标题": title,
            "概念": (it.get("concept") or "")[:60],
            "来源": it.get("source", ""),
            "标签": "、".join(it.get("tags") or []) if isinstance(it.get("tags"), list)
                    else str(it.get("tags") or ""),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    for it in items:
        title = it.get("title") or it.get("concept") or it.get("kb_id", "")
        with st.expander(f"📖 {title}"):
            st.markdown(f"**概念**：{it.get('concept') or '—'}")
            if it.get("summary"):
                st.markdown(f"**摘要**：{it['summary']}")
            if it.get("source"):
                st.markdown(f"**来源**：{it['source']}")
            tags = it.get("tags")
            if tags:
                tag_html = " ".join(badge(str(t), "violet") for t in (tags if isinstance(tags, list) else [tags]))
                st.markdown(tag_html, unsafe_allow_html=True)
            content = it.get("content")
            if content:
                st.markdown("---")
                st.markdown("**全文**")
                st.write(content)


# ---------------------------------------------------------------- 分类浏览
section("按类型浏览")

tabs = st.tabs([f"{icon} {label}" for _, label, icon in KINDS])
for tab, (kind, label, _icon) in zip(tabs, KINDS):
    with tab:
        if st.button(f"🔄 加载{label}", type="secondary", key=f"load_{kind}"):
            st.session_state[f"kb_{kind}"] = None
        cached = st.session_state.get(f"kb_{kind}")
        if cached is None:
            with st.spinner(f"加载 {label} …"):
                resp = APIClient.knowledge_list(kind=kind, limit=200)
            if guard_error(resp, f"加载{label}"):
                st.session_state[f"kb_{kind}"] = {"__error__": True}
            else:
                st.session_state[f"kb_{kind}"] = resp
        resp = st.session_state.get(f"kb_{kind}")
        if isinstance(resp, dict) and resp.get("__error__"):
            continue
        items = _extract_items(resp)
        total = resp.get("total", len(items)) if isinstance(resp, dict) else len(items)
        st.caption(f"共 {total} 条")
        if kind == "methodology":
            _render_methodology(items)
        elif kind == "strategy":
            _render_strategy(items)
        else:
            if items:
                st.dataframe(
                    pd.DataFrame([_render_common(it) for it in items]),
                    use_container_width=True, hide_index=True,
                )
            else:
                note(f"暂无{label}条目。", "info")


# ---------------------------------------------------------------- 语义检索
section("关键词语义检索", "POST /knowledge/search")

col_s, col_k = st.columns([3, 1])
with col_s:
    query = st.text_input("检索内容", "缠论 三买 或 趋势跟踪")
with col_k:
    kind_filter = st.selectbox("类型过滤", ["全部", "methodology", "factor", "strategy", "research_log"])

if st.button("🔎 检索", type="primary"):
    if not query.strip():
        st.warning("请输入检索关键词")
        st.stop()
    with st.spinner("检索中…"):
        sres = APIClient.knowledge_search(
            query.strip(),
            top_k=10,
            kind=None if kind_filter == "全部" else kind_filter,
        )
    if guard_error(sres, "知识库检索"):
        st.stop()
    results = sres.get("results") or sres.get("hits") or []
    if results:
        kpi_row([
            {"label": "命中条数", "value": str(len(results)), "tone": "accent"},
            {"label": "检索方式", "value": "语义检索", "tone": "neutral"},
        ])
        sdf = pd.DataFrame([{
            "类型": (r.get("kind") or r.get("kb_type") or ""),
            "得分": round(float(r.get("score") or 0), 3),
            "内容": (r.get("text") or r.get("concept") or r.get("title") or "")[:110],
        } for r in results])
        st.dataframe(sdf, use_container_width=True, hide_index=True)
        with st.expander("🔎 检索详情", expanded=False):
            st.json(sres)
    else:
        note("无检索结果，换一组关键词试试。", "info")

st.caption("💡 方法论条目与因子研究相互印证：先查「方法论」建立理论框架，再到因子挖掘页落地为可回测因子。")
