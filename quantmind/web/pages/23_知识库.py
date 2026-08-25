"""知识库浏览页：概览 → 检索 → 分类浏览三层结构。

重构目标：
1. 顶部概览：KPI 卡片显示各类条数
2. 语义检索置顶：实时搜索，卡片式结果
3. 分类浏览：按来源/状态/类别分组，卡片式展示
4. 自动加载：去掉"加载"按钮，页面打开自动加载
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


@st.cache_data(ttl=300)
def _load_all_counts():
    """加载各类条目数量（缓存 5 分钟）。"""
    try:
        from quantmind.knowledge import KnowledgeStore
        ks = KnowledgeStore()
        counts = {}
        for kind in ("methodology", "factor", "strategy", "research_log"):
            try:
                counts[kind] = len(ks.list_items(kind=kind, limit=500))
            except Exception:
                counts[kind] = 0
        try:
            counts["lifecycle"] = len(ks.list_strategy_lifecycles())
        except Exception:
            counts["lifecycle"] = 0
        return counts
    except Exception:
        return {"methodology": 0, "factor": 0, "strategy": 0, "research_log": 0, "lifecycle": 0}


setup_page("知识库", "🗂️")
page_header(
    "知识库",
    "交易理论与研究沉淀的结构化存储与检索",
    "🗂️",
)

# ============================================================================
# 1. 顶部概览：KPI 卡片显示各类条数
# ============================================================================
counts = _load_all_counts()
kpi_row([
    {"label": "📖 方法论", "value": counts.get("methodology", 0), "tone": "accent"},
    {"label": "🔬 因子", "value": counts.get("factor", 0), "tone": "accent"},
    {"label": "🧠 策略", "value": counts.get("strategy", 0), "tone": "accent"},
    {"label": "📝 研究日志", "value": counts.get("research_log", 0), "tone": "accent"},
    {"label": "🔄 生命周期", "value": counts.get("lifecycle", 0), "tone": "accent"},
])

st.divider()

# ============================================================================
# 2. 语义检索置顶：实时搜索
# ============================================================================
section("🔎 语义检索", "输入关键词，实时搜索知识库")

col_s, col_k = st.columns([3, 1])
with col_s:
    query = st.text_input(
        "搜索",
        placeholder="缠论 三买、趋势跟踪、动量因子...",
        help="支持多关键词，用空格分隔",
    )
with col_k:
    kind_filter = st.selectbox(
        "类型过滤",
        ["全部", "methodology", "factor", "strategy", "research_log"],
        format_func=lambda x: "全部" if x == "全部" else {
            "methodology": "方法论",
            "factor": "因子",
            "strategy": "策略",
            "research_log": "研究日志",
        }.get(x, x),
    )

# 实时搜索：输入即搜（用 session_state 缓存结果）
if query.strip():
    search_key = f"search_{query}_{kind_filter}"
    if search_key not in st.session_state:
        with st.spinner("检索中..."):
            sres = APIClient.knowledge_search(
                query.strip(),
                top_k=20,
                kind=None if kind_filter == "全部" else kind_filter,
            )
        if guard_error(sres, "知识库检索"):
            st.session_state[search_key] = []
        else:
            results = sres.get("results") or sres.get("hits") or []
            st.session_state[search_key] = results
    
    results = st.session_state.get(search_key, [])
    if results:
        st.caption(f"找到 {len(results)} 条结果")
        for r in results:
            score = float(r.get("score") or 0)
            kind = r.get("kind") or r.get("kb_type") or ""
            title = r.get("title") or r.get("concept") or r.get("text", "")[:80]
            text = r.get("text") or r.get("summary") or ""
            
            # 卡片式展示
            with st.container():
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.markdown(f"**{title}**")
                    st.caption(text[:200] + "..." if len(text) > 200 else text)
                with col2:
                    st.markdown(f"<div style='text-align:right'>{badge(kind, 'info')}<br>得分: {score:.2f}</div>", unsafe_allow_html=True)
                st.divider()
    else:
        note("无检索结果，换一组关键词试试。", "info")
else:
    note("输入关键词开始搜索，支持多关键词用空格分隔。", "info")

st.divider()

@st.cache_data(ttl=300)
def _load_items(kind: str, limit: int = 200):
    """加载指定类型的条目（缓存 5 分钟）。"""
    try:
        resp = APIClient.knowledge_list(kind=kind, limit=limit)
        if isinstance(resp, dict):
            return resp.get("items") or []
        elif isinstance(resp, list):
            return resp
        return []
    except Exception:
        return []


# ============================================================================
# 3. 分类浏览：双栏布局（列表 + 详情）
# ============================================================================
section("📚 分类浏览", "点击左侧条目查看详情")

# Tab 切换类型
type_tabs = st.tabs(["📖 方法论", "🔬 因子", "🧠 策略", "📝 研究日志"])

for tab, kind in zip(type_tabs, ["methodology", "factor", "strategy", "research_log"]):
    with tab:
        # 加载数据
        items = _load_items(kind)
        if not items:
            note(f"暂无{kind}条目。", "info")
            continue
        
        # 提取元数据用于筛选
        sources = set()
        tags = set()
        for it in items:
            meta = it.get("metadata") or {}
            src = meta.get("source") or it.get("source") or "未知"
            sources.add(src)
            for tag in (meta.get("tags") or it.get("tags") or []):
                tags.add(tag)
        
        # 顶部筛选区
        filter_cols = st.columns([2, 1, 1])
        with filter_cols[0]:
            search_key = f"search_{kind}"
            search_text = st.text_input(
                "🔎 搜索",
                placeholder="输入关键词过滤...",
                key=search_key,
            )
        with filter_cols[1]:
            source_filter = st.selectbox(
                "来源筛选",
                ["全部"] + sorted(sources),
                key=f"source_filter_{kind}",
            )
        with filter_cols[2]:
            tag_filter = st.selectbox(
                "标签筛选",
                ["全部"] + sorted(tags),
                key=f"tag_filter_{kind}",
            )
        
        # 过滤条目
        filtered_items = []
        for it in items:
            meta = it.get("metadata") or {}
            title = meta.get("title") or meta.get("concept") or it.get("text", "")[:80]
            content = meta.get("content") or meta.get("summary") or it.get("text") or ""
            source = meta.get("source") or it.get("source") or "未知"
            item_tags = meta.get("tags") or it.get("tags") or []
            
            # 应用筛选
            if search_text and search_text.lower() not in title.lower() and search_text.lower() not in content.lower():
                continue
            if source_filter != "全部" and source != source_filter:
                continue
            if tag_filter != "全部" and tag_filter not in item_tags:
                continue
            
            filtered_items.append({
                "title": title,
                "content": content,
                "source": source,
                "tags": item_tags,
                "data": it,
            })
        
        st.caption(f"共 {len(filtered_items)} 条（总计 {len(items)} 条）")
        
        if not filtered_items:
            note("无匹配条目，请调整筛选条件。", "info")
            continue
        
        # 双栏布局
        list_col, detail_col = st.columns([1, 2])
        
        with list_col:
            # 条目列表
            selected_idx = None
            for idx, item in enumerate(filtered_items):
                # 构建列表项显示
                title_short = item["title"][:40] + "..." if len(item["title"]) > 40 else item["title"]
                tags_str = ", ".join(item["tags"][:2]) if item["tags"] else ""
                
                # 使用按钮模拟选择
                btn_key = f"select_{kind}_{idx}"
                if st.button(
                    f"▸ {title_short}\n  来源: {item['source']} | {tags_str}",
                    key=btn_key,
                    use_container_width=True,
                ):
                    st.session_state[f"selected_{kind}"] = idx
            
        with detail_col:
            # 详情面板
            selected_idx = st.session_state.get(f"selected_{kind}", 0)
            if selected_idx >= len(filtered_items):
                selected_idx = 0
            
            selected_item = filtered_items[selected_idx]
            
            # 标题
            st.markdown(f"### {selected_item['title']}")
            
            # 元数据
            meta_info = []
            meta_info.append(f"**来源**: {selected_item['source']}")
            if selected_item["tags"]:
                tags_html = " ".join([f"`{t}`" for t in selected_item["tags"]])
                meta_info.append(f"**标签**: {tags_html}")
            st.markdown(" | ".join(meta_info))
            
            st.divider()
            
            # 内容
            if selected_item["content"]:
                st.markdown("**内容**:")
                st.markdown(selected_item["content"])
            else:
                note("无详细内容。", "info")

