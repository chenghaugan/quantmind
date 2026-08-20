"""QuantMind 统一视觉主题与 UI 组件库。

所有页面统一调用 :func:`setup_page` 完成「页面配置 + 全局样式 + 侧边栏品牌区」，
再用 :func:`page_header` / :func:`section` / :func:`kpi_row` 等组件组织内容，
避免每个页面各写一套 markdown 样式导致风格割裂。

配色遵循中国市场习惯：**红涨绿跌**。
"""
from __future__ import annotations

import html
from typing import Any, Dict, Iterable, List, Optional, Sequence

import streamlit as st

# --------------------------------------------------------------------------
# 设计令牌（Design Tokens）
# --------------------------------------------------------------------------
COLORS = {
    "bg": "#0b1220",
    "surface": "#111b2e",
    "surface_alt": "#16233c",
    "border": "#22304d",
    "border_soft": "#1a2740",
    "text": "#e2e8f0",
    "text_muted": "#94a3b8",
    "text_dim": "#64748b",
    "primary": "#3b82f6",
    "primary_dark": "#1d4ed8",
    "violet": "#8b5cf6",
    "cyan": "#06b6d4",
    "amber": "#f59e0b",
    # 中国市场：红涨绿跌
    "up": "#f2483e",
    "down": "#12b886",
    "success": "#12b886",
    "danger": "#f2483e",
    "warning": "#f59e0b",
    "info": "#3b82f6",
}

#: Plotly 统一配色序列
PLOTLY_COLORWAY = [
    "#3b82f6", "#8b5cf6", "#06b6d4", "#f59e0b",
    "#f2483e", "#12b886", "#ec4899", "#a3e635",
]

_GLOBAL_CSS = """
<style>
/* ---------- 字体与基底 ---------- */
html, body, [class*="css"] {
    font-family: "Inter", "HarmonyOS Sans SC", "PingFang SC",
                 "Microsoft YaHei", -apple-system, sans-serif;
}
.stApp {
    background:
        radial-gradient(1200px 600px at 12% -10%, rgba(59,130,246,.13), transparent 60%),
        radial-gradient(1000px 500px at 88% 0%, rgba(139,92,246,.10), transparent 55%),
        #0b1220;
}
.block-container {
    padding-top: 2.2rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}
#MainMenu, footer {visibility: hidden;}

/* 隐藏 Streamlit 默认 pages/ 自动导航，避免与自定义分组导航重复 */
[data-testid="stSidebarNav"] { display: none; }
section[data-testid="stSidebar"] [data-testid="stSidebarNavSeparator"] { display: none; }

/* ---------- 侧边栏 ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0e1729 0%, #0b1220 100%);
    border-right: 1px solid #1a2740;
}
/* Streamlit 1.59.1 sidebar padding - 针对 stSidebarUserContent */
[data-testid="stSidebarUserContent"] { padding: .5rem 0 !important; }
/* 让品牌区上移到与折叠按钮同一水平线，减少顶部空白 */
/* pointer-events:none 让品牌区不拦截点击，保证折叠按钮可点（品牌区无交互元素） */
[data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"] > div:first-child {
    margin-top: -5.0rem;
    pointer-events: none;
}
[data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"] > div:first-child * { pointer-events: none; }
/* 品牌区上移后可能遮挡折叠按钮，提升其层级保证可点击 */
[data-testid="stBaseButton-headerNoPadding"] { z-index: 1000 !important; }
section[data-testid="stSidebar"] .block-container { padding: .5rem 0 !important; }

.qm-brand {
    display:flex; align-items:center; gap:.7rem;
    padding: .2rem 0 .9rem 0;
}
.qm-brand-logo {
    width: 40px; height: 40px; border-radius: 11px; flex: none;
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
    display:flex; align-items:center; justify-content:center;
    font-size: 20px; box-shadow: 0 6px 18px rgba(59,130,246,.35);
}
.qm-brand-name { font-size: 1.12rem; font-weight: 750; color:#f1f5f9; letter-spacing:.3px; line-height:1.15;}
.qm-brand-sub  { font-size: .70rem; color:#64748b; letter-spacing:.6px; text-transform:uppercase; }

.qm-nav-group {
    font-size:.68rem; font-weight:700; color:#475569;
    letter-spacing:1.2px; text-transform:uppercase;
    margin: .9rem 0 .25rem .15rem;
}

/* ---------- 页头 ---------- */
.qm-hero {
    position: relative; overflow: hidden;
    border: 1px solid #22304d; border-radius: 16px;
    background: linear-gradient(120deg, #16233c 0%, #111b2e 55%, #0f1a2d 100%);
    padding: 1.15rem 1.4rem; margin-bottom: 1.3rem;
}
.qm-hero::after {
    content:""; position:absolute; right:-60px; top:-70px;
    width: 240px; height: 240px; border-radius: 50%;
    background: radial-gradient(circle, rgba(59,130,246,.18), transparent 68%);
}
.qm-hero-row { display:flex; align-items:center; gap:.9rem; }
.qm-hero-icon {
    width:46px; height:46px; border-radius:13px; flex:none;
    background: rgba(59,130,246,.14); border:1px solid rgba(59,130,246,.32);
    display:flex; align-items:center; justify-content:center; font-size:24px;
}
.qm-hero-title { font-size:1.62rem; font-weight:760; color:#f8fafc; line-height:1.2; margin:0; }
.qm-hero-sub   { font-size:.87rem; color:#94a3b8; margin:.28rem 0 0 0; max-width: 900px; line-height:1.55;}

/* ---------- 分区标题 ---------- */
.qm-section { display:flex; align-items:baseline; gap:.55rem; margin: 1.5rem 0 .7rem 0; }
.qm-section-bar { width:3px; height:17px; border-radius:2px;
    background: linear-gradient(180deg,#3b82f6,#8b5cf6); transform: translateY(2px); }
.qm-section-title { font-size:1.04rem; font-weight:700; color:#e2e8f0; letter-spacing:.2px; }
.qm-section-desc  { font-size:.78rem; color:#64748b; }

/* ---------- KPI 卡片 ---------- */
.qm-kpi {
    border:1px solid #22304d; border-radius:13px; padding:.85rem .95rem;
    background: linear-gradient(160deg, #16233c 0%, #111b2e 100%);
    height: 100%; transition: border-color .18s ease, transform .18s ease;
}
.qm-kpi:hover { border-color:#3b5680; transform: translateY(-1px); }
.qm-kpi-label { font-size:.74rem; color:#94a3b8; font-weight:600; letter-spacing:.3px;
    display:flex; align-items:center; gap:.3rem; }
.qm-kpi-value { font-size:1.52rem; font-weight:740; color:#f1f5f9; line-height:1.28;
    margin-top:.18rem; font-variant-numeric: tabular-nums; word-break: break-all;}
.qm-kpi-delta { font-size:.74rem; font-weight:640; margin-top:.1rem; }
.qm-kpi-hint  { font-size:.70rem; color:#64748b; margin-top:.18rem; }
.qm-up   { color:#f2483e !important; }
.qm-down { color:#12b886 !important; }
.qm-neu  { color:#94a3b8 !important; }
.qm-accent { color:#60a5fa !important; }

/* ---------- 徽章 ---------- */
.qm-badge {
    display:inline-flex; align-items:center; gap:.3rem;
    padding:.16rem .55rem; border-radius:999px;
    font-size:.72rem; font-weight:650; letter-spacing:.2px; white-space:nowrap;
}
.qm-badge-success{ background:rgba(18,184,134,.14); color:#34d399; border:1px solid rgba(18,184,134,.35);}
.qm-badge-danger { background:rgba(242,72,62,.14);  color:#fb7185; border:1px solid rgba(242,72,62,.35);}
.qm-badge-warning{ background:rgba(245,158,11,.14); color:#fbbf24; border:1px solid rgba(245,158,11,.35);}
.qm-badge-info   { background:rgba(59,130,246,.14); color:#60a5fa; border:1px solid rgba(59,130,246,.35);}
.qm-badge-muted  { background:rgba(100,116,139,.14);color:#94a3b8; border:1px solid rgba(100,116,139,.30);}
.qm-badge-violet { background:rgba(139,92,246,.14); color:#a78bfa; border:1px solid rgba(139,92,246,.35);}

/* ---------- 信息条 ---------- */
.qm-note {
    border:1px solid #22304d; border-left:3px solid #3b82f6; border-radius:10px;
    background: rgba(30,41,59,.42); padding:.7rem .95rem;
    font-size:.82rem; color:#cbd5e1; line-height:1.62; margin-bottom:.6rem;
}
.qm-note b, .qm-note strong { color:#f1f5f9; }
.qm-note code {
    background: rgba(59,130,246,.13); color:#93c5fd;
    padding:.05rem .3rem; border-radius:4px; font-size:.79rem;
}
.qm-note-warn { border-left-color:#f59e0b; }
.qm-note-ok   { border-left-color:#12b886; }
.qm-note-err  { border-left-color:#f2483e; }

/* ---------- 因子/条目卡片 ---------- */
.qm-item {
    border:1px solid #1e2d4a; border-radius:11px; padding:.62rem .8rem;
    background:#111b2e; margin-bottom:.5rem; transition: border-color .15s ease;
}
.qm-item:hover { border-color:#3b5680; }
.qm-item-name { font-family: "JetBrains Mono", "Cascadia Code", monospace;
    font-size:.85rem; font-weight:650; color:#93c5fd; }
.qm-item-desc { font-size:.77rem; color:#94a3b8; margin-top:.22rem; line-height:1.5;}

/* ---------- 原生组件微调 ---------- */
div[data-testid="stMetric"] {
    background: linear-gradient(160deg,#16233c 0%,#111b2e 100%);
    border:1px solid #22304d; border-radius:12px; padding:.7rem .85rem;
}
div[data-testid="stMetricLabel"] p { font-size:.76rem !important; color:#94a3b8 !important; font-weight:600;}
div[data-testid="stMetricValue"] { font-size:1.4rem !important; font-variant-numeric: tabular-nums;}

.stTabs [data-baseweb="tab-list"] { gap:.35rem; border-bottom:1px solid #1e2d4a; }
.stTabs [data-baseweb="tab"] {
    height:38px; padding:0 1.05rem; border-radius:9px 9px 0 0;
    background:transparent; color:#94a3b8; font-size:.86rem; font-weight:600;
}
.stTabs [aria-selected="true"] {
    background:rgba(59,130,246,.12) !important; color:#f1f5f9 !important;
    border-bottom:2px solid #3b82f6;
}

.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
    border-radius:9px; font-weight:640; font-size:.86rem; border:1px solid #2a3a5c;
    transition: all .15s ease;
}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
    background: linear-gradient(135deg,#3b82f6 0%,#6366f1 100%);
    border:none; box-shadow:0 4px 14px rgba(59,130,246,.30);
}
.stButton > button[kind="primary"]:hover { box-shadow:0 6px 20px rgba(59,130,246,.45); }

div[data-testid="stExpander"] details {
    border:1px solid #1e2d4a !important; border-radius:11px !important; background:#111b2e;
}
div[data-testid="stExpander"] summary { font-size:.86rem; font-weight:640; }

div[data-testid="stDataFrame"], div[data-testid="stTable"] {
    border:1px solid #1e2d4a; border-radius:11px; overflow:hidden;
}

.stSlider label, .stSelectbox label, .stTextInput label,
.stNumberInput label, .stTextArea label, .stMultiSelect label, .stRadio label {
    font-size:.80rem !important; font-weight:640 !important; color:#cbd5e1 !important;
}
div[data-testid="stCaptionContainer"] p { color:#64748b !important; font-size:.76rem !important;}
hr { border-color:#1a2740 !important; margin:1.1rem 0 !important; }

/* 图表容器 */
div[data-testid="stPlotlyChart"] {
    border:1px solid #1e2d4a; border-radius:12px; padding:.35rem; background:#0f1a2d;
}

/* 带边框容器（模块卡 / 表单卡） */
div[data-testid="stVerticalBlockBorderWrapper"] > div > div[data-testid="stVerticalBlock"] {
    border-radius: 12px;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-color:#1e2d4a !important; border-radius:12px !important;
    background: linear-gradient(160deg,#141f36 0%,#111b2e 100%);
}

/* ---------- 首页模块卡 ---------- */
.qm-card {
    display:block; text-decoration:none; color:inherit;
    border:1px solid #22304d; border-radius:12px;
    background:linear-gradient(160deg,#141f36 0%,#111b2e 100%);
    padding:1rem 1.1rem; height:100%;
    transition:transform .16s ease, box-shadow .16s ease, border-color .16s ease, background .16s ease;
}
.qm-card:hover, .qm-card:focus-visible {
    transform:translateY(-3px);
    border-color:rgba(59,130,246,.65);
    background:linear-gradient(160deg,#182646 0%,#141f38 100%);
    box-shadow:0 6px 22px rgba(59,130,246,.22);
    outline:none;
}
.qm-card .qm-mod-icon { font-size:1.25rem; }
.qm-card .qm-mod-name { font-size:.98rem; font-weight:700; color:#e8eefc; }
.qm-card .qm-mod-head { display:flex; align-items:center; gap:.5rem; }
.qm-card .qm-mod-desc { display:block; font-size:.78rem; color:#8ea0bb; margin-top:.4rem;
    line-height:1.55; min-height:2.4em; }

/* 旧样式保留（供其他页面引用） */
.qm-mod { display:flex; align-items:center; gap:.5rem; }
.qm-mod-icon { font-size:1.15rem; }
.qm-mod-name { font-size:.96rem; font-weight:700; color:#e8eefc; }
.qm-mod-desc { font-size:.77rem; color:#8ea0bb; margin:.32rem 0 .5rem 0;
    line-height:1.55; min-height:2.4em; }

/* ---------- 流水线 ---------- */
.qm-flow { display:flex; flex-wrap:wrap; align-items:center; gap:.42rem; margin:.35rem 0 .2rem 0; }
.qm-flow-step {
    padding:.32rem .78rem; border-radius:8px; font-size:.78rem; font-weight:680;
    letter-spacing:.6px; color:#bfd3f5;
    background:rgba(59,130,246,.10); border:1px solid rgba(59,130,246,.28);
}
.qm-flow-arrow { color:#40527a; font-size:.9rem; }

/* ---------- 结论条 ---------- */
.qm-verdict {
    display:flex; align-items:center; gap:.6rem;
    border-radius:12px; padding:.75rem 1rem; margin:.2rem 0 .7rem 0;
    font-size:.9rem; font-weight:650; border:1px solid;
}
.qm-verdict-ok   { background:rgba(18,184,134,.10); border-color:rgba(18,184,134,.38); color:#34d399; }
.qm-verdict-bad  { background:rgba(242,72,62,.10);  border-color:rgba(242,72,62,.38);  color:#fb7185; }
.qm-verdict-warn { background:rgba(245,158,11,.10); border-color:rgba(245,158,11,.38); color:#fbbf24; }
.qm-verdict-icon { font-size:1.3rem; }

/* ---------- 代码块 ---------- */
div[data-testid="stCode"] { border:1px solid #1e2d4a; border-radius:11px; }
</style>
"""


# --------------------------------------------------------------------------
# 页面骨架
# --------------------------------------------------------------------------
#: 侧边栏导航分组（label, page path, icon）
NAV_GROUPS: List[tuple] = [
    ("总览", [
        ("首页", "streamlit_app.py", "🏠"),
        ("仪表盘", "pages/1_仪表盘.py", "📊"),
    ]),
    ("数据", [
        ("行情数据", "pages/2_行情数据.py", "📈"),
        ("数据质量", "pages/13_数据质量.py", "🧪"),
        ("数据管理", "pages/16_数据管理.py", "🗂️"),
        ("行情仓库总览", "pages/19_行情仓库总览.py", "🗄️"),
    ]),
    ("研究·流水线", [
        ("端到端流水线", "pages/20_端到端流水线.py", "🚀"),
        ("AI 研究", "pages/5_AI_研究.py", "🤖"),
        ("因子挖掘流水线", "pages/18_因子挖掘流水线.py", "⛏️"),
    ]),
    ("研究·评估与库", [
        ("因子研究", "pages/3_因子研究.py", "🔬"),
        ("截面研究", "pages/12_截面研究.py", "🧬"),
        ("席位因子", "pages/15_席位因子.py", "🪑"),
        ("因子库", "pages/9_FactorLibrary.py", "📚"),
        ("知识库", "pages/23_知识库.py", "📖"),
    ]),
    ("回测", [
        ("策略回测", "pages/4_策略回测.py", "⚙️"),
        ("参数优化", "pages/11_参数优化.py", "🎛️"),
        ("Walk-Forward", "pages/8_WalkForward.py", "🔁"),
    ]),
    ("策略", [
        ("LLM 策略挖掘", "pages/22_LLM策略挖掘.py", "🧠"),
    ]),
    ("交易", [
        ("风控中心", "pages/10_风控中心.py", "🛡️"),
        ("生命周期", "pages/6_生命周期.py", "🔄"),
        ("实时监控", "pages/7_实时监控.py", "📡"),
    ]),
    ("监测", [
        ("因子衰减监控", "pages/21_因子衰减监控.py", "🩻"),
    ]),
    ("系统", [
        ("设置", "pages/14_设置.py", "⚙️"),
    ]),
]


def setup_page(title: str, icon: str = "📊", layout: str = "wide") -> None:
    """页面初始化：set_page_config + 注入全局 CSS + 渲染侧边栏。

    必须在页面的**第一条** streamlit 调用之前执行。
    """
    st.set_page_config(
        page_title=f"{title} · QuantMind",
        page_icon=icon,
        layout=layout,
        initial_sidebar_state="expanded",
    )
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)
    render_sidebar()


def render_sidebar(show_nav: bool = True) -> None:
    """侧边栏：品牌区 + 分组导航 + 后端连接状态。"""
    with st.sidebar:
        st.markdown(
            """
            <div class="qm-brand">
              <div class="qm-brand-logo">📐</div>
              <div>
                <div class="qm-brand-name">QuantMind</div>
                <div class="qm-brand-sub">AI 量化研究</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if show_nav:
            for group, items in NAV_GROUPS:
                st.markdown(f'<div class="qm-nav-group">{group}</div>', unsafe_allow_html=True)
                for label, target, icon in items:
                    _safe_page_link(target, label, icon)

        st.divider()
        _sidebar_status()


def _safe_page_link(target: str, label: str, icon: str = "") -> None:
    """``st.page_link`` 在 AppTest 沙箱下会抛异常，兜底为纯文本。"""
    try:
        st.page_link(target, label=label, icon=icon or None)
    except Exception:  # noqa: BLE001
        st.markdown(f"{icon} {label}")


@st.cache_data(ttl=20, show_spinner=False)
def _cached_health() -> dict:
    from .api_client import APIClient

    return APIClient.health(timeout=4)


def _sidebar_status() -> None:
    st.markdown('<div class="qm-nav-group">后端状态</div>', unsafe_allow_html=True)
    try:
        health = _cached_health()
    except Exception as exc:  # noqa: BLE001
        health = {"error": str(exc)}

    if "error" in health:
        st.markdown(badge("离线", "danger"), unsafe_allow_html=True)
        st.caption("请先启动 API：`uvicorn quantmind.api.app:app`")
        return

    comps = health.get("components", {})
    feeds = health.get("feeds", [])
    st.markdown(
        badge("在线", "success")
        + " "
        + badge(f"数据源 {len(feeds)}", "info")
        + " "
        + badge(
            "引擎 " + ("运行" if comps.get("event_engine") == "running" else "停止"),
            "success" if comps.get("event_engine") == "running" else "warning",
        ),
        unsafe_allow_html=True,
    )
    if feeds:
        st.caption("· " + " · ".join(feeds[:6]))


# --------------------------------------------------------------------------
# 内容组件
# --------------------------------------------------------------------------
def page_header(title: str, subtitle: str = "", icon: str = "📊") -> None:
    """渐变页头。"""
    sub = f'<p class="qm-hero-sub">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f"""
        <div class="qm-hero">
          <div class="qm-hero-row">
            <div class="qm-hero-icon">{icon}</div>
            <div>
              <p class="qm-hero-title">{html.escape(title)}</p>
              {sub}
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title: str, desc: str = "") -> None:
    """带竖条的分区标题。"""
    d = f'<span class="qm-section-desc">{html.escape(desc)}</span>' if desc else ""
    st.markdown(
        f'<div class="qm-section"><div class="qm-section-bar"></div>'
        f'<span class="qm-section-title">{html.escape(title)}</span>{d}</div>',
        unsafe_allow_html=True,
    )


def badge(text: str, tone: str = "info") -> str:
    """返回徽章 HTML 片段（需配合 ``unsafe_allow_html=True`` 使用）。"""
    tone = tone if tone in ("success", "danger", "warning", "info", "muted", "violet") else "info"
    return f'<span class="qm-badge qm-badge-{tone}">{html.escape(str(text))}</span>'


def note(text: str, tone: str = "info") -> None:
    """信息条（支持内联 HTML/markdown 片段，调用方自行保证安全）。"""
    cls = {"info": "", "warning": " qm-note-warn", "success": " qm-note-ok", "error": " qm-note-err"}
    st.markdown(f'<div class="qm-note{cls.get(tone, "")}">{text}</div>', unsafe_allow_html=True)


def kpi_card(
    label: str,
    value: Any,
    delta: Optional[str] = None,
    tone: str = "neutral",
    hint: str = "",
) -> str:
    """单个 KPI 卡片 HTML。``tone``: up / down / neutral / accent。"""
    tone_cls = {"up": "qm-up", "down": "qm-down", "accent": "qm-accent"}.get(tone, "qm-neu")
    d = f'<div class="qm-kpi-delta {tone_cls}">{html.escape(str(delta))}</div>' if delta else ""
    h = f'<div class="qm-kpi-hint">{html.escape(hint)}</div>' if hint else ""
    return (
        f'<div class="qm-kpi"><div class="qm-kpi-label">{html.escape(str(label))}</div>'
        f'<div class="qm-kpi-value">{html.escape(str(value))}</div>{d}{h}</div>'
    )


def kpi_row(items: Sequence[Dict[str, Any]], per_row: int = 0) -> None:
    """一行 KPI 卡片。

    ``items`` 每项支持键：``label`` / ``value`` / ``delta`` / ``tone`` / ``hint``。
    """
    items = [i for i in items if i]
    if not items:
        return
    n = per_row or len(items)
    for start in range(0, len(items), n):
        chunk = items[start:start + n]
        cols = st.columns(len(chunk), gap="small")
        for col, item in zip(cols, chunk):
            with col:
                st.markdown(
                    kpi_card(
                        item.get("label", ""),
                        item.get("value", "—"),
                        item.get("delta"),
                        item.get("tone", "neutral"),
                        item.get("hint", ""),
                    ),
                    unsafe_allow_html=True,
                )


def item_card(name: str, desc: str = "", tags: Optional[Iterable[str]] = None) -> str:
    """列表条目卡片 HTML（因子库等场景）。"""
    tag_html = ""
    if tags:
        tag_html = " " + " ".join(badge(t, "muted") for t in tags if t)
    d = f'<div class="qm-item-desc">{html.escape(desc)}</div>' if desc else ""
    return (
        f'<div class="qm-item"><span class="qm-item-name">{html.escape(name)}</span>'
        f'{tag_html}{d}</div>'
    )


# --------------------------------------------------------------------------
# 格式化助手
# --------------------------------------------------------------------------
def fmt_pct(x: Optional[float], digits: int = 2, dash: str = "—") -> str:
    if x is None or (isinstance(x, float) and x != x):
        return dash
    return f"{x * 100:.{digits}f}%"


def fmt_num(x: Optional[float], digits: int = 3, dash: str = "—") -> str:
    if x is None or (isinstance(x, float) and x != x):
        return dash
    return f"{x:,.{digits}f}"


def fmt_money(x: Optional[float], dash: str = "—") -> str:
    if x is None or (isinstance(x, float) and x != x):
        return dash
    return f"¥{x:,.0f}"


def tone_of(x: Optional[float], positive_is_up: bool = True) -> str:
    """按数值正负返回红涨/绿跌 tone。"""
    if x is None or (isinstance(x, float) and x != x) or x == 0:
        return "neutral"
    good = x > 0 if positive_is_up else x < 0
    return "up" if good else "down"


def verdict(text: str, tone: str = "ok", icon: str = "") -> None:
    """结论条：一句话给出「能不能上 / 有没有过拟合 / 通不通过风控」。

    ``tone``: ok / bad / warn
    """
    cls = {"ok": "qm-verdict-ok", "bad": "qm-verdict-bad", "warn": "qm-verdict-warn"}.get(
        tone, "qm-verdict-ok"
    )
    ico = icon or {"ok": "✅", "bad": "⛔", "warn": "⚠️"}.get(tone, "✅")
    st.markdown(
        f'<div class="qm-verdict {cls}"><span class="qm-verdict-icon">{ico}</span>'
        f'<span>{html.escape(text)}</span></div>',
        unsafe_allow_html=True,
    )


def guard_error(payload: Any, context: str = "请求") -> bool:
    """统一错误渲染。返回 True 表示存在错误、调用方应中止后续渲染。"""
    if isinstance(payload, dict) and payload.get("error"):
        note(f"<b>{html.escape(context)}失败</b>：{html.escape(str(payload['error']))}", "error")
        return True
    return False
