"""QuantMind 统一视觉主题与 UI 组件库 v2 — Precision Terminal 风格。

所有页面统一调用 :func:`setup_page` 完成「页面配置 + 全局样式 + 侧边栏品牌区」，
再用 :func:`page_header` / :func:`section` / :func:`kpi_row` 等组件组织内容，
避免每个页面各写一套 markdown 样式导致风格割裂。

配色遵循中国市场习惯：**红涨绿跌**。
设计语言：Precision Terminal — 深色精密终端 + 毛玻璃质感 + 动态光晕。
"""
from __future__ import annotations

import html
from typing import Any, Dict, Iterable, List, Optional, Sequence

import streamlit as st

# --------------------------------------------------------------------------
# 设计令牌（Design Tokens）v2
# --------------------------------------------------------------------------
COLORS = {
    "bg": "#06090f",
    "surface": "#0c1220",
    "surface_alt": "#111a2e",
    "surface_glass": "rgba(14, 22, 40, 0.65)",
    "border": "#1a2744",
    "border_soft": "#131d33",
    "border_glow": "rgba(59, 130, 246, 0.25)",
    "text": "#e8edf5",
    "text_muted": "#8b9dc1",
    "text_dim": "#4a5e82",
    "primary": "#4f8ff7",
    "primary_dark": "#2563eb",
    "primary_glow": "rgba(79, 143, 247, 0.15)",
    "violet": "#8b5cf6",
    "violet_glow": "rgba(139, 92, 246, 0.12)",
    "cyan": "#22d3ee",
    "amber": "#f59e0b",
    "amber_glow": "rgba(245, 158, 11, 0.12)",
    # 中国市场：红涨绿跌
    "up": "#ef4444",
    "up_soft": "rgba(239, 68, 68, 0.12)",
    "down": "#10b981",
    "down_soft": "rgba(16, 185, 129, 0.12)",
    "success": "#10b981",
    "danger": "#ef4444",
    "warning": "#f59e0b",
    "info": "#4f8ff7",
}

#: Plotly 统一配色序列
PLOTLY_COLORWAY = [
    "#4f8ff7", "#8b5cf6", "#22d3ee", "#f59e0b",
    "#ef4444", "#10b981", "#ec4899", "#a3e635",
]

_GLOBAL_CSS = """
<style>
/* ======================================================================
   QuantMind v2 — Precision Terminal Theme
   ====================================================================== */

/* ---------- Google Fonts ---------- */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap');

/* ---------- CSS Variables ---------- */
:root {
    --qm-bg: #06090f;
    --qm-surface: #0c1220;
    --qm-surface-alt: #111a2e;
    --qm-glass: rgba(14, 22, 40, 0.55);
    --qm-glass-hover: rgba(18, 28, 50, 0.72);
    --qm-border: #1a2744;
    --qm-border-soft: #131d33;
    --qm-border-glow: rgba(79, 143, 247, 0.2);
    --qm-text: #e8edf5;
    --qm-text-muted: #8b9dc1;
    --qm-text-dim: #4a5e82;
    --qm-primary: #4f8ff7;
    --qm-primary-glow: rgba(79, 143, 247, 0.15);
    --qm-violet: #8b5cf6;
    --qm-cyan: #22d3ee;
    --qm-amber: #f59e0b;
    --qm-up: #ef4444;
    --qm-down: #10b981;
    --qm-radius: 14px;
    --qm-radius-sm: 10px;
    --qm-radius-lg: 20px;
    --qm-font-display: 'Space Grotesk', 'Inter', 'PingFang SC', 'Microsoft YaHei', sans-serif;
    --qm-font-body: 'Inter', 'PingFang SC', 'Microsoft YaHei', -apple-system, sans-serif;
    --qm-font-mono: 'IBM Plex Mono', 'Cascadia Code', 'JetBrains Mono', monospace;
    --qm-shadow-card: 0 4px 24px rgba(0,0,0,0.3), 0 1px 3px rgba(0,0,0,0.2);
    --qm-shadow-glow: 0 0 40px rgba(79, 143, 247, 0.08);
    --qm-transition: 0.22s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ---------- 字体与基底 ---------- */
html, body, [class*="css"] {
    font-family: var(--qm-font-body);
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

.stApp {
    background:
        radial-gradient(ellipse 1400px 700px at 8% -5%, rgba(79,143,247,0.07), transparent 65%),
        radial-gradient(ellipse 1200px 600px at 92% 5%, rgba(139,92,246,0.05), transparent 60%),
        radial-gradient(ellipse 900px 500px at 50% 100%, rgba(34,211,238,0.03), transparent 55%),
        var(--qm-bg);
}

/* 微妙的网格纹理叠加 */
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(79,143,247,0.02) 1px, transparent 1px),
        linear-gradient(90deg, rgba(79,143,247,0.02) 1px, transparent 1px);
    background-size: 60px 60px;
    pointer-events: none;
    z-index: 0;
}

.block-container {
    padding-top: 2.0rem;
    padding-bottom: 3.5rem;
    max-width: 1560px;
    position: relative;
    z-index: 1;
}

#MainMenu, footer {visibility: hidden;}

/* 隐藏 Streamlit 默认 pages/ 自动导航 */
[data-testid="stSidebarNav"] { display: none; }
section[data-testid="stSidebar"] [data-testid="stSidebarNavSeparator"] { display: none; }

/* ---------- 侧边栏 ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080d18 0%, #06090f 100%);
    border-right: 1px solid var(--qm-border-soft);
    backdrop-filter: blur(20px);
}

[data-testid="stSidebarUserContent"] { padding: .5rem 0 !important; }
[data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"] > div:first-child {
    margin-top: -5.0rem;
    pointer-events: none;
}
[data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"] > div:first-child * { pointer-events: none; }
[data-testid="stBaseButton-headerNoPadding"] { z-index: 1000 !important; }
section[data-testid="stSidebar"] .block-container { padding: .5rem 0 !important; }

/* 品牌区 */
.qm-brand {
    display: flex; align-items: center; gap: .75rem;
    padding: .2rem 0 .9rem 0;
}
.qm-brand-logo {
    width: 42px; height: 42px; border-radius: 12px; flex: none;
    background: linear-gradient(135deg, #4f8ff7 0%, #8b5cf6 50%, #22d3ee 100%);
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
    box-shadow: 0 6px 20px rgba(79,143,247,0.3), inset 0 1px 0 rgba(255,255,255,0.1);
    animation: qm-logo-pulse 4s ease-in-out infinite;
}
@keyframes qm-logo-pulse {
    0%, 100% { box-shadow: 0 6px 20px rgba(79,143,247,0.3), inset 0 1px 0 rgba(255,255,255,0.1); }
    50% { box-shadow: 0 8px 28px rgba(79,143,247,0.45), inset 0 1px 0 rgba(255,255,255,0.15); }
}
.qm-brand-name {
    font-family: var(--qm-font-display);
    font-size: 1.18rem; font-weight: 700; color: #f1f5f9;
    letter-spacing: -.2px; line-height: 1.15;
}
.qm-brand-sub {
    font-family: var(--qm-font-mono);
    font-size: .65rem; color: var(--qm-text-dim);
    letter-spacing: 1.5px; text-transform: uppercase; margin-top: 2px;
}

.qm-nav-group {
    font-family: var(--qm-font-mono);
    font-size: .62rem; font-weight: 600; color: var(--qm-text-dim);
    letter-spacing: 1.8px; text-transform: uppercase;
    margin: 1.1rem 0 .3rem .15rem;
}

/* ---------- 页头 Hero ---------- */
.qm-hero {
    position: relative; overflow: hidden;
    border: 1px solid var(--qm-border);
    border-radius: var(--qm-radius-lg);
    background:
        linear-gradient(135deg, rgba(14,22,40,0.9) 0%, rgba(12,18,32,0.95) 100%);
    backdrop-filter: blur(16px);
    padding: 1.4rem 1.6rem; margin-bottom: 1.5rem;
    box-shadow: var(--qm-shadow-card), var(--qm-shadow-glow);
}
.qm-hero::before {
    content: "";
    position: absolute; inset: 0;
    border-radius: inherit;
    background: linear-gradient(135deg, rgba(79,143,247,0.04) 0%, transparent 50%, rgba(139,92,246,0.03) 100%);
    pointer-events: none;
}
.qm-hero::after {
    content: ""; position: absolute; right: -40px; top: -50px;
    width: 280px; height: 280px; border-radius: 50%;
    background: radial-gradient(circle, rgba(79,143,247,0.12), transparent 68%);
    animation: qm-hero-glow 6s ease-in-out infinite alternate;
}
@keyframes qm-hero-glow {
    0% { transform: translate(0, 0) scale(1); opacity: 0.8; }
    100% { transform: translate(-15px, 10px) scale(1.1); opacity: 1; }
}
.qm-hero-row { display: flex; align-items: center; gap: 1rem; position: relative; z-index: 1; }
.qm-hero-icon {
    width: 52px; height: 52px; border-radius: 14px; flex: none;
    background: linear-gradient(135deg, rgba(79,143,247,0.12), rgba(139,92,246,0.08));
    border: 1px solid rgba(79,143,247,0.25);
    display: flex; align-items: center; justify-content: center; font-size: 26px;
    box-shadow: 0 4px 16px rgba(79,143,247,0.1);
}
.qm-hero-title {
    font-family: var(--qm-font-display);
    font-size: 1.72rem; font-weight: 700; color: #f8fafc;
    line-height: 1.2; margin: 0; letter-spacing: -.3px;
}
.qm-hero-sub {
    font-size: .88rem; color: var(--qm-text-muted);
    margin: .35rem 0 0 0; max-width: 900px; line-height: 1.6;
}

/* ---------- 分区标题 ---------- */
.qm-section {
    display: flex; align-items: baseline; gap: .6rem;
    margin: 1.8rem 0 .8rem 0;
}
.qm-section-bar {
    width: 3px; height: 18px; border-radius: 2px;
    background: linear-gradient(180deg, var(--qm-primary), var(--qm-violet));
    transform: translateY(2px);
    box-shadow: 0 0 8px rgba(79,143,247,0.3);
}
.qm-section-title {
    font-family: var(--qm-font-display);
    font-size: 1.08rem; font-weight: 700; color: var(--qm-text);
    letter-spacing: -.1px;
}
.qm-section-desc {
    font-size: .78rem; color: var(--qm-text-dim);
}

/* ---------- KPI 卡片 ---------- */
.qm-kpi {
    border: 1px solid var(--qm-border);
    border-radius: var(--qm-radius);
    padding: 1rem 1.1rem;
    background: var(--qm-glass);
    backdrop-filter: blur(12px);
    height: 100%;
    transition: all var(--qm-transition);
    position: relative;
    overflow: hidden;
}
.qm-kpi::before {
    content: "";
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--qm-primary), transparent);
    opacity: 0;
    transition: opacity var(--qm-transition);
}
.qm-kpi:hover {
    border-color: var(--qm-border-glow);
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3), 0 0 20px rgba(79,143,247,0.06);
}
.qm-kpi:hover::before { opacity: 1; }
.qm-kpi-label {
    font-family: var(--qm-font-mono);
    font-size: .70rem; color: var(--qm-text-muted); font-weight: 600;
    letter-spacing: .5px; text-transform: uppercase;
    display: flex; align-items: center; gap: .35rem;
}
.qm-kpi-value {
    font-family: var(--qm-font-display);
    font-size: 1.62rem; font-weight: 700; color: #f1f5f9;
    line-height: 1.3; margin-top: .25rem;
    font-variant-numeric: tabular-nums; word-break: break-all;
}
.qm-kpi-delta { font-size: .74rem; font-weight: 640; margin-top: .15rem; }
.qm-kpi-hint {
    font-size: .68rem; color: var(--qm-text-dim);
    margin-top: .22rem; font-family: var(--qm-font-mono);
}
.qm-up   { color: var(--qm-up) !important; }
.qm-down { color: var(--qm-down) !important; }
.qm-neu  { color: var(--qm-text-muted) !important; }
.qm-accent { color: var(--qm-primary) !important; }

/* ---------- 徽章 ---------- */
.qm-badge {
    display: inline-flex; align-items: center; gap: .3rem;
    padding: .18rem .6rem; border-radius: 999px;
    font-family: var(--qm-font-mono);
    font-size: .68rem; font-weight: 600; letter-spacing: .3px; white-space: nowrap;
    backdrop-filter: blur(8px);
}
.qm-badge-success { background: rgba(16,185,129,.12); color: #34d399; border: 1px solid rgba(16,185,129,.3); }
.qm-badge-danger  { background: rgba(239,68,68,.12);  color: #f87171; border: 1px solid rgba(239,68,68,.3); }
.qm-badge-warning { background: rgba(245,158,11,.12); color: #fbbf24; border: 1px solid rgba(245,158,11,.3); }
.qm-badge-info    { background: rgba(79,143,247,.12); color: #93c5fd; border: 1px solid rgba(79,143,247,.3); }
.qm-badge-muted   { background: rgba(74,94,130,.12);  color: var(--qm-text-muted); border: 1px solid rgba(74,94,130,.25); }
.qm-badge-violet  { background: rgba(139,92,246,.12); color: #a78bfa; border: 1px solid rgba(139,92,246,.3); }

/* ---------- 信息条 ---------- */
.qm-note {
    border: 1px solid var(--qm-border);
    border-left: 3px solid var(--qm-primary);
    border-radius: var(--qm-radius-sm);
    background: var(--qm-glass);
    backdrop-filter: blur(8px);
    padding: .8rem 1.1rem;
    font-size: .83rem; color: #c8d5e8; line-height: 1.65; margin-bottom: .7rem;
}
.qm-note b, .qm-note strong { color: #f1f5f9; }
.qm-note code {
    font-family: var(--qm-font-mono);
    background: rgba(79,143,247,.1); color: #93c5fd;
    padding: .08rem .35rem; border-radius: 5px; font-size: .78rem;
    border: 1px solid rgba(79,143,247,.15);
}
.qm-note-warn { border-left-color: var(--qm-amber); }
.qm-note-ok   { border-left-color: var(--qm-down); }
.qm-note-err  { border-left-color: var(--qm-up); }

/* ---------- 因子/条目卡片 ---------- */
.qm-item {
    border: 1px solid var(--qm-border);
    border-radius: var(--qm-radius-sm);
    padding: .7rem .9rem;
    background: var(--qm-glass);
    backdrop-filter: blur(8px);
    margin-bottom: .5rem;
    transition: all var(--qm-transition);
}
.qm-item:hover {
    border-color: var(--qm-border-glow);
    background: var(--qm-glass-hover);
}
.qm-item-name {
    font-family: var(--qm-font-mono);
    font-size: .85rem; font-weight: 600; color: #93c5fd;
}
.qm-item-desc { font-size: .77rem; color: var(--qm-text-muted); margin-top: .25rem; line-height: 1.55; }

/* ---------- 原生组件微调 ---------- */
div[data-testid="stMetric"] {
    background: var(--qm-glass);
    backdrop-filter: blur(12px);
    border: 1px solid var(--qm-border);
    border-radius: var(--qm-radius);
    padding: .8rem 1rem;
    transition: all var(--qm-transition);
}
div[data-testid="stMetric"]:hover {
    border-color: var(--qm-border-glow);
}
div[data-testid="stMetricLabel"] p {
    font-family: var(--qm-font-mono) !important;
    font-size: .72rem !important; color: var(--qm-text-muted) !important;
    font-weight: 600 !important; letter-spacing: .3px !important;
    text-transform: uppercase !important;
}
div[data-testid="stMetricValue"] {
    font-family: var(--qm-font-display) !important;
    font-size: 1.45rem !important; font-variant-numeric: tabular-nums;
}

.stTabs [data-baseweb="tab-list"] { gap: .3rem; border-bottom: 1px solid var(--qm-border); }
.stTabs [data-baseweb="tab"] {
    height: 40px; padding: 0 1.15rem; border-radius: var(--qm-radius-sm) var(--qm-radius-sm) 0 0;
    background: transparent; color: var(--qm-text-muted);
    font-family: var(--qm-font-body);
    font-size: .86rem; font-weight: 600;
    transition: all var(--qm-transition);
}
.stTabs [aria-selected="true"] {
    background: var(--qm-primary-glow) !important;
    color: #f1f5f9 !important;
    border-bottom: 2px solid var(--qm-primary);
}

.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
    font-family: var(--qm-font-body);
    border-radius: var(--qm-radius-sm); font-weight: 640; font-size: .86rem;
    border: 1px solid #2a3a5c;
    transition: all var(--qm-transition);
}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
    background: linear-gradient(135deg, #4f8ff7 0%, #6366f1 100%);
    border: none;
    box-shadow: 0 4px 16px rgba(79,143,247,.25), inset 0 1px 0 rgba(255,255,255,.1);
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 24px rgba(79,143,247,.4), inset 0 1px 0 rgba(255,255,255,.15);
    transform: translateY(-1px);
}

div[data-testid="stExpander"] details {
    border: 1px solid var(--qm-border) !important;
    border-radius: var(--qm-radius) !important;
    background: var(--qm-glass);
    backdrop-filter: blur(8px);
}
div[data-testid="stExpander"] summary {
    font-family: var(--qm-font-body);
    font-size: .86rem; font-weight: 640;
}

div[data-testid="stDataFrame"], div[data-testid="stTable"] {
    border: 1px solid var(--qm-border);
    border-radius: var(--qm-radius);
    overflow: hidden;
}

.stSlider label, .stSelectbox label, .stTextInput label,
.stNumberInput label, .stTextArea label, .stMultiSelect label, .stRadio label {
    font-size: .80rem !important; font-weight: 640 !important;
    color: #c8d5e8 !important;
}
div[data-testid="stCaptionContainer"] p {
    color: var(--qm-text-dim) !important; font-size: .76rem !important;
}
hr { border-color: var(--qm-border-soft) !important; margin: 1.2rem 0 !important; }

/* 图表容器 */
div[data-testid="stPlotlyChart"] {
    border: 1px solid var(--qm-border);
    border-radius: var(--qm-radius);
    padding: .4rem;
    background: rgba(8,13,24,0.6);
    backdrop-filter: blur(8px);
}

/* 带边框容器 */
div[data-testid="stVerticalBlockBorderWrapper"] > div > div[data-testid="stVerticalBlock"] {
    border-radius: var(--qm-radius);
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: var(--qm-border) !important;
    border-radius: var(--qm-radius) !important;
    background: var(--qm-glass) !important;
    backdrop-filter: blur(8px);
}

/* ---------- 首页模块卡 ---------- */
.qm-card {
    display: block; text-decoration: none; color: inherit;
    border: 1px solid var(--qm-border);
    border-radius: var(--qm-radius);
    background: var(--qm-glass);
    backdrop-filter: blur(12px);
    padding: 1.1rem 1.2rem;
    height: 100%;
    transition: all var(--qm-transition);
    position: relative;
    overflow: hidden;
}
.qm-card::before {
    content: "";
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--qm-primary), var(--qm-violet), var(--qm-cyan));
    opacity: 0;
    transition: opacity var(--qm-transition);
}
.qm-card:hover, .qm-card:focus-visible {
    transform: translateY(-4px);
    border-color: rgba(79,143,247,.45);
    background: var(--qm-glass-hover);
    box-shadow: 0 12px 40px rgba(0,0,0,0.4), 0 0 30px rgba(79,143,247,.08);
    outline: none;
}
.qm-card:hover::before { opacity: 1; }
.qm-card .qm-mod-icon { font-size: 1.35rem; }
.qm-card .qm-mod-name {
    font-family: var(--qm-font-display);
    font-size: 1rem; font-weight: 700; color: #edf2fc;
}
.qm-card .qm-mod-head { display: flex; align-items: center; gap: .55rem; }
.qm-card .qm-mod-desc {
    display: block; font-size: .78rem; color: var(--qm-text-muted);
    margin-top: .45rem; line-height: 1.6; min-height: 2.5em;
}

/* 旧样式兼容 */
.qm-mod { display: flex; align-items: center; gap: .5rem; }
.qm-mod-icon { font-size: 1.15rem; }
.qm-mod-name { font-size: .96rem; font-weight: 700; color: #edf2fc; }
.qm-mod-desc {
    font-size: .77rem; color: var(--qm-text-muted);
    margin: .32rem 0 .5rem 0; line-height: 1.55; min-height: 2.4em;
}

/* ---------- 流水线 ---------- */
.qm-flow {
    display: flex; flex-wrap: wrap; align-items: center;
    gap: .45rem; margin: .4rem 0 .3rem 0;
}
.qm-flow-step {
    padding: .35rem .85rem; border-radius: 9px;
    font-family: var(--qm-font-mono);
    font-size: .75rem; font-weight: 600;
    letter-spacing: .8px; color: #bfd3f5;
    background: rgba(79,143,247,.08);
    border: 1px solid rgba(79,143,247,.22);
    transition: all var(--qm-transition);
}
.qm-flow-step:hover {
    background: rgba(79,143,247,.14);
    border-color: rgba(79,143,247,.4);
    transform: translateY(-1px);
}
.qm-flow-arrow {
    color: var(--qm-text-dim); font-size: .9rem;
    font-family: var(--qm-font-mono);
}

/* ---------- 结论条 ---------- */
.qm-verdict {
    display: flex; align-items: center; gap: .65rem;
    border-radius: var(--qm-radius);
    padding: .85rem 1.15rem; margin: .2rem 0 .8rem 0;
    font-size: .9rem; font-weight: 650;
    border: 1px solid;
    backdrop-filter: blur(8px);
}
.qm-verdict-ok   { background: rgba(16,185,129,.08);  border-color: rgba(16,185,129,.32);  color: #34d399; }
.qm-verdict-bad  { background: rgba(239,68,68,.08);   border-color: rgba(239,68,68,.32);   color: #f87171; }
.qm-verdict-warn { background: rgba(245,158,11,.08);  border-color: rgba(245,158,11,.32);  color: #fbbf24; }
.qm-verdict-icon { font-size: 1.35rem; }

/* ---------- 代码块 ---------- */
div[data-testid="stCode"] {
    border: 1px solid var(--qm-border);
    border-radius: var(--qm-radius);
}

/* ---------- 滚动条美化 ---------- */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(79,143,247,.2);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(79,143,247,.35); }

/* ---------- 选中文字颜色 ---------- */
::selection {
    background: rgba(79,143,247,.3);
    color: #fff;
}

/* ---------- 页面载入动画 ---------- */
@keyframes qm-fade-in {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}
.block-container > div {
    animation: qm-fade-in 0.4s ease-out;
}

/* ---------- Selectbox / TextInput 输入框 ---------- */
.stSelectbox [data-baseweb="select"],
.stTextInput input, .stNumberInput input, .stTextArea textarea {
    border-radius: var(--qm-radius-sm) !important;
    border-color: var(--qm-border) !important;
    background: rgba(12,18,32,0.6) !important;
    transition: all var(--qm-transition) !important;
}
.stSelectbox [data-baseweb="select"]:focus-within,
.stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {
    border-color: var(--qm-primary) !important;
    box-shadow: 0 0 0 2px var(--qm-primary-glow) !important;
}

/* ---------- Spinner ---------- */
.stSpinner > div {
    border-top-color: var(--qm-primary) !important;
}

/* ======================================================================
   v2.1 — 增强组件
   ====================================================================== */

/* ---------- 状态指示灯（脉冲动画） ---------- */
.qm-status-dot {
    display: inline-block; width: 8px; height: 8px;
    border-radius: 50%; flex: none;
    position: relative;
}
.qm-status-dot::after {
    content: ""; position: absolute; inset: -3px;
    border-radius: 50%; opacity: 0;
    animation: qm-pulse 2s ease-in-out infinite;
}
.qm-status-dot--ok { background: var(--qm-down); }
.qm-status-dot--ok::after { background: var(--qm-down); }
.qm-status-dot--warn { background: var(--qm-amber); }
.qm-status-dot--warn::after { background: var(--qm-amber); }
.qm-status-dot--err { background: var(--qm-up); }
.qm-status-dot--err::after { background: var(--qm-up); }
.qm-status-dot--info { background: var(--qm-primary); }
.qm-status-dot--info::after { background: var(--qm-primary); }
@keyframes qm-pulse {
    0% { opacity: 0; transform: scale(0.8); }
    50% { opacity: 0.4; transform: scale(1.6); }
    100% { opacity: 0; transform: scale(0.8); }
}
.qm-status-row {
    display: flex; align-items: center; gap: .55rem;
    padding: .55rem .8rem;
    border: 1px solid var(--qm-border);
    border-radius: var(--qm-radius-sm);
    background: var(--qm-glass);
    backdrop-filter: blur(8px);
    transition: all var(--qm-transition);
}
.qm-status-row:hover {
    border-color: var(--qm-border-glow);
    background: var(--qm-glass-hover);
}
.qm-status-label {
    font-size: .82rem; font-weight: 600; color: var(--qm-text);
}
.qm-status-value {
    font-family: var(--qm-font-mono);
    font-size: .75rem; color: var(--qm-text-muted);
    margin-left: auto;
}

/* ---------- 速选按钮卡片（行情数据等页的标的速选） ---------- */
.qm-preset-btn {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    gap: .15rem; padding: .55rem .4rem;
    border: 1px solid var(--qm-border);
    border-radius: var(--qm-radius-sm);
    background: var(--qm-glass);
    backdrop-filter: blur(6px);
    cursor: pointer;
    transition: all var(--qm-transition);
    text-align: center;
    height: 100%;
}
.qm-preset-btn:hover {
    border-color: var(--qm-border-glow);
    background: var(--qm-glass-hover);
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.25);
}
.qm-preset-name {
    font-size: .78rem; font-weight: 600; color: var(--qm-text);
    line-height: 1.2;
}
.qm-preset-code {
    font-family: var(--qm-font-mono);
    font-size: .68rem; color: var(--qm-text-dim);
    letter-spacing: .3px;
}

/* ---------- 表单区容器（带顶部渐变条） ---------- */
.qm-form-area {
    border: 1px solid var(--qm-border);
    border-radius: var(--qm-radius);
    background: var(--qm-glass);
    backdrop-filter: blur(10px);
    padding: 1.2rem 1.3rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}
.qm-form-area::before {
    content: "";
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--qm-primary), var(--qm-violet), var(--qm-cyan));
    opacity: 0.6;
}

/* ---------- 分隔线（带标签） ---------- */
.qm-divider {
    display: flex; align-items: center; gap: .8rem;
    margin: 1.5rem 0 1rem 0;
    color: var(--qm-text-dim);
    font-family: var(--qm-font-mono);
    font-size: .68rem; letter-spacing: 1px; text-transform: uppercase;
}
.qm-divider::before, .qm-divider::after {
    content: ""; flex: 1; height: 1px;
    background: linear-gradient(90deg, transparent, var(--qm-border), transparent);
}

/* ---------- 连接状态条 ---------- */
.qm-conn-bar {
    display: flex; align-items: center; gap: .7rem;
    padding: .7rem 1rem;
    border: 1px solid var(--qm-border);
    border-radius: var(--qm-radius);
    background: var(--qm-glass);
    backdrop-filter: blur(8px);
    margin-bottom: 1rem;
}
.qm-conn-bar--ok { border-left: 3px solid var(--qm-down); }
.qm-conn-bar--warn { border-left: 3px solid var(--qm-amber); }
.qm-conn-bar--err { border-left: 3px solid var(--qm-up); }
.qm-conn-text {
    font-size: .84rem; font-weight: 600; color: var(--qm-text);
}
.qm-conn-sub {
    font-family: var(--qm-font-mono);
    font-size: .72rem; color: var(--qm-text-dim);
    margin-left: auto;
}

/* ---------- 生命周期流水线可视化 ---------- */
.qm-pipeline {
    display: flex; align-items: stretch; gap: 0;
    margin: .5rem 0 1.2rem 0;
    overflow-x: auto;
}
.qm-pipe-step {
    flex: 1; min-width: 100px;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    gap: .3rem; padding: .7rem .5rem;
    border: 1px solid var(--qm-border);
    position: relative;
    transition: all var(--qm-transition);
}
.qm-pipe-step:first-child { border-radius: var(--qm-radius-sm) 0 0 var(--qm-radius-sm); }
.qm-pipe-step:last-child { border-radius: 0 var(--qm-radius-sm) var(--qm-radius-sm) 0; }
.qm-pipe-step + .qm-pipe-step { border-left: none; }
.qm-pipe-step::after {
    content: "›"; position: absolute; right: -5px; top: 50%; transform: translateY(-50%);
    font-size: 1rem; color: var(--qm-text-dim); z-index: 1;
}
.qm-pipe-step:last-child::after { display: none; }
.qm-pipe-step--active {
    background: var(--qm-primary-glow);
    border-color: rgba(79,143,247,0.4);
    z-index: 1;
}
.qm-pipe-step--done {
    background: rgba(16,185,129,0.06);
    border-color: rgba(16,185,129,0.25);
}
.qm-pipe-label {
    font-family: var(--qm-font-mono);
    font-size: .72rem; font-weight: 700; letter-spacing: .8px;
    color: var(--qm-text-muted);
}
.qm-pipe-step--active .qm-pipe-label { color: var(--qm-primary); }
.qm-pipe-step--done .qm-pipe-label { color: var(--qm-down); }
.qm-pipe-desc {
    font-size: .65rem; color: var(--qm-text-dim);
    text-align: center; line-height: 1.3;
    max-width: 120px;
}

/* ---------- 订单预览卡 ---------- */
.qm-order-preview {
    border: 1px solid var(--qm-border);
    border-radius: var(--qm-radius);
    background: var(--qm-glass);
    backdrop-filter: blur(8px);
    padding: 1rem 1.2rem;
    font-size: .84rem;
    line-height: 1.8;
    color: var(--qm-text-muted);
}
.qm-order-preview b, .qm-order-preview strong { color: var(--qm-text); }
.qm-order-preview code {
    font-family: var(--qm-font-mono);
    background: rgba(79,143,247,0.1);
    color: #93c5fd;
    padding: .05rem .3rem;
    border-radius: 4px;
    font-size: .8rem;
}

/* ---------- 数据表增强 ---------- */
.qm-table-wrapper {
    border: 1px solid var(--qm-border);
    border-radius: var(--qm-radius);
    overflow: hidden;
    background: var(--qm-glass);
    backdrop-filter: blur(6px);
}

/* ---------- 统计条（页面底部） ---------- */
.qm-footer {
    display: flex; align-items: center; justify-content: space-between;
    padding: .6rem 1rem;
    border-top: 1px solid var(--qm-border-soft);
    margin-top: 2rem;
    font-family: var(--qm-font-mono);
    font-size: .68rem;
    color: var(--qm-text-dim);
    letter-spacing: .3px;
}
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
        ("LLM策略挖掘", "pages/24_LLM策略挖掘.py", "🎯"),
    ]),
    ("研究·评估与库", [
        ("因子研究", "pages/3_因子研究.py", "🔬"),
        ("因子库", "pages/9_FactorLibrary.py", "📚"),
        ("知识库", "pages/23_知识库.py", "📖"),
    ]),
    ("回测", [
        ("策略回测", "pages/4_策略回测.py", "⚙️"),
        ("参数优化", "pages/11_参数优化.py", "🎛️"),
        ("Walk-Forward", "pages/8_WalkForward.py", "🔁"),
    ]),
    ("策略", [
        ("因子组合策略", "pages/22_因子组合策略.py", "🧠"),
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
                <div class="qm-brand-sub">Precision Terminal</div>
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


# --------------------------------------------------------------------------
# v2.1 增强组件
# --------------------------------------------------------------------------
def status_dot(tone: str = "ok") -> str:
    """返回脉冲状态指示灯 HTML。tone: ok / warn / err / info。"""
    cls = {"ok": "qm-status-dot--ok", "warn": "qm-status-dot--warn",
           "err": "qm-status-dot--err", "info": "qm-status-dot--info"}.get(tone, "qm-status-dot--info")
    return f'<span class="qm-status-dot {cls}"></span>'


def status_row(label: str, value: str = "", tone: str = "ok") -> str:
    """返回带状态指示灯的行组件 HTML。"""
    return (
        f'<div class="qm-status-row">{status_dot(tone)}'
        f'<span class="qm-status-label">{html.escape(str(label))}</span>'
        f'<span class="qm-status-value">{html.escape(str(value))}</span></div>'
    )


def status_cards(items: Sequence[Dict[str, str]]) -> None:
    """一行状态卡片。每项支持 label / value / tone。"""
    items = [i for i in items if i]
    if not items:
        return
    cols = st.columns(len(items), gap="small")
    for col, item in zip(cols, items):
        with col:
            st.markdown(
                status_row(item.get("label", ""), item.get("value", ""), item.get("tone", "ok")),
                unsafe_allow_html=True,
            )


def divider(label: str = "") -> None:
    """带标签的分隔线。"""
    if label:
        st.markdown(f'<div class="qm-divider">{html.escape(label)}</div>', unsafe_allow_html=True)
    else:
        st.divider()


def conn_bar(text: str, sub: str = "", tone: str = "ok") -> None:
    """连接状态条。tone: ok / warn / err。"""
    cls = {"ok": "qm-conn-bar--ok", "warn": "qm-conn-bar--warn", "err": "qm-conn-bar--err"}.get(tone, "")
    sub_html = f'<span class="qm-conn-sub">{html.escape(sub)}</span>' if sub else ""
    st.markdown(
        f'<div class="qm-conn-bar {cls}">{status_dot(tone)}'
        f'<span class="qm-conn-text">{html.escape(text)}</span>{sub_html}</div>',
        unsafe_allow_html=True,
    )


def pipeline(steps: Sequence[Dict[str, str]], active: str = "") -> None:
    """生命周期流水线可视化。steps: [{label, desc}, ...]；active 为当前阶段。"""
    labels = [s.get("label", "") for s in steps]
    descs = [s.get("desc", "") for s in steps]
    active_idx = labels.index(active) if active in labels else -1

    html_parts = ['<div class="qm-pipeline">']
    for i, (lbl, desc) in enumerate(zip(labels, descs)):
        cls = "qm-pipe-step"
        if i < active_idx:
            cls += " qm-pipe-step--done"
        elif i == active_idx:
            cls += " qm-pipe-step--active"
        html_parts.append(
            f'<div class="{cls}">'
            f'<span class="qm-pipe-label">{html.escape(lbl)}</span>'
            f'<span class="qm-pipe-desc">{html.escape(desc)}</span></div>'
        )
    html_parts.append('</div>')
    st.markdown("".join(html_parts), unsafe_allow_html=True)


def order_preview(lines: Sequence[str]) -> None:
    """订单预览卡。"""
    content = "<br>".join(lines)
    st.markdown(f'<div class="qm-order-preview">{content}</div>', unsafe_allow_html=True)


def form_area() -> Any:
    """返回一个带顶部渐变条的表单容器（用 st.form 包裹）。"""
    return st.form
