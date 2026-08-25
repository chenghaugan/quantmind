"""布局组件 v2 — Precision Terminal 风格。"""

import streamlit as st


def render_sidebar():
    """渲染侧边栏（已被 theme.render_sidebar 替代，保留兼容接口）。"""
    pass


def render_metric_card(label: str, value: str, delta: str = None, delta_color: str = "normal"):
    """渲染指标卡片"""
    st.metric(label=label, value=value, delta=delta, delta_color=delta_color)


def render_error_box(error_msg: str):
    """渲染错误提示"""
    st.error(f"❌ {error_msg}")


def render_success_box(msg: str):
    """渲染成功提示"""
    st.success(f"✅ {msg}")


def render_info_box(msg: str):
    """渲染信息提示"""
    st.info(msg)
