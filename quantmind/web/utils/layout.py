"""布局组件"""

import streamlit as st


def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/chart.png", width=80)
        st.title("QuantMind")
        st.caption("AI 驱动量化投研框架")
        st.divider()
        st.markdown("### 导航")
        st.markdown("📊 仪表盘 - 系统概览")
        st.markdown("📈 行情数据 - 市场浏览")
        st.markdown("🔬 因子研究 - Alpha 评估")
        st.markdown("⚙️ 策略回测 - 历史验证")
        st.markdown("🤖 AI 研究 - 智能因子挖掘")
        st.markdown("🔄 生命周期 - 策略晋升")
        st.markdown("📡 实时监控 - 实盘管理")
        st.divider()
        st.markdown("### 系统状态")
        # 这里可以调用 /health 接口显示状态


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
