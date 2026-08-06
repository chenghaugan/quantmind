"""QuantMind Web 控制台 - 首页

启动：``streamlit run quantmind/web/streamlit_app.py``
依赖后端：``uvicorn quantmind.api.app:app --port 8000``
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st  # noqa: E402

from utils.theme import (  # noqa: E402
    setup_page, page_header, section, kpi_row, badge, note, item_card,
)
from utils.api_client import APIClient  # noqa: E402

setup_page("首页", "🏠")

page_header(
    "QuantMind 控制台",
    "面向中国市场的 AI 驱动量化投研框架 · 商品期货 / 金融期货 / A股 / 港股 / 期权",
    "📐",
)

# ---------------------------------------------------------------- 后端状态
health = APIClient.health(timeout=4)
online = "error" not in health

if online:
    comps = health.get("components", {})
    feeds = health.get("feeds", [])
    kpi_row([
        {"label": "后端服务", "value": "在线", "tone": "up", "hint": health.get("timestamp", "")[:19].replace("T", " ")},
        {"label": "数据源", "value": len(feeds), "tone": "accent", "hint": "、".join(feeds[:3]) or "无"},
        {"label": "事件引擎", "value": "运行中" if comps.get("event_engine") == "running" else "已停止",
         "tone": "up" if comps.get("event_engine") == "running" else "down"},
        {"label": "生命周期", "value": "已激活" if comps.get("lifecycle") == "active" else "未激活",
         "tone": "up" if comps.get("lifecycle") == "active" else "neutral"},
    ])
else:
    note(
        "<b>后端未连接</b>：请先启动 API 服务 —— "
        "<code>uvicorn quantmind.api.app:app --host 0.0.0.0 --port 8000</code>"
        f"<br><span style='opacity:.7'>{health.get('error', '')}</span>",
        "error",
    )

# ---------------------------------------------------------------- 能力总览
section("功能模块", "按投研流水线组织，从数据到实盘全链路打通")

MODULES = [
    ("📊", "仪表盘", "系统健康度、数据源、策略与因子总览", "pages/1_仪表盘.py"),
    ("📈", "行情数据", "多市场 K 线查询、技术指标与导出", "pages/2_行情数据.py"),
    ("🧪", "数据质量", "间隙 / 异常尖峰 / 换月跳变 / 新鲜度体检", "pages/13_数据质量.py"),
    ("🗄️", "行情仓库总览", "本地 Parquet 缓存运维、覆盖区间可视化", "pages/19_行情仓库总览.py"),
    ("🔬", "因子研究", "单标的因子 IC / IR / 衰减 / 分位收益评估", "pages/3_因子研究.py"),
    ("🧬", "截面研究", "多标的面板 Alpha 因子与多空组合回测", "pages/12_截面研究.py"),
    ("🔎", "因子搜索", "co/ea/tot 三种算法迭代挖掘更强因子", "pages/17_因子搜索.py"),
    ("🧬", "因子挖掘流水线", "端到端：搜索→去冗余→OOS回测→复合alpha", "pages/18_因子挖掘流水线.py"),
    ("🚀", "端到端流水线", "Idea→证据→挖掘→OOS→代码→知识库一键跑通", "pages/20_端到端流水线.py"),
    ("📚", "因子库", "内置 Alpha101 / Alpha191 因子检索", "pages/9_FactorLibrary.py"),
    ("🤖", "AI 研究", "自然语言想法 → 假设 / 因子 / 策略代码", "pages/5_AI_研究.py"),
    ("⚙️", "策略回测", "回测 / 模拟 / 实盘三路线同一套策略代码", "pages/4_策略回测.py"),
    ("🎛️", "参数优化", "网格搜索寻优，附带过拟合提示", "pages/11_参数优化.py"),
    ("🔁", "Walk-Forward", "滚动窗口样本外验证与过拟合诊断", "pages/8_WalkForward.py"),
    ("🛡️", "风控中心", "限额档位、委托预检、交易日历", "pages/10_风控中心.py"),
    ("🔄", "生命周期", "IDEA → RESEARCH → … → LIVE 晋升闸门", "pages/6_生命周期.py"),
    ("📡", "实时监控", "WebSocket 事件流与手动下单", "pages/7_实时监控.py"),
    ("⚙️", "设置", "配置 AI 模型 API Key / Base URL / 模型", "pages/14_设置.py"),
]

cols = st.columns(3, gap="small")
for i, (icon, name, desc, target) in enumerate(MODULES):
    # Streamlit 多页路由：页面标题 = 去数字前缀的文件名（不含 .py）
    page_slug = re.sub(r"^\d+_", "", Path(target).stem)
    with cols[i % 3]:
        st.markdown(
            f"<a class='qm-card' href='/{page_slug}' target='_self'>"
            f"<span class='qm-mod-head'><span class='qm-mod-icon'>{icon}</span>"
            f"<span class='qm-mod-name'>{name}</span></span>"
            f"<span class='qm-mod-desc'>{desc}</span>"
            f"</a>",
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------- 快速上手
section("快速上手")

c1, c2 = st.columns(2, gap="medium")
with c1:
    st.markdown("**本地启动**")
    st.code(
        "# 1) 后端\n"
        "uvicorn quantmind.api.app:app --host 0.0.0.0 --port 8000\n\n"
        "# 2) 前端\n"
        "streamlit run quantmind/web/streamlit_app.py\n\n"
        "# 或一键容器化\n"
        "docker compose up -d",
        language="bash",
    )
with c2:
    st.markdown("**命令行通道**")
    st.code(
        "python -m quantmind.cli e2e        # 端到端冒烟\n"
        "python -m quantmind.cli risk       # 风控体检\n"
        "python -m quantmind.cli backtest --strategy dual_ma --symbol rb0\n"
        "python -m pytest                   # 全量测试",
        language="bash",
    )

section("投研流水线")
st.markdown(
    "<div class='qm-flow'>"
    + "".join(
        f"<span class='qm-flow-step'>{s}</span>"
        + ("<span class='qm-flow-arrow'>→</span>" if i < 5 else "")
        for i, s in enumerate(["IDEA", "RESEARCH", "BACKTEST", "PAPER", "APPROVED", "LIVE"])
    )
    + "</div>",
    unsafe_allow_html=True,
)
st.caption(
    "每一次状态跃迁都必须通过晋升闸门（夏普、回撤、模拟盘天数、风控评审），"
    "AI 生成的代码在落地前会先过 AST 沙箱校验。"
)
