"""因子库浏览页面：分类统计 + 搜索 + 详情卡片。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st  # noqa: E402

from utils.theme import (  # noqa: E402
    setup_page, page_header, section, note, kpi_row, badge,
)
from utils.api_client import APIClient  # noqa: E402

setup_page("因子库", "📚")
page_header(
    "因子库",
    "内置动量 / 均值回归 / 波动率 / 成交量 / 期货专属 / WorldQuant Alpha101·191 系列因子，覆盖多资产。",
    "📚",
)

note(
    "**因子库浏览**：QuantMind 内置多类因子。每个因子都可在「因子研究」直接评估 IC / IR / 衰减 / 分位收益。",
    "info",
)

# ----------------------------------------------------------------- 加载
@st.cache_data(ttl=60, show_spinner=False)
def load_factors():
    return APIClient.factors(timeout=10)

data = load_factors()
if isinstance(data, dict) and data.get("error"):
    note(f"未能从后端加载因子清单：{data['error']}。请确认 API 已启动。", "error")
    st.stop()

factors = (data.get("factors") if isinstance(data, dict) else None) or []
if not factors:
    note("未获取到因子列表，请确认后端服务已启动。", "warning")
    st.stop()

# ----------------------------------------------------------------- 概览
categories = {}
for f in factors:
    cat = f.get("category", "other")
    categories.setdefault(cat, []).append(f)

cat_names = {
    "momentum": "动量因子", "reversion": "均值回归因子", "volatility": "波动率因子",
    "volume": "成交量因子", "futures": "期货专属因子", "alpha101": "Alpha101 因子",
    "alpha191": "Alpha191 因子", "technical": "技术指标因子", "other": "其他因子",
}
section("概览")
kpi_row([
    {"label": "因子总数", "value": len(factors), "tone": "accent"},
    {"label": "因子类别", "value": len(categories), "tone": "accent"},
    {"label": "Alpha 系列",
     "value": len(categories.get("alpha101", [])) + len(categories.get("alpha191", [])),
     "tone": "accent"},
])

# ----------------------------------------------------------------- 筛选
section("筛选与搜索")
cf1, cf2 = st.columns([1, 2], gap="medium")
with cf1:
    selected_cats = st.multiselect(
        "筛选类别", list(categories.keys()),
        format_func=lambda c: f"{cat_names.get(c, c)} ({len(categories[c])})",
        default=list(categories.keys()),
    )
with cf2:
    search_text = st.text_input("搜索因子名称或描述", placeholder="输入关键词…")

displayed = []
for cat in selected_cats:
    for f in categories[cat]:
        name = f.get("name", "")
        desc = f.get("description", "")
        if search_text and search_text.lower() not in name.lower() and search_text.lower() not in desc.lower():
            continue
        displayed.append({**f, "category_cn": cat_names.get(cat, cat)})

if not displayed:
    note("没有匹配的因子。", "info")
    st.stop()

st.caption(f"共 {len(displayed)} 个因子")

# ----------------------------------------------------------------- 分组展示
grouped = {}
for f in displayed:
    grouped.setdefault(f["category_cn"], []).append(f)

for cat_cn, items in grouped.items():
    with st.expander(f"**{cat_cn}**　（{len(items)} 个）", expanded=True):
        for f in items:
            cols = st.columns([3, 1, 6])
            with cols[0]:
                st.markdown(f"**`{f.get('name', '')}`**")
            with cols[1]:
                params = f.get("params", {})
                window = params.get("window")
                st.markdown(f"窗口: {window}" if window else "—")
            with cols[2]:
                st.markdown(f.get("description", "无描述"))

# ----------------------------------------------------------------- 跳转
section("快速跳转")


def _safe_link(target, label):
    try:
        st.page_link(target, label=label)
    except Exception:
        st.markdown(f"**{label}** → `{target}`")


l1, l2, l3 = st.columns(3, gap="medium")
with l1:
    _safe_link("pages/3_因子研究.py", "🔬 因子研究 — 评估 IC/IR/衰减")
with l2:
    _safe_link("pages/8_WalkForward.py", "🔁 Walk-Forward — 滚动样本外")
with l3:
    _safe_link("pages/4_策略回测.py", "⚙️ 策略回测 — 构建组合")

st.caption("💡 点击因子名称可在因子研究页面直接评估其预测能力。")
