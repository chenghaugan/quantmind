"""仪表盘：系统健康度 / 数据源 / 策略 / 因子总览"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st  # noqa: E402

from utils.theme import (  # noqa: E402
    setup_page, page_header, section, kpi_row, badge, note, item_card, guard_error,
)
from utils.api_client import APIClient  # noqa: E402
from utils.charts import create_event_timeline  # noqa: E402


def _fmt_param(v):
    """将默认参数值格式化为可显示字符串（避免混合数字类型导致 pyarrow 转换失败）。"""
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "是" if v else "否"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


setup_page("仪表盘", "📊")
page_header("系统仪表盘", "后端健康度、数据源、策略与因子资产的一屏总览", "📊")

col_a, col_b = st.columns([1, 5])
with col_a:
    if st.button("🔄 刷新", width="stretch"):
        st.cache_data.clear()
        st.rerun()

health = APIClient.health(timeout=5)

# ---------------------------------------------------------------- 健康度
section("运行状态")
if guard_error(health, "后端健康检查"):
    note(
        "启动后端：<code>uvicorn quantmind.api.app:app --host 0.0.0.0 --port 8000</code>；"
        "或使用 <code>docker compose up -d</code> 一次拉起全部依赖。",
        "warning",
    )
    st.stop()

comps = health.get("components", {})
feeds = health.get("feeds", []) or []
factors_res = APIClient.factors(timeout=10)
strategies_res = APIClient.strategies(timeout=10)

factors = factors_res.get("factors", []) if isinstance(factors_res, dict) else []
strategies = strategies_res if isinstance(strategies_res, list) else []

kpi_row([
    {"label": "服务状态", "value": health.get("status", "?").upper(), "tone": "up",
     "hint": health.get("timestamp", "")[:19].replace("T", " ")},
    {"label": "数据源", "value": len(feeds), "tone": "accent", "hint": "已注册 Feed"},
    {"label": "内置因子", "value": len(factors), "tone": "accent", "hint": "单标的时序因子"},
    {"label": "可用策略", "value": len(strategies), "tone": "accent", "hint": "回测/模拟/实盘通用"},
])

st.write("")
c1, c2, c3 = st.columns(3, gap="small")
labels = {
    "data_manager": ("数据管理器", "💾"),
    "event_engine": ("事件引擎", "⚡"),
    "lifecycle": ("生命周期管理", "🔄"),
}
for col, (key, (label, icon)) in zip((c1, c2, c3), labels.items()):
    val = comps.get(key, "unknown")
    ok = val in ("active", "running")
    with col:
        with st.container(border=True):
            st.markdown(
                f"<div class='qm-mod'><span class='qm-mod-icon'>{icon}</span>"
                f"<span class='qm-mod-name'>{label}</span></div>"
                f"<div style='margin-top:.45rem'>"
                + badge("正常" if ok else val, "success" if ok else "danger")
                + "</div>",
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------- 数据源
section("数据源", f"共 {len(feeds)} 个已注册 Feed")
if feeds:
    FEED_DESC = {
        "mock": "离线模拟数据（无网络时兜底，保证流程可跑通）",
        "akshare": "AKShare · 期货 / A股 / 指数日线",
        "mootdx": "通达信本地行情",
        "yfinance": "Yahoo Finance · 海外与港股",
        "csv": "本地 CSV 目录（商品期货日线）",
        "parquet": "本地 Parquet 目录（A股 / 港股 / 期权）",
        "seat": "席位持仓数据（商品期货 F1-F8 因子）",
    }
    cols = st.columns(3, gap="small")
    for i, f in enumerate(feeds):
        desc = next((v for k, v in FEED_DESC.items() if k in f.lower()), "自定义数据源")
        with cols[i % 3]:
            st.markdown(item_card(f, desc, ["Feed"]), unsafe_allow_html=True)
else:
    note("未注册任何数据源，系统将使用内置 MockFeed 兜底。", "warning")

# ---------------------------------------------------------------- 策略
section("策略清单")
if guard_error(strategies_res, "策略清单查询"):
    pass
elif strategies:
    for s in strategies:
        with st.expander(f"⚙️ {s.get('name')} — {s.get('description', '')}", expanded=False):
            params = s.get("parameters", {}) or {}
            if params:
                st.dataframe(
                    [
                        {"参数": k, "默认值": _fmt_param(v)}
                        for k, v in params.items()
                    ],
                    width="stretch", hide_index=True,
                )
            else:
                st.caption("该策略无可调参数。")
            st.caption("在「策略回测」页可直接选用；在「参数优化」页可对其做网格寻优。")
else:
    note("暂无可用策略。", "warning")

# ---------------------------------------------------------------- 因子
section("因子资产", f"内置 {len(factors)} 个单标的因子")
if factors:
    cats = {}
    for f in factors:
        cats[f.get("category", "其他")] = cats.get(f.get("category", "其他"), 0) + 1
    left, right = st.columns([1, 1], gap="medium")
    with left:
        st.plotly_chart(
            create_event_timeline(cats, title="因子分类分布", height=300),
            width="stretch", key="factor_cat",
        )
    with right:
        st.markdown("**分类明细**")
        st.dataframe(
            [{"分类": k, "因子数": v} for k, v in sorted(cats.items(), key=lambda x: -x[1])],
            width="stretch", hide_index=True, height=300,
        )
    st.caption("完整因子检索与评估请前往「因子库」与「因子研究」页面。")
else:
    note("未加载到因子，请检查 FactorRegistry 注册是否正常。", "warning")
