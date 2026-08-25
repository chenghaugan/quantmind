"""风控中心：限额档位查看 / 委托预检试算 / 交易日历。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from utils.theme import (  # noqa: E402
    setup_page, page_header, section, note, verdict, guard_error,
    kpi_row, badge, fmt_num, fmt_pct, fmt_money, tone_of,
    divider, conn_bar, order_preview,
)
from utils.api_client import APIClient  # noqa: E402
from utils.constants import EXCHANGE_NAMES  # noqa: E402

setup_page("风控中心", "🛡️")
page_header(
    "风控中心",
    "实盘前可视化风控闸门：查看限额档位、试算委托是否会被拦截、查交易日历与交易时段。",
    "🛡️",
)

note(
    "下单前先在这里跑一遍 **委托预检**（`check_order`），就能看到会不会被哪一条限额拦下、"
    "估算的名义市值与保证金占用。`unlimited` 档仅供测试回放，禁止实盘。",
    "info",
)

# ================================================================= 档位限额
section("限额档位")
profiles = APIClient.risk_profiles()
if guard_error(profiles, "风控档位"):
    st.stop()

profile_list = profiles.get("profiles", [])
limit_labels = profiles.get("labels", {})
code_labels = profiles.get("codes", {})

def _fmt_limit(v):
    """把限额值转成可安全展示的字符串。"""
    if isinstance(v, (list, tuple)):
        if not v:
            return "—"
        return "、".join(str(x) for x in v)
    if isinstance(v, dict):
        return "、".join(f"{k}:{val}" for k, val in v.items()) or "—"
    if isinstance(v, set):
        return "、".join(str(x) for x in v) or "—"
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "是" if v else "否"
    return str(v)

tabs = st.tabs([f"{p['name']}" for p in profile_list] or ["默认"])
for i, p in enumerate(profile_list):
    with tabs[i]:
        st.markdown(f"**{p.get('label', p['name'])}**")
        limits = p.get("limits", {})
        rows = [{"限额项": limit_labels.get(k, k), "值": _fmt_limit(v)}
                for k, v in limits.items()]
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        else:
            st.caption("无限额配置。")

st.caption("拒单代码对照：" + " · ".join(
    f"{k}={v}" for k, v in list(code_labels.items())[:8]
))

# ================================================================= 委托预检
divider("委托预检")
section("委托预检（试算）")
cl, cr = st.columns([1, 2], gap="medium")
with cl:
    profile = st.selectbox("限额档位",
                           [p["name"] for p in profile_list] or ["default"],
                           format_func=lambda x: {
                               "default": "默认档", "conservative": "保守档",
                               "unlimited": "不限档(测试)"
                           }.get(x, x))
    vt_symbol = st.text_input("合约代码", "rb0.SHFE", help="格式 symbol.exchange")
    direction = st.selectbox("方向", ["多", "空"], index=0)
    offset = st.selectbox("开平", ["开", "平", "平今", "平昨"], index=0)
    volume = st.number_input("手数", value=1.0, min_value=0.0, step=1.0)
    price = st.number_input("价格", value=3500.0, step=1.0, help="0 表示市价")
    equity = st.number_input("账户权益", value=1_000_000.0, step=100_000.0, format="%f")
    position_volume = st.number_input("当前净持仓", value=0.0, step=1.0,
                                      help="正=多、负=空")
    last_price = st.number_input("最新价", value=3500.0, step=1.0)
    check_btn = st.button("🛡️ 试算风控", type="primary", width="stretch")

with cr:
    st.markdown("**试算说明**")
    st.markdown(
        "- 选择档位与委托参数后点击试算，得到 `PASS` 或具体拒单代码。\n"
        "- `context` 会返回合约乘数、名义市值、预估保证金占用率。\n"
        "- 下方还会展示当前档位下该合约的实际限额清单。"
    )
    if profile_list:
        cur = next((p for p in profile_list if p["name"] == profile), None)
        if cur:
            st.markdown(f"**当前档位：{cur.get('label', profile)}**")
            lim = cur.get("limits", {})
            st.caption(" · ".join(f"{limit_labels.get(k, k)}={_fmt_limit(v)}"
                                  for k, v in list(lim.items())[:6]))

if check_btn:
    payload = {
        "profile": profile, "vt_symbol": vt_symbol, "direction": direction,
        "offset": offset, "volume": volume, "price": price,
        "last_price": last_price, "equity": equity,
        "position_volume": position_volume,
        "check_session": False,
    }
    with st.spinner("正在试算风控闸门…"):
        res = APIClient.risk_check(payload)
    if guard_error(res, "风控试算"):
        st.stop()

    decision = res.get("decision", {})
    passed = decision.get("passed", False)
    code = decision.get("code", "")
    reason = decision.get("reason", "")
    ctx = res.get("context", {})
    stt = res.get("state", {})

    if passed:
        verdict(f"✅ 通过（{code}）：委托不会被拦截。", "ok", icon="✅")
    else:
        verdict(f"⛔ 被拦截（{code} · {code_labels.get(code, code)}）：{reason}",
                "bad", icon="⛔")

    kpi_row([
        {"label": "合约乘数", "value": ctx.get("contract_size"), "tone": "accent"},
        {"label": "名义市值", "value": "¥{:,}".format(ctx.get("notional", 0)),
         "tone": tone_of(ctx.get("notional"))},
        {"label": "预估保证金", "value": "¥{:,}".format(ctx.get("margin_estimate", 0)),
         "tone": "accent"},
        {"label": "保证金率", "value": fmt_num(ctx.get("margin_rate"), 2),
         "tone": "neutral"},
        {"label": "已用保证金", "value": "¥{:,}".format(stt.get("margin_used", 0)),
         "tone": "neutral"},
    ])
    with st.expander("🔎 风控试算原始返回", expanded=False):
        st.json(res)

# ================================================================= 交易日历
divider("交易日历")
section("交易日历与交易时段")
cld, clr = st.columns([1, 2], gap="medium")
with cld:
    csymbol = st.text_input("标的", "IF0", key="cal_symbol")
    cexch = st.selectbox("交易所", list(EXCHANGE_NAMES.keys()),
                         format_func=lambda x: f"{x} · {EXCHANGE_NAMES.get(x, '')}",
                         key="cal_exch")
    chorizon = st.slider("展望天数", 7, 30, 14)
    cal_btn = st.button("📅 查询日历", type="primary", width="stretch")

with clr:
    st.markdown("**说明**")
    st.markdown(
        "- 查询某标的当日是否为交易日、是否有夜盘、当前是否处于交易时段。\n"
        "- 列出未来 N 天的交易/休市安排，便于排布调仓与风控监控。"
    )

if cal_btn:
    with st.spinner("正在查询交易日历…"):
        cal = APIClient.risk_calendar(day=None, symbol=csymbol,
                                       exchange=cexch, horizon=chorizon)
    if guard_error(cal, "交易日历"):
        st.stop()

    is_td = cal.get("is_trading_day", False)
    has_night = cal.get("has_night_session", False)
    trading_now = cal.get("now_is_trading_time", False)

    # 连接状态条风格展示
    if is_td and trading_now:
        conn_bar(f"{cal.get('date')} · 交易中", "当前处于交易时段", "ok")
    elif is_td:
        conn_bar(f"{cal.get('date')} · 交易日", "当前非交易时段", "warn")
    else:
        conn_bar(f"{cal.get('date')} · 非交易日", "休市", "err")

    kpi_row([
        {"label": "下一交易日", "value": cal.get("next_trading_day") or "—",
         "tone": "accent"},
        {"label": "上一交易日", "value": cal.get("prev_trading_day") or "—",
         "tone": "accent"},
        {"label": "夜盘收盘", "value": cal.get("night_close") or "—", "tone": "neutral"},
        {"label": "节假日总数", "value": cal.get("holiday_count", 0),
         "tone": "neutral"},
    ])

    sessions = cal.get("day_sessions", [])
    if sessions:
        st.markdown("**日盘交易时段**")
        st.code(" · ".join(f"{a}–{b}" for a, b in sessions))

    upcoming = cal.get("upcoming", [])
    if upcoming:
        df = pd.DataFrame(upcoming)
        df["星期"] = df["weekday"].map({0: "一", 1: "二", 2: "三", 3: "四",
                                        4: "五", 5: "六", 6: "日"})
        df["交易日"] = df["is_trading_day"].map({True: "✅", False: "❌"})
        df["夜盘"] = df["has_night_session"].map({True: "🌙", False: "—"})
        st.dataframe(df[["date", "星期", "交易日", "夜盘"]],
                     width="stretch", hide_index=True)

st.caption("💡 实盘下单会经过同一套风控闸门：拒单不会抛异常，而是返回决策对象供网关处理。")
