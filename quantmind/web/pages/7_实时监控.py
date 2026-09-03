"""实时监控与手动下单（WebSocket 事件流 + 结构化仪表）。

方案 C（WS 前端消费）：改用可复用的 ``utils/ws_client.WSClient``（基于已安装的
``websockets`` asyncio 客户端 + 后台线程 + 自动重连），从后端 ``/ws`` 接收实时
行情 / 信号 / 持仓 / 成交 / 风控事件，并在**结构化仪表**中即时刷新。

线程安全模型（已验证稳定）：后台 daemon 线程只把消息 **put 进 Queue**，主线程在
``@st.fragment(run_every=2)`` 里 **仅读 Queue、仅写 session_state**，避免跨线程
操作 Streamlit 内部状态导致崩溃。
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st  # noqa: E402

from utils.theme import (  # noqa: E402
    setup_page, page_header, section, note, badge, guard_error, kpi_row,
    conn_bar, order_preview, divider,
)
from utils.api_client import APIClient  # noqa: E402
from utils.ws_client import WSClient, WS_URL  # noqa: E402

setup_page("实时监控", "📡")
page_header(
    "实时监控",
    "通过 WebSocket 订阅后端事件总线：行情 / 信号 / 持仓 / 成交 / 风控事件即时刷新仪表；并支持手动下单试算。",
    "📡",
)

note(
    "**事件流说明**：连接后服务端先回 `hello` 握手；连接断开会自动按退避秒数重连。"
    "支持 `eTick / eBar / eSignal / eOrder / eTrade / ePosition / eAccount / eRisk / eLog`。",
    "info",
)

with st.expander("🔌 WebSocket 连接说明", expanded=False):
    st.markdown(
        f"**端点**：`{WS_URL}`\n\n"
        "**握手**：`{\"type\": \"hello\", \"msg\": \"QuantMind WebSocket 已连接\"}`\n\n"
        "**自动重连**：连接失败会按 1s→2s→…→30s 退避重连，直至手动断开。\n\n"
        "```python\nfrom utils.ws_client import WSClient, connect_ws\n"
        "client = connect_ws()\n# client.messages 读取事件流（Queue）\n```"
    )

# ===== Session State：后台线程只写 Queue，主线程只写 state =====
for key, default in [
    ("ws_client", None),
    ("ws_connected", False),
    ("ws_messages", []),          # 滚动原始事件日志
    ("ws_latest", {}),            # 结构化最新状态：{数据类型: {symbol: data}}
    ("ws_trades", []),            # 最近成交（eTrade）
    ("ws_risk_events", []),       # 最近风控事件（eRisk）
    ("ws_error", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def connect():
    """创建并启动 WSClient（后台线程自动重连）。"""
    client = WSClient(WS_URL)
    client.start()
    st.session_state.ws_client = client
    st.session_state.ws_connected = True
    st.session_state.ws_error = None
    st.session_state.ws_messages = []
    st.session_state.ws_latest = {}
    st.session_state.ws_trades = []
    st.session_state.ws_risk_events = []


def disconnect():
    st.session_state.ws_connected = False
    client = st.session_state.ws_client
    if client is not None:
        client.stop()
    st.session_state.ws_client = None


def _symbol_of(data) -> str:
    """从事件 data 里尽量取出完整标的标识（symbol.exchange，优先 vt_symbol）。"""
    if isinstance(data, dict):
        vt = data.get("vt_symbol")
        if vt:
            return str(vt)
        sym = data.get("symbol")
        exch = data.get("exchange")
        if sym and exch:
            return f"{sym}.{exch}"
        return str(sym or "")
    return ""


# ===== 连接控制 =====
c1, c2 = st.columns(2, gap="medium")
with c1:
    if st.button("🔌 连接 WebSocket", type="primary",
                 disabled=st.session_state.ws_connected, width="stretch"):
        try:
            connect()
        except Exception as e:
            st.session_state.ws_error = str(e)
            st.session_state.ws_connected = False
        st.rerun()
with c2:
    if st.button("⏹ 断开连接", disabled=not st.session_state.ws_connected, width="stretch"):
        disconnect()
        st.rerun()

client = st.session_state.ws_client
if st.session_state.ws_connected and client is not None:
    # 同步后台线程的连接状态（断线重连期间会回到 False）
    st.session_state.ws_connected = client.connected
    if client.error:
        st.session_state.ws_error = client.error
    if client.connected:
        conn_bar("WebSocket 已连接", WS_URL, "ok")
    else:
        conn_bar("连接中断，正在自动重连…", "退避重连中", "warn")
elif st.session_state.ws_error:
    conn_bar("连接错误", st.session_state.ws_error, "err")
else:
    conn_bar("未连接", "点击上方按钮开始接收实时事件", "warn")

st.markdown("---")

# ===== 结构化实时仪表（M2：即时刷新） =====
if st.session_state.ws_connected and client is not None:
    section("实时仪表", "从事件流即时刷新：行情 / 信号 / 持仓 / 成交 / 风控")


    @st.fragment(run_every=2)
    def dashboard():
        q = client.messages
        try:
            while True:
                m = q.get_nowait()
                t = m.get("type", "")
                d = m.get("data", m)
                # 原始日志（滚动，最多 200 条）
                st.session_state.ws_messages.append(m)
                # 结构化最新状态：eSignal 用 dict 全文，其余按 symbol 记最新
                if t == "eSignal":
                    st.session_state.ws_latest["eSignal"] = d
                elif t in ("eBar", "eTick", "ePosition"):
                    sym = _symbol_of(d)
                    if sym:
                        st.session_state.ws_latest.setdefault(t, {})[sym] = d
                elif t == "eTrade":
                    st.session_state.ws_trades.append(d)
                elif t == "eRisk":
                    st.session_state.ws_risk_events.append(d)
                elif t == "eAccount":
                    st.session_state.ws_latest["eAccount"] = d
        except Exception:
            pass

        # 裁剪
        if len(st.session_state.ws_messages) > 200:
            st.session_state.ws_messages = st.session_state.ws_messages[-200:]
        if len(st.session_state.ws_trades) > 50:
            st.session_state.ws_trades = st.session_state.ws_trades[-50:]
        if len(st.session_state.ws_risk_events) > 20:
            st.session_state.ws_risk_events = st.session_state.ws_risk_events[-20:]

        latest = st.session_state.ws_latest

        # ---- 账户卡（eAccount）----
        acct = latest.get("eAccount")
        if isinstance(acct, dict) and (acct.get("balance") is not None
                                       or acct.get("available") is not None):
            kpi_row([
                {"label": "账户余额", "value": acct.get("balance", 0.0), "tone": "accent"},
                {"label": "可用资金", "value": acct.get("available", 0.0)},
                {"label": "冻结", "value": acct.get("frozen", 0.0)},
            ])
            st.write("")

        # ---- 行情卡（eBar / eTick）----
        bars = latest.get("eBar") or {}
        ticks = latest.get("eTick") or {}
        if bars or ticks:
            rows = []
            for sym, b in bars.items():
                rows.append({
                    "合约": sym,
                    "开": b.get("open", ""),
                    "高": b.get("high", ""),
                    "低": b.get("low", ""),
                    "收": b.get("close", ""),
                    "量": b.get("volume", ""),
                })
            for sym, tk in ticks.items():
                rows.append({"合约": sym, "开": "", "高": "", "低": "",
                             "收": tk.get("last_price", ""), "量": tk.get("volume", "")})
            import pandas as pd
            st.markdown(badge(f"最新行情 {len(rows)}", "info"), unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        # ---- 信号（eSignal）----
        sig = latest.get("eSignal")
        if isinstance(sig, dict):
            target = sig.get("target")
            vt = sig.get("vt_symbol", "")
            tone = "up" if (target or 0) > 0 else "down" if (target or 0) < 0 else "neutral"
            kpi_row([{"label": f"信号 · {vt}", "value": target, "tone": tone}])
            st.write("")

        # ---- 持仓（ePosition）----
        positions = latest.get("ePosition") or {}
        if positions:
            import pandas as pd
            pos_rows = [{
                "合约": sym,
                "净持仓": p.get("volume", 0),
                "均价": p.get("price", ""),
                "浮动盈亏": p.get("pnl", ""),
            } for sym, p in positions.items() if p.get("volume") != 0]
            if pos_rows:
                st.markdown(badge(f"持仓 {len(pos_rows)}", "violet"), unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(pos_rows), width="stretch", hide_index=True)

        # ---- 最近成交（eTrade）----
        if st.session_state.ws_trades:
            import pandas as pd
            t_rows = [{
                "合约": (tr.get("symbol", "") if isinstance(tr, dict) else ""),
                "方向": (tr.get("direction", "") if isinstance(tr, dict) else ""),
                "价格": (tr.get("price", "") if isinstance(tr, dict) else ""),
                "手数": (tr.get("volume", "") if isinstance(tr, dict) else ""),
            } for tr in st.session_state.ws_trades[-10:]]
            st.markdown(badge("最近成交", "success"), unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(t_rows), width="stretch", height=220, hide_index=True)

        # ---- 最近风控（eRisk）----
        if st.session_state.ws_risk_events:
            st.markdown(badge(f"风控事件 {len(st.session_state.ws_risk_events)}", "danger"),
                        unsafe_allow_html=True)
            for ev in st.session_state.ws_risk_events[-5:]:
                st.caption(json.dumps(ev, ensure_ascii=False)[:160])


    dashboard()

st.markdown("---")

# ===== 实时事件流（原始日志） =====
section("实时事件流")


@st.fragment(run_every=2)
def live_feed():
    if not st.session_state.ws_connected:
        st.caption("未连接。点击上方「连接 WebSocket」开始接收事件。")
        return
    msgs = st.session_state.ws_messages
    if msgs:
        types = {}
        for m in msgs:
            t = m.get("type", "unknown")
            types[t] = types.get(t, 0) + 1
        cols = st.columns(min(len(types), 6) if types else 1)
        for i, (t, cnt) in enumerate(types.items()):
            with cols[i % len(cols)]:
                st.metric(t, cnt)
    else:
        st.caption("已连接，等待事件…")

    if msgs and st.button("🧹 清空事件", key="clear_ws"):
        st.session_state.ws_messages = []
        st.rerun()

    if msgs:
        rows = [
            {"类型": m.get("type", ""),
             "内容": json.dumps(m.get("data", m), ensure_ascii=False)[:200]}
            for m in reversed(msgs[-50:])
        ]
        st.dataframe(rows, width="stretch", height=360, hide_index=True)


live_feed()

st.markdown("---")

# ===== 手动下单 =====
section("手动下单")
cl, cr = st.columns([1, 2], gap="medium")
with cl:
    vt_symbol = st.text_input("合约代码", "rb0.SHFE", help="格式：symbol.exchange")
    direction = st.selectbox("方向", ["多", "空"], index=0)
    offset = st.selectbox("开平", ["开", "平", "平今", "平昨"], index=0)
    volume = st.number_input("手数", value=1, min_value=1, step=1)
    price = st.number_input("价格", value=0.0, step=0.01, help="0 表示市价单")
with cr:
    order_preview([
        f"<b>合约</b>：<code>{vt_symbol}</code>",
        f"<b>方向</b>：{'🔴 做多' if direction == '多' else '🟢 做空'}",
        f"<b>开平</b>：{offset}",
        f"<b>手数</b>：{volume}",
        f"<b>价格</b>：{'市价' if price == 0 else f'限价 {price}'}",
    ])

if st.button("🚀 提交订单", type="primary", width="stretch"):
    with st.spinner("正在提交订单…"):
        result = APIClient.order(vt_symbol, direction, volume, offset, price)
    if guard_error(result, "下单"):
        st.stop()
    if result.get("ok"):
        st.markdown(badge("已提交", "success"), unsafe_allow_html=True)
        st.json(result)
    else:
        st.markdown(badge("被拒绝", "danger"), unsafe_allow_html=True)
        st.error(result.get("msg", "未知错误"))

st.caption("💡 启动后端后点击「连接 WebSocket」接收实时事件；下单前建议先在「风控中心」试算是否被拦截。")

st.markdown("---")

# ===== 持仓与订单（实盘状态管理）=====
section("持仓与订单")
c1, c2 = st.columns(2, gap="medium")
with c1:
    if st.button("🔄 刷新持仓 / 订单", type="primary", width="stretch"):
        st.session_state.pop("qm_refresh", None)
        st.rerun()
with c2:
    st.caption("持仓与订单由后端内存台账维护，下单后刷新即可查看。")

# ---- 持仓表 ----
pos_res = APIClient.positions(timeout=10)
if guard_error(pos_res, "持仓查询"):
    st.stop()
positions = pos_res.get("positions", []) or []
if positions:
    import pandas as pd
    pos_rows = [{
        "合约": p.get("vt_symbol", ""),
        "净持仓": p.get("net_volume", 0),
        "更新": (p.get("updated") or "")[11:19],
    } for p in positions]
    st.markdown(badge(f"当前持仓 {len(positions)}", "info"), unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(pos_rows), width="stretch", hide_index=True)
else:
    st.markdown(badge("暂无持仓", "muted"), unsafe_allow_html=True)

st.write("")

# ---- 订单历史 + 撤单 ----
order_res = APIClient.orders(timeout=10)
if guard_error(order_res, "订单查询"):
    st.stop()
orders = order_res.get("orders", []) or []
if orders:
    import pandas as pd
    order_rows = [{
        "订单号": o.get("order_id", ""),
        "合约": o.get("vt_symbol", ""),
        "方向": o.get("direction", ""),
        "开平": o.get("offset", ""),
        "手数": o.get("volume", ""),
        "价格": o.get("price", ""),
        "状态": o.get("status", ""),
        "时间": (pd.to_datetime(o.get("datetime"), utc=True).tz_convert("Asia/Shanghai")
                 .strftime("%H:%M:%S") if o.get("datetime") else ""),
    } for o in orders]
    st.markdown(badge(f"订单 {len(orders)}", "violet"), unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(order_rows), width="stretch", height=280, hide_index=True)

    # 撤单选择器
    active = [o.get("order_id") for o in orders if o.get("status") != "已撤"]
    if active:
        st.write("")
        with st.form("cancel_form", border=True):
            ca, cb = st.columns([3, 1])
            with ca:
                cancel_id = st.selectbox("选择要撤销的订单", active)
            with cb:
                st.write("")
                cancel_btn = st.form_submit_button("❌ 撤单", width="stretch")
        if cancel_btn:
            cres = APIClient.cancel_order(cancel_id)
            if guard_error(cres, "撤单"):
                st.stop()
            if cres.get("ok"):
                st.toast(f"订单 {cancel_id} 已撤销", icon="✅")
            else:
                st.error(cres.get("msg", "撤单失败"))
            st.rerun()
else:
    st.markdown(badge("暂无订单", "muted"), unsafe_allow_html=True)
