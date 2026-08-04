"""实时监控与手动下单（真实 WebSocket 事件流）。

线程安全模型（已验证稳定）：后台 daemon 线程只把消息 **put 进 Queue**，
主线程在 ``@st.fragment(run_every=2)`` 里 **仅读 Queue、仅写 session_state**，
避免跨线程操作 Streamlit 内部状态导致崩溃。
"""
import sys
import json
import threading
import queue
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st  # noqa: E402

from utils.theme import (  # noqa: E402
    setup_page, page_header, section, note, badge, guard_error,
)
from utils.api_client import APIClient  # noqa: E402

WS_URL = "ws://127.0.0.1:8000/ws"

setup_page("实时监控", "📡")
page_header(
    "实时监控",
    "通过 WebSocket 真实连接后端事件总线，接收实时行情 / 信号 / 成交事件；并支持手动下单试算。",
    "📡",
)

note(
    "**事件流说明**：连接后服务端会先回 `hello` 握手；运行回测 / 模拟交易时策略事件经事件引擎广播到所有客户端。"
    "支持 `eTick / eBar / eSignal / eOrder / eTrade / ePosition / eAccount / eLog`。",
    "info",
)

with st.expander("🔌 WebSocket 连接说明", expanded=False):
    st.markdown(
        f"**端点**：`{WS_URL}`\n\n"
        "**握手**：`{\"type\": \"hello\", \"msg\": \"QuantMind WebSocket 已连接\"}`\n\n"
        "**回显测试**：发送任意 JSON 会被服务端原样回显。\n\n"
        "```python\nimport websocket\nws = websocket.WebSocketApp(\"" + WS_URL + "\")\nws.run_forever()\n```"
    )

# ===== Session State：后台线程只写 Queue，主线程只写 messages =====
for key, default in [
    ("ws", None),
    ("ws_connected", False),
    ("ws_queue", None),
    ("ws_messages", []),
    ("ws_error", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default if key != "ws_queue" else queue.Queue()


def _reader():
    ws = st.session_state.ws
    if ws is None:
        return
    ws.settimeout(1.0)
    while st.session_state.ws_connected:
        try:
            raw = ws.recv()
        except Exception:
            if st.session_state.ws_connected:
                continue
            break
        try:
            msg = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            msg = {"type": "raw", "data": raw}
        st.session_state.ws_queue.put(msg)


def connect():
    import websocket  # 延迟导入，无依赖时页面不崩

    ws = websocket.create_connection(WS_URL, timeout=10)
    st.session_state.ws = ws
    st.session_state.ws_connected = True
    st.session_state.ws_error = None
    st.session_state.ws_messages = []
    st.session_state.ws_queue = queue.Queue()
    threading.Thread(target=_reader, daemon=True).start()


def disconnect():
    st.session_state.ws_connected = False
    ws = st.session_state.ws
    if ws is not None:
        try:
            ws.close()
        except Exception:
            pass
    st.session_state.ws = None


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

if st.session_state.ws_connected:
    st.markdown(badge("已连接", "success"), unsafe_allow_html=True)
    st.caption(WS_URL)
elif st.session_state.ws_error:
    st.markdown(badge("连接错误", "danger"), unsafe_allow_html=True)
    st.caption(st.session_state.ws_error)

# ===== 发送测试消息 =====
if st.session_state.ws_connected:
    with st.form("ws_send_form", border=True):
        ca, cb = st.columns([4, 1])
        with ca:
            payload = st.text_input("发送测试消息 (JSON)", '{"action": "ping"}')
        with cb:
            st.write("")
            submitted = st.form_submit_button("发送", width="stretch")
    if submitted:
        try:
            st.session_state.ws.send(payload)
            st.toast("已发送，等待 echo 回包…")
        except Exception as e:
            st.error(f"发送失败：{e}")

st.markdown("---")

# ===== 实时事件流（自动刷新）=====
section("实时事件流")


@st.fragment(run_every=2)
def live_feed():
    q = st.session_state.ws_queue
    while not q.empty():
        st.session_state.ws_messages.append(q.get())
    msgs = st.session_state.ws_messages
    if len(msgs) > 200:
        st.session_state.ws_messages = msgs[-200:]
        msgs = st.session_state.ws_messages

    if msgs:
        types = {}
        for m in msgs:
            t = m.get("type", "unknown")
            types[t] = types.get(t, 0) + 1
        n = len(types)
        cols = st.columns(min(n, 6) if n else 1)
        for i, (t, cnt) in enumerate(types.items()):
            with cols[i % len(cols)]:
                st.metric(t, cnt)
    else:
        st.caption("尚无事件。连接后将自动收到 hello 握手；发送测试消息可触发 echo 回显。")

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
    st.markdown("**订单预览**")
    st.markdown(
        f"- **合约**：`{vt_symbol}`\n"
        f"- **方向**：{'🔴 做多' if direction == '多' else '🟢 做空'}\n"
        f"- **开平**：{offset}\n"
        f"- **手数**：{volume}\n"
        f"- **价格**：{'市价' if price == 0 else f'限价 {price}'}"
    )

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
        "时间": (o.get("datetime") or "")[11:19],
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
