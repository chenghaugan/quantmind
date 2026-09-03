"""AI 投资助手：右下角悬浮对话面板（弹窗式，参考主观投资-docker「AI 修正助手」）。

用法（放在页面 setup_page() 之后、任何 st.stop() 之前）::

    from utils.assistant_widget import render_assistant
    render_assistant()

特性：
- 关闭时：右下角悬浮 💬 气泡按钮（position:fixed，不占页面布局）；
- 打开后：固定右下角对话面板（标题栏 + 消息区 + 输入框 + 发送）；
- 自动携带当前页工作流上下文（策略思想 / 当前代码 / 最近回测摘要）；
- **执行过程可视化**：AI 调用工具（读回测/校验代码/查知识库/列策略库）时，
  实时渲染 🔧工具调用 + ↳结果摘要（对应参考实现的 tool_call/tool_result）；
- 🟢/🔴 连接状态灯；⚙️ 系统提示词查看/编辑/AI 自动再生成；
- ⏹ 停止取消运行；🗑️ 删除聊天记录；🆕 新建对话（归档当前，可回溯）；
- 超时提示：>90s 提醒检查配置；
- 回复含 ```python 代码块时可「📋 应用到策略代码区」（先沙箱校验）。
"""
from __future__ import annotations

import json
import re
import time

import streamlit as st
from streamlit.components.v1 import html as _components_html

from .api_client import APIClient

_MAX_HISTORY = 20        # 每次请求携带的最大对话条数
_POLL_SECONDS = 3        # 任务轮询间隔
_TIMEOUT_WARN_S = 90     # 超时提醒阈值（对齐参考实现的 90s 看门狗）

_CSS = """
<style>
/* 依据 Streamlit 1.62 真实 DOM：fragment 内容渲染为
   stLayoutWrapper > stVerticalBlock（无 stFragment testid）；
   用 "stLayoutWrapper 的直接子块且包含锚点" 唯一定位，避免命中主内容块 */

/* ---- 关闭态：右下角悬浮气泡 ---- */
div[data-testid="stLayoutWrapper"]:has(#qm-ai-closed) {
    height: 0 !important; min-height: 0 !important; overflow: visible !important;
}
div[data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlock"]:has(#qm-ai-closed) {
    position: fixed; right: 22px; bottom: 22px; z-index: 999999;
    width: auto !important; gap: 0 !important; margin: 0; padding: 0 !important;
    background: transparent; border: none;
}
div[data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlock"]:has(#qm-ai-closed) .stButton button {
    width: 56px !important; height: 56px !important; min-height: 56px !important;
    padding: 0; border-radius: 50%; font-size: 26px; line-height: 1;
    border: none; box-shadow: 0 4px 18px rgba(0,0,0,.28);
    background: var(--primary-color, #2f6bff); color: #fff;
    display: block; margin-left: auto;
}
div[data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlock"]:has(#qm-ai-closed) .stButton button:hover {
    transform: scale(1.06); box-shadow: 0 6px 22px rgba(0,0,0,.35);
}
div[data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlock"]:has(#qm-ai-closed) [data-testid="stElementContainer"] {
    width: 56px;
}

/* ---- 打开态：右下角固定对话面板 ---- */
div[data-testid="stLayoutWrapper"]:has(#qm-ai-open) {
    height: 0 !important; min-height: 0 !important; overflow: visible !important;
}
div[data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlock"]:has(#qm-ai-open) {
    position: fixed; right: 20px; bottom: 20px; top: auto; z-index: 999999;
    height: calc(100vh - 40px); max-height: none !important;
    display: flex; flex-direction: column; overflow: hidden;
    width: 520px; max-width: calc(100vw - 40px);
    background: var(--qm-surface, #0c1220);
    color: var(--qm-text, #e8edf5);
    border: 1px solid var(--qm-border, #1a2744); border-radius: 14px;
    box-shadow: 0 10px 36px rgba(0,0,0,.28);
    padding: 12px 14px !important; gap: .35rem !important;
    font-size: 14px;
}
/* 面板内直接子元素容器：默认不伸缩 */
div[data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlock"]:has(#qm-ai-open) > div[data-testid="stElementContainer"] {
    flex: 0 0 auto;
}
/* 消息容器：撑满中间，内部滚动 */
div[data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlock"]:has(#qm-ai-open) > div[data-testid="stLayoutWrapper"]:has(#qm-ai-msgs) {
    flex: 1 1 auto !important; min-height: 0 !important; overflow: hidden;
}
div[data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlock"]:has(#qm-ai-open) > div[data-testid="stLayoutWrapper"]:has(#qm-ai-msgs) > div[data-testid="stVerticalBlock"]:has(#qm-ai-msgs) {
    height: 100% !important; overflow-y: auto !important;
}
/* 面板内文字用主题浅色（fallback 白底时保持可读） */
div[data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlock"]:has(#qm-ai-open) [data-testid="stMarkdownContainer"],
div[data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlock"]:has(#qm-ai-open) [data-testid="stChatMessage"] {
    color: var(--qm-text, #e8edf5);
}
/* 压缩面板内标题字号 */
div[data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlock"]:has(#qm-ai-open) h1 { font-size: 1.3em; }
div[data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlock"]:has(#qm-ai-open) h2 { font-size: 1.2em; }
div[data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlock"]:has(#qm-ai-open) h3 { font-size: 1.1em; }
/* 面板内消息更紧凑 */
div[data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlock"]:has(#qm-ai-open) [data-testid="stChatMessage"] {
    padding-top: .4rem; padding-bottom: .4rem;
}
/* 隐藏锚点 */
#qm-ai-closed, #qm-ai-open { display: none; }
</style>
"""

# 空面板时的快捷提问
_QUICK_PROMPTS = [
    "🧠 解释当前策略代码的意图",
    "📼 最近回测结果靠谱吗？帮我解读",
    "🛠️ 如何改进这个策略？",
    "✏️ 给策略加一个 ATR 跟踪止损",
]

_TOOL_LABELS = {
    "read_backtest": "📕 读取回测结果",
    "validate_code": "🔍 沙箱校验代码",
    "search_knowledge": "📚 检索知识库",
    "list_strategies": "🗂️ 查询策略库",
}


def _result_summary() -> str:
    """把当前回测结果（session_state.val_result）压缩成文本摘要，控制 token。"""
    result = st.session_state.get("val_result")
    if not isinstance(result, dict) or not result:
        return ""
    lines = []
    if result.get("promoted"):
        lines.append(f"已达标入库：{result.get('promoted_symbols')}")
    for p in result.get("per_symbol") or []:
        sym = p.get("symbol", "?")
        iv = p.get("interval") or "1d"
        if p.get("error"):
            lines.append(f"- {sym}·{iv}：回测失败 {str(p['error'])[:60]}")
            continue
        r = p.get("report") or {}
        g = p.get("gate") or {}
        eq = p.get("equity_curve") or []
        days = len({str(e.get("date", ""))[:10] for e in eq})
        ntr = p.get("trades", 0)
        warns = []
        if days and days < 60:
            warns.append(f"仅{days}个交易日")
        if ntr < 20:
            warns.append(f"仅{ntr}笔")
        lines.append(
            f"- {sym}·{iv}：总收益 {r.get('total_return', 0):+.2%}，"
            f"年化 {r.get('annual_return', 0):+.2%}，Sharpe {r.get('sharpe', 0):.2f}，"
            f"回撤 {r.get('max_drawdown', 0):.2%}，胜率 {r.get('win_rate', 0):.0%}，"
            f"{ntr}笔，成本占比 {r.get('cost_ratio', 0):.1%}，"
            f"门槛 {g.get('status', '-')}"
            + (f"（样本警示：{'、'.join(warns)}）" if warns else ""))
    optim = result.get("optim")
    if optim:
        ratio = optim.get("is_ratio", 0.7) or 0.7
        lines.append(f"参数优化：试验 {optim.get('n_trials', 0)} 次，"
                     f"IS/OOS={ratio:.0%}/{1 - ratio:.0%}")
    return "\n".join(lines) if lines else ""


def _build_context() -> dict:
    """组装注入系统提示词的工作流上下文（思想 / 代码 / 回测摘要）。"""
    ctx = {}
    if st.session_state.get("val_idea"):
        ctx["idea"] = str(st.session_state.val_idea)[:800]
    if st.session_state.get("val_generated_code"):
        ctx["code"] = str(st.session_state.val_generated_code)[:6000]
    summary = _result_summary()
    if summary:
        ctx["result_summary"] = summary[:3000]
    return ctx


def _extract_last_code(text: str):
    blocks = re.findall(r"```(?:python|py)\n(.*?)```", text or "", re.S)
    return blocks[-1].strip() if blocks else None


def _fmt_tool_call(ev: dict) -> str:
    name = ev.get("name", "")
    label = _TOOL_LABELS.get(name, f"🔧 {name}")
    args = ev.get("input") or {}
    brief = ", ".join(f"{k}={str(v)[:30]}" for k, v in list(args.items())[:2])
    return f"🔧 {label}" + (f"（{brief}）" if brief else "")


def render_assistant() -> None:
    """渲染 AI 投资助手悬浮面板（在页面 setup_page() 之后调用）。"""
    st.markdown(_CSS, unsafe_allow_html=True)  # 幂等注入

    @st.fragment(run_every=_POLL_SECONDS)
    def _assistant_fragment():
        # ---- 首次渲染：从服务器恢复会话 ----
        if "ai_messages" not in st.session_state:
            st.session_state.ai_messages = []
            s = APIClient.get("/assistant/session", timeout=5) or {}
            if isinstance(s.get("messages"), list) and s["messages"]:
                st.session_state.ai_messages = s["messages"][-2 * _MAX_HISTORY:]

        # ---- 检测 AI 自动应用的代码 ----
        _pending = APIClient.get("/assistant/pending_code", timeout=3) or {}
        if _pending.get("code"):
            st.session_state.val_generated_code = _pending["code"]
            st.session_state.val_code_sandbox_ok = True
            st.session_state["code_check_done"] = "🤖 AI 助手已自动更新策略代码（沙箱校验通过）"
            # 删除 widget 状态，强制重新创建以显示新代码
            st.session_state.pop("val_code_editor", None)
            APIClient.delete("/assistant/pending_code", timeout=3)
            st.rerun()

        def _start_send(text: str) -> None:
            """发起一轮助手对话（后台任务 + 轮询）。"""
            history = [m for m in st.session_state.ai_messages
                       if m.get("role") in ("user", "assistant")][-_MAX_HISTORY:]
            st.session_state.ai_messages.append({"role": "user", "content": text})
            r = APIClient.post("/assistant/chat/start", json={
                "message": text, "history": history,
                "context": _build_context()}, timeout=30)
            tid = (r or {}).get("task_id")
            if tid:
                st.session_state["ai_task_id"] = tid
                st.session_state["ai_submitted_at"] = time.time()
            else:
                st.session_state.ai_messages.append(
                    {"role": "assistant",
                     "content": f"⚠️ 启动失败：{(r or {}).get('error', '未知错误')}"})
            st.session_state.pop("ai_input", None)

        # ================================================== 关闭态：悬浮气泡
        if not st.session_state.get("ai_open", False):
            st.markdown('<span id="qm-ai-closed" class="qm-ai-anchor"></span>',
                        unsafe_allow_html=True)
            if st.button("💬", key="ai_bubble", help="AI 投资助手"):
                st.session_state.ai_open = True
                st.rerun()
            return

        # ================================================== 打开态：对话面板
        st.markdown('<span id="qm-ai-open" class="qm-ai-anchor"></span>',
                    unsafe_allow_html=True)

        # 标题栏：状态灯 + 标题 + 🆕🗑️✕
        hc1, hc2, hc3, hc4 = st.columns([4, 1, 1, 1])
        with hc1:
            # 状态灯：LLM Provider 就绪为绿，否则红
            if "ai_llm_ok" not in st.session_state:
                st.session_state.ai_llm_ok = None
            if st.session_state.ai_llm_ok is None:
                st.session_state.ai_llm_ok = None
                _st = APIClient.get("/assistant/status", timeout=5) or {}
                st.session_state.ai_llm_ok = bool(_st.get("llm_ok"))
            dot = "🟢" if st.session_state.ai_llm_ok else "🔴"
            st.markdown(f"**💬 AI 投资助手** {dot}")
        with hc2:
            if st.button("🆕", key="ai_new", help="新建对话（归档当前）"):
                APIClient.post("/assistant/session/new", timeout=5)
                st.session_state.ai_messages = []
                st.session_state.pop("ai_task_id", None)
                st.rerun()
        with hc3:
            if st.button("🗑️", key="ai_clear", help="删除聊天记录"):
                APIClient.delete("/assistant/session", timeout=5)
                st.session_state.ai_messages = []
                st.session_state.pop("ai_task_id", None)
                st.rerun()
        with hc4:
            if st.button("✕", key="ai_close", help="收起面板"):
                st.session_state.ai_open = False
                st.rerun()

        # ⚙️ 系统提示词编辑（折叠区）
        with st.expander("🏷️ 系统提示词（AI 行为与知识）", expanded=False):
            # AI 更新完成后：刷新编辑器内容（text_area 的 key 状态需先弹出才能重置）
            if st.session_state.pop("ai_prompt_reload", False):
                _p = APIClient.get("/assistant/system_prompt", timeout=5) or {}
                st.session_state.ai_prompt_text = _p.get("content", "")
                st.session_state.pop("ai_prompt_editor", None)
            if "ai_prompt_text" not in st.session_state:
                _p = APIClient.get("/assistant/system_prompt", timeout=5) or {}
                st.session_state.ai_prompt_text = _p.get("content", "")
            st.caption("自定义提示词优先于内置默认；可手动编辑保存，或让 AI 自动再生成。")
            st.session_state.ai_prompt_text = st.text_area(
                "提示词", value=st.session_state.ai_prompt_text,
                height=160, key="ai_prompt_editor", label_visibility="collapsed")
            pc1, pc2 = st.columns(2)
            with pc1:
                if st.button("💾 保存", key="ai_prompt_save", width="stretch"):
                    APIClient.put("/assistant/system_prompt",
                                  json={"content": st.session_state.ai_prompt_text},
                                  timeout=10)
                    st.success("已保存")
            with pc2:
                if st.button("🤖 AI 更新", key="ai_prompt_regen", width="stretch",
                             help="让 AI 扫描平台知识重新生成提示词"):
                    r = APIClient.post("/assistant/system_prompt/regenerate",
                                       timeout=30)
                    tid = (r or {}).get("task_id")
                    if tid:
                        st.session_state["ai_regen_task_id"] = tid
                        st.session_state["ai_regen_submitted_at"] = time.time()
                        st.rerun()
                    else:
                        st.error(f"启动失败：{(r or {}).get('error', '未知错误')}")
            # 提示词再生成任务的轮询
            ptid = st.session_state.get("ai_regen_task_id")
            if ptid:
                s = APIClient.get(f"/assistant/chat/status/{ptid}", timeout=10) or {}
                if s.get("status") == "success":
                    st.session_state.ai_prompt_text = (s.get("result") or {}).get("content", "")
                    st.session_state.pop("ai_regen_task_id", None)
                    st.session_state["ai_prompt_reload"] = True
                    st.rerun()
                elif s.get("status") in ("error", "cancelled", "not_found"):
                    st.session_state.pop("ai_regen_task_id", None)
                    st.error(f"再生成失败：{s.get('message') or '任务丢失'}")

        # ---- 消息列表（独立滚动区） ----
        msgs = st.session_state.ai_messages
        with st.container():
            st.markdown('<span id="qm-ai-msgs"></span>', unsafe_allow_html=True)
            if not msgs:
                st.markdown(
                    "<div style='text-align:center;color:gray;padding:24px 8px'>"
                    "🙂 你好！我是 AI 投资助手<br>"
                    "<small>我会解释策略代码、解读回测结果、帮你修改策略</small>"
                    "</div>", unsafe_allow_html=True)
                for i, qp in enumerate(_QUICK_PROMPTS):
                    if st.button(qp, key=f"ai_quick_{i}", width="stretch"):
                        _start_send(qp)
                        st.rerun()
            for i, msg in enumerate(msgs):
                content = (msg.get("content") or "").strip()
                if not content:
                    continue
                is_last_assistant = (i == len(msgs) - 1 and msg.get("role") == "assistant")
                with st.chat_message("user" if msg.get("role") == "user" else "assistant"):
                    st.markdown(content)
                    # 最后一条助手回复含代码块时，提供一键应用（页面 24 语境）
                    code = _extract_last_code(content)
                    if (is_last_assistant and code
                            and "val_generated_code" in st.session_state):
                        if st.button("📋 应用到策略代码区（先沙箱校验）",
                                     key=f"ai_apply_{i}", width="stretch"):
                            _vr = APIClient.post("/strategy/draft/validate",
                                                 {"code": code}, timeout=10)
                            if _vr.get("ok"):
                                st.session_state.val_generated_code = code
                                st.session_state.val_code_sandbox_ok = True
                                st.session_state["code_check_done"] = (
                                    "✅ 助手修改的代码已通过沙箱校验并应用")
                                st.rerun()
                            else:
                                st.error(f"沙箱校验失败，未应用：{_vr.get('error', '未知错误')}")
            # 自动滚动到最新消息（同源 iframe 内脚本操作父页面消息区）
            _components_html(
                "<script>const el=parent.document.getElementById('qm-ai-msgs');"
                "if(el)el.scrollTop=el.scrollHeight;</script>", height=0)

        # ---- 消息区之后：运行中状态/输入区（固定在面板底部） ----
        if st.session_state.get("ai_task_id"):
            tid = st.session_state.get("ai_task_id")
            submitted = st.session_state.get("ai_submitted_at", time.time())
            s = APIClient.get(f"/assistant/chat/status/{tid}", timeout=10) or {}
            status = s.get("status")
            if status is None and s.get("error"):
                status = "not_found"
            if status == "success":
                st.session_state.ai_messages.append(
                    {"role": "assistant",
                     "content": (s.get("result") or {}).get("reply", "")})
                st.session_state.pop("ai_task_id", None)
                st.rerun()
            elif status in ("error", "cancelled", "not_found"):
                if status == "cancelled":
                    body = "⏹ 已停止本轮回答。"
                else:
                    body = f"⚠️ 出错了：{s.get('message') or s.get('error') or status}"
                st.session_state.ai_messages.append({"role": "assistant", "content": body})
                st.session_state.pop("ai_task_id", None)
                st.rerun()
            else:
                elapsed = int(time.time() - submitted)
                with st.chat_message("assistant"):
                    # 工具调用过程（🔧 调用 + ↳ 结果摘要）
                    for ev in (s.get("progress", {}).get("events") or [])[-8:]:
                        if ev.get("type") == "tool_call":
                            st.caption(_fmt_tool_call(ev))
                        elif ev.get("type") == "tool_result":
                            st.caption(f"　　:silver[↳] :green[{(ev.get('result') or '')[:80]}]")
                    if s.get("status") == "running":
                        st.caption(f"💭 思考中… 已运行 {elapsed}s")
                if elapsed >= _TIMEOUT_WARN_S:
                    st.caption("⏳ 已超过 90s：LLM 可能仍在执行多步工具；"
                               "若长时间无响应请点击 ⏹ 停止并检查 AI 配置。")
                c1, c2 = st.columns([3, 1])
                with c2:
                    if st.button("⏹ 停止", key="ai_stop", width="stretch"):
                        APIClient.post(f"/assistant/chat/cancel/{tid}", timeout=10)
                        st.session_state.pop("ai_task_id", None)
                        st.session_state.ai_messages.append(
                            {"role": "assistant", "content": "⏹ 已停止本轮回答。"})
                        st.rerun()

        # ---- 输入区 ----
        st.text_area("输入", height=68, key="ai_input",
                     placeholder="告诉 AI 要做什么，例如：解释当前策略代码…",
                     label_visibility="collapsed")
        send = st.button("➤ 发送", type="primary", width="stretch", key="ai_send")
        if send:
            text = (st.session_state.get("ai_input") or "").strip()
            if text:
                _start_send(text)
                st.rerun()

    _assistant_fragment()


def _fmt_tool_call(ev: dict) -> str:
    """把工具调用事件格式化为一行说明。"""
    name = ev.get("name", "")
    label = _TOOL_LABELS.get(name, f"🔧 {name}")
    args = ev.get("input") or {}
    try:
        brief = json.dumps(args, ensure_ascii=False)[:60]
    except Exception:  # noqa: BLE001
        brief = ""
    return f"{label} {brief}" if brief else label
