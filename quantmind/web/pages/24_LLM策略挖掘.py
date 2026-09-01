"""LLM策略挖掘：策略思想 → LLM 预编程 → 多品种回测 → 门槛 → 有效策略库。

核心范式（与「端到端流水线」区分）：
  - 端到端流水线：idea → 拆成**因子**（多品种截面挖掘）；
  - 本页面：完整的**策略思路**（如布林带回穿+止损规则）→ LLM 预编程为策略代码
    → 沙箱校验 → 在所选品种上逐品种回测对比 → 门槛判定 → 达标自动入有效策略库。
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.graph_objects as go
import streamlit as st

from utils.api_client import APIClient
from utils.theme import kpi_row, note, page_header, setup_page, verdict

# 历史运行记录存储路径
_HISTORY_FILE = Path(__file__).resolve().parent.parent.parent.parent / "data_cache" / "strategy_validation_runs.json"


def _load_history() -> list:
    """加载历史运行记录。"""
    try:
        if _HISTORY_FILE.exists():
            return json.loads(_HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return []


def _save_to_history(result: dict, idea: str, symbols: list, interval: str) -> None:
    """保存本次运行到历史记录。"""
    try:
        _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        history = _load_history()
        run_id = f"val_{int(time.time())}"
        entry = {
            "run_id": run_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "idea": idea[:200],
            "symbols": symbols,
            "interval": interval,
            "result": result,
        }
        history.insert(0, entry)  # 最新的在前
        history = history[:50]  # 最多保留 50 条
        _HISTORY_FILE.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        # 保存失败不影响主流程
        pass

setup_page("LLM策略挖掘", "🎯")
page_header(
    "LLM策略挖掘",
    "输入策略思想（自然语言规则），LLM 预编程为策略代码，在多品种真实数据上回测验证",
    "🎯",
)


def _render_optim_block(result: dict) -> None:
    """参数优化结果：IS/OOS 对比 + DSR/高原徽章（有 optim 字段才渲染）。"""
    optim = result.get("optim")
    if not optim:
        return
    st.markdown("### 🔍 参数优化详情（防过拟合三防线）")
    grid = optim.get("param_grid") or {}
    grid_txt = "，".join(f"{k}: {v}" for k, v in grid.items())
    ratio = optim.get("is_ratio", 0.7) or 0.7
    src_map = {"request": "请求指定", "code": "策略代码声明", "auto": "策略类自动推导",
               "builtin": "内置模板"}
    grid_src = src_map.get(optim.get("grid_source"), "自动")
    st.caption(
        f"试验数 **{optim.get('n_trials', 0)}** · IS/OOS = {ratio:.0%}/{1 - ratio:.0%} · "
        f"网格（{grid_src}）：{grid_txt or '自动'} · "
        f"入库判据：{'Deflated Sharpe ≥ 0.9' if optim.get('use_dsr') else 'OOS Sharpe'}")
    rows = []
    for p in result.get("per_symbol") or []:
        if p.get("error"):
            continue
        d = p.get("optim_detail") or {}
        if not d:
            continue
        is_s = d.get("is_sharpe") or 0.0
        oos_s = d.get("oos_sharpe") or 0.0
        decay = (oos_s / is_s) if is_s else 0.0
        dsr = d.get("dsr")
        plateau = d.get("plateau") or {}
        rows.append({
            "品种": p["symbol"],
            "最优参数": ", ".join(f"{k}={v}" for k, v in (d.get("best_combo") or {}).items()),
            "IS Sharpe": f"{is_s:.2f}",
            "OOS Sharpe": f"{oos_s:.2f}",
            "OOS保持度": f"{decay:.0%}",
            "DSR": (f"{dsr:.2f} {'✅' if dsr >= 0.9 else '⚠️'}") if dsr is not None else "-",
            "参数高原": ("✅ 高原" if plateau.get("ok") else "⚠️ 尖峰"),
        })
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.caption("解读：**OOS/IS 保持度**低 → 样本外衰减大；**DSR < 0.9** → 多次试验后"
                   "的 Sharpe 不可信；**尖峰** → 参数邻域表现差，最优只是样本内噪声。"
                   "三者任一不达标都不会入库。")
        # Top-K 组合明细
        for p in result.get("per_symbol") or []:
            d = (p.get("optim_detail") or {})
            top = d.get("top") or []
            if top:
                with st.expander(f"🗂️ {p['symbol']} · IS 段 Top-{len(top)} 组合", expanded=False):
                    st.dataframe(
                        [{"参数": ", ".join(f"{k}={v}" for k, v in t["combo"].items()),
                          "IS Sharpe": f"{t['is_sharpe']:.2f}"} for t in top],
                        use_container_width=True, hide_index=True)


def _render_history():
    """历史运行报告：双栏布局（左侧列表 + 右侧详情）+ 完整指标 + 可视化。"""
    st.markdown("#### 📜 历史运行报告")
    history = _load_history()
    if not history:
        note("暂无历史运行记录。跑完一次「LLM策略挖掘」即会留存。", "info")
        return
    
    st.caption(f"共 {len(history)} 条记录")
    
    # 顶部筛选区
    filter_cols = st.columns([2, 1])
    with filter_cols[0]:
        search_text = st.text_input("🔎 搜索策略思想", placeholder="输入关键词过滤...", key="val_hist_search")
    with filter_cols[1]:
        sort_by = st.selectbox("排序", ["时间倒序", "时间正序"], key="val_hist_sort")
    
    # 过滤和排序
    filtered = []
    for h in history:
        idea = h.get("idea") or ""
        if search_text and search_text.lower() not in idea.lower():
            continue
        filtered.append(h)
    
    if sort_by == "时间正序":
        filtered.sort(key=lambda x: x.get("created_at") or "")
    else:
        filtered.sort(key=lambda x: x.get("created_at") or "", reverse=True)
    
    st.caption(f"显示 {len(filtered)} / {len(history)} 条")
    
    if not filtered:
        note("无匹配记录，请调整筛选条件。", "info")
        return
    
    # ---- 记录列表：点击即查看详情，点击其他行切换 ----
    _rows = []
    for h in filtered:
        _result = h.get("result") or {}
        _per = _result.get("per_symbol") or []
        _verified = any((p.get("gate") or {}).get("status") == "verified" for p in _per)
        _errs = [p for p in _per if "error" in p]
        _status = "✅ 达标" if _verified else ("❌ 有错误" if _errs and len(_errs) == len(_per) else "◻ 待复核")
        _rows.append({
            "label": (f"🕘 {(h.get('created_at') or '')[:16]} │ 💡 {(h.get('idea') or '—')[:30]}"
                      f" │ 📊 {', '.join(h.get('symbols') or [])[:24]} · {h.get('interval') or '1d'}"
                      f" │ {_status}"),
            "run_id": h.get("run_id"),
        })

    _picked = st.radio(
        "运行记录（点击查看详情）",
        [r["label"] for r in _rows],
        key="hist_pick",
        label_visibility="collapsed",
    )
    selected_id = next((r["run_id"] for r in _rows if r["label"] == _picked),
                       None) if _picked is not None else None

    # ---- 批量删除（独立折叠区，不干扰点击查看）----
    _batch_done = st.session_state.pop("batch_delete_done", None)
    if _batch_done:
        st.success(_batch_done) if _batch_done.startswith("✅") else st.error(_batch_done)

    with st.expander(f"🗑️ 批量删除（当前 {len(filtered)} 条）", expanded=False):
        _labels = {f"{i+1}. {(h.get('created_at') or '')[:16]} · {(h.get('idea') or '—')[:30]}": h.get("run_id")
                   for i, h in enumerate(filtered)}
        _to_del = st.multiselect("选择要删除的记录", list(_labels.keys()), key="hist_del_multi")
        if st.button("🗑️ 删除所选", key="hist_del_btn", type="primary",
                     width="stretch", disabled=not _to_del):
            _ok, _fail = 0, 0
            for _label in _to_del:
                _rid = _labels[_label]
                try:
                    _r = APIClient.delete(f"/strategy/validate/history/{_rid}", timeout=10)
                    if _r.get("deleted_history") or _r.get("deleted_lifecycle"):
                        _ok += 1
                    else:
                        _fail += 1
                except Exception:
                    _fail += 1
            st.session_state.pop("val_hist_selected", None)
            st.session_state["batch_delete_done"] = (
                f"✅ 已删除 {_ok} 条记录" if not _fail
                else f"⚠️ 删除完成：成功 {_ok} 条，失败 {_fail} 条")
            st.rerun()

    # ---- 详情（点击表格行即展开）----
    if not selected_id:
        note("点击上方表格任意一行，即可查看完整运行详情。", "info")
        return

    selected = next((h for h in filtered if h.get("run_id") == selected_id), None)
    if not selected:
        note("未找到该记录", "warning")
        return

    result = selected.get("result") or {}
    per_symbol = result.get("per_symbol") or []
    llm_code = result.get("code") or ""

    # 概览
    st.markdown("### 📊 运行概览")
    idea_text = selected.get("idea") or "—"
    st.info(f"💡 **策略思想**：{idea_text}")

    kpi_row([
        {"label": "运行时间", "value": (selected.get("created_at") or "")[:19], "tone": "neutral"},
        {"label": "品种", "value": ", ".join(selected.get("symbols") or []), "tone": "accent"},
        {"label": "周期", "value": ", ".join(selected.get("intervals") or [selected.get("interval") or "1d"]), "tone": "neutral"},
        {"label": "策略", "value": result.get("strategy_desc") or result.get("strategy") or "LLM 预编程", "tone": "neutral"},
    ])

    # 多品种对比表
    if per_symbol:
        st.markdown("### 📊 多品种回测对比")
        rows = []
        for p in per_symbol:
            if "error" in p:
                rows.append({"品种": p["symbol"], "周期": p.get("interval", "-"), "K线": "-", "交易": "-", "总收益": "-",
                             "年化": "-", "Sharpe": "-", "回撤": "-", "门槛": f"❌ {p['error'][:20]}"})
                continue
            r = p.get("report") or {}
            g = p.get("gate") or {}
            _gm = g.get("metrics") or {}
            rows.append({
                "品种": p["symbol"],
                "周期": p.get("interval") or result.get("interval") or "1d",
                "K线": f"{p.get("bars", 0):,}",
                "交易": f"{p.get("trades", 0):,}",
                "总收益": f"{r.get('total_return', 0):+.2%}",
                "年化": f"{r.get('annual_return', 0):+.2%}",
                'Sharpe': f"{r.get('sharpe', 0):.2f}",
                "回撤": f"{r.get('max_drawdown', 0):.2%}",
                "成本/收益": (f"{g.get('metrics', {}).get('cost_ratio', 0):.1%}"
                             if g else f"{r.get('cost_ratio', 0):.1%}"),
                "总成本": f"{g.get('metrics', {}).get('total_cost') or r.get('total_cost', 0):,.0f}",
                "门槛": (g.get("status") or "-").upper(),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
    _render_optim_block(result)
    
    # 净值曲线
    curves = [(f'{p["symbol"]}·{p.get("interval") or result.get("interval") or "1d"}',
               p.get("equity_curve") or []) for p in per_symbol
              if not p.get("error") and p.get("equity_curve")]
    if curves:
        st.markdown("### 📈 净值曲线对比")
        fig = go.Figure()
        for sym, eq in curves:
            dates = [e.get("date") for e in eq]
            equity = [e.get("equity") for e in eq]
            fig.add_trace(go.Scatter(x=dates, y=equity, mode="lines", name=sym,
                                     line=dict(width=1.5)))
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10),
                          xaxis_title="日期", yaxis_title="净值")
        st.plotly_chart(fig, use_container_width=True)
    
    # LLM 生成代码
    if llm_code:
        with st.expander("📝 LLM 生成的策略代码", expanded=False):
            st.code(llm_code, language="python")
    
    # 门槛判定详情
    if result.get("gate_enabled"):
        st.markdown("### 🎯 门槛判定详情")
        for p in per_symbol:
            g = p.get("gate") or {}
            if not g:
                continue
            _m = g.get("metrics") or {}
            cols = st.columns([1, 3, 3, 3])
            cols[0].markdown(f"**{p['symbol']}**")
            cols[1].markdown(f"Sharpe {_m.get('sharpe')}")
            cols[2].markdown(f"回撤 {_m.get('max_drawdown')}")
            cols[3].markdown(f"判定 **{g.get('status','?').upper()}**")
            _cr = _m.get("cost_ratio")
            if _cr is not None:
                st.caption(f"成本/净收益 {_cr:.1%} · 总成本 {_m.get('total_cost', 0):,.0f}")
    
    # 原始数据
    with st.expander("🔎 原始数据", expanded=False):
        st.json(selected)



# 视图切换
_view = st.radio("视图", ["🚀 运行策略挖掘", "📜 历史运行报告"], horizontal=True, key="val_hist_view")
if _view == "📜 历史运行报告":
    _render_history()
    st.stop()

note(
    "**流程**：策略思想 → LLM 预编程（AST 沙箱校验）→ 逐品种真实回测 → "
    "门槛判定（Sharpe/回撤/成本占比）→ 达标自动入有效策略库（生命周期页可见）。<br>"
    "⚠️ 勾选「真实成本模型」后按品种差异化计费（手续费/平今/印花税/滑点），并拦截成本占比过高的高换手策略；"
    "参数未优化，结果仅供研究参考。",
    "info",
)



def _start_draft(history_msgs: list) -> None:
    """发起一轮 LLM 策略编程（生成/修改），进入草稿轮询。"""
    try:
        started = APIClient.post("/strategy/draft/start", json={
            "idea": idea.strip(), "history": history_msgs}, timeout=30)
    except Exception as exc:  # noqa: BLE001
        note(f"启动失败：{exc}", "error")
        st.stop()
    tid = (started or {}).get("task_id")
    if not tid:
        note((started or {}).get("error") or "未返回任务 ID", "error")
        st.stop()
    st.session_state["draft_task_id"] = tid
    st.session_state["draft_submitted_at"] = time.time()
    st.rerun()


@st.fragment(run_every=3)
def _draft_poll_fragment() -> None:
    """轮询 LLM 编程任务；完成后直接更新代码并整页刷新。"""
    tid = st.session_state.get("draft_task_id")
    submitted = st.session_state.get("draft_submitted_at", time.time())
    s = APIClient.get(f"/strategy/draft/status/{tid}", timeout=30)
    status = (s or {}).get("status")
    if status is None and (s or {}).get("error"):
        status = "not_found"
    if status == "success":
        res = (s or {}).get("result") or {}
        # 备份当前代码，供「恢复上一版」使用
        _prev = st.session_state.get("val_generated_code")
        if _prev:
            st.session_state["val_prev_code"] = _prev
        st.session_state["val_generated_code"] = res.get("code", "")
        st.session_state["val_code_sandbox_ok"] = bool(res.get("sandbox_ok"))
        if res.get("error"):
            st.session_state["draft_error"] = res["error"]
        _persist_workflow()
        st.session_state.pop("draft_task_id", None)
        # 一键运行：代码生成成功且通过沙箱 → 自动开始回测
        auto = st.session_state.pop("draft_auto_backtest", False)
        if auto and res.get("sandbox_ok") and not res.get("error"):
            st.session_state["val_auto_start"] = True
        st.rerun()
        return
    if status in ("error", "cancelled", "not_found"):
        st.session_state["draft_error"] = (s or {}).get("message") or f"任务{status}"
        st.session_state.pop("draft_task_id", None)
        st.rerun()
        return
    prog = (s or {}).get("progress") or {}
    elapsed = int(time.time() - submitted)
    st.progress(0.92, text=f"🧬 {prog.get('message') or 'LLM 编程中…'}"
                           f"（已运行 {elapsed}s，生成完整代码约需 10~60s，切页不中断）")

# ------------------------------------------------------------ 统一工作流：思想 → 代码 → 回测
# 第一性原理：session_state 易失（刷新/重连/重启即丢），以服务器端文件为唯一可信源。

# ---- 每次会话首次加载，从服务器恢复上次工作状态 ----
if "val_state_restored" not in st.session_state:
    st.session_state.val_state_restored = False
try:
    if not st.session_state.val_state_restored:
        st.session_state.val_state_restored = True
        _saved = APIClient.get("/strategy/draft/state", timeout=5) or {}
        if _saved.get("saved_at"):
            for _k in ("val_idea", "val_generated_code", "val_code_sandbox_ok",
                       "val_symbols", "val_intervals",
                       "use_cost_checkbox", "gate_enable_checkbox",
                       "val_gate_sharpe", "val_gate_mdd", "promote_checkbox",
                       "optim_enable_checkbox", "opt_ratio", "opt_topk", "opt_grid"):
                if _k not in st.session_state and _saved.get(_k) is not None:
                    st.session_state[_k] = _saved[_k]
            # 兼容旧格式：单周期字符串 → 多周期列表
            if "val_intervals" not in st.session_state and _saved.get("val_interval"):
                st.session_state.val_intervals = [_saved["val_interval"]]
            # 日期字段：ISO 字符串 → date 对象（date_input 要求 date 类型）
            for _k in ("val_start", "val_end"):
                _v = _saved.get(_k)
                if _k not in st.session_state and _v:
                    try:
                        st.session_state[_k] = datetime.strptime(str(_v)[:10], "%Y-%m-%d").date()
                    except (ValueError, TypeError):
                        pass
            # 兼容旧格式：代码摘要存在于 draft_pending 中（旧版中间态）
            if (st.session_state.val_generated_code is None
                    and (_saved.get("draft_pending") or {}).get("code")):
                st.session_state.val_generated_code = _saved["draft_pending"]["code"]
                st.session_state.setdefault("val_code_sandbox_ok",
                                            _saved["draft_pending"].get("sandbox_ok", True))
            st.info("🔄 已从服务器恢复上次的工作状态（策略思想/代码/参数）。")
except Exception:
    pass


def _persist_workflow():
    """把当前工作流状态保存到服务器（失败静默，不影响主流程）。"""
    def _jsonable(v):
        # date/datetime 对象无法 JSON 序列化，转 ISO 字符串
        return v.isoformat() if hasattr(v, "isoformat") else v
    try:
        APIClient.put("/strategy/draft/state", json={
            "val_idea": st.session_state.get("val_idea", ""),
            "val_generated_code": st.session_state.get("val_generated_code"),
            "val_code_sandbox_ok": st.session_state.get("val_code_sandbox_ok"),
            "val_symbols": st.session_state.get("val_symbols", []),
            "val_intervals": st.session_state.get("val_intervals", ["15m"]),
            "val_start": _jsonable(st.session_state.get("val_start")),
            "val_end": _jsonable(st.session_state.get("val_end")),
            "use_cost_checkbox": st.session_state.get("use_cost_checkbox", True),
            "gate_enable_checkbox": st.session_state.get("gate_enable_checkbox", True),
            "val_gate_sharpe": st.session_state.get("val_gate_sharpe", 1.0),
            "val_gate_mdd": st.session_state.get("val_gate_mdd", -0.15),
            "promote_checkbox": st.session_state.get("promote_checkbox", True),
            "optim_enable_checkbox": st.session_state.get("optim_enable_checkbox", False),
            "opt_ratio": st.session_state.get("opt_ratio", 0.7),
            "opt_topk": st.session_state.get("opt_topk", 5),
            "opt_grid": st.session_state.get("opt_grid", ""),
        }, timeout=5)
    except Exception:
        pass


def _clear_workflow():
    """清空会话内所有工作流状态，并删除服务器端持久化。"""
    for _k in ("val_idea", "val_generated_code", "val_code_sandbox_ok",
               "val_symbols", "val_intervals", "val_start", "val_end",
               "draft_history", "draft_task_id",
               "draft_submitted_at", "draft_error", "draft_feedback"):
        st.session_state.pop(_k, None)
    try:
        APIClient.delete("/strategy/draft/state", timeout=5)
    except Exception:
        pass


if "val_generated_code" not in st.session_state:
    st.session_state.val_generated_code = None
if "val_code_sandbox_ok" not in st.session_state:
    st.session_state.val_code_sandbox_ok = None

# ---- 策略思想（始终显示，可随时迭代） ----
if "val_idea" not in st.session_state:
    st.session_state.val_idea = (
        "布林带策略：收盘价跌破下轨后，5个交易日内收盘重新上穿下轨则买入做多；"
        "持多时收盘价跌破中轨卖出；止损5%。"
        "对称：收盘价突破上轨后，5个交易日内收盘跌回上轨下方则卖空；"
        "持空时收盘价上穿中轨平仓；止损5%。"
    )

idea = st.text_area(
    "💡 策略思想（自然语言规则，LLM 将编程实现）",
    height=120,
    placeholder="描述完整的交易规则：入场、离场、止损、参数…",
    key="val_idea",
    on_change=_persist_workflow,
)

# ---- 生成入口（始终显示；已有代码时为重新生成）----
_has_code_now = st.session_state.val_generated_code is not None
_g1, _g2 = st.columns(2)
with _g1:
    if st.button("🧬 重新生成策略代码" if _has_code_now else "🧬 生成策略代码",
                 type="primary", width="stretch"):
        if not idea.strip():
            note("请先填写策略思想。", "warning")
            st.stop()
        _start_draft([])
with _g2:
    if st.button("🚀 一键运行（生成 → 回测）", width="stretch"):
        if not idea.strip():
            note("请先填写策略思想。", "warning")
            st.stop()
        st.session_state["draft_auto_backtest"] = True
        _start_draft([])
if _has_code_now:
    st.caption("💡 重新生成将替换当前代码；旧代码会自动备份，可在代码区点「恢复上一版」找回。")

# ---- LLM 编程中：轮询进度 ----
if st.session_state.get("draft_task_id"):
    _draft_poll_fragment()
    st.stop()

# ---- 错误提示 ----
if st.session_state.get("draft_error"):
    note(f"上一轮失败：{st.session_state['draft_error']}", "error")
    st.session_state.pop("draft_error", None)

# ================================================================ 代码编辑器（手动编辑 + LLM 修改双通道）
_has_code = st.session_state.val_generated_code is not None

if _has_code:
    _sandbox_raw = st.session_state.get("val_code_sandbox_ok")
    _sandbox_ok = bool(_sandbox_raw)
    if _sandbox_raw is None:
        st.warning("✏️ 代码已恢复，尚未检验：请点击「🔍 检验修改」确认沙箱状态")
    elif _sandbox_ok:
        st.success("📝 策略代码（可手动编辑，或填写修改意见让 LLM 修改）")
    else:
        st.error("⚠️ 当前代码未通过沙箱校验，请修正后点击「🔍 检验修改」，"
                 "或填写修改意见让 LLM 修复")

    edited_code = st.text_area(
        "策略代码",
        value=st.session_state.val_generated_code,
        height=400,
        key="val_code_editor",
        label_visibility="collapsed",
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🔍 检验修改", type="secondary", width="stretch"):
            if edited_code != st.session_state.val_generated_code:
                try:
                    _r = APIClient.post("/strategy/draft/validate",
                                        {"code": edited_code}, timeout=10)
                    if _r.get("ok"):
                        st.session_state.val_generated_code = edited_code
                        st.session_state.val_code_sandbox_ok = True
                        _persist_workflow()
                        st.rerun()
                    else:
                        st.error(f"❌ 沙箱校验失败：{_r.get('error', '未知错误')}")
                except Exception as e:
                    st.error(f"检验请求失败：{e}")
            else:
                st.info("代码未修改")

    st.text_area(
        "💬 修改意见（让 LLM 按此修改当前代码）",
        height=70,
        key="draft_feedback",
        placeholder="例：止损改成 2% 固定止损；参数加一个回看窗口",
    )
    feedback = st.session_state.get("draft_feedback", "")
    with col2:
        if st.button("🧬 让 LLM 修改", width="stretch"):
            if not feedback.strip():
                st.warning("请先输入修改意见。")
            else:
                history = [
                    {"role": "assistant", "content": st.session_state.val_generated_code},
                    {"role": "user", "content": feedback.strip()}]
                _start_draft(history)
    with col3:
        if st.button("↩️ 恢复上一版", width="stretch",
                     disabled="val_prev_code" not in st.session_state,
                     help="恢复最近一次生成/修改前的代码备份"):
            st.session_state.val_generated_code, st.session_state["val_prev_code"] = (
                st.session_state["val_prev_code"], st.session_state.val_generated_code)
            st.session_state.val_code_sandbox_ok = None  # 备份代码需重新检验
            _persist_workflow()
            st.rerun()
    with col4:
        if st.button("🗑️ 清空重新开始", width="stretch"):
            _clear_workflow()
            st.rerun()

    # ================================================================ 阶段二：回测参数
    st.markdown("---")
    st.markdown("### 📊 选择回测参数")


    # 多品种选择
    st.markdown("**测试品种（多选，逐品种独立回测对比）**")
    if "val_symbols" not in st.session_state:
        st.session_state.val_symbols = ["IC0", "IF0"]
    symbols = st.multiselect(
        "标的",
        ["IC0", "IF0", "IH0", "IM0", "rb0", "cu0", "au0", "ag0",
         "m0", "ta0", "i0", "j0", "y0", "SR0", "CF0", "MA0"],
        key="val_symbols",
        on_change=_persist_workflow,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        if "val_intervals" not in st.session_state:
            st.session_state.val_intervals = ["15m"]
        intervals = st.multiselect(
            "周期（可多选对比）",
            ["1d", "1h", "30m", "15m", "5m", "1m"],
            key="val_intervals",
            on_change=_persist_workflow,
            help="选择多个周期时，逐品种×逐周期独立回测对比",
        )
    with col2:
        start = st.date_input("起始日期", value=None, format="YYYY-MM-DD", key="val_start",
                              on_change=_persist_workflow)
    with col3:
        end = st.date_input("结束日期", value=None, format="YYYY-MM-DD", key="val_end",
                            on_change=_persist_workflow)

    # 周期与策略类型匹配提示
    # 周期与策略类型匹配提示（多周期任选 1d 时提醒）
    if "1d" in intervals and set(intervals) != {"1d"}:
        st.caption("⚠️ 日线数据（1d）仅适用于日线级别策略（如持仓 N 天、日线交叉）。"
                   "如果策略包含日内逻辑（如 14:55 平仓、分钟级止损），请改用内日数据（1h/30m/15m/5m/1m）。")

    use_cost = st.checkbox("真实成本模型", value=True, key="use_cost_checkbox",
                           on_change=_persist_workflow,
                           help="按品种差异化费率/平今/印花税/滑点估算成本，并拦截成本占比过高的高换手策略")

    st.markdown("---")
    st.markdown("**🎯 门槛判定与有效策略库（可选）**")
    col4, col5, col6 = st.columns(3)
    with col4:
        gate_enable = st.checkbox("启用门槛判定", value=True, key="gate_enable_checkbox",
                                  on_change=_persist_workflow)
    with col5:
        gate_sharpe = st.number_input("最低 Sharpe", value=1.0, step=0.1,
                                      format="%.2f", key="val_gate_sharpe",
                                      on_change=_persist_workflow)
    with col6:
        gate_mdd = st.number_input("最大回撤下限", value=-0.15, step=0.05,
                                   format="%.2f", key="val_gate_mdd",
                                   on_change=_persist_workflow)
    promote = st.checkbox("达标自动注册到有效策略库（生命周期）", value=True, key="promote_checkbox",
                          on_change=_persist_workflow)

    st.markdown("---")
    st.markdown("**🔍 参数优化（防过拟合三防线，可选）**")
    optim_enable = st.checkbox(
        "启用网格参数优化",
        value=False,
        key="optim_enable_checkbox",
        on_change=_persist_workflow,
        help="IS/OOS 时间切分：网格只在样本内(IS)搜索，样本外(OOS)仅验证 Top-K 组合；"
             "另加参数高原检验与 Deflated Sharpe 多重检验校正，防止挑出来的参数是过拟合噪声")
    optim_grid_text = ""
    if optim_enable:
        _oc1, _oc2 = st.columns(2)
        with _oc1:
            optim_ratio = st.slider("样本内(IS)占比", 0.5, 0.9, 0.7, 0.05,
                                    key="opt_ratio", on_change=_persist_workflow)
        with _oc2:
            optim_topk = st.number_input("OOS 验证 Top-K", 1, 10, 5,
                                         key="opt_topk", on_change=_persist_workflow)
        optim_grid_text = st.text_area(
            "参数网格（JSON；留空 = 自动推导：请求显式 > LLM 代码 PARAM_GRID > 策略默认参数邻域 > 内置模板）",
            value='{"window": [10, 20, 30], "threshold": [0.02, 0.03, 0.04]}',
            height=80, key="opt_grid", on_change=_persist_workflow)

    # 开始回测（可由按钮或一键运行触发）
    def _do_start_backtest():
        if not symbols:
            note("请至少选择一个测试品种。", "warning")
            st.stop()

        # 调用回测 API
        approved_code = st.session_state.val_generated_code
        idea = st.session_state.get("val_idea", "")

        st.session_state["val_result"] = None
        _exch = "CFFEX" if all(s in ("IC0", "IF0", "IH0", "IM0") for s in symbols) else "SHFE"
        payload = {
            "idea": idea.strip(),
            "code": approved_code or "",
            "symbols": symbols,
            "exchange": _exch,
            "intervals": intervals,
            "interval": intervals[0] if intervals else "1d",
            "start": start.isoformat() if start else None,
            "end": end.isoformat() if end else None,
            "cost": use_cost,
            "gate": ({"min_sharpe": gate_sharpe, "min_drawdown": gate_mdd}
                     if gate_enable else None),
            "promote": bool(gate_enable and promote),
        }
        if optim_enable:
            try:
                _grid = json.loads(optim_grid_text) if optim_grid_text.strip() else {}
                if not isinstance(_grid, dict):
                    _grid = {}
            except json.JSONDecodeError:
                note("参数网格 JSON 解析失败，将使用自动网格。", "warning")
                _grid = {}
            payload["optimization"] = {
                "enabled": True,
                "param_grid": _grid,
                "is_ratio": float(optim_ratio),
                "top_k": int(optim_topk),
            }

        # 记下本次上下文，供完成时写入历史
        st.session_state["val_ctx"] = {"idea": idea.strip(), "symbols": list(symbols),
                                       "interval": "+".join(intervals)}
        try:
            _started = APIClient.post("/strategy/validate/start", json=payload, timeout=30)
        except Exception as exc:  # noqa: BLE001
            st.session_state["val_result"] = {"error": f"启动失败：{exc}"}
            st.rerun()
        tid = (_started or {}).get("task_id")
        if not tid:
            st.session_state["val_result"] = {"error": (_started or {}).get("error") or "未返回任务 ID"}
            st.rerun()
        st.session_state["val_task_id"] = tid
        st.session_state["val_submitted_at"] = time.time()
        st.rerun()

    # 一键运行：代码生成完成后自动开始回测
    if st.session_state.pop("val_auto_start", False):
        _do_start_backtest()

    # 开始回测按钮
    if st.button("🧭 开始回测", type="primary", width="stretch"):
        _do_start_backtest()









@st.fragment(run_every=3)
def _val_poll_fragment() -> None:
    tid = st.session_state.get("val_task_id")
    submitted = st.session_state.get("val_submitted_at", time.time())
    s = APIClient.get(f"/strategy/validate/status/{tid}", timeout=30)
    status = (s or {}).get("status")
    if status is None and (s or {}).get("error"):
        status = "not_found"  # 404 被 APIClient 转成 {"error"}，按任务丢失处理
    if status == "success":
        res = (s or {}).get("result") or {}
        st.session_state["val_result"] = res
        st.session_state.pop("val_task_id", None)
        st.session_state.pop("val_submitted_at", None)
        _ctx = st.session_state.get("val_ctx") or {}
        _save_to_history(res, _ctx.get("idea", ""), _ctx.get("symbols", []),
                         _ctx.get("interval", "1d"))
        st.rerun()
        return
    if status in ("error", "cancelled", "not_found"):
        st.session_state["val_result"] = {
            "error": (s or {}).get("message") or f"任务{status}"}
        st.session_state.pop("val_task_id", None)
        st.rerun()
        return
    prog = (s or {}).get("progress") or {}
    elapsed = int(time.time() - submitted)
    if prog.get("total"):
        # 真实进度：后端任务汇报的阶段/步数（LLM 编程 → 逐品种回测 → OOS 验证）
        frac = min(0.98, max(0.02, float(prog["current"]) / float(prog["total"])))
        st.progress(frac, text=f"{prog.get('message') or '运行中…'}（{prog['current']}/{prog['total']}，"
                               f"后台任务已运行 {elapsed}s，切页不中断）")
    else:
        # 回退：阶段未知时按时间估算
        frac = min(0.95, elapsed / 1800.0)
        st.progress(frac, text="LLM 编程 + 多品种回测中…（后台任务，切页不中断；"
                                f"页面每 3s 自动刷新，已运行 {elapsed}s）")

_val_tid = st.session_state.get("val_task_id")
if _val_tid:
    _val_poll_fragment()
    st.stop()

result = st.session_state.get("val_result")
if result is None:
    st.stop()
if not isinstance(result, dict):
    result = {}
if "error" in result:
    note(f"❌ {result['error']}", "error")
    st.stop()
per_symbol = result.get("per_symbol") or []
llm_code = result.get("code") or ""

# ------------------------------------------------------------ 结论
verified = [p for p in per_symbol if (p.get("gate") or {}).get("status") == "verified"]
if result.get("promoted"):
    verdict(f"✅ 门槛达标并已入有效策略库：{result.get('promoted_symbols')}", "ok", icon="✅")
elif result.get("gate_enabled") and verified:
    verdict(f"✅ 部分品种达标：{[p['symbol'] for p in verified]}（未勾选自动入库）",
            "ok", icon="✅")
elif result.get("gate_enabled"):
    verdict("❌ 全部品种未通过门槛（Sharpe/回撤不达标）", "warn", icon="❌")
else:
    verdict(f"回测完成：{result.get('strategy_desc') or result.get('strategy')}",
            "info", icon="📊")

# ------------------------------------------------------------ 多品种对比表
rows = []
for p in per_symbol:
    if "error" in p:
        rows.append({"品种": p["symbol"], "周期": p.get("interval", "-"), "K线": "-", "交易": "-", "总收益": "-",
                     "年化": "-", "Sharpe": "-", "回撤": "-", "门槛": f"❌ {p['error'][:20]}"})
        continue
    r = p.get("report") or {}
    g = p.get("gate") or {}
    rows.append({
        "品种": p["symbol"],
        "周期": p.get("interval") or result.get("interval") or "1d",
        "K线": f"{p.get("bars", 0):,}",
        "交易": f"{p.get("trades", 0):,}",
        "总收益": f"{r.get('total_return', 0):+.2%}",
        "年化": f"{r.get('annual_return', 0):+.2%}",
        'Sharpe': f"{r.get('sharpe', 0):.2f}",
        "回撤": f"{r.get('max_drawdown', 0):.2%}",
        "门槛": (g.get("status") or "-").upper(),
    })
if rows:
    st.markdown("### 📊 多品种回测对比")
    st.dataframe(rows, use_container_width=True)

# 参数优化详情（IS/OOS + DSR + 高原）
_render_optim_block(result)

# ------------------------------------------------------------ 净值曲线（各品种叠加）
curves = [(f'{p["symbol"]}·{p.get("interval") or result.get("interval") or "1d"}',
          p.get("equity_curve") or []) for p in per_symbol
         if not p.get("error") and p.get("equity_curve")]
if curves:
    fig = go.Figure()
    for sym, eq in curves:
        dates = [e.get("date") for e in eq]
        equity = [e.get("equity") for e in eq]
        fig.add_trace(go.Scatter(x=dates, y=equity, mode="lines", name=sym,
                                 line=dict(width=1.5)))
    fig.update_layout(
        title="净值曲线对比（各品种独立回测）",
        xaxis_title="日期", yaxis_title="净值（初始 100 万）", height=420,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------ LLM 生成代码
if llm_code:
    with st.expander("📝 LLM 生成的策略代码（沙箱已校验）", expanded=False):
        st.code(llm_code, language="python")
        st.caption("⚠️ 请核对代码是否忠实于你的策略思想（LLM 可能理解偏差）。")

# ------------------------------------------------------------ 门槛判定详情
if result.get("gate_enabled"):
    st.markdown("### 🎯 门槛判定详情")
    for p in per_symbol:
        g = p.get("gate") or {}
        if not g:
            continue
        _m = g.get("metrics") or {}
        tone = "up" if g.get("status") == "verified" else "down"
        cols = st.columns([1, 3, 3, 3, 3])
        cols[0].markdown(f"**{p['symbol']}**")
        cols[1].markdown(f"Sharpe {_m.get('sharpe')}")
        cols[2].markdown(f"回撤 {_m.get('max_drawdown')}")
        cols[3].markdown(f"判定 **{g.get('status','?').upper()}**")
        cols[4].caption(g.get("reason", "")[:60])
    if result.get("promote_error"):
        note(f"⚠️ 判定通过但自动入库失败：{result['promote_error']}", "warning")

st.caption(f"idea：{result.get('idea', '')[:80]}…" if len(result.get("idea", "")) > 80
           else f"idea：{result.get('idea', '')}")
