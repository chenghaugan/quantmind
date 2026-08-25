"""LLM策略挖掘：策略思想 → LLM 预编程 → 多品种回测 → 门槛 → 有效策略库。

核心范式（与「端到端流水线」区分）：
  - 端到端流水线：idea → 拆成**因子**（多品种截面挖掘）；
  - 本页面：完整的**策略思路**（如布林带回穿+止损规则）→ LLM 预编程为策略代码
    → 沙箱校验 → 在所选品种上逐品种回测对比 → 门槛判定 → 达标自动入有效策略库。

预置模板（动量/缠论3买/缠论1买/布林/双均线）保留为快捷入口，可折叠展开。
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
    
    # 双栏布局
    list_col, detail_col = st.columns([1, 2])
    
    with list_col:
        st.markdown("**运行列表**")
        for h in filtered:
            idea = (h.get("idea") or "—")[:50]
            created = (h.get("created_at") or "")[:16]
            symbols = h.get("symbols") or []
            interval = h.get("interval") or "1d"
            result = h.get("result") or {}
            per_symbol = result.get("per_symbol") or []
            
            # 统计达标情况
            verified = [p for p in per_symbol if (p.get("gate") or {}).get("status") == "verified"]
            status_icon = "✅" if verified else "❌"
            
            btn_label = f"{status_icon} {created}\n{idea}\n{', '.join(symbols[:3])} · {interval}"
            btn_key = f"val_hist_{h.get('run_id')}"
            if st.button(btn_label, key=btn_key, use_container_width=True):
                st.session_state["val_hist_selected"] = h.get("run_id")
    
    with detail_col:
        selected_id = st.session_state.get("val_hist_selected")
        if not selected_id:
            note("点击左侧运行记录查看详情", "info")
            return
        
        # 找到选中的记录
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
            {"label": "周期", "value": selected.get("interval") or "1d", "tone": "neutral"},
            {"label": "策略", "value": result.get("strategy_desc") or result.get("strategy") or "LLM 预编程", "tone": "neutral"},
        ])
        
        # 多品种对比表
        if per_symbol:
            st.markdown("### 📊 多品种回测对比")
            rows = []
            for p in per_symbol:
                if "error" in p:
                    rows.append({"品种": p["symbol"], "K线": "-", "交易": "-", "总收益": "-",
                                 "Sharpe": "-", "回撤": "-", "门槛": f"❌ {p['error'][:20]}"})
                    continue
                r = p.get("report") or {}
                g = p.get("gate") or {}
                rows.append({
                    "品种": p["symbol"],
                    "K线": p.get("bars", 0),
                    "交易": p.get("trades", 0),
                    "总收益": f"{r.get('total_return', 0):+.2%}",
                    "Sharpe": f"{r.get('sharpe', 0):.2f}",
                    "回撤": f"{r.get('max_drawdown', 0):.2%}",
                    "门槛": (g.get("status") or "-").upper(),
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)
        
        # 净值曲线
        curves = [(p["symbol"], p.get("equity_curve") or []) for p in per_symbol
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
        
        # 原始数据
        with st.expander("🔎 原始数据", expanded=False):
            st.json(selected)


# 视图切换
_view = st.radio("视图", ["🚀 运行策略挖掘", "📜 历史运行报告"], horizontal=True)
if _view == "📜 历史运行报告":
    _render_history()
    st.stop()

note(
    "**流程**：策略思想 → LLM 预编程（AST 沙箱校验）→ 逐品种真实回测 → "
    "门槛判定（Sharpe/回撤）→ 达标自动入有效策略库（生命周期页可见）。<br>"
    "⚠️ 回测默认不含交易成本；参数未优化，结果仅供研究参考。",
    "info",
)

# ------------------------------------------------------------ 策略思想输入
idea = st.text_area(
    "💡 策略思想（自然语言规则，LLM 将编程实现）",
    value=(
        "布林带策略：收盘价跌破下轨后，5个交易日内收盘重新上穿下轨则买入做多；"
        "持多时收盘价跌破中轨卖出；止损5%。"
        "对称：收盘价突破上轨后，5个交易日内收盘跌回上轨下方则卖空；"
        "持空时收盘价上穿中轨平仓；止损5%。"
    ),
    height=120,
    placeholder="描述完整的交易规则：入场、离场、止损、参数…",
)

with st.expander("⚙️ 参数设置", expanded=True):
    # 策略来源：LLM 编程为主，预置模板为快捷方式
    use_template = st.toggle("使用预置模板（快捷方式，跳过 LLM 编程）", value=False)
    if use_template:
        template = st.selectbox(
            "预置模板",
            ["momentum", "chan_first_buy", "chan_third_buy", "bollinger_recover", "dual_ma"],
            index=0,
        )
    else:
        template = ""

    # 多品种选择
    st.markdown("**📊 测试品种（多选，逐品种独立回测对比）**")
    symbols = st.multiselect(
        "标的",
        ["IC0", "IF0", "IH0", "IM0", "rb0", "cu0", "au0", "ag0",
         "m0", "ta0", "i0", "j0", "y0", "SR0", "CF0", "MA0"],
        default=["IC0", "IF0"],
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        interval = st.selectbox("周期", ["1d", "1h", "30m", "15m", "5m", "1m"], index=0)
    with col2:
        start = st.date_input("起始日期", value=None, format="YYYY-MM-DD", key="val_start")
    with col3:
        end = st.date_input("结束日期", value=None, format="YYYY-MM-DD", key="val_end")
    use_cost = st.checkbox("真实成本模型", value=False,
                           help="按品种差异化费率/平今/保证金估算成本")

    st.markdown("---")
    st.markdown("**🎯 门槛判定与有效策略库（可选）**")
    col4, col5, col6 = st.columns(3)
    with col4:
        gate_enable = st.checkbox("启用门槛判定", value=True)
    with col5:
        gate_sharpe = st.number_input("最低 Sharpe", value=1.0, step=0.1,
                                      format="%.2f", key="val_gate_sharpe")
    with col6:
        gate_mdd = st.number_input("最大回撤下限", value=-0.15, step=0.05,
                                   format="%.2f", key="val_gate_mdd")
    promote = st.checkbox("达标自动注册到有效策略库（生命周期）", value=True)

run_btn = st.button("🚀 运行策略挖掘", type="primary", width="stretch")

if not run_btn and not st.session_state.get("val_result"):
    note("填写策略思想与品种后点击运行，结果展示：<b>LLM 生成代码</b> / "
         "<b>多品种回测对比</b> / <b>门槛判定</b>。", "info")
    st.stop()

if run_btn:
    if not symbols:
        note("请至少选择一个测试品种。", "warning")
        st.stop()
    st.session_state["val_result"] = None
    _exch = "CFFEX" if all(s in ("IC0", "IF0", "IH0", "IM0") for s in symbols) else "SHFE"
    payload = {
        "idea": idea.strip(),
        "strategy": template,
        "llm_code": not use_template,
        "symbols": symbols,
        "exchange": _exch,
        "interval": interval,
        "start": start.isoformat() if start else None,
        "end": end.isoformat() if end else None,
        "cost": use_cost,
        "gate": ({"min_sharpe": gate_sharpe, "min_drawdown": gate_mdd}
                 if gate_enable else None),
        "promote": bool(gate_enable and promote),
    }
    with st.spinner("LLM 编程 + 多品种回测中…（首次运行较慢）"):
        try:
            result = APIClient.post("/strategy/validate", json=payload)
        except Exception as exc:  # noqa: BLE001
            result = {"error": str(exc)}

    if "error" in result:
        note(f"❌ {result['error']}", "error")
        st.stop()
    st.session_state["val_result"] = result
    # 保存到历史记录
    _save_to_history(result, idea, symbols, interval)

result = st.session_state["val_result"]
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
        rows.append({"品种": p["symbol"], "K线": "-", "交易": "-", "总收益": "-",
                     "Sharpe": "-", "回撤": "-", "门槛": f"❌ {p['error'][:20]}"})
        continue
    r = p.get("report") or {}
    g = p.get("gate") or {}
    rows.append({
        "品种": p["symbol"],
        "K线": p.get("bars", 0),
        "交易": p.get("trades", 0),
        "总收益": f"{r.get('total_return', 0):+.2%}",
        "Sharpe": f"{r.get('sharpe', 0):.2f}",
        "回撤": f"{r.get('max_drawdown', 0):.2%}",
        "门槛": (g.get("status") or "-").upper(),
    })
if rows:
    st.markdown("### 📊 多品种回测对比")
    st.dataframe(rows, use_container_width=True)

# ------------------------------------------------------------ 净值曲线（各品种叠加）
curves = [(p["symbol"], p.get("equity_curve") or []) for p in per_symbol
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
elif result.get("llm_code") is False:
    st.caption(f"使用预置模板：{result.get('strategy')}")

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
