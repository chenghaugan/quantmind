"""端到端流水线页面：Idea → AI证据研究 → 因子挖掘 → OOS复合alpha → 策略代码 → 知识库。

把整条研究链「一次可视化」：输入一个投资想法，一次调用后端 /factor/e2e 跑通
证据研究、因子挖掘、样本外复合 alpha、代码生成与知识库入库。
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Optional  # noqa: E402

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from utils.theme import (  # noqa: E402
    setup_page, page_header, section, note, verdict, guard_error,
    kpi_row, fmt_num, fmt_pct, badge,
)
from utils.api_client import APIClient  # noqa: E402
from utils.constants import (  # noqa: E402
    BASKET_CHOICES, resolve_basket_symbols, ALL_EXCHANGES, EXCHANGE_NAMES,
    INTERVALS, INTERVAL_NAMES,
)


def _kb_conn_note():
    """页面顶部显示当前知识库连接状态：真实 DB 路径 + 各类条数。

    直读本地 KnowledgeStore。任何异常都降级为温和提示，绝不抛异常阻断页面；
    某类目读取慢/异常时仅该类显示 '—'。
    """
    try:
        from quantmind.knowledge import KnowledgeStore
        ks = KnowledgeStore()
        dbp = str(getattr(ks, "db_path", "") or "—")
    except Exception:  # noqa: BLE001 包/库不可用则降级提示
        st.caption("📚 知识库：后端/本地库不可用（页面仍可正常使用，从项目根启动可连到 db/knowledge.db）")
        return
    counts = {}
    for kind in ("factor", "strategy", "methodology", "research_log"):
        try:
            counts[kind] = len(ks.list_items(kind=kind, limit=500))
        except Exception:  # noqa: BLE001 单类读取失败仅该类降级
            counts[kind] = "—"
    try:
        counts["lifecycle"] = len(ks.list_strategy_lifecycles())
    except Exception:  # noqa: BLE001
        counts["lifecycle"] = "—"
    st.caption(
        f"📚 知识库：<code>{dbp}</code> ｜ 因子 {counts['factor']} · "
        f"策略 {counts['strategy']} · 方法论 {counts['methodology']} · "
        f"研究日志 {counts['research_log']} · 生命周期 {counts['lifecycle']}",
        unsafe_allow_html=True,
    )


def _flow(steps):
    """渲染 qm-flow 流水线链路（steps: str 列表）。"""
    st.markdown(
        "<div class='qm-flow'>"
        + "".join(
            f"<span class='qm-flow-step'>{s}</span>"
            + ("<span class='qm-flow-arrow'>→</span>" if i < len(steps) - 1 else "")
            for i, s in enumerate(steps)
        )
        + "</div>",
        unsafe_allow_html=True,
    )


def _status_tone(status: Optional[str]) -> str:
    s = (status or "").lower()
    if "verif" in s or s == "verified":
        return "success"
    if "reject" in s or s == "fail" or s == "rejected":
        return "danger"
    if "propos" in s or s == "proposed" or s == "pending":
        return "info"
    return "muted"


def _history_sources():
    """拿到 (runs 列表, detail 函数)。优先走后端 /runs；后端不可用（旧代码/未启动/
    404）时退回直读本地 KnowledgeStore，保证历史报告始终可看。返回 (runs, detail_fun)。"""
    try:
        data = APIClient.runs(limit=50, timeout=15)
        if data and "error" not in data and data.get("runs") is not None:
            return (data.get("runs") or []), APIClient.run_detail
    except Exception:  # noqa: BLE001 后端异常则走本地兜底，不阻断页面
        pass
    try:
        from quantmind.knowledge import KnowledgeStore
        ks = KnowledgeStore()
        runs = (ks.list_runs(limit=50) or []) if ks else []
        def _detail(run_id, timeout=20):
            run = ks.get_run(run_id) or {}
            run["trials"] = ks.trials_for_run(run_id)
            return run
        return runs, _detail
    except Exception:  # noqa: BLE001 本地库也不可读则返回空
        return [], None


def _render_history():
    """端到端运行历史报告：列出历史 run（idea/时间/复合统计/AI brief），选中可看因子试验明细。"""
    st.markdown("#### 📜 历史运行报告")
    runs, detail_fun = _history_sources()
    if not runs:
        note("暂无历史运行记录。跑完一次「端到端流水线」即会留存（含复合 alpha 统计与 AI 经验简报）。"
             "<br>提示：若刚跑完仍看不到，请重启后端（旧后端无 /runs 接口），再重跑一次。", "info")
        return
    source_hint = "（本地知识库直读）" if detail_fun is not APIClient.run_detail else ""
    labels = {}
    for r in runs:
        md = r.get("metadata") or {}
        idea = str(md.get("idea") or "—")
        created = str(r.get("created_at") or "")[:19]
        status = str(md.get("status") or "")
        labels[r["run_id"]] = f"{created} ｜ {idea} ｜ {status}"
    st.caption(f"数据源：后端 /runs{source_hint}")
    page = st.selectbox("选择运行记录", list(labels),
                        format_func=lambda k: labels.get(k, k))
    if page != st.session_state.get("hist_sel") or st.button("🔄 加载该次报告"):
        st.session_state["hist_sel"] = page
        st.session_state["hist_detail"] = detail_fun(page, timeout=20)
    det = st.session_state.get("hist_detail") or {}
    if guard_error(det, "运行详情"):
        return
    md = det.get("metadata") or {}
    kpi_row([
        {"label": "复合前向IC", "value": fmt_num(md.get("composite_fwd_ic"), 4)},
        {"label": "复合Sharpe", "value": fmt_num(md.get("composite_sharpe"), 3)},
        {"label": "代表因子", "value": str(md.get("n_representative") or 0)},
        {"label": "已验证假设", "value": str(md.get("n_verified_hypotheses") or 0)},
        {"label": "算法", "value": str(md.get("algo") or "—")},
    ])
    with st.expander("📝 AI 经验简报", expanded=False):
        st.write(str(md.get("brief") or "（无简报）"))
    trials = det.get("trials") or []
    if not trials:
        note("该次运行暂无因子试验明细。", "info")
        return
    has_metric = any(
        (t.get("metadata") or {}).get("test_ic") is not None for t in trials)
    if not has_metric:
        note("ℹ️ 该记录为**历史回填**：仅含因子状态与判读，无逐因子试验指标"
             "（train/val/test IC、Sharpe 等）。跑一次新的端到端流水线即可获得完整指标表。", "info")
    rows = []
    for t in trials:
        m = t.get("metadata") or {}
        is_rep = bool(m.get("is_representative"))
        rows.append({
            "表达式": (str(m.get("expression") or ""))[:42],
            "判定": m.get("status") or "",
            "代表": "★" if is_rep else "",
            "trainIC": fmt_num(m.get("train_ic"), 4),
            "valIC": fmt_num(m.get("val_ic"), 4),
            "testIC": fmt_num(m.get("test_ic"), 4),
            "testSharpe": fmt_num(m.get("test_sharpe"), 3),
            "收益%": fmt_num(m.get("test_return"), 2),
            "回撤%": fmt_num(m.get("test_mdd"), 2),
            "判读理由": str(m.get("reason") or ""),
        })
    st.caption(f"共 {len(rows)} 条因子试验（★=去冗余后的代表因子）")
    st.dataframe(rows, use_container_width=True, hide_index=True)



#: 指数/全A 股票池在端到端流水线中默认展开到的标的数上限
#  （截面因子挖掘对每标的跑多轮 OOS 回测，全量沪深300/中证2000会非常慢，故默认截断）
_DEFAULT_POOL = 40


setup_page("端到端流水线", "🚀")
page_header(
    "端到端 AI 投研流水线",
    "输入一个投资想法，把「AI证据研究 → 因子挖掘 → 样本外复合 alpha → 策略代码 → 知识库」"
    "整条研究链一次跑通并整页可视化。",
    "🚀",
)

_kb_conn_note()

# 视图切换：运行流水线 / 历史运行报告（历史上屏时不进入下方运行表单）
_view = st.radio("视图", ["🚀 运行流水线", "📜 历史运行报告"], horizontal=True)
if _view == "📜 历史运行报告":
    _render_history()
    st.stop()

note(
    "**链路**：Idea → AI 证据研究（假设 + 候选因子）→ 多 seed 因子挖掘与去冗余 → "
    "train/val/test 防泄漏切分 → 每代表因子做样本外（OOS）回测 → 按 ICIR/最小方差等 "
    "方案合成复合 alpha 并回测 → 生成可直接回测的策略代码 → （可选）写入知识库。<br>"
    "整个过程一次 POST 完成；下方按标签页查看每一环的结果。",
    "info",
)

# ------------------------------------------------------------- 输入区
_asset_default = "期货"
l, r = st.columns([2, 1], gap="medium")
with l:
    idea = st.text_area(
        "💡 投资想法（Idea）",
        "螺纹钢期货动量与期限结构因子组合策略",
        height=90,
        help="用自然语言描述你的投资假设，后端会据此做 AI 证据研究并挖掘候选因子。",
    )
    asset_class = st.selectbox(
        "资产类别", ["期货", "A股", "港股", "期权"],
        index=["期货", "A股", "港股", "期权"].index(_asset_default),
    )
    c1, c2 = st.columns(2)
    with c1:
        algo = st.selectbox(
            "搜索算法", ["co", "ea", "tot"],
            format_func=lambda x: {
                "co": "链式精炼 (CoT)",
                "ea": "进化算法 (EA，种群变异+选择)",
                "tot": "树状思维 (ToT，分支+剪枝)",
            }[x],
        )
    with c2:
        rounds = st.slider("每 seed 迭代深度", 1, 8, 3)
    c3, c4 = st.columns(2)
    with c3:
        dedup_th = st.slider("去冗余相关阈值", 0.5, 0.95, 0.7, 0.05,
                             help="两因子相关 ≥ 此值视为冗余，每簇仅保留 |IC| 最高者")
    with c4:
        composite_scheme = st.selectbox(
            "复合权重方案", ["icir", "min_var", "equal", "inv_var"],
            format_func=lambda x: {
                "icir": "ICIR 加权（信息比率）",
                "min_var": "最小方差（闭式）",
                "equal": "等权",
                "inv_var": "逆方差",
            }[x],
        )
    run_composite = st.checkbox("启用复合 alpha 组合回测", value=True)

with r:
    interval = st.selectbox(
        "K线颗粒度", INTERVALS, index=INTERVALS.index("1d"),
        format_func=lambda x: INTERVAL_NAMES.get(x, x),
        help="决定 idea 在哪个时间颗粒度上做因子挖掘与 OOS 回测：1分钟/5分钟/1小时/日线等。"
             "注意：分钟级与小时级需对应粒度数据可得",
    )
    basket = st.selectbox("标的篮子", BASKET_CHOICES, format_func=lambda x: str(x))
    _is_pool = str(basket).startswith("指数·")
    _pool_n = _DEFAULT_POOL
    if _is_pool:
        _pool_n = st.number_input("股票池标的数上限", min_value=5, max_value=200,
                                  value=_DEFAULT_POOL, step=5,
                                  help="指数/全A 股票池按此数量截断成分股后再送入流水线")
    symbols, exch = resolve_basket_symbols(basket, max_symbols=_pool_n if _is_pool else None)
    st.caption("篮子：" + " · ".join(symbols[:5]) + ("…" if len(symbols) > 5 else ""))
    custom = st.text_input(
        "自定义标的（逗号分隔，与篮子取并集）",
        "",
        placeholder="如：IF0, IH0, IC0, IM0 或任意品种/代码",
        help="在此追加任意具体品种（如股指期货 IF0/IH0/IC0/IM0、商品期货 rb0/cu0、"
             "股票 600519 等），与上方篮子合并后一起送入流水线；留空则仅用篮子。",
    )
    exchange = st.selectbox("交易所", ALL_EXCHANGES, index=ALL_EXCHANGES.index(exch),
                            format_func=lambda x: f"{x} · {EXCHANGE_NAMES.get(x, '')}")
    forward_periods = st.slider("前向期数", 1, 20, 1)
    c5, c6 = st.columns(2)
    with c5:
        train_frac = st.slider("训练期占比", 0.3, 0.9, 0.6, 0.05)
    with c6:
        val_frac = st.slider("验证期占比", 0.1, 0.4, 0.2, 0.05)
    verify_threshold = st.slider("代码校验阈值", 0.0, 0.2, 0.02, 0.01)
    ingest_knowledge = st.checkbox("研究结果写入知识库", value=True)
    run_btn = st.button("🚀 运行端到端流水线", type="primary", width="stretch")

if not run_btn and not st.session_state.get("e2e_task_id") and not st.session_state.get("e2e_result"):
    note("填写想法与参数后点击运行，结果会整页展示：<b>AI 证据研究</b> / <b>因子挖掘与 OOS</b> / "
         "<b>复合 alpha</b> / <b>策略代码</b> / <b>知识库</b>（标签页切换）。", "info")
    st.stop()

_extra_symbols = [s.strip() for s in custom.split(",") if s.strip()]
final_symbols = list(dict.fromkeys([*list(symbols), *_extra_symbols]))
st.caption("送入流水线的标的数：" + str(len(final_symbols)))
if len(final_symbols) < 2:
    note("端到端流水线至少需要 2 个标的（篮子或自定义均不足 2 个）。", "warning")
    st.stop()
if not idea.strip():
    note("请输入一个投资想法。", "warning")
    st.stop()

payload = {
    "idea": idea.strip(),
    "asset_class": asset_class,
    "seeds": [],
    "symbols": final_symbols,
    "exchange": exchange,
    "interval": interval,
    "start": None,
    "end": None,
    "algo": algo,
    "rounds": rounds,
    "forward_periods": forward_periods,
    "market": "",
    "train_frac": train_frac,
    "val_frac": val_frac,
    "dedup_threshold": dedup_th,
    "min_abs_ic": 0.03,
    "run_composite": run_composite,
    "composite_scheme": composite_scheme,
    "composite_standardize": "zscore",
    "n_groups": 5,
    "long_short": True,
    "cost_rate": 0.0,
    "max_candidates": 8,
    "verify_threshold": verify_threshold,
    "run_search": False,
    "max_rounds": 2,
    "ingest_knowledge": ingest_knowledge,
}

# ---- 异步启动 + 非阻塞轮询（st.fragment run_every 自动刷新，规避长跑超时/会话卡死）----
@st.fragment(run_every=3)
def _e2e_poll_fragment() -> None:
    """轮询后台端到端任务：运行中更新进度条；完成/失败后写入 session_state 并整页重跑。"""
    tid = st.session_state.get("e2e_task_id")
    submitted = st.session_state.get("e2e_submitted_at", time.time())
    s = APIClient.factor_e2e_status(tid, timeout=30)
    status = (s or {}).get("status")
    elapsed = time.time() - submitted
    if status == "success":
        st.session_state["e2e_result"] = (s or {}).get("result") or {}
        st.session_state.pop("e2e_task_id", None)
        st.rerun()
        return
    if status in ("error", "cancelled"):
        st.session_state["e2e_result"] = {"error": (s or {}).get("message") or f"任务{status}"}
        st.session_state.pop("e2e_task_id", None)
        st.rerun()
        return
    if status == "not_found":
        st.session_state["e2e_result"] = {
            "error": (s or {}).get("message") or "任务不存在（后端可能已重启），请重新运行。"}
        st.session_state.pop("e2e_task_id", None)
        st.rerun()
        return
    # 仍在运行：进度条随时间推进（近似，非真实百分比）
    frac = min(0.95, elapsed / 1800.0)
    st.progress(
        frac,
        text=f"正在跑通端到端流水线（{idea[:40]}…，{algo.upper()} × {rounds} 轮）"
             f"· 已运行 {int(elapsed)}s（后台任务，页面每 3s 自动刷新）",
    )


if run_btn:
    # 新一次提交：清空旧结果，启动后台任务，随后整页重跑进入轮询分支
    _started = APIClient.factor_e2e_start(payload, timeout=30)
    tid = (_started or {}).get("task_id")
    if not tid:
        guard_error(_started, "端到端流水线启动")
        st.stop()
    st.session_state["e2e_task_id"] = tid
    st.session_state["e2e_submitted_at"] = time.time()
    st.session_state.pop("e2e_result", None)
    st.rerun()

_e2e_tid = st.session_state.get("e2e_task_id")
if _e2e_tid:
    # 任务进行中：渲染轮询片段（每 3s 自动刷新）并暂不渲染结果
    _e2e_poll_fragment()
    st.stop()

result = st.session_state.get("e2e_result")

if guard_error(result or {"error": "端到端流水线未返回有效结果"}, "端到端流水线"):
    st.stop()

# ---- 方法论知识层·需澄清：库内/联网均无可靠资料 → 回问用户，不编造 ----
_miss = (result.get("needs_input") or {}).get("missing") or []
if _miss:
    note("流水线暂无法从「知识库 + 网络资料」得到该方法论的可信、可计算实现。"
         "为避免编造失真的策略，请补充以下信息后重跑（补充内容会写入知识库，下次直接可用）：", "warning")
    for m in _miss:
        st.markdown(f"- **{m}**")
    user_info = st.text_area(
        "请补充：该方法论的定义 / 计算规则 / 典型示例",
        key="e2e_meth_input", height=160,
        placeholder="例如：黄金分割线——取阶段高H/低L，回撤位=H-(H-L)*r（r=0.382/0.5/0.618），回踩到 0.618 附近止跌且收盘站回该位上方即做多…",
    )
    if st.button("提交补充并重跑", type="primary"):
        payload["methodology_input"] = user_info.strip()
        st.session_state.pop("e2e_result", None)
        _started = APIClient.factor_e2e_start(payload, timeout=30)
        tid = (_started or {}).get("task_id")
        if not tid:
            guard_error(_started, "重跑端到端流水线")
        else:
            st.session_state["e2e_task_id"] = tid
            st.session_state["e2e_submitted_at"] = time.time()
        st.rerun()
    st.stop()

ev = result.get("evidence") or {}
pipeline = result.get("pipeline") or {}
composite = pipeline.get("composite") or {}
summary = pipeline.get("summary") or {}
steps = pipeline.get("steps") or []
strategy = result.get("strategy") or {}
knowledge = result.get("knowledge") or {}

# ------------------------------------------------------------- 结论
_verified_c = sum(1 for h in (ev.get("hypotheses") or []) if _status_tone(h.get("status")) == "success")
_rep_c = summary.get("representative_count")
_verdict_txt = (f"流水线跑通：产出 {_rep_c} 个代表因子、{_verified_c} 条已验证假设，"
                f"复合 alpha 前向 IC={fmt_num((composite.get('ic_report') or {}).get('ic_mean'), 4)}。")
mean_test_ic = summary.get("mean_test_ic")
if mean_test_ic is not None and mean_test_ic > 0:
    verdict(_verdict_txt, "ok", icon="✅")
else:
    verdict("流水线已跑通。样本外（OOS）未出现明显正 IC——这是回测/合成数据上"
            "常见的诚实结果，提示依赖样本外验证而非训练期表现。", "warn", icon="🔁")

# ------------------------------------------------------------- Tabs
tab_overview, tab_ev, tab_pipe, tab_comp, tab_code, tab_kb = st.tabs([
    "概览", "AI 证据研究", "因子挖掘 / OOS", "复合 alpha", "策略代码", "知识库",
])

# ===== 1. 概览 =====
with tab_overview:
    if result.get("cached"):
        note("♻️ 结果来自 20 分钟内的缓存（相同想法/标的/参数），未重新计算。", "info")
    _code_safe = strategy.get("code_safe")
    kpi_row([
        {"label": "代表因子", "value": str(summary.get("representative_count", 0)),
         "tone": "accent"},
        {"label": "平均 Train IC", "value": fmt_num(summary.get("mean_train_ic"), 4),
         "tone": "neutral"},
        {"label": "平均 Test IC", "value": fmt_num(summary.get("mean_test_ic"), 4),
         "tone": "accent" if (summary.get("mean_test_ic") or 0) > 0 else "neutral"},
        {"label": "已验证假设", "value": str(_verified_c), "tone": "neutral"},
        {"label": "策略代码", "value": "安全" if _code_safe else "未通过校验",
         "tone": "up" if _code_safe else "down"},
    ])
    _judged_overview = knowledge.get("judged_trials") or []
    if _judged_overview:
        _ov_ok = sum(1 for t in _judged_overview if _status_tone(t.get("status")) == "success")
        _ov_rej = sum(1 for t in _judged_overview if _status_tone(t.get("status")) == "danger")
        kpi_row([
            {"label": "已验证因子", "value": str(_ov_ok),
             "tone": "up" if _ov_ok else "neutral"},
            {"label": "被拒因子", "value": str(_ov_rej),
             "tone": "down" if _ov_rej else "neutral"},
            {"label": "经验 brief", "value": "有" if (knowledge.get("brief") or "") else "无",
             "tone": "accent" if (knowledge.get("brief") or "") else "neutral"},
        ])
    _flow(["IDEA", "AI 证据研究", "因子挖掘",
           "OOS 复合 alpha", "策略代码", "知识库"])
    st.caption(f"想法：{result.get('idea') or idea.strip()}　·　标的数：{result.get('client_ready') and len(final_symbols) or len(final_symbols)}")
    with st.expander("🤖 AI 假设（证据研究结论）", expanded=True):
        hyps = ev.get("hypotheses") or []
        if hyps:
            for h in hyps:
                st.markdown(
                    badge(f"{(h.get('status') or '?').upper()}", _status_tone(h.get("status")))
                    + f"　<b>{h.get('statement') or ''}</b>",
                    unsafe_allow_html=True,
                )
                if h.get("evidence"):
                    st.caption(h["evidence"])
        else:
            st.caption("无假设产出。")
    with st.expander("🔎 原始返回", expanded=False):
        st.json(result)

# ===== 2. AI 证据研究 =====
with tab_ev:
    section("AI 证据研究", "假设验证与候选因子清单")
    hyps = ev.get("hypotheses") or []
    if hyps:
        hdf = pd.DataFrame([{
            "假设": h.get("statement", ""),
            "状态": h.get("status", ""),
            "证据": h.get("evidence", ""),
        } for h in hyps])
        st.dataframe(hdf, width="stretch", hide_index=True)
        for h in hyps:
            st.markdown(
                badge(f"{(h.get('status') or '?').upper()}", _status_tone(h.get("status")))
                + f"　<b>{h.get('statement') or ''}</b>",
                unsafe_allow_html=True,
            )
            if h.get("evidence"):
                st.caption(h["evidence"])
    else:
        note("暂无假设数据。", "info")

    facs = ev.get("factors") or []
    if facs:
        fdf = pd.DataFrame([{
            "因子": f.get("name", ""),
            "类型": f.get("kind", ""),
            "窗口": f.get("window", ""),
            "权重": fmt_num(f.get("weight"), 3),
            "表达式": f.get("expression", "")[:60],
        } for f in facs])
        st.dataframe(fdf, width="stretch", hide_index=True)

    verified_exprs = ev.get("verified_exprs") or []
    if verified_exprs:
        st.subheader("已验证表达式")
        st.caption("、".join(str(e) for e in verified_exprs))
    fs = ev.get("fact_sheet") or {}
    if fs:
        with st.expander("📋 Fact Sheet", expanded=False):
            st.json(fs)

# ===== 3. 因子挖掘 / OOS =====
with tab_pipe:
    section("因子挖掘与样本外回测", "去冗余后代表因子逐因子的 OOS 表现")
    kpi_row([
        {"label": "候选因子", "value": str(summary.get("candidate_count", 0)), "tone": "accent"},
        {"label": "代表因子", "value": str(summary.get("representative_count", 0)), "tone": "accent"},
        {"label": "回测数", "value": str(summary.get("backtested_count", 0)), "tone": "neutral"},
        {"label": "标的", "value": str(len(final_symbols)), "tone": "neutral"},
    ])
    if steps:
        sdf = pd.DataFrame([{
            "表达式": s.get("expression", ""),
            "算法": s.get("algo", ""),
            "Train IC": fmt_num(s.get("train_ic"), 4),
            "Val IC": fmt_num(s.get("val_ic"), 4),
            "Test IC": fmt_num(s.get("test_ic"), 4),
            "OOS Sharpe": fmt_num(s.get("test_sharpe"), 2),
            "OOS 收益": fmt_pct(s.get("test_return")),
            "OOS 回撤": fmt_pct(s.get("test_mdd")),
            "吸收冗余": str(len(s.get("removed_redundant") or [])),
        } for s in steps])
        st.dataframe(sdf, width="stretch", hide_index=True,
                     height=min(60 + 35 * len(sdf), 420))
    else:
        st.caption("无代表因子产出（可能被 min_abs_ic / 去冗余过滤，或搜索失败）。")

    # IC 时序折线（多因子叠加）
    _ic_fig = go.Figure()
    _palette = ["#60a5fa", "#f472b6", "#34d399", "#fbbf24", "#a78bfa",
                "#fb7185", "#22d3ee", "#a3e635"]
    for _i, s in enumerate(steps):
        _ics = s.get("ic_series") or []
        if not _ics:
            continue
        _sm = [_ics[0]] if _ics else []
        for _j in range(1, len(_ics)):
            _win = [x for x in _ics[max(0, _j - 9): _j + 1] if x is not None]
            _sm.append(sum(_win) / len(_win) if _win else None)
        _label = s.get("expression", "")[:22] + ("…" if len(s.get("expression", "")) > 22 else "")
        _ic_fig.add_trace(go.Scatter(
            x=[f"T+{k}" for k in range(len(_ics))], y=_sm, name=_label,
            line=dict(width=1.6, color=_palette[_i % len(_palette)])))
    if len(_ic_fig.data):
        _ic_fig.update_layout(height=340, title="代表因子样本外截面 IC 时序（10 期滚动均值）",
                              margin=dict(t=44, b=30),
                              legend=dict(font=dict(size=10)), hovermode="x unified")
        _ic_fig.add_hline(y=0, line=dict(color="rgba(148,163,184,.4)", dash="dash"))
        st.plotly_chart(_ic_fig, use_container_width=True, config={"displayModeBar": False})

# ===== 4. 复合 alpha =====
with tab_comp:
    section("复合 alpha 组合", "权重优化 + 样本外净值 + 归因")
    pf = composite.get("portfolio") or {}
    kpi_row([
        {"label": "Sharpe", "value": fmt_num(pf.get("sharpe"), 2),
         "tone": "up" if (pf.get("sharpe") or 0) > 0 else "neutral"},
        {"label": "总收益", "value": fmt_pct(pf.get("total_return")), "tone": "neutral"},
        {"label": "最大回撤", "value": fmt_pct(pf.get("max_drawdown")), "tone": "down"},
        {"label": "前向 IC", "value": fmt_num((composite.get("ic_report") or {}).get("ic_mean"), 4),
         "tone": "neutral"},
        {"label": "标的数", "value": str(composite.get("n_symbols", 0)), "tone": "neutral"},
    ])

    col_eq, col_w = st.columns([2, 1], gap="medium")
    with col_eq:
        daily = pf.get("daily_returns") or []
        if daily:
            nav = []
            cur = 1.0
            for i, r in enumerate(daily):
                cur *= (1.0 + float(r))
                nav.append({"date": f"T+{i}", "equity": cur})
            fig = px.line(pd.DataFrame(nav), x="date", y="equity",
                          title="复合 alpha 净值（OOS）")
            fig.update_layout(xaxis_title="", yaxis_title="净值", height=360,
                              margin=dict(t=44, b=30))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("无日收益数据，无法绘制净值曲线。")
    with col_w:
        weights = composite.get("weights") or {}
        if weights:
            wdf = pd.DataFrame([
                {"因子": k if len(k) <= 26 else k[:25] + "…", "权重": v}
                for k, v in weights.items()
            ]).sort_values("权重", ascending=True)
            figw = px.bar(wdf, x="权重", y="因子", orientation="h", title="组合权重")
            figw.update_layout(height=360, margin=dict(t=44, l=8, b=8),
                               xaxis_title="权重", yaxis_title="")
            figw.update_yaxes(automargin=True)
            st.plotly_chart(figw, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("无权重数据。")

    # 归因表
    contrib = composite.get("contribution") or []
    if contrib:
        cdf = pd.DataFrame([{
            "表达式": r.get("expression", "")[:30],
            "权重": fmt_num(r.get("weight"), 3),
            "Test IC": fmt_num(r.get("test_ic"), 3),
            "OOS Sharpe": fmt_num(r.get("test_sharpe"), 2),
            "OOS 收益": fmt_pct(r.get("test_return")),
            "贡献": fmt_num(r.get("contribution"), 4),
            "贡献占比": fmt_pct(r.get("abs_pct")),
        } for r in contrib])
        st.subheader("组合归因（近似贡献分解）")
        st.caption("`contribution = 权重 × 成分样本外收益`（横截面组合近似）")
        st.dataframe(cdf, width="stretch", hide_index=True)
    else:
        st.caption("无归因数据。")

# ===== 5. 策略代码 =====
with tab_code:
    section("策略代码", "流水线生成的策略 + 沙箱安全校验")
    code = strategy.get("code") or ""
    _safe = strategy.get("code_safe")
    st.markdown(
        badge("代码安全", "success") if _safe else badge("代码未通过校验", "danger"),
        unsafe_allow_html=True,
    )
    if code:
        st.code(code, language="python")
    else:
        note("未生成策略代码。", "info")
    errs = strategy.get("code_errors") or []
    if errs:
        st.subheader("代码错误")
        st.code("\n".join(str(e) for e in errs), language="text")
    la = strategy.get("lookahead") or []
    if la:
        st.subheader("前视偏差提示")
        st.caption("、".join(str(x) for x in la))

    # ---------------- 注册入模拟盘 ----------------
    st.divider()
    section("入驻策略库", "把流水线产出的策略注册进策略库并挂上生命周期（模拟盘入口）")
    if not code:
        note("未生成策略代码，无法注册。", "warning")
    else:
        _idea_short = "".join(ch for ch in str(result.get("idea") or idea or "")[:4] if ch.isalnum())
        reg = st.session_state.setdefault("e2e_reg", {})
        _default_name = reg.get("name", f"e2e_{_idea_short}")
        reg_name = st.text_input("策略注册名", value=_default_name)
        reg.setdefault("name", reg_name)
        if _safe:
            if st.button("📥 注册策略入模拟盘", type="primary"):
                with st.spinner("正在注册并创建生命周期记录…"):
                    r = APIClient.strategy_register(reg_name, code, str(result.get("idea") or idea))
                if guard_error(r, "策略注册"):
                    st.stop()
                reg["result"] = r
        else:
            st.markdown(badge("未通过沙箱，需人工复核", "warning"), unsafe_allow_html=True)
            if st.button("忽略警告强制注册（人工已复核）", type="secondary", help="仅在你已人工审阅源码并确认安全时使用"):
                with st.spinner("正在强行注册…"):
                    r = APIClient.strategy_register(reg_name, code, str(result.get("idea") or idea))
                if guard_error(r, "策略注册"):
                    st.stop()
                reg["result"] = r
        r = reg.get("result")
        if r:
            if r.get("ok"):
                st.markdown(
                    badge("已注册", "success")
                    + f"　<b>strategy_id</b>：<code>{r.get('strategy_id')}</code>　"
                    + f"生命周期状态：{badge(r.get('lifecycle') or 'IDEA', 'info')}",
                    unsafe_allow_html=True,
                )
                note("✅ 注册成功，已挂上生命周期。可到「生命周期」页查看晋升模拟盘（PAPER）。", "success")
            else:
                note(f"注册失败：{r.get('error') or r.get('msg') or r}", "error")

# ===== 6. 知识库 =====
with tab_kb:
    section("知识库", "研究结果入库 + 语义检索")
    kpi_row([
        {"label": "已入库", "value": "是" if knowledge.get("ingested") else "否",
         "tone": "up" if knowledge.get("ingested") else "neutral"},
        {"label": "KB 记录数", "value": str(knowledge.get("kb_records", 0)), "tone": "accent"},
    ])

    # ---- 本次运行 AI 判读（后端追加字段，缺省防御；后端未返回时自然隐藏/显示空态）----
    _judged = knowledge.get("judged_trials") or []
    _brief = knowledge.get("brief") or ""
    _run_id = knowledge.get("run_id")
    if _judged or _brief or _run_id:
        section("本次运行 AI 判读", "代表因子的成功/失败判读 + 经验归纳（AI 知识闭环）")
        if _run_id:
            st.caption(f"run_id: <code>{_run_id}</code>")
        if _judged:
            _n_ok = sum(1 for t in _judged if _status_tone(t.get("status")) == "success")
            _n_rej = sum(1 for t in _judged if _status_tone(t.get("status")) == "danger")
            kpi_row([
                {"label": "已验证因子", "value": str(_n_ok), "tone": "up"},
                {"label": "被拒因子", "value": str(_n_rej), "tone": "down"},
            ])
            # 因子判读表：数值列用 DataFrame，状态用 badge 逐行标注
            _jdf = pd.DataFrame([{
                "表达式": str(t.get("expression") or "")[:48],
                "Train IC": fmt_num(t.get("train_ic"), 4),
                "Test IC": fmt_num(t.get("test_ic"), 4),
                "OOS Sharpe": fmt_num(t.get("test_sharpe"), 2),
                "原因": str(t.get("reason") or "")[:60],
                "标签": "、".join(str(x) for x in (t.get("tags") or [])),
            } for t in _judged])
            st.dataframe(_jdf, width="stretch", hide_index=True)
            for t in _judged:
                st.markdown(
                    badge(str(t.get("status") or "?").upper(), _status_tone(t.get("status")))
                    + f"　<code>{str(t.get('expression') or '')[:60]}</code>"
                    + f"　·　Test IC {fmt_num(t.get('test_ic'), 4)}　OOS Sharpe "
                    + f"{fmt_num(t.get('test_sharpe'), 2)}",
                    unsafe_allow_html=True,
                )
                if t.get("reason"):
                    st.caption("　　" + str(t["reason"]))
        else:
            note("本次运行暂无因子判读（judged_trials 未返回）。", "warn")
        if _brief:
            section("AI 经验归纳", "本次运行沉淀的可复用经验")
            st.markdown(_brief)
        for _name, _label, _tone in (
            ("effective_themes", "✅ 有效主题", "up"),
            ("failure_traps", "⚠️ 失败陷阱", "danger"),
            ("next_suggestions", "➡️ 下一步建议", "info"),
        ):
            _items = knowledge.get(_name) or []
            if _items:
                st.markdown(
                    f"<b style='color: {'#f87171' if _tone == 'danger' else '#60a5fa' if _tone == 'info' else '#34d399'}'>"
                    f"{_label}</b>",
                    unsafe_allow_html=True,
                )
                for _it in _items:
                    st.markdown(f"- {_it}")
            st.caption("")

    # ---- 历史端到端 Run（直读本地知识库 e2e_runs + factor_trials，展示已沉淀闭环）----
    section("🗂️ 历史端到端 Run", "已沉淀的挖掘运行：因子判读 + 经验 brief（从本地知识库直读）")
    try:
        from quantmind.knowledge import KnowledgeStore
        _hks = KnowledgeStore()
        _runs = _hks.list_runs(limit=50)
    except Exception:  # noqa: BLE001 库不可用则温和降级
        _runs = []
    if not _runs:
        note("暂无历史端到端 Run（可在本页跑一次端到端后沉淀，或先执行 backfill 回填脚本）。", "info")
    else:
        # 历史 run 概览表
        _hrows = []
        for _r in _runs:
            _rm = _r.get("metadata") or {}
            _rid = _rm.get("run_id") or _r.get("run_id") or ""
            _n_ok = _rm.get("n_verified_hypotheses") or _rm.get("n_representative") or 0
            _hrows.append({
                "run_id": _rid,
                "想法": str(_rm.get("idea") or "")[:40],
                "已验证因子": _n_ok,
                "状态": _rm.get("status") or "",
            })
        if _hrows:
            st.dataframe(pd.DataFrame(_hrows), width="stretch", hide_index=True)
        # 每个 run 展开：trials 判读表 + brief
        for _r in _runs:
            _rm = _r.get("metadata") or {}
            _rid = _rm.get("run_id") or _r.get("run_id") or ""
            _idea = str(_rm.get("idea") or _rid)
            with st.expander(f"🧪 {_idea[:40]}　<code>{_rid}</code>", expanded=False):
                try:
                    _trials = _hks.trials_for_run(_rid, limit=200)
                except Exception:  # noqa: BLE001
                    _trials = []
                # run 级 KPI
                _brief = str(_rm.get("brief") or "")
                _n_tri = len(_trials)
                _n_ver = sum(1 for _t in _trials if (str((_t.get("metadata") or {}).get("status")) or "").lower() in ("verified", "pass", "passed"))
                kpi_row([
                    {"label": "因子试错总数", "value": str(_n_tri), "tone": "accent"},
                    {"label": "已验证", "value": str(_n_ver), "tone": "up"},
                    {"label": "状态", "value": str(_rm.get("status") or "—"), "tone": "neutral"},
                ])
                if _trials:
                    _tdf = pd.DataFrame([{
                        "表达式": str((_t.get("metadata") or {}).get("expression") or "")[:48],
                        "Test IC": fmt_num((_t.get("metadata") or {}).get("test_ic"), 4),
                        "OOS Sharpe": fmt_num((_t.get("metadata") or {}).get("test_sharpe"), 2),
                        "状态": str((_t.get("metadata") or {}).get("status") or ""),
                        "原因": str((_t.get("metadata") or {}).get("reason") or "")[:60],
                    } for _t in _trials])
                    st.dataframe(_tdf, width="stretch", hide_index=True)
                    for _t in _trials:
                        _tm = _t.get("metadata") or {}
                        _ts = str(_tm.get("status") or "")
                        st.markdown(
                            badge(_ts.upper(), _status_tone(_ts))
                            + f"　<code>{str(_tm.get('expression') or '')[:60]}</code>"
                            + f"　·　Test IC {fmt_num(_tm.get('test_ic'), 4)}　OOS Sharpe "
                            + f"{fmt_num(_tm.get('test_sharpe'), 2)}",
                            unsafe_allow_html=True,
                        )
                        if _tm.get("reason"):
                            st.caption("　　" + str(_tm["reason"]))
                else:
                    note("该 run 暂无因子试错明细。", "info")
                if _brief:
                    st.markdown(f"**AI 经验**：{_brief}")
        st.caption("💡 历史 run 来自本地知识库 e2e_runs / factor_trials 表；缺失时可运行 scripts/backfill_e2e_history.py 回填。")
    st.subheader("语义检索")
    q = st.text_input("检索内容", "螺纹钢 期限结构 动量")
    top_k = st.slider("返回条数", 1, 20, 10)
    if st.button("🔎 检索", type="primary"):
        sres = APIClient.knowledge_search(q, top_k=top_k)
        if guard_error(sres, "知识库检索"):
            st.stop()
        results = sres.get("results") or []
        if results:
            rdf = pd.DataFrame([{
                "ID": r.get("kb_id", ""),
                "类型": r.get("kind", ""),
                "得分": fmt_num(r.get("score"), 3),
                "内容": (r.get("text") or "")[:90],
            } for r in results])
            st.dataframe(rdf, width="stretch", hide_index=True)
            with st.expander("🔎 检索详情", expanded=False):
                st.json(sres)
        else:
            note("无检索结果。", "info")

    st.subheader("知识库条目")
    kind = st.selectbox("类型过滤", ["", "hypothesis", "factor", "strategy", "insight"])
    if st.button("📚 加载知识库", type="secondary"):
        lres = APIClient.knowledge_list(kind=kind or None)
        if guard_error(lres, "知识库列表"):
            st.stop()
        items = lres.get("items") or []
        total = lres.get("total", len(items))
        st.caption(f"共 {total} 条")
        if items:
            ldf = pd.DataFrame([{
                "ID": it.get("kb_id", ""),
                "类型": it.get("kind", ""),
                "内容": (it.get("text") or "")[:90],
            } for it in items])
            st.dataframe(ldf, width="stretch", hide_index=True)
        else:
            note("知识库为空。", "info")

st.caption("提示：整条链一次调用后端 /factor/e2e 跑通；调整上方想法与参数可重复运行。")
