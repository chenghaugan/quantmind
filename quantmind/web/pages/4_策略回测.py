"""策略回测：回测 / 模拟 / 实盘三路线共用同一份策略代码"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from utils.theme import (  # noqa: E402
    setup_page, page_header, section, kpi_row, note, verdict, guard_error,
    fmt_num, fmt_pct, fmt_money, tone_of, badge,
)
from utils.api_client import APIClient  # noqa: E402
from utils.charts import (  # noqa: E402
    create_equity_curve, create_drawdown_chart, create_monthly_heatmap,
    create_returns_histogram, create_gauge,
)
from utils.constants import EXCHANGES, EXCHANGE_NAMES, STRATEGIES, GATEWAYS  # noqa: E402

setup_page("策略回测", "⚙️")
page_header(
    "策略回测",
    "同一套策略代码，通过运行模式切换回测 / 模拟盘 / 实盘路由；可启用品种级真实成本模型",
    "⚙️",
)


@st.cache_data(ttl=600, show_spinner=False)
def load_strategies():
    res = APIClient.strategies(timeout=10)
    return res if isinstance(res, list) else []


remote = load_strategies()
remote_map = {s.get("name"): s for s in remote if isinstance(s, dict)}
options = list(remote_map) or list(STRATEGIES)

# ----------------------------------------------------------------- 参数模板
_TEMPLATE_FILE = Path(__file__).resolve().parent.parent / "config" / "strategy_templates.json"


def _load_templates() -> dict:
    try:
        if _TEMPLATE_FILE.exists():
            return json.loads(_TEMPLATE_FILE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        pass
    return {}


def _save_templates(data: dict) -> None:
    try:
        _TEMPLATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _TEMPLATE_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


templates = _load_templates()

with st.expander("📁 策略参数模板", expanded=False):
    tc1, tc2 = st.columns(2, gap="medium")
    with tc1:
        tmpl_name = st.text_input("模板名称", "", key="tmpl_new",
                                  placeholder="如：rb0 稳健双均线")
        if st.button("💾 保存当前参数为模板", width="stretch"):
            # 当前选中策略的默认参数作为初始模板
            cur_defaults = (remote_map.get(strategy, {}).get("parameters")
                            or STRATEGIES.get(strategy, {}).get("params", {})) if 'strategy' in dir() else {}
            name = tmpl_name.strip()
            if not name:
                name = f"{strategy or 'generic'}_{len(templates) + 1}"
            templates[name] = {"strategy": strategy if 'strategy' in dir() else "dual_ma",
                               "parameters": cur_defaults}
            _save_templates(templates)
            st.session_state["qm_tmpl_saved"] = name
            st.rerun()
        if st.session_state.get("qm_tmpl_saved"):
            st.success(f"模板「{st.session_state['qm_tmpl_saved']}」已保存")
    with tc2:
        if templates:
            tnames = [f"{n}（{v.get('strategy', '')}）" for n, v in templates.items()]
            sel = st.selectbox("加载历史模板", [""] + tnames, key="tmpl_pick",
                               format_func=lambda x: x or "— 选择模板 —")
            if st.button("📂 加载所选模板", width="stretch") and sel:
                real = sel.split("（")[0]
                if real in templates:
                    st.session_state["qm_tmpl"] = templates[real]
                    st.rerun()
        else:
            st.caption("暂无保存的模板。先在左侧保存一个。")

section("回测设置")
with st.form("bt_form"):
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        strategy = st.selectbox(
            "策略", options,
            format_func=lambda x: f"{STRATEGIES.get(x, {}).get('name', x)}（{x}）",
        )
        meta = remote_map.get(strategy, {})
        st.caption("📖 " + (meta.get("description") or STRATEGIES.get(strategy, {}).get("desc", "")))
        mode = st.radio("运行模式", ["backtest", "paper", "live"], horizontal=True,
                        format_func=lambda m: {"backtest": "回测", "paper": "模拟盘",
                                               "live": "实盘路由"}[m])
    with c2:
        symbol = st.text_input("合约代码", "rb0")
        all_ex = [e for exs in EXCHANGES.values() for e in exs]
        exchange = st.selectbox("交易所", all_ex,
                                format_func=lambda x: f"{x} · {EXCHANGE_NAMES.get(x, '')}")
        gateway = st.selectbox("网关（实盘模式生效）", list(GATEWAYS),
                               format_func=lambda g: GATEWAYS[g],
                               disabled=(mode != "live"))
    with c3:
        capital = st.number_input("初始资金（元）", value=1_000_000.0, step=100_000.0, min_value=10_000.0)
        commission = st.number_input("手续费率", value=0.0002, step=0.0001, format="%.4f",
                                     help="简化成本；勾选真实成本模型后由品种费率表接管")
        use_cost = st.checkbox("启用真实成本模型", value=True,
                               help="按品种差异化费率 / 平今 / 印花税 / 保证金计算")

    section("策略参数")
    defaults = meta.get("parameters") or STRATEGIES.get(strategy, {}).get("params", {})
    # 若用户加载了匹配当前策略的模板，则用模板参数覆盖默认值
    _tmpl = st.session_state.get("qm_tmpl")
    if isinstance(_tmpl, dict) and _tmpl.get("strategy") == strategy and _tmpl.get("parameters"):
        defaults = {**defaults, **_tmpl["parameters"]}
        st.markdown(badge(f"已加载模板参数：{strategy}", "violet"), unsafe_allow_html=True)
    setting = {}
    if defaults:
        pcols = st.columns(min(4, max(1, len(defaults))), gap="small")
        for i, (k, v) in enumerate(defaults.items()):
            with pcols[i % len(pcols)]:
                if isinstance(v, bool):
                    setting[k] = st.checkbox(k, value=v)
                elif isinstance(v, int):
                    setting[k] = st.number_input(k, value=int(v), step=1)
                elif isinstance(v, float):
                    setting[k] = st.number_input(k, value=float(v), step=0.01, format="%.4f")
                else:
                    st.caption(f"{k} = {v}")
    else:
        st.caption("该策略无可调参数。")

    submitted = st.form_submit_button("🚀 开始运行", type="primary", width="stretch")

if not submitted:
    note(
        "<b>模式说明</b>：<br>"
        "· <b>回测</b> — 批量历史撮合，产出完整绩效报告；<br>"
        "· <b>模拟盘</b> — 逐根回放并实时广播事件，可在「实时监控」页看到事件流；<br>"
        "· <b>实盘路由</b> — 走 LiveEngine + 网关，下单前会先过风控闸门（可在「风控中心」预检）。",
        "info",
    )
    st.stop()

with st.spinner(f"正在以「{mode}」模式运行 {strategy} …"):
    res = APIClient.backtest(strategy, symbol, exchange, mode=mode, setting=setting,
                             capital=capital, commission=commission, cost=use_cost)

if guard_error(res, "运行"):
    st.stop()

# ------------------------------------------------------------ 非回测模式
if res.get("mode") == "live":
    verdict(f"实盘路由链路已打通：网关 {GATEWAYS.get(res.get('gateway'), res.get('gateway'))} 接单成功。", "ok")
    st.json(res)
    note("实盘模式仅验证下单链路，不产出绩效报告。上线前请在「风控中心」确认限额档位。", "warning")
    st.stop()

if res.get("mode") == "paper":
    summary = res.get("summary") or {}
    verdict(f"模拟盘回放完成，共成交 {res.get('trades', 0)} 笔。", "ok")
    section("模拟盘摘要")
    if summary:
        st.dataframe([{"指标": k, "数值": v} for k, v in summary.items()],
                     width="stretch", hide_index=True)
    else:
        st.caption("引擎未返回摘要。")
    note("模拟盘运行时事件已广播到 WebSocket，可到「实时监控」页查看实时事件流。", "info")
    st.stop()

# ------------------------------------------------------------ 回测结果
report = res.get("report") or {}
curve = res.get("equity_curve") or []

sharpe = report.get("sharpe") or 0.0
mdd = report.get("max_drawdown") or 0.0
tot = report.get("total_return") or 0.0
trades = report.get("trade_count") or res.get("trades") or 0

if trades == 0:
    verdict("策略在该区间未产生任何成交 —— 参数可能过于严格，或数据长度不足。", "warn")
elif sharpe >= 1.0 and mdd > -0.25:
    verdict(f"表现良好：夏普 {sharpe:.2f}，最大回撤 {mdd * 100:.1f}%，可推进到 Walk-Forward 验证。", "ok")
elif sharpe >= 0.5:
    verdict(f"勉强达标：夏普 {sharpe:.2f}，建议先做参数寻优与滚动验证再谈晋升。", "warn")
else:
    verdict(f"未达晋升门槛：夏普 {sharpe:.2f}（要求 ≥ 0.5），不建议进入模拟盘。", "bad")

section("核心绩效")
kpi_row([
    {"label": "总收益率", "value": fmt_pct(tot), "tone": tone_of(tot)},
    {"label": "年化收益", "value": fmt_pct(report.get("annual_return")),
     "tone": tone_of(report.get("annual_return"))},
    {"label": "夏普比率", "value": fmt_num(sharpe, 2), "tone": tone_of(sharpe - 0.5)},
    {"label": "最大回撤", "value": fmt_pct(mdd), "tone": "down" if mdd < -0.2 else "neutral"},
    {"label": "卡玛比率", "value": fmt_num(report.get("calmar"), 2),
     "tone": tone_of(report.get("calmar"))},
])
st.write("")
kpi_row([
    {"label": "胜率", "value": fmt_pct(report.get("win_rate"), 1)},
    {"label": "盈亏比", "value": fmt_num(report.get("profit_factor"), 2),
     "tone": tone_of((report.get("profit_factor") or 0) - 1)},
    {"label": "成交笔数", "value": f"{trades}", "tone": "accent"},
    {"label": "期末权益", "value": fmt_money(report.get("final_equity"))},
    {"label": "索提诺", "value": fmt_num(report.get("sortino"), 2),
     "tone": tone_of(report.get("sortino"))},
])

section("净值与回撤")
c1, c2 = st.columns([1.6, 1], gap="medium")
with c1:
    st.plotly_chart(create_equity_curve(curve, title="资金曲线", height=360),
                    width="stretch", key="bt_eq")
with c2:
    st.plotly_chart(create_gauge(sharpe, "夏普比率（门槛 0.5）", vmin=-1, vmax=3,
                                 good=0.5, height=180),
                    width="stretch", key="bt_g1")
    st.plotly_chart(create_gauge(abs(mdd), "最大回撤（红线 30%）", vmin=0, vmax=0.6,
                                 good=0.0, height=180),
                    width="stretch", key="bt_g2")

st.plotly_chart(create_drawdown_chart(curve, title="水下回撤曲线", height=260),
                width="stretch", key="bt_dd")

section("收益分布")
c1, c2 = st.columns(2, gap="medium")
with c1:
    st.plotly_chart(create_monthly_heatmap(curve, title="月度收益热力图", height=300),
                    width="stretch", key="bt_month")
with c2:
    if curve and len(curve) > 2:
        eq = pd.Series([p.get("equity", 0) for p in curve])
        rets = eq.pct_change().dropna().tolist()
    else:
        rets = []
    st.plotly_chart(create_returns_histogram(rets, title="日收益分布", height=300),
                    width="stretch", key="bt_hist")

section("交易成本拆解", "启用真实成本模型后按品种费率表精算" if use_cost else "简化成本模式")
cost_rows = [
    ("手续费", report.get("total_commission")),
    ("印花税", report.get("total_stamp_tax")),
    ("冲击成本", report.get("total_impact")),
    ("滑点", report.get("total_slippage")),
    ("合计成本", report.get("total_cost")),
]
kpi_row([{"label": k, "value": fmt_money(v)} for k, v in cost_rows])
cr = report.get("cost_ratio")
if cr is not None:
    tone = "bad" if cr > 0.3 else ("warn" if cr > 0.15 else "ok")
    verdict(f"成本占毛收益比 {cr * 100:.1f}%"
            + {"bad": " —— 成本吞噬严重，需降低换手或换品种。",
               "warn": " —— 成本偏高，注意换手率。",
               "ok": " —— 成本可控。"}[tone], tone)
st.caption(f"保证金占用峰值：{fmt_money(report.get('margin_used'))}")

with st.expander("📄 完整报告字段", expanded=False):
    st.dataframe([{"字段": k, "值": v} for k, v in report.items()],
                 width="stretch", hide_index=True)

if curve:
    st.download_button(
        "⬇️ 导出净值曲线 CSV",
        pd.DataFrame(curve).to_csv(index=False).encode("utf-8-sig"),
        file_name=f"equity_{strategy}_{symbol}.csv", mime="text/csv",
    )


def _build_html_report() -> str:
    """生成自包含 HTML 研报（含核心指标与净值表）。"""
    import plotly.io as pio

    fig = create_equity_curve(curve, title=f"{strategy} · {symbol} 资金曲线", height=400)
    eq_chart = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")
    rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>"
        for k, v in report.items()
        if v is not None
    )
    html = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>QuantMind 研报 · {strategy} {symbol}</title>
<style>
body{{font-family:Segoe UI,Microsoft YaHei,sans-serif;margin:32px auto;max-width:960px;color:#1f2937}}
h1{{font-size:22px}} .sub{{color:#6b7280;font-size:14px}}
table{{border-collapse:collapse;width:100%;margin-top:16px}}
th,td{{border:1px solid #e5e7eb;padding:7px 10px;text-align:left;font-size:13px}}
th{{background:#f3f4f6}}
.sec{{margin-top:26px;font-weight:700;font-size:16px;border-left:4px solid #3b82f6;padding-left:8px}}
</style></head><body>
<h1>QuantMind 策略回测研报</h1>
<div class="sub">策略：{strategy} · 合约：{symbol} · 模式：{mode} · 生成：{pd.Timestamp.now():%Y-%m-%d %H:%M}</div>
<div class="sec">① 资金曲线</div>{eq_chart}
<div class="sec">② 绩效指标</div><table><thead><tr><th>指标</th><th>值</th></tr></thead><tbody>{rows}</tbody></table>
</body></html>"""
    return html


if curve:
    st.download_button(
        "📄 导出 HTML 研报",
        _build_html_report().encode("utf-8"),
        file_name=f"report_{strategy}_{symbol}.html", mime="text/html",
        help="生成含资金曲线与全绩效指标的可分享 HTML 研报",
    )

st.caption("下一步：「参数优化」找更优参数 → 「Walk-Forward」验证样本外稳定性 → 「生命周期」申请晋升。")
