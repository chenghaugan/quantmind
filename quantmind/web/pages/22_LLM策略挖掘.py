"""LLM 策略挖掘：因子 → 策略规格 → 自动回测 → 生命周期注册。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.api_client import APIClient
from utils.theme import kpi_row, note, page_header, section, setup_page, verdict

setup_page("LLM策略挖掘", "🧠")
page_header(
    "LLM策略挖掘",
    "从因子库选择因子，LLM 设计策略规格，自动回测验证，一键注册到生命周期",
    "🧠",
)

note(
    "**流程**：选择因子（含 IC/IR 指标）→ LLM 分析并设计策略 → 确定性编译 → "
    "自动回测（最多 3 次迭代）→ 通过闸门则自动注册到生命周期",
    "info",
)

# ------------------------------------------------------------ 0. 领域知识获取（阶段A）
def _render_knowledge_block(knowledge: dict) -> None:
    """渲染「领域知识获取」区块：知识来源 → 定义/买卖规则 → 候选因子方向。

    数据来自后端 result['knowledge']（{concept, definition, buy_signal_rules,
    candidate_factors, sources, kb_hits}）。缺失任一字段都静默跳过，绝不让页面报错。
    """
    if not isinstance(knowledge, dict) or not knowledge:
        note("⚠️ 本次结果未启用知识增强（后端未返回 knowledge 字段）。", "warning")
        return

    with st.expander("📚 领域知识获取（阶段A）", expanded=True):
        # 1. 知识来源
        sources = knowledge.get("sources") or []
        kb_hits = knowledge.get("kb_hits")
        if sources or kb_hits:
            st.markdown("**① 知识来源**")
            if isinstance(sources, list) and sources:
                for src in sources:
                    if isinstance(src, str) and src:
                        st.caption(f"· {src}")
                    elif isinstance(src, dict):
                        st.caption("· " + " | ".join(str(v) for v in src.values() if v))
            if kb_hits:
                st.caption(f"· 库内方法论命中 {kb_hits} 条")
        else:
            st.caption("① 知识来源：无")

        # 2. 定义 + 买卖规则
        definition = knowledge.get("definition")
        concept = knowledge.get("concept")
        rules = knowledge.get("buy_signal_rules")
        if definition or concept:
            st.markdown("**② 定义 (definition)**")
            st.write(definition or concept)
        if rules:
            st.markdown("**③ 买卖规则 (buy_signal_rules)**")
            if isinstance(rules, str):
                st.write(rules)
            else:
                st.json(rules)

        # 3. 候选因子方向
        cands = knowledge.get("candidate_factors")
        if cands:
            st.markdown("**④ 候选因子方向 (candidate_factors)**")
            if isinstance(cands, str):
                st.write(cands)
            else:
                st.dataframe(
                    pd.DataFrame([{"候选因子": c if isinstance(c, str) else str(c)}
                                  for c in cands]),
                    use_container_width=True, hide_index=True,
                )


# 页面上方展示：优先用本次/上次挖掘返回的知识字段，否则提示未启用增强
_render_knowledge_block(st.session_state.get("mined_knowledge") or {})

# ---------------------------------------------------------------- 1. 因子选择
section("1️⃣ 选择因子")

# 加载因子库
factor_lib = APIClient.factors()
if "error" in factor_lib:
    verdict("无法加载因子库", "bad")
    st.stop()

# 构建因子 DataFrame
if isinstance(factor_lib, list):
    factor_df = pd.DataFrame(factor_lib)
else:
    factor_df = pd.DataFrame(factor_lib.get("factors", []))

if factor_df.empty:
    verdict("因子库为空", "bad")
    st.stop()

# 显示因子表格（带指标）
display_cols = ["name", "kind", "window", "description"]
metric_cols = [c for c in ["ic_mean", "icir", "sharpe"] if c in factor_df.columns]
if metric_cols:
    display_cols.extend(metric_cols)

st.dataframe(
    factor_df[[c for c in display_cols if c in factor_df.columns]],
    use_container_width=True,
    hide_index=True,
)

# 多选因子
selected_factors = st.multiselect(
    "选择因子（至少 1 个）",
    options=factor_df["name"].tolist(),
    default=factor_df["name"].tolist()[:2] if len(factor_df) >= 2 else [],
)

if not selected_factors:
    st.warning("请至少选择一个因子")
    st.stop()

# 获取选中因子的详细信息
selected_factor_data = factor_df[factor_df["name"].isin(selected_factors)].to_dict("records")

# ---------------------------------------------------------------- 2. 策略约束
section("2️⃣ 策略约束（可选）")

col1, col2 = st.columns(2)
with col1:
    constraint = st.text_input(
        "策略偏好",
        placeholder="如：偏动量、低换手、稳健型",
        help="自然语言描述你对策略的偏好",
    )
with col2:
    template_pref = st.selectbox(
        "模板偏好（可选）",
        ["自动选择", "dual_ma", "multifactor", "vol_target", "pair_trading"],
    )
    if template_pref == "自动选择":
        template_pref = None

# 标的和交易所
col3, col4 = st.columns(2)
with col3:
    symbol = st.text_input("交易标的", value="rb0")
with col4:
    exchange = st.selectbox("交易所", ["SHFE", "DCE", "CZCE", "INE", "SSE", "SZSE"])

# ---------------------------------------------------------------- 3. 执行挖掘
section("3️⃣ 执行策略挖掘")

if st.button("🚀 开始挖掘", type="primary", use_container_width=True):
    with st.spinner("LLM 正在分析因子并设计策略..."):
        result = APIClient.strategy_mining_architect(
            factors=selected_factor_data,
            constraint=constraint,
            template_preference=template_pref,
            symbol=symbol,
            exchange=exchange,
            timeout=60,
        )

    if not result.get("ok"):
        verdict(f"策略设计失败：{result.get('error', '未知错误')}", "bad")
        st.stop()

    # 保存知识字段到 session state（供页面上方「领域知识获取」区块展示）
    st.session_state["mined_knowledge"] = result.get("knowledge") or {}

    # 保存 spec 到 session state
    spec = result["spec"]
    st.session_state["mined_spec"] = spec

    # 显示策略规格
    with st.expander("📋 策略规格", expanded=True):
        st.json(spec)

    # 显示验证状态
    if result["valid"]:
        verdict("策略规格验证通过", "ok")
    else:
        verdict(f"策略规格验证失败：{result['errors']}", "bad")
        st.stop()

    # 显示设计理由
    if result.get("rationale"):
        st.info(f"**设计理由**：{result['rationale']}")

# ---------------------------------------------------------------- 4. 自动回测
if "mined_spec" in st.session_state:
    section("4️⃣ 自动回测")

    col1, col2 = st.columns(2)
    with col1:
        max_iter = st.slider("最大迭代次数", 1, 5, 3)
    with col2:
        min_sharpe = st.number_input("最低 Sharpe", value=0.5, step=0.1, format="%.2f")

    if st.button("⚙️ 运行自动回测", use_container_width=True):
        with st.spinner("自动回测中..."):
            backtest_result = APIClient.strategy_mining_auto_backtest(
                spec=st.session_state["mined_spec"],
                strategy_id=f"mined-{hash(str(st.session_state['mined_spec'])) % 10000:04d}",
                max_iterations=max_iter,
                min_sharpe=min_sharpe,
                max_drawdown=-0.30,
                timeout=120,
            )

        if not backtest_result.get("ok"):
            verdict(f"回测失败：{backtest_result.get('error', '未知错误')}", "bad")
            st.stop()

        # 显示结果
        report = backtest_result.get("report", {})

        kpi_row(
            [
                ("Sharpe", f"{report.get('sharpe', 0):.3f}"),
                ("年化收益", f"{report.get('annual_return', 0):.2%}"),
                ("最大回撤", f"{report.get('max_drawdown', 0):.2%}"),
                ("胜率", f"{report.get('win_rate', 0):.2%}"),
            ]
        )

        if backtest_result["passed"]:
            verdict(
                f"✅ 策略通过闸门！已注册到生命周期（迭代 {backtest_result['iteration']} 次）",
                "ok",
            )
        else:
            verdict(
                f"⚠️ 策略未通过闸门（迭代 {backtest_result['iteration']} 次）",
                "warn",
            )
            if backtest_result.get("adjustment_notes"):
                st.write("**调整说明**:", backtest_result.get("adjustment_notes", ""))

        # 显示权益曲线
        if "equity_curve" in report and report["equity_curve"]:
            section("权益曲线")
            eq_df = pd.DataFrame(report["equity_curve"])
            if "date" in eq_df.columns and "equity" in eq_df.columns:
                fig = px.line(eq_df, x="date", y="equity", title="权益曲线")
                st.plotly_chart(fig, use_container_width=True)

st.caption(
    "💡 通过自动回测的策略会自动注册到「生命周期」页面，可继续晋升到模拟盘/实盘。"
)
