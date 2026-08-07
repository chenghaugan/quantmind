"""AI 因子挖掘页面：idea → 假设 / 推荐因子 / 可执行代码（AST 安全校验）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st  # noqa: E402

from utils.theme import (  # noqa: E402
    setup_page, page_header, section, note, verdict, guard_error, badge,
)
from utils.api_client import APIClient  # noqa: E402
from utils.constants import ASSET_CLASS_CHOICES, ASSET_CLASS_DESC  # noqa: E402

setup_page("AI 研究", "🤖")
page_header(
    "AI 因子挖掘",
    "输入投资想法，自动生成假设、风险说明、推荐因子组合与可执行的策略因子代码（AST 沙箱校验）。",
    "🤖",
)

note(
    "**AI 驱动因子挖掘**：把一句中文投资逻辑转化为结构化研究规格。"
    "当前使用内置 Mock LLM，生产环境可切换为 OpenAI / 本地大模型。",
    "info",
)

# ----------------------------------------------------------------- 输入区
col_left, col_right = st.columns([2, 1], gap="medium")
with col_left:
    idea = st.text_area(
        "💡 投资想法",
        "螺纹钢期货的动量与期限结构因子组合策略，利用短期动量捕捉趋势，同时用期限结构过滤假突破",
        height=170,
        help="描述你的交易逻辑、资产类别、预期因子类型",
    )
with col_right:
    asset_class = st.selectbox(
        "资产类别", ASSET_CLASS_CHOICES, index=0,
        format_func=lambda x: f"{x}　{ASSET_CLASS_DESC.get(x, '')}",
    )
    st.write("")
    st.write("")
    st.write("")
    generate_btn = st.button("🚀 生成因子规格", type="primary", width="stretch")

# ----------------------------------------------------------------- 生成
if not generate_btn:
    note(
        "提示：点击「生成因子规格」后，AI 会产出 <b>假设与风险</b> / <b>推荐因子</b> / "
        "<b>可执行代码</b> 三块内容，并对代码做 AST 安全校验。",
        "info",
    )
    st.stop()

with st.spinner("AI 正在分析您的投资想法…"):
    result = APIClient.research(idea, asset_class)

if guard_error(result, "AI 研究"):
    st.stop()

hypothesis = result.get("hypothesis", "无")
risk_notes = result.get("risk_notes", []) or []
suggested = result.get("suggested_factors", []) or []
generated = result.get("generated_factors", []) or []
code_safe = result.get("code_safe", False)
code_errors = result.get("code_errors", []) or []

# ----------------------------------------------------------------- 结论
if code_safe:
    verdict("规格已生成且代码通过 AST 沙箱校验，可直接送入因子研究 / 回测管线。", "ok",
            icon="✅")
else:
    verdict("规格已生成，但生成的代码未通过安全校验，请人工复核后再使用。", "warn",
            icon="⚠️")

# ----------------------------------------------------------------- 假设与风险
section("假设与风险")
hcol, rcol = st.columns(2, gap="medium")
with hcol:
    st.markdown("**📝 投资假设**")
    st.markdown(f"> {hypothesis}" if hypothesis else "> 无")
with rcol:
    st.markdown("**⚠️ 风险说明**")
    if risk_notes:
        for n in risk_notes:
            st.markdown(f"- {n}")
    else:
        st.caption("无特别说明")

# ----------------------------------------------------------------- 推荐因子
section("推荐因子")
if suggested:
    for f in suggested:
        st.code(f, language="python")
else:
    st.caption("AI 未推荐具体因子表达式。")

# ----------------------------------------------------------------- 生成代码
section("生成的因子代码")
if generated:
    for f in generated:
        with st.expander(f"📊 {f.get('name', '未命名')}　·　{f.get('kind', '')}"):
            st.code(f.get("code", ""), language="python")
            st.json({
                "因子名": f.get("name"),
                "类型": f.get("kind"),
                "窗口": f.get("window"),
                "权重": f.get("weight"),
            })
else:
    st.caption("无生成代码。")

# ----------------------------------------------------------------- 安全校验
section("代码安全校验")
if code_safe:
    st.markdown(badge("通过 AST 沙箱校验", "success"), unsafe_allow_html=True)
    st.caption("代码不含危险操作，可安全执行。")
else:
    st.markdown(badge("未通过校验", "danger"), unsafe_allow_html=True)
    if code_errors:
        for e in code_errors:
            st.error(f"  · {e}")

# ----------------------------------------------------------------- 研究溯源
section("研究溯源")
prov = result.get("provenance")
if isinstance(prov, dict) and prov:
    note(
        "**溯源 = 可追溯证据链**（对标 Vibe-Trading 的 evidence chain）：记录本次研究的数据来源、"
        "AI 工具调用与假设演进，结论可审计、可复现。",
        "info",
    )
    srcs = prov.get("data_sources") or []
    if srcs:
        st.markdown(
            "数据来源：" + " ".join(badge(s, "info") for s in srcs),
            unsafe_allow_html=True,
        )
    gen_at = prov.get("generated_at")
    if gen_at:
        st.caption(f"生成时间：{gen_at}")

    sub_sections = [
        ("工具调用", prov.get("tool_calls")),
        ("证据链", prov.get("evidence_chain")),
        ("研究假设", prov.get("hypotheses")),
        ("研究日志", prov.get("research_log")),
    ]
    for name, items in sub_sections:
        if not isinstance(items, list) or not items:
            continue
        rows = []
        for it in items:
            if isinstance(it, dict):
                for k, v in it.items():
                    rows.append({"字段": k, "值": str(v) if v is not None else ""})
            else:
                rows.append({"字段": name, "值": str(it)})
        if rows:
            st.markdown(f"**{name}（{len(items)} 条）**")
            st.dataframe(rows, width="stretch", hide_index=True, height=min(60 + 35 * len(rows), 320))
else:
    note("本次研究未附带溯源信息。", "info")

with st.expander("🔎 原始返回", expanded=False):
    st.json(result)

st.caption("下一步：把推荐因子拿到「因子研究」验证 IC/IR，或接入「策略回测」构建组合。")
