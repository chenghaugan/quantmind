"""设置页：配置 AI 研究功能使用的模型供应商 / API Key / Base URL / 模型 / 温度。

配置保存在后端 ``config/ai_settings.json``，保存后即时生效，无需重启服务。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st  # noqa: E402

from utils.theme import (  # noqa: E402
    setup_page, page_header, section, note, verdict, guard_error, badge,
    fmt_pct,
)
from utils.api_client import APIClient  # noqa: E402

setup_page("设置", "⚙️")
page_header(
    "设置",
    "在此配置 AI 研究功能（自然语言 → 因子 / 策略代码）所调用的模型。"
    "支持所有 OpenAI 兼容服务：DeepSeek、OpenAI、通义千问、OpenRouter 等。",
    "⚙️",
)

note(
    "未填写 API Key 时，系统使用内置 **Mock 模型**，可在无网络环境下完整演示研究流程；"
    "填写真实凭据后，AI 研究将调用你配置的模型。配置保存在后端，**同时写入项目根 `.env`**，"
    "刷新页面或重启服务都不会丢失，容器化部署也能通过环境变量生效（与本地配置文件双向同步）。",
    "info",
)

# ----------------------------------------------------------------- 读取现状
with st.spinner("读取当前配置…"):
    cur = APIClient.ai_settings()
if guard_error(cur, "AI 设置"):
    st.stop()

PROVIDER_LABELS = {
    "mock": "Mock（离线演示，无需 Key）",
    "openai": "OpenAI 兼容（DeepSeek / 通义 / OpenRouter 等）",
}
provider_val = cur.get("provider", "mock")
api_key_val = cur.get("api_key", "") or ""
base_url_val = cur.get("base_url", "") or ""
model_val = cur.get("model", "") or ""
temp_val = float(cur.get("temperature", 0.7) or 0.7)

# 当前生效状态
is_real = provider_val != "mock" and bool(api_key_val)
st.markdown(
    badge(f"当前模型：{PROVIDER_LABELS.get(provider_val, provider_val)}",
          "success" if is_real else "info"),
    unsafe_allow_html=True,
)
st.caption("真实模型已生效" if is_real else "当前为离线 Mock 模型，AI 研究返回确定性演示结果")

# 配置来源
source = cur.get("source", "default")
SOURCE_LABELS = {
    "json": "本地配置文件（config/ai_settings.json）",
    "env": "环境变量文件（项目根 .env）",
    "default": "代码默认值（未单独配置）",
}
st.info(f"配置来源：{SOURCE_LABELS.get(source, source)}", icon="📂")

# ================================================================= 配置表单
section("AI 模型配置")
with st.container(border=True):
    provider = st.selectbox(
        "模型供应商",
        ["mock", "openai"],
        index=["mock", "openai"].index(provider_val) if provider_val in ("mock", "openai") else 0,
        format_func=lambda x: PROVIDER_LABELS.get(x, x),
        help="选择后下方字段含义不同：Mock 无需填写任何凭据。",
    )
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        api_key = st.text_input(
            "API Key", value=api_key_val, type="password",
            placeholder="sk-...（DeepSeek / OpenAI / 通义 等）",
            help="以明文保存在后端配置文件中，仅你自己可见。",
        )
        base_url = st.text_input(
            "Base URL", value=base_url_val,
            placeholder="https://api.deepseek.com/v1",
            help="OpenAI 兼容的接口地址，通常需以 /v1 结尾。",
        )
    with col2:
        model = st.text_input(
            "模型名称", value=model_val,
            placeholder="deepseek-chat / gpt-4o-mini / qwen-plus",
        )
        temperature = st.slider(
            "采样温度", min_value=0.0, max_value=1.0, value=temp_val,
            step=0.05, help="越低越确定，越高越发散。",
        )

    save_btn, test_btn = st.columns([1, 1], gap="medium")
    with save_btn:
        save_clicked = st.button("💾 保存配置", type="primary", width="stretch")
    with test_btn:
        test_clicked = st.button("🔌 测试连接", width="stretch")

payload = {
    "provider": provider,
    "api_key": api_key,
    "base_url": base_url,
    "model": model,
    "temperature": temperature,
}

if save_clicked:
    with st.spinner("保存中…"):
        res = APIClient.ai_settings_save(payload)
    if isinstance(res, dict) and res.get("ok"):
        new_provider = res.get("provider", provider)
        st.toast("配置已保存并即时生效", icon="✅")
        synced = res.get("synced_env")
        tail = "（真实模型）" if new_provider != "mock" else ""
        extra = "，并已同步写入项目根 `.env`（重启 / 容器部署仍生效）" if synced else ""
        verdict(
            f"已保存。当前生效模型：{PROVIDER_LABELS.get(new_provider, new_provider)}{tail}{extra}",
            "ok", icon="✅",
        )
        st.rerun()
    else:
        verdict(f"保存失败：{res}", "bad", icon="⛔")

if test_clicked:
    with st.spinner("正在向模型发送测试请求…"):
        test_res = APIClient.ai_settings_test(payload)
    if isinstance(test_res, dict) and test_res.get("ok"):
        sample = test_res.get("sample", "")
        st.toast(f"连接成功（{test_res.get('provider')}）", icon="✅")
        verdict("连接成功，模型返回了有效响应。", "ok", icon="✅")
        with st.expander("查看模型返回片段", expanded=True):
            st.code(sample, language="text")
    else:
        err = test_res.get("error", "未知错误") if isinstance(test_res, dict) else str(test_res)
        verdict(f"连接失败：{err}", "bad", icon="⛔")
        st.caption("请检查 API Key、Base URL 与模型名称是否正确，以及网络是否可达。")

# ================================================================= 本地数据路径
st.divider()
section("本地数据路径")
with st.spinner("读取本地数据配置…"):
    roots = APIClient.data_roots(timeout=10)
if isinstance(roots, dict) and not roots.get("error"):
    st.markdown(
        "配置各资产类别的本地数据根目录。填入后，对应数据源（期货 CSV / A股 Parquet / 港股 / 期权 / 席位）"
        "会**优先于在线源**被使用。保存后同步写入项目根 `.env`，重启不丢失。",
        unsafe_allow_html=True,
    )
    FIELD_META = [
        ("local_data_root", "本地数据根目录（公共）", "如 china-futures CSV 克隆路径"),
        ("local_stock_root", "A 股数据根目录", "如 astock-data-toolkit 的 Parquet/CSV"),
        ("local_hk_root", "港股数据根目录", "如东方财富导出的港股日频 Parquet/CSV"),
        ("local_option_root", "期权数据根目录", "如股指/ETF/商品期权日频 Parquet/CSV"),
        ("seat_data_root", "期货席位数据根目录", "如 TradingAgents qihuo/database/positioning"),
    ]
    with st.container(border=True):
        vals = {}
        for field, label, hint in FIELD_META:
            vals[field] = st.text_input(
                label, value=(roots.get(field) or ""), key=f"local_{field}",
                placeholder="留空则使用在线数据源", help=hint,
            )
        if st.button("💾 保存本地数据路径", type="primary", width="stretch"):
            payload = {k: (v or "").strip() for k, v in vals.items()}
            with st.spinner("保存中…"):
                res = APIClient.data_roots_save(payload)
            if isinstance(res, dict) and not res.get("error"):
                synced = res.get("synced_env")
                extra = "，并已同步写入 `.env`" if synced else ""
                verdict(f"本地数据路径已保存{extra}。需重启后端使数据源注册生效。", "ok", icon="✅")
            else:
                verdict(f"保存失败：{res}", "bad", icon="⛔")
else:
    note("无法读取本地数据配置，请确认后端已启动。", "warning")

# ================================================================= 告警通知
st.divider()
section("告警通知")
with st.spinner("读取告警配置…"):
    alert = APIClient.alert_settings(timeout=10)
if isinstance(alert, dict) and not alert.get("error"):
    st.markdown(
        "配置监控告警的推送通道（当前支持 Telegram 风格 Webhook）。启用后，"
        "风险管理 / 成交等事件将按此配置推送。",
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        enabled = st.toggle("启用告警推送", value=bool(alert.get("enabled")))
        channel = st.selectbox("推送渠道", ["telegram", "generic_webhook"],
                               index=0, help="telegram：webhook+chat_id；generic_webhook：仅 webhook")
        a1, a2 = st.columns(2, gap="medium")
        with a1:
            webhook_url = st.text_input("Webhook URL", value=(alert.get("webhook_url") or ""),
                                        help="如 https://api.telegram.org/bot<TOKEN>/sendMessage")
            chat_id = st.text_input("Chat ID", value=(alert.get("chat_id") or ""),
                                    help="Telegram 会话/群组 ID")
        with a2:
            secret = st.text_input("密钥（可选）", value=(alert.get("secret") or ""),
                                   type="password", help="部分渠道签名需要")
        if st.button("💾 保存告警配置", type="primary", width="stretch"):
            ares = APIClient.alert_settings_save({
                "enabled": enabled, "channel": channel,
                "webhook_url": webhook_url, "chat_id": chat_id, "secret": secret,
            })
            if isinstance(ares, dict) and not ares.get("error"):
                verdict("告警配置已保存。", "ok", icon="✅")
            else:
                verdict(f"保存失败：{ares}", "bad", icon="⛔")
else:
    note("无法读取告警配置，请确认后端已启动。", "warning")

# ================================================================= 说明
section("支持的模型服务")
st.markdown(
    "- **DeepSeek**：Base URL 填 `https://api.deepseek.com/v1`，模型填 `deepseek-chat` 或 `deepseek-reasoner`。\n"
    "- **OpenAI**：Base URL 填 `https://api.openai.com/v1`，模型填 `gpt-4o-mini` 等。\n"
    "- **通义千问**：Base URL 填 `https://dashscope.aliyuncs.com/compatible-mode/v1`，模型填 `qwen-plus` 等。\n"
    "- **OpenRouter / 其他**：填入对应的兼容 `/v1` 地址与模型名即可。\n"
    "- 所有服务均通过标准 `POST /v1/chat/completions` 调用，无需安装额外 SDK。"
)

st.divider()
st.caption("💡 设置仅影响「AI 研究」页面调用的模型；行情、回测、风控等功能不依赖此配置。")
