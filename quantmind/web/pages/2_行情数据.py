"""行情数据：多市场 K 线查询、指标叠加、导出"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from utils.theme import (  # noqa: E402
    setup_page, page_header, section, kpi_row, note, guard_error, fmt_num, tone_of,
)
from utils.api_client import APIClient  # noqa: E402
from utils.charts import create_price_chart, create_returns_histogram  # noqa: E402
from utils.constants import (  # noqa: E402
    EXCHANGES, EXCHANGE_NAMES, INTERVALS, INTERVAL_NAMES, SYMBOL_PRESETS,
)

setup_page("行情数据", "📈")
page_header("行情数据", "商品期货 / 金融期货 / A股 / 港股 多市场历史行情查询与可视化", "📈")

if "md_symbol" not in st.session_state:
    st.session_state.md_symbol = "IF0"
if "md_exchange" not in st.session_state:
    st.session_state.md_exchange = "CFFEX"

# ---------------------------------------------------------------- 速选（迷你卡片按钮）
section("常用标的", "点击即可填入查询条件")
tabs = st.tabs(list(SYMBOL_PRESETS.keys()))
for tab, (cls, items) in zip(tabs, SYMBOL_PRESETS.items()):
    with tab:
        cols = st.columns(min(len(items), 6), gap="small")
        for i, (sym, exch, name) in enumerate(items):
            with cols[i % len(cols)]:
                if st.button(
                    f"**{name}**\n`{sym}`",
                    key=f"preset_{cls}_{sym}",
                    width="stretch",
                    help=f"{name} ({sym}.{exch})",
                ):
                    st.session_state.md_symbol = sym
                    st.session_state.md_exchange = exch
                    st.rerun()

# ---------------------------------------------------------------- 查询条件
section("查询条件")
with st.form("md_query"):
    c1, c2, c3, c4 = st.columns([2, 2, 1.4, 1.4])
    with c1:
        symbol = st.text_input("合约代码", st.session_state.md_symbol,
                               help="期货主连 rb0 / 具体合约 rb2501 / A股 600519 / 港股 00700")
    with c2:
        all_ex = [e for exs in EXCHANGES.values() for e in exs]
        idx = all_ex.index(st.session_state.md_exchange) if st.session_state.md_exchange in all_ex else 0
        exchange = st.selectbox("交易所", all_ex, index=idx,
                                format_func=lambda x: f"{x} · {EXCHANGE_NAMES.get(x, '')}")
    with c3:
        interval = st.selectbox("周期", INTERVALS, index=6,
                                format_func=lambda x: INTERVAL_NAMES.get(x, x))
    with c4:
        limit = st.selectbox("最多加载", [200, 500, 1000], index=1)

    c5, c6, c7 = st.columns([1.5, 1.5, 2])
    with c5:
        use_range = st.checkbox("指定日期区间", value=False)
    with c6:
        ma_opt = st.multiselect("均线", [5, 10, 20, 60, 120], default=[5, 20, 60])
    with c7:
        d1, d2 = st.columns(2)
        start = d1.date_input("开始", value=None, disabled=not use_range, format="YYYY-MM-DD")
        end = d2.date_input("结束", value=None, disabled=not use_range, format="YYYY-MM-DD")

    submitted = st.form_submit_button("🔍 查询行情", type="primary", width="stretch")


@st.cache_data(ttl=60, show_spinner=False)
def fetch(symbol, exchange, interval, start, end, limit):
    # 先取第 1 页拿到总条数，再请求最后一页，确保展示的是"最新" limit 根，
    # 而不是最旧的 limit 根（page=1 对长历史序列会倒退到多年前的旧数据）。
    first = APIClient.data(symbol, exchange, interval, start=start, end=end,
                           page=1, page_size=limit, timeout=60)
    total = (first.get("pagination") or {}).get("total", 0) or len(first.get("data") or [])
    last_page = max(1, -(-total // limit)) if total else 1
    if last_page <= 1:
        return first
    last = APIClient.data(symbol, exchange, interval, start=start, end=end,
                          page=last_page, page_size=limit, timeout=60)
    return last


if not submitted:
    note(
        "选择标的后点击「查询行情」。系统优先读取本地数据目录，其次在线数据源"
        "（AKShare / mootdx / yfinance），完全离线时自动降级为内置 MockFeed，"
        "保证流程始终可跑通。",
        "info",
    )
    st.stop()

st.session_state.md_symbol = symbol
st.session_state.md_exchange = exchange
s = start.isoformat() if (use_range and start) else None
e = end.isoformat() if (use_range and end) else None

with st.spinner(f"正在获取 {symbol}.{exchange} …"):
    res = fetch(symbol, exchange, interval, s, e, limit)

if guard_error(res, "行情查询"):
    st.stop()

bars = res.get("data") or []
pg = res.get("pagination", {})
if not bars:
    note(f"<b>{symbol}.{exchange}</b> 在所选条件下没有数据，请检查代码、交易所或日期区间。", "warning")
    st.stop()

df = pd.DataFrame(bars)
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)

# ---------------------------------------------------------------- 概览
latest = df.iloc[-1]
prev_close = df.iloc[-2]["close"] if len(df) > 1 else latest["close"]
chg = latest["close"] - prev_close
chg_pct = chg / prev_close if prev_close else 0.0
rng = (df["high"].max() - df["low"].min())

section("最新行情", f"{EXCHANGE_NAMES.get(exchange, exchange)} · {INTERVAL_NAMES.get(interval, interval)}")
kpi_row([
    {"label": "最新价", "value": fmt_num(latest["close"], 2),
     "delta": f"{chg:+.2f}  ({chg_pct * 100:+.2f}%)", "tone": tone_of(chg)},
    {"label": "最高 / 最低", "value": f"{latest['high']:.2f} / {latest['low']:.2f}",
     "hint": f"区间振幅 {rng:.2f}"},
    {"label": "成交量", "value": f"{latest['volume']:,.0f}", "tone": "accent"},
    {"label": "样本数", "value": f"{len(df)}", "hint": f"服务端共 {pg.get('total', len(df))} 根"},
    {"label": "数据区间", "value": df['datetime'].iloc[0].strftime('%Y-%m-%d'),
     "hint": f"至 {df['datetime'].iloc[-1].strftime('%Y-%m-%d')}"},
])

# ---------------------------------------------------------------- 图表
section("K 线图", "红涨绿跌 · 含成交量副图")
st.plotly_chart(
    create_price_chart(bars, title=f"{symbol}.{exchange} · {INTERVAL_NAMES.get(interval, interval)}",
                       ma=tuple(sorted(ma_opt)) if ma_opt else ()),
    width="stretch", key="md_kline",
)

# ---------------------------------------------------------------- 统计
section("收益统计")
rets = df["close"].pct_change().dropna()
left, right = st.columns([1.4, 1], gap="medium")
with left:
    st.plotly_chart(
        create_returns_histogram(rets.tolist(), title="单期收益率分布", height=320),
        width="stretch", key="md_hist",
    )
with right:
    # 年化因子 = 每年期数（1d=252 交易日，1w=52 周，日内按实际交易时长折算）
    _ppy = {"1d": 252, "1w": 52, "1h": 252 * 4, "30m": 252 * 8,
            "15m": 252 * 16, "5m": 252 * 48, "1m": 252 * 240}
    ann = _ppy.get(interval, 252)
    stats = {
        "样本数": f"{len(rets)}",
        "平均收益": f"{rets.mean() * 100:.4f}%",
        "标准差": f"{rets.std() * 100:.4f}%",
        "年化波动": f"{rets.std() * (ann ** 0.5) * 100:.2f}%",
        "最大单期涨幅": f"{rets.max() * 100:.2f}%",
        "最大单期跌幅": f"{rets.min() * 100:.2f}%",
        "上涨占比": f"{(rets > 0).mean() * 100:.1f}%",
        "偏度": f"{rets.skew():.3f}",
        "峰度": f"{rets.kurtosis():.3f}",
    }
    st.dataframe(
        [{"指标": k, "数值": v} for k, v in stats.items()],
        width="stretch", hide_index=True, height=340,
    )

# ---------------------------------------------------------------- 数据表
section("原始数据")
show = df.sort_values("datetime", ascending=False).copy()
show["datetime"] = show["datetime"].dt.strftime("%Y-%m-%d %H:%M")
show = show.rename(columns={
    "datetime": "时间", "open": "开盘", "high": "最高", "low": "最低",
    "close": "收盘", "volume": "成交量", "symbol": "代码", "exchange": "交易所",
    "interval": "周期",
})
st.dataframe(show[["时间", "开盘", "最高", "最低", "收盘", "成交量"]],
             width="stretch", hide_index=True, height=380)

st.download_button(
    "⬇️ 导出 CSV",
    df.to_csv(index=False).encode("utf-8-sig"),
    file_name=f"{symbol}_{exchange}_{interval}.csv",
    mime="text/csv",
)
st.caption("提示：数据完整性存疑时，可到「数据质量」页对同一标的做一次体检。")
