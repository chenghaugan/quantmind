"""数据管理：数据源概览 / 下载入库 / 本地数据文件清单。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from utils.theme import (  # noqa: E402
    setup_page, page_header, section, note, verdict, guard_error, kpi_row,
)
from utils.api_client import APIClient  # noqa: E402
from utils.constants import EXCHANGES, EXCHANGE_NAMES, INTERVALS, INTERVAL_NAMES  # noqa: E402

setup_page("数据管理", "🗂️")
page_header(
    "数据管理",
    "查看已注册数据源、配置的本地数据根目录，下载/更新标的数据入库，并浏览本地 parquet/csv 数据文件。",
    "🗂️",
)

note(
    "<b>说明</b>：下载经由 DataManager 的多源回退链拉取并回写缓存/存储。"
    "本地数据路径（期货 / A股 / 港股 / 期权 / 席位）可在「设置」页统一配置。",
    "info",
)


# ---------------------------------------------------------------- 数据源概览
section("数据源与本地根目录")
feeds_res = APIClient.feeds(timeout=5)
roots_res = APIClient.data_roots(timeout=10)
if isinstance(feeds_res, dict) and not feeds_res.get("error"):
    feeds = feeds_res.get("feeds", [])
    kpi_row([{"label": "已注册数据源", "value": len(feeds), "tone": "accent",
              "hint": " · ".join(feeds[:5])}])
else:
    note("无法获取数据源清单，请确认后端已启动。", "warning")

if isinstance(roots_res, dict) and not roots_res.get("error"):
    rows = [{"路径": k, "取值": (roots_res.get(k) or "（未配置）")} for k in
            ["local_data_root", "local_stock_root", "local_hk_root",
             "local_option_root", "seat_data_root"]]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

st.markdown("---")

# ---------------------------------------------------------------- 下载入库
section("下载 / 更新入库")
with st.form("dl_form"):
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        symbol = st.text_input("合约代码", "rb0")
        all_ex = [e for exs in EXCHANGES.values() for e in exs]
        exchange = st.selectbox("交易所", all_ex, index=0,
                                format_func=lambda x: f"{x} · {EXCHANGE_NAMES.get(x, '')}")
    with c2:
        interval = st.selectbox("周期", INTERVALS,
                                format_func=lambda x: INTERVAL_NAMES.get(x, x),
                                index=INTERVALS.index("1d"))
        start = st.text_input("开始日期", "", placeholder="YYYY-MM-DD，留空表示不限")
    with c3:
        end = st.text_input("结束日期", "", placeholder="YYYY-MM-DD，留空表示最新")
        submit = st.form_submit_button("⬇️ 下载入库", type="primary", width="stretch")

if submit:
    payload = {"symbol": symbol, "exchange": exchange, "interval": interval}
    if start:
        payload["start"] = start
    if end:
        payload["end"] = end
    with st.spinner(f"正在拉取 {symbol}.{exchange} {interval} 并入库…"):
        dl = APIClient.data_download(payload)
    if guard_error(dl, "下载"):
        st.stop()
    if dl.get("downloaded"):
        verdict(f"入库成功：{symbol}.{exchange} 共 {dl['downloaded']} 根 K 线 "
                f"（{dl.get('start', '—')} ~ {dl.get('end', '—')}）", "ok", icon="✅")
    else:
        note(f"未取到数据：{dl.get('error', '未知原因')}", "error")

st.markdown("---")

# ---------------------------------------------------------------- 本地文件清单
section("本地数据文件")
if st.button("🔄 刷新文件清单"):
    st.cache_data.clear()
if isinstance(roots_res, dict) and roots_res.get("error"):
    st.stop()

files_res = APIClient.data_files(timeout=15)
if guard_error(files_res, "读取文件清单"):
    st.stop()

total = files_res.get("total_files", 0)
groups = files_res.get("groups", []) or []
kpi_row([{"label": "本地数据文件数", "value": total, "tone": "accent"}])

if not groups:
    note("未读取到本地数据文件。请先在「设置」页配置本地数据路径。", "warning")
    st.stop()

for g in groups:
    label = g.get("label", "")
    root = g.get("root") or "（未配置）"
    cnt = g.get("count", 0)
    with st.expander(f"{label} · {root} · {cnt} 个文件", expanded=False):
        fl = g.get("files", [])
        if not fl:
            st.caption("该根目录下无 parquet/csv 文件。")
            continue
        df = pd.DataFrame(fl)
        st.dataframe(df, width="stretch", height=280, hide_index=True)
