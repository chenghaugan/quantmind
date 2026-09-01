"""数据管理：数据源概览 / 下载入库 / 本地数据文件清单。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from utils.theme import (  # noqa: E402
    setup_page, page_header, section, note, verdict, guard_error, kpi_row, fmt_num,
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
section("数据源")
feeds_res = APIClient.feeds(timeout=5)
if isinstance(feeds_res, dict) and not feeds_res.get("error"):
    feeds = feeds_res.get("feeds", [])
    kpi_row([{"label": "已注册数据源", "value": len(feeds), "tone": "accent",
              "hint": " · ".join(feeds)}])
    note(
        "系统按优先级依次尝试数据源，单源失败自动降级到下一个：<br>"
        "<b>TqSdk</b>（8000根，首选）→ <b>akshare_future</b>（新浪）→ <b>efinance</b>（东财）→ ... → mock（兜底）。",
        "info",
    )
else:
    note("无法获取数据源清单，请确认后端已启动。", "warning")

st.markdown("---")

# ---------------------------------------------------------------- 期货数据自动下载
section("期货数据下载")
note(
    "<b>支持股指期货 + 商品期货</b>，可自定义品种、交易所、周期。"
    "调度器每个交易日 16:30 自动下载默认配置（股指期货全周期）。"
    "也可点按下方按钮手动触发，或自定义下载。",
    "info",
)

# 期货品种配置
FUTURES_CONFIG = {
    "股指期货": {
        "IF0": "沪深300",
        "IC0": "中证500",
        "IH0": "上证50",
        "IM0": "中证1000",
    },
    "商品期货-黑色系": {
        "rb0": "螺纹钢",
        "hc0": "热卷",
        "i0": "铁矿石",
        "j0": "焦炭",
        "jm0": "焦煤",
    },
    "商品期货-有色金属": {
        "cu0": "铜",
        "al0": "铝",
        "zn0": "锌",
        "pb0": "铅",
        "ni0": "镍",
        "sn0": "锡",
    },
    "商品期货-贵金属": {
        "au0": "黄金",
        "ag0": "白银",
    },
    "商品期货-能源化工": {
        "sc0": "原油",
        "fu0": "燃料油",
        "lu0": "低硫燃料油",
        "bu0": "沥青",
        "ru0": "橡胶",
        "ma0": "甲醇",
        "TA0": "PTA",
        "PP0": "聚丙烯",
        "L0": "塑料",
        "V0": "PVC",
        "eg0": "乙二醇",
        "eb0": "苯乙烯",
    },
    "商品期货-农产品": {
        "m0": "豆粕",
        "y0": "豆油",
        "a0": "豆一",
        "p0": "棕榈油",
        "OI0": "菜油",
        "RM0": "菜粕",
        "SR0": "白糖",
        "CF0": "棉花",
        "AP0": "苹果",
    },
}

# 交易所映射
EXCHANGE_MAP = {
    "IF0": "CFFEX", "IC0": "CFFEX", "IH0": "CFFEX", "IM0": "CFFEX",
    "rb0": "SHFE", "hc0": "SHFE", "cu0": "SHFE", "al0": "SHFE", "zn0": "SHFE",
    "pb0": "SHFE", "ni0": "SHFE", "sn0": "SHFE", "au0": "SHFE", "ag0": "SHFE",
    "fu0": "SHFE", "lu0": "SHFE", "bu0": "SHFE", "ru0": "SHFE",
    "i0": "DCE", "j0": "DCE", "jm0": "DCE", "m0": "DCE", "y0": "DCE",
    "a0": "DCE", "p0": "DCE", "PP0": "DCE", "L0": "DCE", "V0": "DCE",
    "eg0": "DCE", "eb0": "DCE",
    "TA0": "CZCE", "MA0": "CZCE", "OI0": "CZCE", "RM0": "CZCE",
    "SR0": "CZCE", "CF0": "CZCE", "AP0": "CZCE",
    "sc0": "INE",
}

# 周期配置
INTERVAL_OPTIONS = {
    "1d": "日线（~10年）",
    "4h": "4小时（~6.6年）",
    "2h": "2小时（~6.6年）",
    "60m": "60分钟（~6.6年）",
    "30m": "30分钟（~4年）",
    "15m": "15分钟（~2年）",
    "5m": "5分钟（~8个月）",
    "3m": "3分钟（~3个月）",
    "1m": "1分钟（~48天）",
}

# ================================================================
# 一键全量下载（全期货 × 全周期，增量到最新）
# ================================================================
# 汇总所有品种
ALL_FUTURES = []
for _cat, _syms in FUTURES_CONFIG.items():
    for _sym, _name in _syms.items():
        if _sym not in [x[0] for x in ALL_FUTURES]:
            ALL_FUTURES.append((_sym, _name, _cat))
ALL_SYMBOLS = [s for s, _, _ in ALL_FUTURES]
ALL_INTERVALS = list(INTERVAL_OPTIONS.keys())

# 定时配置获取
fd_cfg = APIClient.get("/settings/futures-download", timeout=10)
cur_cron = (fd_cfg.get("schedule_cron") if isinstance(fd_cfg, dict) else None) or "30 16 * * 1-5"
fd_enabled = (fd_cfg.get("enabled") if isinstance(fd_cfg, dict) else True)

# ---- 一键全量下载按钮 ----
st.markdown("### ⬇️ 立即下载（增量到最新）")
b1, b2, b3 = st.columns([2, 2, 1])
with b1:
    full_download = st.button("🚀 一键全量下载（全部期货 × 全周期）", type="primary", width="stretch")
with b2:
    default_download = st.button("📦 仅股指期货全周期", width="stretch")
with b3:
    refresh_btn = st.button("🔄 刷新", width="stretch")

st.caption(f"全量 = {len(ALL_SYMBOLS)} 个品种 × {len(ALL_INTERVALS)} 个周期 = {len(ALL_SYMBOLS)*len(ALL_INTERVALS)} 个任务；已下载的数据自动跳过（增量），只拉当天新增。")

st.markdown("---")
with st.expander("⚙️ 自定义选择（品种 / 周期）", expanded=False):
    st.markdown("**选择期货类别**（不选=全部）")
    selected_categories = st.multiselect(
        "期货类别",
        options=list(FUTURES_CONFIG.keys()),
        default=list(FUTURES_CONFIG.keys()),
        help="选择要下载的期货类别",
    )
    st.markdown("**选择周期**（不选=全部）")
    selected_intervals = st.multiselect(
        "数据周期",
        options=list(INTERVAL_OPTIONS.keys()),
        default=ALL_INTERVALS,
        format_func=lambda x: INTERVAL_OPTIONS[x],
        help="TqSdk 免费版每个周期最多 8000 根",
    )
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        custom_download = st.button("⬇️ 下载所选", type="primary", width="stretch")
    with col_btn2:
        custom_download = st.button("⬇️ 下载所选", type="primary", width="stretch", key="cd2")

# ---- 定时更新配置（期货 / A股 / 港股 / 美股分市场独立设置）----
st.markdown("---")
st.markdown("### ⏰ 定时自动更新")
st.caption("每个市场可独立设置开关与更新时间（周一至周五自动触发，收盘后增量更新到最新）。A股/港股/美股仅更新仓库内已缓存标的（首次建库用下方「全市场预热」）。")


def _parse_hm(cron: str, default_h: int, default_m: int):
    """从 cron 表达式解析 (小时, 分钟)；解析失败用默认值。"""
    try:
        parts = str(cron).split()
        return int(parts[1]), int(parts[0])
    except Exception:  # noqa: BLE001
        return default_h, default_m


mu_cfg = APIClient.get("/settings/market-update", timeout=10)
mu_cfg = mu_cfg if isinstance(mu_cfg, dict) else {}

# (key, 名称, 默认(时,分), 说明)
_MARKET_ROWS = [
    ("futures", "📈 期货", (16, 30), "收盘后增量更新全部品种 × 周期（默认 16:30）"),
    ("a_stock", "A股（沪深）", (23, 0), "15:00 收盘后更新已缓存 A 股（日线/60m/30m）"),
    ("hk_stock", "🏝️ 港股", (23, 0), "16:00 收盘后更新已缓存港股（日线）"),
    ("us_stock", "🗽 美股", (5, 0), "北京时间凌晨更新已缓存美股（日线，16:00 ET 收盘后）"),
]

_schedule_rows = []  # (key, enabled, hour, minute) 收集后统一保存
with st.form("market_schedule_form"):
    # 表头
    hc1, hc2, hc3, hc4 = st.columns([2, 1, 1, 1])
    hc1.markdown("**市场**")
    hc2.markdown("**启用**")
    hc3.markdown("**小时**")
    hc4.markdown("**分钟**")
    for key, name, (dh, dm_), desc in _MARKET_ROWS:
        if key == "futures":
            enabled0 = bool(fd_cfg.get("enabled", True)) if isinstance(fd_cfg, dict) else True
            h0, m0 = _parse_hm(cur_cron, dh, dm_)
        else:
            mc = mu_cfg.get(key) or {}
            enabled0 = bool(mc.get("enabled", False))
            h0, m0 = _parse_hm(mc.get("schedule_cron", ""), dh, dm_)
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        with c1:
            st.markdown(f"**{name}**")
            st.caption(desc)
        with c2:
            en = st.toggle("启用", value=enabled0, key=f"sch_{key}_enabled",
                           label_visibility="collapsed")
        with c3:
            hh = st.number_input("时", min_value=0, max_value=23, value=h0, step=1,
                                 key=f"sch_{key}_h", label_visibility="collapsed")
        with c4:
            mm = st.number_input("分", min_value=0, max_value=59, value=m0, step=1,
                                 key=f"sch_{key}_m", label_visibility="collapsed")
        _schedule_rows.append((key, en, int(hh), int(mm)))

    if st.form_submit_button("💾 保存定时配置", type="primary"):
        ok_msgs, err_msgs = [], []
        stock_payload = {}
        for key, en, hh, mm in _schedule_rows:
            cron = f"{mm} {hh} * * 1-5"
            if key == "futures":
                sres = APIClient.put("/settings/futures-download",
                                     json={"enabled": bool(en), "schedule_cron": cron}, timeout=10)
                if isinstance(sres, dict) and not sres.get("error"):
                    ok_msgs.append(f"期货 {'✅' if en else '⏸️'} {hh:02d}:{mm:02d}")
                else:
                    err_msgs.append(f"期货: {sres}")
            else:
                stock_payload[key] = {"enabled": bool(en), "schedule_cron": cron}
                ok_msgs.append(
                    {"a_stock": "A股", "hk_stock": "港股", "us_stock": "美股"}[key]
                    + (" ✅" if en else " ⏸️") + f" {hh:02d}:{mm:02d}")
        if stock_payload:
            sres = APIClient.put("/settings/market-update", json=stock_payload, timeout=10)
            if not (isinstance(sres, dict) and not sres.get("error")):
                err_msgs.append(f"股票市场: {sres}")
        if err_msgs:
            verdict(f"保存失败：{'；'.join(err_msgs)}", "bad", icon="⛔")
        else:
            verdict(f"定时配置已保存：{'；'.join(ok_msgs)}。", "ok", icon="✅")

# ---- 确定本次下载的任务参数 ----
submit_download = full_download or default_download or 'custom_download' in locals() and custom_download
if submit_download:
    if full_download or ('custom_download' in locals() and custom_download and selected_categories == list(FUTURES_CONFIG.keys()) and selected_intervals == ALL_INTERVALS):
        # 全量：全部品种 × 全部周期
        symbols_list = ALL_SYMBOLS
        intervals_list = ALL_INTERVALS
        is_full = True
    elif default_download:
        # 仅股指期货全周期
        symbols_list = list(FUTURES_CONFIG["股指期货"].keys())
        intervals_list = ALL_INTERVALS
        is_full = False
    else:
        # 自定义选择
        if not selected_categories or not selected_intervals:
            st.error("请至少选择一个类别和一个周期")
            st.stop()
        symbols_list = []
        for cat in selected_categories:
            if cat in FUTURES_CONFIG:
                for sym, name in FUTURES_CONFIG[cat].items():
                    if sym not in symbols_list:
                        symbols_list.append(sym)
        intervals_list = selected_intervals
        is_full = False

    if not symbols_list or not intervals_list:
        st.error("请至少选择一个品种和一个周期")
        st.stop()

    # 提交异步任务
    payload = {"symbols": symbols_list, "intervals": intervals_list}
    try:
        start_result = APIClient.post("/data/futures/download/start", json=payload, timeout=10)
        if start_result.get("error"):
            st.error(f"提交任务失败：{start_result.get('error')}")
            st.stop()
        task_id = start_result.get("task_id")
        if not task_id:
            st.error("未返回任务 ID")
            st.stop()
        st.session_state["futures_download_task_id"] = task_id
        st.session_state["futures_download_started"] = True
        st.session_state["futures_download_desc"] = f"{len(symbols_list)} 品种 × {len(intervals_list)} 周期"
    except Exception as e:
        st.error(f"提交任务失败：{e}")
        st.stop()

# 轮询任务状态
if st.session_state.get("futures_download_started"):
    task_id = st.session_state.get("futures_download_task_id")
    if task_id:
        import time

        desc = st.session_state.get("futures_download_desc", "下载")
        st.markdown(f"### 🔄 正在执行：{desc}")
        progress_placeholder = st.empty()
        max_polls = 900  # 最多轮询 15 分钟
        poll_count = 0

        while poll_count < max_polls:
            try:
                status = APIClient.get(f"/data/futures/download/status/{task_id}", timeout=10)
                if status.get("error"):
                    progress_placeholder.error(f"查询状态失败：{status.get('error')}")
                    break

                task_status = status.get("status")
                progress = status.get("progress", {})
                current = progress.get("current", 0)
                total = progress.get("total", 0)
                message = progress.get("message", "处理中...")

                with progress_placeholder.container():
                    if total > 0:
                        st.progress(current / total, text=f"{message} ({current}/{total})")
                    else:
                        st.info(message)

                if task_status == "success":
                    progress_placeholder.empty()
                    result = status.get("result", {})
                    downloaded = result.get("downloaded", 0)
                    failed = result.get("failed", 0)
                    skipped = result.get("skipped", 0)
                    up_to_date = result.get("up_to_date", 0)

                    kpi_row([
                        {"label": "新增下载", "value": downloaded, "tone": "up"},
                        {"label": "已是最新(跳过)", "value": up_to_date, "tone": "neutral"},
                        {"label": "失败", "value": failed, "tone": "down" if failed else "neutral"},
                        {"label": "总任务", "value": result.get("total", 0), "tone": "accent"},
                    ])

                    results = result.get("results", [])
                    if results:
                        df_results = pd.DataFrame(results)
                        st.dataframe(df_results, width="stretch", height=400, hide_index=True)

                    if failed == 0:
                        verdict("下载完成，已增量更新到最新。", "ok")
                    else:
                        verdict(f"完成，但有 {failed} 个任务失败", "warn")

                    st.session_state.pop("futures_download_task_id", None)
                    st.session_state.pop("futures_download_started", None)
                    st.session_state.pop("futures_download_desc", None)
                    break

                elif task_status == "error":
                    progress_placeholder.empty()
                    st.error(f"下载失败：{status.get('message')}")
                    st.session_state.pop("futures_download_task_id", None)
                    st.session_state.pop("futures_download_started", None)
                    st.session_state.pop("futures_download_desc", None)
                    break

                elif task_status == "cancelled":
                    progress_placeholder.empty()
                    st.warning("任务已取消")
                    st.session_state.pop("futures_download_task_id", None)
                    st.session_state.pop("futures_download_started", None)
                    st.session_state.pop("futures_download_desc", None)
                    break

                time.sleep(1)
                poll_count += 1

            except Exception as e:
                progress_placeholder.error(f"轮询失败：{e}")
                break
        else:
            progress_placeholder.empty()
            st.warning("下载任务仍在后台运行中（已离开当前页不会中断）。请稍后刷新页面到「行情仓库总览」查看结果。")

# ---------------------------------------------------------------- A股数据更新
section("A股数据更新（日线 / 60分钟 / 30分钟）")
note(
    "<b>A股增量更新</b>：把已缓存的 A 股标的的 日线 / 60分钟 / 30分钟 更新到最新。"
    "日线走腾讯完整历史，分钟走腾讯分钟接口。可指定单只股票，或更新全部已缓存 A 股。",
    "info",
)

astk_col1, astk_col2 = st.columns([2, 1], gap="medium")
with astk_col1:
    stock_input = st.text_input(
        "A股代码（可多只，逗号分隔，留空=更新全部已缓存）",
        "600000",
        placeholder="如 600000,000001",
        help="留空则更新行情仓库内所有已缓存的 A 股标的",
    )
ascol1, ascol2 = st.columns(2)
with ascol1:
    stock_intervals = st.multiselect(
        "周期", options=["1d", "1h", "30m"],
        default=["1d", "1h", "30m"],
        format_func=lambda x: {"1d": "日线", "1h": "60分钟", "30m": "30分钟"}[x],
    )
with ascol2:
    st.markdown("")
    st.markdown("")  # 占位对齐
    run_stock = st.button("⬇️ 更新 A股数据", type="primary", width="stretch")

if run_stock:
    if not stock_intervals:
        st.error("请至少选择一个周期")
        st.stop()
    # 解析代码 → vt 格式
    vts = []
    if stock_input.strip():
        for c in [x.strip().zfill(6) for x in stock_input.split(",") if x.strip()]:
            ex_name = "SSE" if c.startswith(('6', '9')) else "SZSE"
            vts.append(f"{c}.{ex_name}")
    else:
        vts = None
    payload = {"symbols": vts, "intervals": list(stock_intervals), "manual": True}
    try:
        start_res = APIClient.post("/data/stock/download/start", json=payload, timeout=10)
        if start_res.get("error"):
            st.error(f"提交失败：{start_res.get('error')}")
            st.stop()
        task_id = start_res.get("task_id")
        if not task_id:
            st.error("未返回任务 ID")
            st.stop()
        st.session_state["stock_download_task_id"] = task_id
        st.session_state["stock_download_started"] = True
    except Exception as e:
        st.error(f"提交失败：{e}")
        st.stop()

# 轮询 A股任务
if st.session_state.get("stock_download_started"):
    task_id = st.session_state.get("stock_download_task_id")
    if task_id:
        import time
        sp = st.empty()
        for _ in range(600):
            status = APIClient.get(f"/data/stock/download/status/{task_id}", timeout=10)
            ts = status.get("status")
            if ts is None and status.get("error"):
                # 后端 404/网络错误被 APIClient 转成 {"error"}：任务已丢失，
                # 必须清理状态退出，否则每轮都会阻塞空转 10 分钟
                sp.empty()
                st.error("任务已丢失（后端可能已重启），请重新点击更新：" + status.get("error", ""))
                st.session_state.pop("stock_download_task_id", None)
                st.session_state.pop("stock_download_started", None)
                break
            p = status.get("progress", {})
            with sp.container():
                if p.get("total"):
                    st.progress(p["current"] / p["total"], text=f"{(p.get('message') or '')} ({p['current']}/{p['total']})")
                else:
                    st.info(p.get("message", "处理中..."))
            if ts == "success":
                sp.empty()
                r = status.get("result", {})
                kpi_row([
                    {"label": "更新", "value": r.get("updated", 0), "tone": "up"},
                    {"label": "已最新", "value": r.get("up_to_date", 0), "tone": "neutral"},
                    {"label": "失败", "value": r.get("failed", 0), "tone": "down" if r.get("failed") else "neutral"},
                ])
                if r.get("results"):
                    st.dataframe(pd.DataFrame(r["results"]), width="stretch", height=300, hide_index=True)
                verdict("A股数据更新完成", "ok" if not r.get("failed") else "warn")
                st.session_state.pop("stock_download_task_id", None)
                st.session_state.pop("stock_download_started", None)
                break
            elif ts == "not_found":
                sp.empty()
                st.error(status.get("message") or "任务不存在（后端可能已重启），请重新点击更新")
                st.session_state.pop("stock_download_task_id", None)
                st.session_state.pop("stock_download_started", None)
                break
            elif ts in ("error", "cancelled"):
                sp.empty()
                st.error(status.get("message"))
                st.session_state.pop("stock_download_task_id", None)
                st.session_state.pop("stock_download_started", None)
                break
            time.sleep(1)

# ---------------------------------------------------------------- 全市场预热
section("全市场预热（区分 A股 / 港股）")
note(
    "<b>点一次 = 自动建完所有未缓存的标的</b>（不再是一趟 50 只）。A股与港股分开处理，"
    "可选覆盖范围、各自进度独立展示；已缓存的自动跳过，断点可续。"
    "开启 <code>QM_MARKET_WARM_ENABLED=true</code> 后定时调度也会继续兜底推进。",
    "info",
)
warm_markets = st.multiselect(
    "预热市场",
    options=["A", "HK"],
    default=["A", "HK"],
    format_func=lambda m: {"A": "🇨🇳 A股（沪深）", "HK": "🇭🇰 港股"}[m],
    help="选择本次预热哪些市场；A股=沪深两市，港股=香港主板",
)
mcol1, mcol2 = st.columns([1, 3], gap="medium")
if mcol1.button("🚀 全市场预热（增量）", type="primary", width="stretch"):
    if not warm_markets:
        st.error("请至少选择一个市场")
        st.stop()
    start_res = APIClient.cache_warm_market_start(markets=warm_markets, full=True)
    if start_res.get("error"):
        st.error(f"提交失败：{start_res.get('error')}")
        st.stop()
    task_id = start_res.get("task_id")
    if not task_id:
        st.error("未返回任务 ID")
        st.stop()
    st.session_state["market_warm_task_id"] = task_id
    st.session_state["market_warm_started"] = True

# 轮询全市场预热任务（后台异步，实时进度条，切页不中断）
if st.session_state.get("market_warm_started"):
    task_id = st.session_state.get("market_warm_task_id")
    if task_id:
        import time
        sp = st.empty()
        for _ in range(7200):  # 最多轮询 ~2 小时（全市场 5000+ 只约需 1 小时+）
            status = APIClient.cache_warm_market_status(task_id)
            ts = status.get("status")
            if ts is None and status.get("error"):
                # 任务丢失（后端重启/裁剪）：立即退出，否则空转满 7200 次（约 2 小时）
                sp.empty()
                st.error("任务已丢失（后端可能已重启），请重新启动预热：" + status.get("error", ""))
                st.session_state.pop("market_warm_task_id", None)
                st.session_state.pop("market_warm_started", None)
                break
            p = status.get("progress", {})
            with sp.container():
                if p.get("total"):
                    st.progress(p["current"] / p["total"],
                                text=f"{(p.get('message') or '')} ({p['current']}/{p['total']})")
                else:
                    st.info(p.get("message", "处理中..."))
            if ts == "success":
                sp.empty()
                mw = status.get("result", {}) or {}
                if mw.get("skipped"):
                    verdict(f"已跳过：{mw.get('reason', '未知原因')}", "warn")
                else:
                    kpi_row([
                        {"label": "本趟目标", "value": mw.get("target", 0), "tone": "accent"},
                        {"label": "成功拉取", "value": mw.get("warmed", 0), "tone": "up"},
                        {"label": "失败", "value": mw.get("failed", 0), "tone": "down" if mw.get("failed") else "neutral"},
                        {"label": "剩余待建", "value": mw.get("pending_left", 0), "tone": "accent"},
                    ])
                    # 分市场明细
                    by_market = mw.get("by_market") or {}
                    if by_market:
                        st.markdown("**分市场明细**")
                        rows = []
                        for m, r in by_market.items():
                            name = {"A": "A股（沪深）", "HK": "港股"}.get(m, m)
                            if r.get("skipped"):
                                rows.append({"市场": name, "本趟目标": 0, "成功": 0, "失败": 0,
                                             "剩余待建": 0, "状态": f"跳过：{r.get('reason','')}"})
                            else:
                                rows.append({"市场": name, "本趟目标": r.get("target", 0),
                                             "成功": r.get("warmed", 0), "失败": r.get("failed", 0),
                                             "剩余待建": r.get("pending_left", 0),
                                             "状态": "✅ 已全部建库" if r.get("done") else "⏳ 后续自推进"})
                        if rows:
                            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
                    done = mw.get("done")
                    verdict("所选市场已全部建库" if done else "本趟完成，剩余标的将随后续调度继续建库", "ok")
                st.session_state.pop("market_warm_task_id", None)
                st.session_state.pop("market_warm_started", None)
                break
            elif ts in ("error", "cancelled"):
                sp.empty()
                st.error(status.get("message") or "任务异常")
                st.session_state.pop("market_warm_task_id", None)
                st.session_state.pop("market_warm_started", None)
                break
            time.sleep(1)

st.markdown("---")

# ---------------------------------------------------------------- 本地文件清单
section("本地数据文件（行情仓库）")
if st.button("🔄 刷新文件清单"):
    st.cache_data.clear()
    st.session_state.pop("qm_local_cache", None)

# 优先展示行情仓库（data_cache）——期货数据下载的实际落盘位置
# 几千只标的下不再逐标的铺开：只给 KPI 聚合 + 跳转，逐标的明细到「行情仓库总览」下钻。
if "qm_local_cache" not in st.session_state:
    st.session_state.qm_local_cache = APIClient.cache_stats(timeout=30)
cache_res = st.session_state.qm_local_cache
if isinstance(cache_res, dict) and not cache_res.get("error") and cache_res.get("enabled"):
    total = cache_res.get("files", 0)
    rows_total = cache_res.get("rows", 0)
    last = cache_res.get("last_datetime") or "—"
    _agg = cache_res.get("agg") or {}
    _fr = _agg.get("freshness") or {}
    _n_stale = _fr.get("stale_1_3d", 0) + _fr.get("stale_gt3d", 0)
    kpi_row([
        {"label": "行情仓库文件数", "value": total, "tone": "accent"},
        {"label": "总 K 线数", "value": fmt_num(rows_total), "tone": "accent"},
        {"label": "最新数据", "value": last[:10], "tone": "up"},
        {"label": "落后标的", "value": f"{_n_stale}", "tone": "danger" if _n_stale else "neutral"},
    ])
    _by_market = {}
    for b in _agg.get("by_exchange") or []:
        m = b.get("market") or "其他"
        mm = _by_market.setdefault(m, [0, 0])
        mm[0] += b.get("symbols", 0)
        mm[1] += b.get("rows", 0)
    if _by_market:
        st.markdown("**按市场聚合**")
        st.dataframe(pd.DataFrame([
            {"市场": m, "标的数": cnt, "K线数": fmt_num(r, 0)}
            for m, (cnt, r) in sorted(_by_market.items(), key=lambda x: -x[1][1])
        ]), width="stretch", hide_index=True)
    st.caption("💡 逐标的覆盖区间 / 数据量 Top / 落后清单，见左侧「行情仓库总览」页的聚合与下钻。")
else:
    st.info("行情仓库为空，请先使用上方「期货数据下载」拉取数据。")