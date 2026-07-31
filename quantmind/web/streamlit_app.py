"""QuantMind Web 前端（MVP = Streamlit）。

通过 REST 调用 FastAPI 后端（quantmind.api），覆盖：行情浏览、AI 研究、因子评估、
回测/模拟/实盘（切路线）、生命周期晋升、实时监控。
热更新：``streamlit run --server.runOnSave true``。
"""
from __future__ import annotations

import os
import httpx
import streamlit as st

API_URL = os.getenv("QM_API_URL", "http://localhost:8000").rstrip("/")


def _api(path: str, **kw):
    try:
        with httpx.Client(timeout=30) as c:
            if kw.get("method") == "POST":
                r = c.post(f"{API_URL}{path}", json=kw.get("json", {}))
            else:
                r = c.get(f"{API_URL}{path}", params=kw.get("params"))
        return r.json()
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def main():
    st.set_page_config(page_title="QuantMind", layout="wide")
    st.title("QuantMind · AI 驱动量化投研")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["行情", "AI 研究", "因子", "回测/模拟/实盘", "生命周期", "监控"]
    )

    with tab1:
        st.header("行情浏览")
        sym = st.text_input("合约", "rb0")
        exch = st.selectbox("交易所", ["SHFE", "CFFEX", "DCE", "CZCE", "INE", "SSE", "SZSE", "HKEX"])
        if st.button("查询", key="data"):
            data = _api("/data", params={"symbol": sym, "exchange": exch, "interval": "1d"})
            if isinstance(data, list) and data:
                st.line_chart({b["datetime"]: b["close"] for b in data})
                st.write(f"共 {len(data)} 根，末价 {data[-1]['close']:.2f}")
            else:
                st.error("无数据（请确认后端已启动）")

    with tab2:
        st.header("AI 研究")
        idea = st.text_area("投资想法", "螺纹钢期货的动量与期限结构因子组合策略")
        ac = st.text_input("资产类别(可选)", "期货")
        if st.button("研究", key="research"):
            res = _api("/research", method="POST", json={"idea": idea, "asset_class": ac})
            if "error" in res:
                st.error(res["error"])
            else:
                st.json(res)

    with tab3:
        st.header("因子评估")
        fsym = st.text_input("因子合约", "rb0", key="fsym")
        fexch = st.selectbox("因子交易所", ["SHFE", "CFFEX", "DCE"], key="fexch")
        fname = st.text_input("因子名", "momentum_20")
        fexpr = st.text_input("或表达式", "(close/ref(close,60)-1)")
        if st.button("评估", key="factor"):
            payload = {"symbol": fsym, "exchange": fexch, "factor": fname,
                       "expression": fexpr or None}
            res = _api("/factor", method="POST", json=payload)
            st.json(res)

    with tab4:
        st.header("回测 / 模拟 / 实盘（切换路线）")
        bsym = st.text_input("策略合约", "rb0", key="bsym")
        bexch = st.selectbox("策略交易所", ["SHFE", "CFFEX"], key="bexch")
        strat = st.selectbox("策略", ["multifactor", "dual_ma"])
        mode = st.selectbox("模式", ["backtest", "paper", "live"])
        gw = st.selectbox("实盘网关", ["ctp", "xtp", "ib"])
        if st.button("运行", key="run"):
            res = _api("/backtest", method="POST", json={
                "strategy": strat, "symbol": bsym, "exchange": bexch,
                "mode": mode, "gateway": gw})
            if "equity_curve" in res:
                st.line_chart({p["date"]: p["equity"] for p in res["equity_curve"]})
            st.json({k: v for k, v in res.items() if k != "equity_curve"})

    with tab5:
        st.header("生命周期晋升")
        sid = st.text_input("策略ID", "strat-001", key="sid")
        to = st.selectbox("晋升到", ["BACKTEST", "PAPER", "APPROVED", "LIVE"])
        sharpe = st.number_input("夏普", value=0.8)
        mdd = st.number_input("最大回撤", value=-0.15)
        if st.button("晋升", key="promote"):
            res = _api("/lifecycle", method="POST", json={
                "strategy_id": sid, "to": to,
                "metrics": {"sharpe": sharpe, "max_drawdown": mdd}})
            st.json(res)

    with tab6:
        st.header("实时监控")
        st.info(f"WebSocket: {API_URL}/ws —— 实时推送 bar/signal/position/trade/account/log 事件")
        st.write("启动后端后，通过 ``/ws`` 订阅即可在自定义面板接收实时事件。"
                 "本页同时提供手动下单入口：")
        osym = st.text_input("下单合约", "rb0.SHFE", key="osym")
        ovol = st.number_input("手数", value=1)
        if st.button("手动下单", key="order"):
            res = _api("/order", method="POST", json={
                "vt_symbol": osym, "direction": "多", "offset": "开",
                "volume": ovol, "price": 0.0})
            st.json(res)


if __name__ == "__main__":
    main()
